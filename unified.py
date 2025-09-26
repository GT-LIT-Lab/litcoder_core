#!/usr/bin/env python3
"""
Unified trainer that works across datasets.
"""

import argparse
import logging
from typing import Dict, List, Union, Any, Optional
import numpy as np
from datetime import datetime

from encoding.assembly.assembly_generator import AssemblyGenerator
from encoding.features.factory import FeatureExtractorFactory
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.utils import ActivationCache, ModelSaver, zs
from encoding.features.FIR_expander import FIR
from encoding.plotting.plotting_utils import (
    BrainPlotter,
    TensorBoardLogger,
    WandBLogger,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """A unified trainer that works across LPP, Lebel, and Narratives datasets."""
    
    # Dataset-specific configurations
    DATASET_CONFIGS = {
        "lpp": {
            "use_train_test_split": False,
            "trimming": {
                "features_start": 5, "features_end": -5,
                "targets_start": 5, "targets_end": -5,
            }
        },
        "lebel": {
            "use_train_test_split": True,
            "trimming": {
                "train_features_start": 10, "train_features_end": -5,
                "train_targets_start": 0, "train_targets_end": None,
                "test_features_start": 50, "test_features_end": -5,
                "test_targets_start": 40, "test_targets_end": None,
            }
        },
        "narratives": {
            "use_train_test_split": False,
            "trimming": {
                "features_start": 14, "features_end": -9,
                "targets_start": 14, "targets_end": -9,
            }
        }
    }

    def __init__(self, config: Dict):
        """Initialize the trainer with configuration parameters."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get dataset-specific config
        self.dataset_config = self.DATASET_CONFIGS[config["dataset_type"]]
        
        self.setup_logger()
        self.setup_assembly()
        self.setup_models()
        self.activation_cache = ActivationCache(cache_dir=self.config["cache_dir"])
        self.model_saver = ModelSaver(
            base_dir=self.config.get("results_dir", "results")
        )

    def setup_logger(self):
        """Initialize experiment logger (wandb or tensorboard)."""
        backend = self.config.get("logger_backend", "wandb").lower()
        run_name = f"{self.config['dataset_type']}-{self.config['subject']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if backend == "wandb":
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "wandb selected as logger_backend but not installed. Install with: pip install wandb"
                ) from e
            project_name = self.config.get("wandb_project_name", "lit-encoding")
            wandb.init(project=project_name, config=self.config, name=run_name)
            self.experiment_logger = WandBLogger()
        elif backend == "tensorboard":
            run_dir = f"{self.config.get('results_dir', 'results')}/runs/{run_name}"
            self.experiment_logger = TensorBoardLogger(log_dir=run_dir)
        else:
            raise ValueError(
                f"Unsupported logger_backend '{backend}'. Use 'wandb' or 'tensorboard'."
            )
        self.brain_plotter = BrainPlotter(self.experiment_logger)

    def setup_assembly(self):
        """Initialize the assembly based on dataset type."""
        self.assembly = AssemblyGenerator.generate_assembly(
            dataset_type=self.config["dataset_type"],
            data_dir=self.config["data_dir"],
            subject=self.config["subject"],
            tr=self.config["tr"],
            lookback=self.config["lookback"],
            context_type=self.config["context_type"],
            use_volume=self.config["use_volume"],
        )
        self.logger.info(f"Assembly loaded with {len(self.assembly.stories)} stories")
        self.logger.info(f"Dataset: {self.config['dataset_type']}")
        self.logger.info(f"Using context type: {self.config['context_type']}")

    def setup_models(self):
        """Initialize feature extractors and other models."""
        # Handle multiple modalities
        modalities = self.config.get("modalities", [self.config.get("modality")])
        model_names = self.config.get("model_names", [self.config.get("model_name")])
        
        # Ensure we have matching lists
        if len(model_names) == 1 and len(modalities) > 1:
            model_names = model_names * len(modalities)
        elif len(model_names) != len(modalities):
            raise ValueError(f"Number of model_names ({len(model_names)}) must match modalities ({len(modalities)})")
        
        self.feature_extractors = []
        
        for modality, model_name in zip(modalities, model_names):
            # Prepare feature extractor config for each modality
            feature_config = {}
            if modality == "language_model":
                feature_config = {
                    "model_name": model_name,
                    "layer_idx": self.config["layer_idx"],
                    "last_token": self.config["last_token"],
                    "lookback": self.config["lookback"],
                }
            elif modality == "speech":
                feature_config = {
                    "chunk_size": self.config.get("chunk_size", 0.1),
                    "context_size": self.config.get("context_size", 16.0),
                    "layer": self.config["layer_idx"],
                    "pool": "last",
                    "target_sample_rate": 16000,
                    "device": "cpu",
                }
            elif modality == "embeddings":
                feature_config = {
                    "vector_path": self.config.get("vector_path"),
                    "binary": self.config.get("binary", True),
                    "lowercase": self.config.get("lowercase", False),
                    "oov_handling": "copy_prev",
                    "use_tqdm": True,
                }
            elif modality == "wordrate":
                feature_config = {}

            # Create feature extractor using factory
            extractor = FeatureExtractorFactory.create_extractor(
                modality=modality,
                model_name=model_name,
                config=feature_config,
                cache_dir=self.config["cache_dir"],
            )
            self.feature_extractors.append(extractor)
        
        self.logger.info(f"Created {len(self.feature_extractors)} feature extractors: {modalities}")
        
        self.downsampler = Downsampler()
        self.model = NestedCVModel(model_name="ridge_regression")

    def prepare_data(self) -> Dict[str, np.ndarray]:
        """Universal data preparation method that works across all datasets."""
        
        # Determine which stories to process
        if self.config["dataset_type"] == "lpp" and self.config.get("story_idx"):
            # Single story for LPP
            story_idx = self.config["story_idx"] - 1
            stories_to_process = [self.assembly.stories[story_idx]]
        else:
            # All stories for other datasets
            stories_to_process = self.assembly.stories
        
        # Step 1: Extract and downsample features for all stories (multiple extractors)
        all_downsampled_features = {}
        brain_data = {}
        
        for story in stories_to_process:
            idx = self.assembly.stories.index(story)
            story_features = []
            
            # Extract features from each extractor
            for extractor in self.feature_extractors:
                features = FeatureExtractorFactory.extract_features_with_caching(
                    extractor, self.assembly, story, idx, self.config['layer_idx'] ,self.config['lookback'], self.config['dataset_type']
                )

                is_wordrate = hasattr(extractor, '__class__') and 'wordrate' in extractor.__class__.__name__.lower()
                if is_wordrate:
                    downsampled = features

                
                elif isinstance(features, tuple):
                    features, times = features
                    tr_times = self.assembly.get_tr_times()[idx]
                    split_indices = self.assembly.get_split_indices()[idx]
                    
                    downsampled = self.downsampler.downsample(
                        data=features,
                        data_times=times,
                        tr_times=tr_times,
                        method=self.config["downsample_method"],
                        window=self.config["lanczos_window"],
                        cutoff_mult=self.config["lanczos_cutoff_mult"],
                        split_indices=split_indices,
                    )
                else:
                    # Text-based features
                    split_indices = self.assembly.get_split_indices()[idx]
                    data_times = self.assembly.get_data_times()[idx]
                    tr_times = self.assembly.get_tr_times()[idx]
                    
                    downsampled = self.downsampler.downsample(
                        data=features,
                        data_times=data_times,
                        tr_times=tr_times,
                        method=self.config["downsample_method"],
                        split_indices=(
                            split_indices
                            if any(method in self.config["downsample_method"] 
                                   for method in ["average", "sum", "last"])
                            else None
                        ),
                        window=self.config["lanczos_window"],
                        cutoff_mult=self.config["lanczos_cutoff_mult"],
                    )
                
                story_features.append(downsampled)
            
            # Concatenate features from all extractors
            if len(story_features) > 1:
                min_length = min(feat.shape[0] for feat in story_features)
                story_features = [feat[:min_length] for feat in story_features]
                combined_features = np.concatenate(story_features, axis=1)
            else:
                combined_features = story_features[0]
            
            all_downsampled_features[story] = combined_features
            brain_data[story] = self.assembly.get_brain_data()[idx]
        
        # Step 2: Apply FIR delays
        delays = range(1, self.config["ndelays"] + 1)
        delayed_features = {}
        for story in stories_to_process:
            delayed_features[story] = FIR.make_delayed(all_downsampled_features[story], delays)
        
        # Step 3: Handle dataset-specific data structuring with flexible trimming
        trimming = self.dataset_config["trimming"]
        
        if self.dataset_config["use_train_test_split"]:
            # Lebel-style: separate train/test stories
            train_stories = stories_to_process[:-1]
            test_stories = stories_to_process[-1:]
            
            # Training data with flexible trimming
            train_feat_start = trimming.get("train_features_start", 0)
            train_feat_end = trimming.get("train_features_end", None)
            train_targ_start = trimming.get("train_targets_start", 0) 
            train_targ_end = trimming.get("train_targets_end", None)

            print(train_feat_start, train_feat_end, train_targ_start, train_targ_end)
            
            X_train = np.nan_to_num(np.vstack([
                zs(delayed_features[story][train_feat_start:train_feat_end]) 
                for story in train_stories
            ]))
            Y_train = np.vstack([
                zs(brain_data[story][train_targ_start:train_targ_end]) 
                for story in train_stories
            ])
            print(X_train.shape, Y_train.shape)

            
            # Test data with flexible trimming
            test_feat_start = trimming.get("test_features_start", 0)
            test_feat_end = trimming.get("test_features_end", None)
            test_targ_start = trimming.get("test_targets_start", 0)
            test_targ_end = trimming.get("test_targets_end", None)
            
            X_test = np.nan_to_num(np.vstack([
                zs(delayed_features[story][test_feat_start:test_feat_end]) 
                for story in test_stories
            ]))
            Y_test = np.vstack([
                zs(brain_data[story][test_targ_start:test_targ_end]) 
                for story in test_stories
            ])
            print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
            
            return {
                "Rstim": X_train, "Rresp": Y_train,
                "Pstim": X_test, "Presp": Y_test
            }
        else:
            # LPP/Narratives style: concatenate and trim
            story_order = self.config.get("story_order", stories_to_process)
            
            X = np.concatenate([delayed_features[story] for story in story_order], axis=0)
            Y = np.concatenate([brain_data[story] for story in story_order], axis=0)
            
            # Apply flexible trimming
            feat_start = trimming.get("features_start", 0)
            feat_end = trimming.get("features_end", None)
            targ_start = trimming.get("targets_start", 0)
            targ_end = trimming.get("targets_end", None)
            
            X = X[feat_start:feat_end]
            Y = Y[targ_start:targ_end]
            
            return {"X": X, "Y": Y}

    def train(self) -> Dict[str, Any]:
        """Run the training process."""
        try:
            # Prepare data using universal method
            data = self.prepare_data()
            
            # Train based on data structure
            if "Rstim" in data:
                # Train/test split (Lebel)
                metrics, weights, best_alphas = self.model.fit_predict(
                    features=data["Rstim"],
                    targets=data["Rresp"],
                    X_test=data["Pstim"],
                    y_test=data["Presp"],
                    groups=self.assembly.get_coord("stimulus_id"),
                    folding_type=self.config["folding_type"],
                    n_outer_folds=self.config["n_outer_folds"],
                    n_inner_folds=self.config["n_inner_folds"],
                    chunk_length=self.config["chunk_length"],
                    singcutoff=self.config["singcutoff"],
                    use_gpu=self.config["use_gpu"],
                    single_alpha=True,
                    normalpha=True,
                    use_corr=True,
                    normalize_features=self.config["normalize_features"],
                    normalize_targets=self.config["normalize_targets"],
                )
            else:
                # Cross-validation (LPP/Narratives)
                metrics, weights, best_alphas = self.model.fit_predict(
                    features=data["X"],
                    targets=data["Y"],
                    folding_type=self.config["folding_type"],
                    n_outer_folds=self.config["n_outer_folds"],
                    n_inner_folds=self.config["n_inner_folds"],
                    chunk_length=self.config["chunk_length"],
                    singcutoff=self.config["singcutoff"],
                    use_gpu=self.config["use_gpu"],
                    single_alpha=True,
                    normalpha=True,
                    use_corr=True,
                    normalize_features=self.config["normalize_features"],
                    normalize_targets=self.config["normalize_targets"],
                )

            # Log and save
            self.log_metrics(metrics)
            
            hyperparams = {
                **self.config,
                "single_alpha": True,
                "normalpha": True, 
                "use_corr": True,
            }

            self.model_saver.save_encoding_model(
                weights=weights,
                best_alphas=best_alphas,
                hyperparams=hyperparams,
                metrics=metrics,
            )

            # Print results
            self.logger.info("\nTraining Results:")
            self.logger.info(f"Median correlation: {metrics['median_score']:.3f}")
            self.logger.info(f"Significant voxels: {metrics['n_significant']}/{len(metrics['correlations'])}")
            
            return metrics

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def log_metrics(self, metrics: Dict[str, Union[float, List[float]]]):
        """Log metrics to the configured backend."""
        # Scalar summaries
        self.experiment_logger.log_scalar("median_correlation", float(metrics["median_score"]))
        self.experiment_logger.log_scalar("mean_correlation", float(metrics["mean_score"]))
        self.experiment_logger.log_scalar("std_correlation", float(metrics["std_score"]))
        self.experiment_logger.log_scalar("min_correlation", float(metrics["min_score"]))
        self.experiment_logger.log_scalar("max_correlation", float(metrics["max_score"]))

        # Brain plots
        if "correlations" in metrics and "significant_mask" in metrics:
            correlations = np.array(metrics["correlations"])
            significant_mask = np.array(metrics["significant_mask"], dtype=bool)

            self.brain_plotter.log_plots(
                correlations=correlations,
                significant_mask=significant_mask,
                prefix="",
                is_volume=self.config["use_volume"],
            )

        # Additional metrics
        if "best_alpha" in metrics:
            self.experiment_logger.log_scalar("best_alpha", float(metrics["best_alpha"]))
        if "n_significant" in metrics:
            self.experiment_logger.log_scalar("n_significant_voxels", float(metrics["n_significant"]))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified trainer for encoding models")

    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, required=True, 
                       choices=["lpp", "lebel", "narratives"], 
                       help="Dataset type")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to dataset directory")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--tr", type=float, required=True, help="TR value")
    parser.add_argument("--context_type", type=str, default="fullcontext",
                       choices=["fullcontext", "nocontext", "halfcontext"],
                       help="Context window type")
    parser.add_argument("--use_volume", action="store_true", help="Use volume data")
    
    # LPP-specific
    parser.add_argument("--story_idx", type=int, help="Story index for LPP (1-based)")

    # Modality and model parameters
    parser.add_argument("--modality", type=str, help="Single modality (for backward compatibility)")
    parser.add_argument("--modalities", type=str, nargs="+", 
                       help="Multiple modalities (e.g., --modalities language_model wordrate)")
    parser.add_argument("--model_name", type=str, help="Single model name (for backward compatibility)")
    parser.add_argument("--model_names", type=str, nargs="+",
                       help="Multiple model names (e.g., --model_names gpt2-small word2vec)")
    parser.add_argument("--layer_idx", type=int, default=9, help="Layer index")
    parser.add_argument("--last_token", action="store_true", help="Use last token only")

    # Training parameters
    parser.add_argument("--n_outer_folds", type=int, default=5, help="Outer CV folds")
    parser.add_argument("--n_inner_folds", type=int, default=5, help="Inner CV folds")
    parser.add_argument("--folding_type", type=str, default="chunked", help="CV folding type")
    parser.add_argument("--chunk_length", type=int, default=20, help="Chunk length")
    parser.add_argument("--singcutoff", type=float, default=1e-10, help="Singular value cutoff")

    # Preprocessing parameters
    parser.add_argument("--downsample_method", type=str, default="lanczos", help="Downsampling method")
    parser.add_argument("--lanczos_cutoff_mult", type=float, default=1.0, help="Lanczos cutoff multiplier")
    parser.add_argument("--lanczos_window", type=int, default=3, help="Lanczos window")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features")
    parser.add_argument("--normalize_targets", action="store_true", help="Normalize targets")
    parser.add_argument("--ndelays", type=int, required=True, help="Number of FIR delays")
    parser.add_argument("--lookback", type=int, required=True, help="Context lookback")

    # System parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory") 
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")

    # Logging
    parser.add_argument("--logger_backend", type=str, default="wandb",
                       choices=["wandb", "tensorboard"], help="Logging backend")
    parser.add_argument("--wandb_project_name", type=str, default="lit-encoding",
                       help="Wandb project name")

    # Modality-specific parameters
    parser.add_argument("--vector_path", type=str, help="Vector file path (embeddings)")
    parser.add_argument("--binary", action="store_true", help="Binary vectors (embeddings)")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase (embeddings)")
    parser.add_argument("--chunk_size", type=float, default=0.1, help="Chunk size (speech)")
    parser.add_argument("--context_size", type=float, default=16.0, help="Context size (speech)")
    parser.add_argument("--story_order", type=str, nargs="+", help="Story processing order")
    
    # Custom trimming parameters (override dataset defaults)
    parser.add_argument("--features_start", type=int, help="Features trim start")
    parser.add_argument("--features_end", type=int, help="Features trim end") 
    parser.add_argument("--targets_start", type=int, help="Targets trim start")
    parser.add_argument("--targets_end", type=int, help="Targets trim end")
    parser.add_argument("--train_features_start", type=int, help="Train features trim start")
    parser.add_argument("--train_features_end", type=int, help="Train features trim end")
    parser.add_argument("--train_targets_start", type=int, help="Train targets trim start") 
    parser.add_argument("--train_targets_end", type=int, help="Train targets trim end")
    parser.add_argument("--test_features_start", type=int, help="Test features trim start")
    parser.add_argument("--test_features_end", type=int, help="Test features trim end")
    parser.add_argument("--test_targets_start", type=int, help="Test targets trim start")
    parser.add_argument("--test_targets_end", type=int, help="Test targets trim end")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    config = vars(args)
    
    if not config.get("modalities") and not config.get("modality"):
        raise ValueError("Must specify either --modality or --modalities")
    if not config.get("model_names") and not config.get("model_name"):
        raise ValueError("Must specify either --model_name or --model_names")
    
    if config.get("modality") and not config.get("modalities"):
        config["modalities"] = [config["modality"]]
    if config.get("model_name") and not config.get("model_names"):
        config["model_names"] = [config["model_name"]]
    
    custom_trimming = {}
    trimming_params = [
        "features_start", "features_end", "targets_start", "targets_end",
        "train_features_start", "train_features_end", "train_targets_start", "train_targets_end",
        "test_features_start", "test_features_end", "test_targets_start", "test_targets_end"
    ]
    for param in trimming_params:
        if config.get(param) is not None:
            custom_trimming[param] = config[param]
    
    if custom_trimming:
        # Update dataset config with custom trimming
        UnifiedTrainer.DATASET_CONFIGS[config["dataset_type"]]["trimming"].update(custom_trimming)
        logger.info(f"Using custom trimming parameters: {custom_trimming}")

    logger.info(f"Starting training for {config['dataset_type']} dataset")
    logger.info(f"Subject: {config['subject']}")
    logger.info(f"TR: {config['tr']}")
    logger.info(f"Modalities: {config['modalities']}")
    logger.info(f"Models: {config['model_names']}")
    logger.info(f"N delays: {config['ndelays']}")
    logger.info(f"Lookback: {config['lookback']}")
    
    # Initialize and train
    trainer = UnifiedTrainer(config)
    metrics = trainer.train()
    
    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics['median_score']:.4f}")
    if "n_significant" in metrics:
        logger.info(f"Significant voxels: {metrics['n_significant']}")


if __name__ == "__main__":
    main()