
"""
Abstract trainer that accepts components as dependencies.
"""

import logging
from typing import Dict, List, Union, Any, Optional
import numpy as np
from datetime import datetime

from encoding.utils import ModelSaver, zs
from encoding.features.FIR_expander import FIR
from encoding.plotting.plotting_utils import BrainPlotter, TensorBoardLogger, WandBLogger

logger = logging.getLogger(__name__)


class AbstractTrainer:
    """
    A completely abstract trainer that accepts all components as dependencies.
    
    This trainer doesn't know about datasets, assemblies, or specific feature types.
    It just orchestrates the pipeline: extract → downsample → FIR → trim → train.
    """
    
    def __init__(
        self,
        assembly: Any,                    # Data source
        feature_extractors: List[Any],    # List of feature extractors
        downsampler: Any,                 # Downsampling component
        model: Any,                       # Training model
        fir_delays: List[int],            # FIR delay parameters
        trimming_config: Dict,            # Trimming configuration
        use_train_test_split: bool = False,  # Data structuring mode
        # Feature extraction parameters
        layer_idx: int = 9,
        lookback: int = 256,
        dataset_type: str = "unknown",
        # Logging parameters
        logger_backend: str = "wandb",
        wandb_project_name: str = "abstract-trainer",
        wandb_mode: Optional[str] = None,
        results_dir: str = "results",
        run_name: Optional[str] = None,
        # Processing parameters
        downsample_config: Optional[Dict] = None,
        story_selection: Optional[List[str]] = None,
    ):
        """
        Initialize with all components as dependencies.
        
        Args:
            assembly: Data assembly (has .stories, .get_brain_data(), etc.)
            feature_extractors: List of feature extraction components
            downsampler: Downsampling component
            model: Model with fit_predict() method
            fir_delays: List of FIR delays to apply
            trimming_config: Dict specifying how to trim data
            use_train_test_split: Whether to use train/test split vs concatenation
            layer_idx: Layer index for feature extraction
            lookback: Context lookback for feature extraction
            dataset_type: Dataset type for caching
            logger_backend: "wandb" or "tensorboard"
            wandb_project_name: Project name for wandb
            wandb_mode: Mode for wandb ('online' or 'offline'). 
            results_dir: Directory for results
            run_name: Custom run name
            downsample_config: Downsampling parameters
            story_selection: Specific stories to process (None = all)
        """
        self.assembly = assembly
        self.feature_extractors = feature_extractors
        self.downsampler = downsampler
        self.model = model
        self.fir_delays = fir_delays
        self.trimming_config = trimming_config
        self.use_train_test_split = use_train_test_split
        self.downsample_config = downsample_config or {}
        
        # Feature extraction parameters
        self.layer_idx = layer_idx
        self.lookback = lookback
        self.dataset_type = dataset_type
        
        # Story selection
        if story_selection is None:
            self.stories_to_process = self.assembly.stories
        elif isinstance(story_selection, int):
            # Single story index (1-based)
            self.stories_to_process = [self.assembly.stories[story_selection - 1]]
        else:
            # List of story names
            self.stories_to_process = story_selection
        
        # Setup logging
        self.setup_logger(logger_backend, wandb_project_name, wandb_mode, results_dir, run_name)
        self.model_saver = ModelSaver(base_dir=results_dir)
        self.brain_plotter = BrainPlotter(self.experiment_logger)
        
        logger.info(f"Abstract trainer initialized")
        logger.info(f"Feature extractors: {len(self.feature_extractors)}")
        logger.info(f"Stories to process: {len(self.stories_to_process)}")
        logger.info(f"Layer idx: {self.layer_idx}")
        logger.info(f"Lookback: {self.lookback}")
        logger.info(f"Dataset type: {self.dataset_type}")
        logger.info(f"FIR delays: {self.fir_delays}")
        logger.info(f"Use train/test split: {self.use_train_test_split}")
    
    def setup_logger(self, backend: str, project_name: str, wandb_mode: Optional[str], results_dir: str, run_name: Optional[str]):
        """Setup experiment logger.

        Args: 
            backend (str): logger backend to use ('wandb' or 'tensorboard')
            project_name (str): project name
            wandb_mode (str): Mode for wandb ('online' or 'offline'). 
                    If None, reads from WANDB_MODE env var (defaults to 'offline')
            results_dir (str): Directory to store the results
            run_name (str): custom run_name
        
        """
        if run_name is None:
            run_name = f"abstract-trainer-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if backend == "wandb":
            try:
                import wandb
                import os
            
                if wandb_mode is None:
                    wandb_mode = os.environ.get('WANDB_MODE', 'offline')
                
                os.environ['WANDB_MODE'] = wandb_mode
                

                if wandb_mode == "offline":
                    wandb_dir = f'{results_dir}/wandb'
                    tmpdir = os.environ.get('TMPDIR', results_dir)
                    wandb_cache_dir = f'{tmpdir}/wandb_cache'
                    os.makedirs(wandb_dir, exist_ok=True)
                    os.makedirs(wandb_cache_dir, exist_ok=True)
                    os.environ.setdefault('WANDB_DIR', wandb_dir)
                    os.environ.setdefault('WANDB_CACHE_DIR', wandb_cache_dir)

                wandb.init(project=project_name, name=run_name,start_method="thread")
                self.experiment_logger = WandBLogger()

            except ImportError as e:
                raise ImportError("wandb not installed. Install with: pip install wandb") from e
            
        elif backend == "tensorboard":
            run_dir = f"{results_dir}/runs/{run_name}"
            self.experiment_logger = TensorBoardLogger(log_dir=run_dir)
        else:
            raise ValueError(f"Unsupported logger_backend '{backend}'")
    
    def extract_and_downsample_features(self) -> Dict[str, np.ndarray]:
        """Extract and downsample features for all stories."""
        all_features = {}
        
        for story in self.stories_to_process:
            idx = self.assembly.stories.index(story)
            story_features = []
            
            # Extract features from each extractor
            for extractor in self.feature_extractors:
                features = self._extract_single_features(extractor, story, idx)
                
                # Downsample if needed
                if self._should_downsample(extractor):
                    downsampled = self._downsample_features(features, idx)
                else:
                    downsampled = features
                
                story_features.append(downsampled)
            
            # Concatenate features from multiple extractors
            if len(story_features) > 1:
                # Align timepoints
                min_length = min(feat.shape[0] for feat in story_features)
                story_features = [feat[:min_length] for feat in story_features]
                combined = np.concatenate(story_features, axis=1)
            else:
                combined = story_features[0]
            
            all_features[story] = combined
            logger.info(f"Story {story}: feature shape {combined.shape}")
        
        return all_features
    
    def _extract_single_features(self, extractor, story: str, idx: int):
        """Extract features from a single extractor."""
        from encoding.features.factory import FeatureExtractorFactory
        
        # Use the factory's caching method
        return FeatureExtractorFactory.extract_features_with_caching(
            extractor, self.assembly, story, idx, self.layer_idx, self.lookback, self.dataset_type
        )
    
    def _should_downsample(self, extractor) -> bool:
        """Determine if this extractor needs downsampling."""
        # Simple heuristic: wordrate doesn't need downsampling
        extractor_name = extractor.__class__.__name__.lower()
        return 'wordrate' not in extractor_name
    
    def _downsample_features(self, features, story_idx: int):
        """Downsample features for a story."""
        if isinstance(features, tuple):
            # Speech features
            features, times = features
            tr_times = self.assembly.get_tr_times()[story_idx]
            split_indices = self.assembly.get_split_indices()[story_idx]
            
            return self.downsampler.downsample(
                data=features,
                data_times=times,
                tr_times=tr_times,
                split_indices=split_indices,
                **self.downsample_config
            )
        else:
            # Text-based features
            split_indices = self.assembly.get_split_indices()[story_idx]
            data_times = self.assembly.get_data_times()[story_idx]
            tr_times = self.assembly.get_tr_times()[story_idx]
            
            return self.downsampler.downsample(
                data=features,
                data_times=data_times,
                tr_times=tr_times,
                split_indices=split_indices,
                **self.downsample_config
            )
    
    def apply_fir_delays(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply FIR delays to features."""
        delayed_features = {}
        for story, feat in features.items():
            delayed_features[story] = FIR.make_delayed(feat, self.fir_delays)
            logger.info(f"Story {story}: delayed feature shape {delayed_features[story].shape}")
        return delayed_features
    
    def structure_data(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Structure data according to training paradigm."""
        brain_data = {}
        for story in self.stories_to_process:
            idx = self.assembly.stories.index(story)
            brain_data[story] = self.assembly.get_brain_data()[idx]
        
        if self.use_train_test_split:
            return self._create_train_test_split(features, brain_data)
        else:
            return self._create_concatenated_data(features, brain_data)
    
    def _create_train_test_split(self, features: Dict, brain_data: Dict) -> Dict[str, np.ndarray]:
        """Create train/test split (Lebel style)."""
        stories = list(features.keys())
        train_stories = stories[:-1]
        test_stories = stories[-1:]
        
        # Training data
        train_feat_start = self.trimming_config.get("train_features_start", 0)
        train_feat_end = self.trimming_config.get("train_features_end", None)
        train_targ_start = self.trimming_config.get("train_targets_start", 0)
        train_targ_end = self.trimming_config.get("train_targets_end", None)
        
        X_train = np.nan_to_num(np.vstack([
            zs(features[story][train_feat_start:train_feat_end])
            for story in train_stories
        ]))
        Y_train = np.vstack([
            zs(brain_data[story][train_targ_start:train_targ_end])
            for story in train_stories
        ])
        
        # Test data
        test_feat_start = self.trimming_config.get("test_features_start", 0)
        test_feat_end = self.trimming_config.get("test_features_end", None)
        test_targ_start = self.trimming_config.get("test_targets_start", 0)
        test_targ_end = self.trimming_config.get("test_targets_end", None)
        
        X_test = np.nan_to_num(np.vstack([
            zs(features[story][test_feat_start:test_feat_end])
            for story in test_stories
        ]))
        Y_test = np.vstack([
            zs(brain_data[story][test_targ_start:test_targ_end])
            for story in test_stories
        ])
        
        logger.info(f"Train: X{X_train.shape}, Y{Y_train.shape}")
        logger.info(f"Test: X{X_test.shape}, Y{Y_test.shape}")
        
        return {"Rstim": X_train, "Rresp": Y_train, "Pstim": X_test, "Presp": Y_test}
    
    def _create_concatenated_data(self, features: Dict, brain_data: Dict) -> Dict[str, np.ndarray]:
        """Create concatenated data (LPP/Narratives style)."""
        story_order = self.stories_to_process
        
        X = np.concatenate([features[story] for story in story_order], axis=0)
        Y = np.concatenate([brain_data[story] for story in story_order], axis=0)
        
        # Apply trimming
        feat_start = self.trimming_config.get("features_start", 0)
        feat_end = self.trimming_config.get("features_end", None)
        targ_start = self.trimming_config.get("targets_start", 0)
        targ_end = self.trimming_config.get("targets_end", None)
        
        X = X[feat_start:feat_end]
        Y = Y[targ_start:targ_end]
        
        logger.info(f"Final: X{X.shape}, Y{Y.shape}")
        
        return {"X": X, "Y": Y}
    
    def train(self, **model_kwargs) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        # Step 1: Extract and downsample features
        features = self.extract_and_downsample_features()
        
        # Step 2: Apply FIR delays
        delayed_features = self.apply_fir_delays(features)
        
        # Step 3: Structure data for training
        data = self.structure_data(delayed_features)
        
        # Step 4: Train model
        logger.info("Starting model training...")
        if "Rstim" in data:
            # Train/test split
            metrics, weights, best_alphas = self.model.fit_predict(
                features=data["Rstim"],
                targets=data["Rresp"],
                X_test=data["Pstim"],
                y_test=data["Presp"],
                **model_kwargs
            )
        else:
            # Cross-validation
            metrics, weights, best_alphas = self.model.fit_predict(
                features=data["X"],
                targets=data["Y"],
                **model_kwargs
            )
        
        # Step 5: Log and save results
        self.log_metrics(metrics)
        self.save_model(weights, best_alphas, metrics, model_kwargs)
        
        logger.info(f"Training complete. Median correlation: {metrics['median_score']:.4f}")
        
        return metrics
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to configured backend."""
        self.experiment_logger.log_scalar("median_correlation", float(metrics["median_score"]))
        self.experiment_logger.log_scalar("mean_correlation", float(metrics["mean_score"]))
        self.experiment_logger.log_scalar("std_correlation", float(metrics["std_score"]))
        
        if "correlations" in metrics and "significant_mask" in metrics:
            correlations = np.array(metrics["correlations"])
            significant_mask = np.array(metrics["significant_mask"], dtype=bool)
            self.brain_plotter.log_plots(correlations = correlations, 
                                         significant_mask=significant_mask, 
                                         prefix="", 
                                         is_volume=self.assembly.is_volume)
        
        if "best_alpha" in metrics:
            self.experiment_logger.log_scalar("best_alpha", float(metrics["best_alpha"]))
        if "n_significant" in metrics:
            self.experiment_logger.log_scalar("n_significant_voxels", float(metrics["n_significant"]))
    
    def save_model(self, weights, best_alphas, metrics, model_kwargs):
        """Save model results."""
        hyperparams = {
            "fir_delays": self.fir_delays,
            "trimming_config": self.trimming_config,
            "use_train_test_split": self.use_train_test_split,
            "downsample_config": self.downsample_config,
            "layer_idx": self.layer_idx,
            "lookback": self.lookback,
            "dataset_type": self.dataset_type,
            "stories_processed": len(self.stories_to_process),
            **model_kwargs
        }
        
        self.model_saver.save_encoding_model(
            weights=weights,
            best_alphas=best_alphas,
            hyperparams=hyperparams,
            metrics=metrics,
        )

