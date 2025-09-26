#!/usr/bin/env python3
"""
Simple end-to-end Lebel training script using AbstractTrainer.
Shows how to use the abstract trainer architecture for any dataset.
"""

import argparse
import logging

from encoding.assembly.assembly_generator import AssemblyGenerator
from encoding.features.factory import FeatureExtractorFactory
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel

from encoding.trainer import AbstractTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Simple Lebel training using AbstractTrainer."""
    args = parse_args()
    
    logger.info("Setting up components...")
    
    # 1. Create assembly (data source)
    assembly = AssemblyGenerator.generate_assembly(
        dataset_type="lebel",
        data_dir=args.data_dir,
        subject=args.subject,
        tr=args.tr,
        lookback=args.lookback,
        context_type=args.context_type,
        use_volume=args.use_volume,
    )
    
    # 2. Create feature extractors
    feature_extractors = []
    for modality, model_name in zip(args.modalities, args.model_names):
        feature_config = {}
        if modality == "language_model":
            feature_config = {
                "model_name": model_name,
                "layer_idx": args.layer_idx,
                "last_token": args.last_token,
            }
        elif modality == "wordrate":
            feature_config = {}
        elif modality == "embeddings":
            feature_config = {
                "vector_path": args.vector_path,
                "binary": args.binary,
                "lowercase": args.lowercase,
            }
        
        extractor = FeatureExtractorFactory.create_extractor(
            modality=modality,
            model_name=model_name,
            config=feature_config,
            cache_dir=args.cache_dir,
        )
        feature_extractors.append(extractor)
    
    # 3. Create other components
    downsampler = Downsampler()
    model = NestedCVModel(model_name="ridge_regression")
    
    # 4. Define configurations
    fir_delays = list(range(1, args.ndelays + 1))
    
    trimming_config = {
        "train_features_start": 10, "train_features_end": -5,
        "train_targets_start": 0, "train_targets_end": None,
        "test_features_start": 50, "test_features_end": -5,
        "test_targets_start": 40, "test_targets_end": None,
    }
    
    downsample_config = {
        "method": args.downsample_method,
        "window": args.lanczos_window,
        "cutoff_mult": args.lanczos_cutoff_mult,
    }
    
    # 5. Create AbstractTrainer
    trainer = AbstractTrainer(
        assembly=assembly,
        feature_extractors=feature_extractors,
        downsampler=downsampler,
        model=model,
        fir_delays=fir_delays,
        trimming_config=trimming_config,
        use_train_test_split=True,  # Lebel uses train/test split
        logger_backend=args.logger_backend,
        wandb_project_name=args.wandb_project_name,
        dataset_type="lebel",
        results_dir=args.results_dir,
        downsample_config=downsample_config,
    )
    
    # 6. Train!
    logger.info("Starting training...")
    metrics = trainer.train(
        # Pass model training parameters
        folding_type=args.folding_type,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        chunk_length=args.chunk_length,
        singcutoff=args.singcutoff,
        use_gpu=args.use_gpu,
        single_alpha=True,
        normalpha=True,
        use_corr=True,
        normalize_features=args.normalize_features,
        normalize_targets=args.normalize_targets,
    )
    
    # 7. Results
    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics['median_score']:.4f}")
    if "n_significant" in metrics:
        logger.info(f"Significant voxels: {metrics['n_significant']}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Lebel trainer using AbstractTrainer")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Lebel dataset")
    parser.add_argument("--subject", type=str, default="UTS03", help="Subject ID")
    parser.add_argument("--tr", type=float, default=2.0, help="TR value")
    parser.add_argument("--context_type", type=str, default="fullcontext", 
                       choices=["fullcontext", "nocontext", "halfcontext"])
    parser.add_argument("--lookback", type=int, default=128, help="Context lookback")
    parser.add_argument("--use_volume", action="store_true", help="Use volume data")
    
    # Feature extraction
    parser.add_argument("--modalities", type=str, nargs="+", default=["language_model"],
                       choices=["language_model", "wordrate", "embeddings"],
                       help="Feature modalities")
    parser.add_argument("--model_names", type=str, nargs="+", default=["gpt2-small"],
                       help="Model names")
    parser.add_argument("--layer_idx", type=int, default=6, help="Layer index")
    parser.add_argument("--last_token", action="store_true", help="Use last token only")
    
    # Embeddings-specific
    parser.add_argument("--vector_path", type=str, help="Path to embedding vectors")
    parser.add_argument("--binary", action="store_true", help="Binary vector file")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase tokens")
    
    # Processing
    parser.add_argument("--ndelays", type=int, default=4, help="Number of FIR delays")
    parser.add_argument("--downsample_method", type=str, default="lanczos", help="Downsampling method")
    parser.add_argument("--lanczos_window", type=int, default=3, help="Lanczos window")
    parser.add_argument("--lanczos_cutoff_mult", type=float, default=1.0, help="Lanczos cutoff")
    
    # Training
    parser.add_argument("--folding_type", type=str, default="kfold", help="CV folding type")
    parser.add_argument("--n_outer_folds", type=int, default=5, help="Outer folds")
    parser.add_argument("--n_inner_folds", type=int, default=5, help="Inner folds")
    parser.add_argument("--chunk_length", type=int, default=20, help="Chunk length")
    parser.add_argument("--singcutoff", type=float, default=1e-10, help="Singular cutoff")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features")
    parser.add_argument("--normalize_targets", action="store_true", help="Normalize targets")
    
    # System
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--logger_backend", type=str, default="wandb", 
                       choices=["wandb", "tensorboard"])
    parser.add_argument("--wandb_project_name", type=str, default="lebel-simple",
                       help="Wandb project name")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()