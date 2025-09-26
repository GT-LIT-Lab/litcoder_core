#!/usr/bin/env python3

import logging

from encoding.assembly.assembly_loader import load_assembly
from encoding.features.factory import FeatureExtractorFactory
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # 1) Load the packaged assembly
    assembly_path = "/storage/coda1/p-aivanova7/0/shared/litcoder_core/scripts/assembly_lebel_uts03.pkl"
    logger.info(f"Loading assembly from {assembly_path}")
    assembly = load_assembly(assembly_path)

    # 2) Create the wordrate-only feature extractor
    extractor = FeatureExtractorFactory.create_extractor(
        modality="wordrate",
        model_name="wordrate",
        config={},
        cache_dir="cache",
    )

    # 3) Set up other components
    downsampler = Downsampler()
    model = NestedCVModel(model_name="ridge_regression")

    fir_delays = [1, 2, 3, 4]
    # Correct Lebel trimming configuration (matches train_lebel.py/unified.py)
    trimming_config = {
        "train_features_start": 10, "train_features_end": -5,
        "train_targets_start": 0, "train_targets_end": None,
        "test_features_start": 50, "test_features_end": -5,
        "test_targets_start": 40, "test_targets_end": None,
    }
    downsample_config = {}

    trainer = AbstractTrainer(
        assembly=assembly,
        feature_extractors=[extractor],
        downsampler=downsampler,
        model=model,
        fir_delays=fir_delays,
        trimming_config=trimming_config,
        use_train_test_split=True,
        logger_backend="wandb",
        wandb_project_name="lebel-wordrate",
        dataset_type="lebel",
        results_dir="results",
        downsample_config=downsample_config,
    )

    logger.info("Starting training (wordrate only, no extra kwargs)...")
    metrics = trainer.train()

    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")
    if "n_significant" in metrics:
        logger.info(f"Significant voxels: {metrics['n_significant']}")


if __name__ == "__main__":
    main() 