Word Rate Feature Tutorial
=========================

This tutorial shows how to train encoding models using word rate features with the LeBel assembly. Word rate features are simple but effective baselines that measure the rate of word presentation.

Overview
--------

Word rate features capture the temporal dynamics of language presentation by measuring how many words are presented per time unit. This is one of the simplest but an effective feature for brain encoding models.

Key Components
--------------

- **Assembly**: Pre-packaged LeBel assembly containing brain data and stimuli
- **Feature Extractor**: WordRateFeatureExtractor for computing word presentation rates
- **Downsampler**: Aligns word-level features with brain data timing
- **Model**: Ridge regression with nested cross-validation
- **Trainer**: AbstractTrainer orchestrates the entire pipeline

Step-by-Step Tutorial
---------------------

1. **Load the Assembly**

   .. code-block:: python

      from encoding.assembly.assembly_loader import load_assembly
      
      # Load the pre-packaged LeBel assembly
      assembly = load_assembly("assembly_lebel_uts03.pkl")

2. **Create Word Rate Feature Extractor**

   .. code-block:: python

      from encoding.features.factory import FeatureExtractorFactory
      
      extractor = FeatureExtractorFactory.create_extractor(
          modality="wordrate",
          model_name="wordrate",
          config={},
          cache_dir="cache",
      )

3. **Set Up Downsampler and Model**

   .. code-block:: python

      from encoding.downsample.downsampling import Downsampler
      from encoding.models.nested_cv import NestedCVModel
      
      downsampler = Downsampler()
      model = NestedCVModel(model_name="ridge_regression")

4. **Configure Training Parameters**

   .. code-block:: python

      # FIR delays for hemodynamic response modeling
      fir_delays = [1, 2, 3, 4]
      
      # Trimming configuration for LeBel dataset
      trimming_config = {
          "train_features_start": 10,
          "train_features_end": -5,
          "train_targets_start": 0,
          "train_targets_end": None,
          "test_features_start": 50,
          "test_features_end": -5,
          "test_targets_start": 40,
          "test_targets_end": None,
      }
      
      downsample_config = {}

5. **Create and Run Trainer**

   .. code-block:: python

      from encoding.trainer import AbstractTrainer
      
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
      
      metrics = trainer.train()
      print(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")

Understanding Word Rate Features
--------------------------------

Word rate features are computed by:

1. **Counting words per TR**: The assembly pre-computes word rates for each TR
2. **No additional processing needed**: Word rates are already aligned with brain data
3. **Simple but effective**: Captures temporal dynamics of language presentation

The word rate extractor simply returns the pre-computed word rates from the assembly, making it the fastest feature type to compute.

Key Parameters
--------------

- **modality**: "wordrate" - specifies the feature type
- **model_name**: "wordrate" - identifier for the extractor
- **config**: {} - no additional configuration needed
- **cache_dir**: "cache" - directory for caching (though word rates don't need caching)

Training Configuration
----------------------

- **fir_delays**: [1, 2, 3, 4] - temporal delays to account for hemodynamic response
- **trimming_config**: LeBel-specific trimming to avoid boundary effects



Word rate features provide an excellent foundation for understanding the LITcoder pipeline before moving to more complex feature types.
