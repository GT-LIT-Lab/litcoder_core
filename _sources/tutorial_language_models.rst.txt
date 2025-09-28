Language Model Features Tutorial
================================

This tutorial shows how to train encoding models using language model features with the LeBel assembly. Language model features capture rich semantic representations from transformer models.

Overview
--------

Language model features extract high-dimensional representations from transformer models like GPT-2. These features capture semantic, syntactic, and contextual information that can be highly predictive of brain activity.

Key Components
--------------

- **Assembly**: Pre-packaged LeBel assembly containing brain data and stimuli
- **Feature Extractor**: LanguageModelFeatureExtractor using transformer models
- **Caching**: Multi-layer activation caching for efficient training
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

2. **Create Language Model Feature Extractor**

   .. code-block:: python

      from encoding.features.factory import FeatureExtractorFactory
      
      extractor = FeatureExtractorFactory.create_extractor(
          modality="language_model",
          model_name="gpt2-small",  # Can be changed to other models
          config={
              "model_name": "gpt2-small",
              "layer_idx": 9,  # Layer to extract features from
              "last_token": True,  # Use last token only
              "lookback": 256,  # Context lookback
              "context_type": "fullcontext",
          },
          cache_dir="cache_language_model",
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
          wandb_project_name="lebel-language-model",
          dataset_type="lebel",
          results_dir="results",
          layer_idx=9,  # Pass layer_idx to trainer
          lookback=256,  # Pass lookback to trainer
      )
      
      metrics = trainer.train()
      print(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")

Understanding Language Model Features
-------------------------------------

Language model features are extracted by:

1. **Text Processing**: Each stimulus text is tokenized and processed
2. **Transformer Forward Pass**: The model processes the text through all layers
3. **Feature Extraction**: Features are extracted from the specified layer
4. **Caching**: Multi-layer activations are cached for efficiency
5. **Downsampling**: Features are aligned with brain data timing

Key Parameters
--------------

- **modality**: "language_model" - specifies the feature type
- **model_name**: "gpt2-small" - transformer model to use
- **layer_idx**: 9 - which layer to extract features from
- **last_token**: True - use only the last token's features (we recommend using this)
- **lookback**: 256 - context window size
- **context_type**: "fullcontext" - how to handle context
- **cache_dir**: "cache_language_model" - directory for caching

Model Options
-------------

Supported models include:
- **gpt2-small**: Fast, good baseline
- **gpt2-medium**: Better performance, slower
- **facebook/opt-125m**: Alternative architecture
- **Other TransformerLens models**: Any compatible model from `TransformerLens model properties table <https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html>`_


Caching System
--------------

The language model extractor uses a sophisticated caching system:

1. **Multi-layer caching**: All layers are cached together
2. **Lazy loading**: Layers are loaded on-demand
3. **Efficient storage**: Compressed storage of activations
4. **Cache validation**: Ensures cached data matches parameters

This makes it efficient to experiment with different layers without recomputing features.

Training Configuration
----------------------

- **fir_delays**: [1, 2, 3, 4] - temporal delays for hemodynamic response
- **trimming_config**: LeBel-specific trimming to avoid boundary effects
- **layer_idx**: 9 - which layer to use for training
- **lookback**: 256 - context window size

