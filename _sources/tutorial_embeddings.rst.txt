Static Embeddings Tutorial
==========================

This tutorial shows how to train encoding models using static word embeddings with the LeBel assembly. Static embeddings provide pre-trained word representations that can be highly predictive of brain activity.

Overview
--------

Static embeddings capture semantic relationships between words using pre-trained models like Word2Vec or GloVe. These embeddings provide rich semantic representations that can be highly predictive of brain activity.

Key Components
--------------

- **Assembly**: Pre-packaged LeBel assembly containing brain data and stimuli
- **Feature Extractor**: StaticEmbeddingFeatureExtractor using pre-trained embeddings
- **Embedding Models**: Word2Vec, GloVe, or other static embedding models
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

2. **Create Static Embedding Feature Extractor**

   .. code-block:: python

      from encoding.features.factory import FeatureExtractorFactory
      
      # You need to provide the path to your embedding file
      vector_path = "/path/to/your/embeddings.bin.gz"  # Replace with your path
      
      extractor = FeatureExtractorFactory.create_extractor(
          modality="embeddings",
          model_name="word2vec",  # Can be "word2vec", "glove", or any identifier
          config={
              "vector_path": vector_path,
              "binary": True,  # Set to True for .bin files, False for .txt files
              "lowercase": False,  # Set to True if your embeddings expect lowercase tokens
              "oov_handling": "copy_prev",  # How to handle out-of-vocabulary words
              "use_tqdm": True,  # Show progress bar
          },
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
          wandb_project_name="lebel-embeddings",
          dataset_type="lebel",
          results_dir="results",
          downsample_config=downsample_config,
      )
      
      metrics = trainer.train()
      print(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")

Understanding Static Embeddings
-------------------------------

Key Parameters
--------------

- **modality**: "embeddings" - specifies the feature type
- **model_name**: "word2vec" - identifier for the extractor
- **vector_path**: Path to the embedding file
- **binary**: True for .bin files, False for .txt files
- **lowercase**: Whether to lowercase tokens before lookup
- **oov_handling**: How to handle out-of-vocabulary words
- **use_tqdm**: Whether to show progress bar
- **cache_dir**: "cache" - directory for caching

Embedding Models
----------------

Supported embedding models include:
- **Word2Vec**: Google News vectors, custom Word2Vec models
- **GloVe**: Stanford GloVe embeddings
- **Custom embeddings**: Any compatible embedding format

File Formats
------------

Supported file formats:
- **Binary files (.bin)**: Set `binary=True`
- **Text files (.txt)**: Set `binary=False`
- **Compressed files (.gz)**: Automatically handled

OOV Handling
------------

Out-of-vocabulary (OOV) word handling strategies:
- **"copy_prev"**: Use the previous word's embedding
- **"zero"**: Use zero vector
- **"random"**: Use random vector
- **"mean"**: Use mean of all embeddings

Choose based on your research question and data characteristics.

Training Configuration
----------------------

- **fir_delays**: [1, 2, 3, 4] - temporal delays for hemodynamic response
- **trimming_config**: LeBel-specific trimming to avoid boundary effects
- **downsample_config**: {} - no additional downsampling configuration needed
