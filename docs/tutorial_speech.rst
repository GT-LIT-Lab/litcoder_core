Speech Features Tutorial
========================

This tutorial shows how to train encoding models using speech features with the LeBel assembly. Speech features extract representations from audio using speech models.

Overview
--------

Speech features capture acoustic and linguistic information from audio stimuli using speech recognition models like Whisper or HuBERT. These features can be highly predictive of brain activity during audio-based experiments.

Key Components
--------------

- **Assembly**: Pre-packaged LeBel assembly containing brain data and audio paths
- **Feature Extractor**: SpeechFeatureExtractor using speech recognition models
- **Audio Processing**: Chunking and resampling of audio files
- **Caching**: Multi-layer activation caching for efficient training
- **Downsampler**: Aligns audio-level features with brain data timing
- **Model**: Ridge regression with nested cross-validation
- **Trainer**: AbstractTrainer orchestrates the entire pipeline

Step-by-Step Tutorial
---------------------

1. **Load the Assembly**

   .. code-block:: python

      from encoding.assembly.assembly_loader import load_assembly
      
      # Load the pre-packaged LeBel assembly
      assembly = load_assembly("assembly_lebel_uts03.pkl")

2. **Set Up Audio Paths**

   .. code-block:: python

      import os
      
      # Set up audio paths for speech model
      base_audio_path = "/path/to/your/audio/files"  # Replace with your audio path
      
      for story_name in assembly.stories:
          # Assuming audio files are named like: story_name.wav
          audio_file_path = os.path.join(base_audio_path, f"{story_name}.wav")
          
          # Set the audio path for this story
          if hasattr(assembly, "story_data") and story_name in assembly.story_data:
              assembly.story_data[story_name].audio_path = audio_file_path
              print(f"Set audio path for {story_name}: {audio_file_path}")

3. **Create Speech Feature Extractor**

   .. code-block:: python

      from encoding.features.factory import FeatureExtractorFactory
      
      extractor = FeatureExtractorFactory.create_extractor(
          modality="speech",
          model_name="openai/whisper-tiny",  # Can be changed to other models
          config={
              "model_name": "openai/whisper-tiny",
              "chunk_size": 0.1,  # seconds between chunk starts (stride)
              "context_size": 16.0,  # seconds of audio per window
              "layer": 3,  # Layer index to extract features from
              "pool": "last",  # Pooling method: 'last' or 'mean'
              "target_sample_rate": 16000,  # Target sample rate for audio
              "device": "cuda",  # Can be "cuda", "cpu"
          },
          cache_dir="cache_speech",
      )

4. **Set Up Downsampler and Model**

   .. code-block:: python

      from encoding.downsample.downsampling import Downsampler
      from encoding.models.nested_cv import NestedCVModel
      
      downsampler = Downsampler()
      model = NestedCVModel(model_name="ridge_regression")

5. **Configure Training Parameters**

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

6. **Create and Run Trainer**

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
          wandb_project_name="lebel-speech-model",
          dataset_type="lebel",
          results_dir="results",
          layer_idx=3,  # Pass layer_idx to trainer
      )
      
      metrics = trainer.train()
      print(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")


You usually would not need to change the wav path for the assembly if you generate your own assembly. But since the assembly is already generated, we need to set the wav path for each story(more detailed tutorial on this coming soon!)

Understanding Speech Features
-----------------------------

Speech features are extracted by:

1. **Audio Loading**: Audio files are loaded and resampled to target sample rate
2. **Chunking**: Audio is divided into overlapping chunks for processing
3. **Model Forward Pass**: Each chunk is processed through the speech model
4. **Feature Extraction**: Features are extracted from the specified layer
5. **Pooling**: Features are pooled across time (last token or mean)
6. **Caching**: Multi-layer activations are cached for efficiency
7. **Downsampling**: Features are aligned with brain data timing

Key Parameters
--------------

- **modality**: "speech" - specifies the feature type
- **model_name**: "openai/whisper-tiny" - speech model to use
- **chunk_size**: 0.1 - seconds between chunk starts (stride)
- **context_size**: 16.0 - seconds of audio per window
- **layer**: 3 - which layer to extract features from
- **pool**: "last" - pooling method ('last' or 'mean')
- **target_sample_rate**: 16000 - target sample rate for audio
- **device**: "cuda" - device to run the model on
- **cache_dir**: "cache_speech" - directory for caching


Caching System
--------------

The speech extractor uses a sophisticated caching system:

1. **Multi-layer caching**: All layers are cached together
2. **Lazy loading**: Layers are loaded on-demand
3. **Efficient storage**: Compressed storage of activations
4. **Cache validation**: Ensures cached data matches parameters

This makes it efficient to experiment with different layers without recomputing features.

Training Configuration
----------------------

- **fir_delays**: [1, 2, 3, 4] - temporal delays for hemodynamic response
- **trimming_config**: LeBel-specific trimming to avoid boundary effects
- **layer_idx**: 3 - which layer to use for training

