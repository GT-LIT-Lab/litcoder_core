Understanding Assemblies in LITcoder
=====================================

An **Assembly** is the core data structure in LITcoder that organizes and manages brain imaging data, stimuli, and metadata for encoding model training. It's the foundation that everything else builds upon.

What is an Assembly?
-------------------

An assembly is a structured container that holds all the data needed to train encoding models:

- **Brain Data**: Recordings aligned with stimuli
- **Stimuli**: Text or audio stimuli presented during the experiment  
- **Timing Information**: Precise timing of when each stimulus was presented
- **Split Indices**: Maps each word/stimulus to its corresponding TR (time repetition)
- **Metadata**: Story names, subject information, and experimental parameters

Think of an assembly as a well-organized database that contains everything needed to train a brain encoding model.

Assembly Structure
-----------------

An assembly contains several key components:

**Stories**: List of story/run names
    Each story represents a continuous experimental session (e.g., listening to a story)

**Story Data**: Dictionary mapping story names to their data
    Contains brain data, stimuli, timing, and metadata for each story

**Timing Information**: 
    - `tr_times`: When each TR (time repetition) occurred
    - `data_times`: Precise timing for each data point (word-level)
    - `split_indices`: Maps each word to its corresponding TR

**Brain Data**: 
    - Preprocessed fMRI data aligned with stimuli
    - Shape: (n_timepoints, n_voxels/vertices)

Working with Assemblies
-----------------------

Let's explore how to work with assemblies using the LeBel assembly:

.. code-block:: python

    from encoding.assembly.assembly_loader import load_assembly
    
    # Load the pre-packaged LeBel assembly
    assembly = load_assembly("assembly_lebel_uts03.pkl")
    
    # Basic information
    print(f"Assembly shape: {assembly.shape}")
    print(f"Stories: {assembly.stories}")
    print(f"Validation method: {assembly.get_validation_method()}")

Key Assembly Methods
-------------------

Here are the most important methods for working with assemblies:

**Data Access**:
- `get_stimuli()`: Get text stimuli for each story
- `get_brain_data()`: Get brain data for each story  
- `get_split_indices()`: Get word-to-TR mapping
- `get_tr_times()`: Get TR timing information
- `get_data_times()`: Get precise word-level timing

**Story-Specific Data**:
- `get_temporal_baseline(story_name)`: Get temporal baseline features
- `get_audio_path()`: Get audio file paths (for speech models)
- `get_words()`: Get individual words for each story
- `get_word_rates()`: Get pre-computed word rates

**Metadata**:
- `get_validation_method()`: Get validation strategy ("inner" or "outer")
- `stories`: List of story names
- `story_data`: Dictionary of story-specific data

Exploring Assembly Contents
---------------------------

Let's examine what's inside an assembly:

.. code-block:: python

    # Load assembly
    assembly = load_assembly("assembly_lebel_uts03.pkl")
    
    # Basic information
    print("=== Assembly Overview ===")
    print(f"Total presentations: {assembly.shape[0]}")
    print(f"Number of voxels/vertices: {assembly.shape[1]}")
    print(f"Stories: {assembly.stories}")
    print(f"Validation method: {assembly.get_validation_method()}")
    
    # Explore each story
    print("\n=== Story Details ===")
    for story in assembly.stories:
        story_data = assembly.story_data[story]
        print(f"\nStory: {story}")
        print(f"  Brain data shape: {story_data.brain_data.shape}")
        print(f"  Number of stimuli: {len(story_data.stimuli)}")
        print(f"  Split indices: {len(story_data.split_indices)} words")
        print(f"  TR times: {len(story_data.tr_times)} TRs")
        print(f"  Data times: {len(story_data.data_times)} words")
        
        # Show first few stimuli
        print(f"  First 3 stimuli: {story_data.stimuli[:3]}")
        
        # Show split indices (these map words to TRs)
        print(f"  First 10 split indices: {story_data.split_indices[:10]}")
        print(f"  Last 10 split indices: {story_data.split_indices[-10:]}")

Understanding the Data Flow
---------------------------

Here's how data flows through an assembly:

1. **Stimuli Extraction**: Text is processed into features (embeddings, word rates, etc.)
2. **Timing Alignment**: Features are aligned with brain data using timing information
3. **Downsampling**: High-resolution features are downsampled to match brain data TR
4. **FIR Delays**: Temporal delays are applied to account for hemodynamic response
5. **Train/Test Split**: Data is split for proper evaluation

Assembly Attributes
-------------------

An assembly has several key attributes:

**Shape**: (n_presentations, n_voxels/vertices)
    Total number of timepoints and brain regions

**Stories**: List of story names
    Each story represents a continuous experimental session

**Story Data**: Dictionary of story-specific data
    Contains all the data for each story

**Coordinates**: Metadata about presentations
    Story IDs, stimulus IDs, etc.

**Validation Method**: "inner" or "outer"
    How the assembly handles train/test splits

Working with Story Data
-----------------------

Each story in an assembly contains:

.. code-block:: python

    # Get data for a specific story
    story_name = assembly.stories[0]
    story_data = assembly.story_data[story_name]
    
    print(f"Story: {story_name}")
    print(f"  Brain data: {story_data.brain_data.shape}")
    print(f"  Stimuli: {len(story_data.stimuli)}")
    print(f"  Split indices: {len(story_data.split_indices)}")
    print(f"  TR times: {len(story_data.tr_times)}")
    print(f"  Data times: {len(story_data.data_times)}")
    
    # Access specific data
    brain_data = story_data.brain_data
    stimuli = story_data.stimuli
    split_indices = story_data.split_indices
    tr_times = story_data.tr_times
    data_times = story_data.data_times

Using Assemblies in Training
----------------------------

Here's how assemblies are used in the training pipeline:

.. code-block:: python

    from encoding.assembly.assembly_loader import load_assembly
    from encoding.features.factory import FeatureExtractorFactory
    from encoding.downsample.downsampling import Downsampler
    from encoding.models.nested_cv import NestedCVModel
    from encoding.trainer import AbstractTrainer
    
    # 1. Load assembly
    assembly = load_assembly("assembly_lebel_uts03.pkl")
    
    # 2. Create feature extractor
    extractor = FeatureExtractorFactory.create_extractor(
        modality="wordrate",
        model_name="wordrate",
        config={},
        cache_dir="cache",
    )
    
    # 3. Set up other components
    downsampler = Downsampler()
    model = NestedCVModel(model_name="ridge_regression")
    
    # 4. Configure training parameters
    fir_delays = [1, 2, 3, 4]
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
    
    # 5. Create trainer
    trainer = AbstractTrainer(
        assembly=assembly,
        feature_extractors=[extractor],
        downsampler=downsampler,
        model=model,
        fir_delays=fir_delays,
        trimming_config=trimming_config,
        use_train_test_split=True,
        logger_backend="wandb",
        wandb_project_name="lebel-tutorial",
        dataset_type="lebel",
        results_dir="results",
    )
    
    # 6. Train the model
    metrics = trainer.train()
    print(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")


This understanding of assemblies is crucial for effectively using LITcoder. The assembly serves as the foundation for all encoding model training, providing the structured interface between your experimental data and the machine learning pipeline.
