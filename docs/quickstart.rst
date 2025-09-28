Quickstart
==========

This minimal example shows how to train an encoding model using the LeBel assembly with word rate features. This is the simplest and fastest way to get started with LITcoder.

.. code-block:: python

   from encoding.assembly.assembly_loader import load_assembly
   from encoding.features.factory import FeatureExtractorFactory
   from encoding.downsample.downsampling import Downsampler
   from encoding.models.nested_cv import NestedCVModel
   from encoding.trainer import AbstractTrainer

   # 1) Load prepackaged assembly
   assembly_path = "assembly_lebel_uts03.pkl"
   assembly = load_assembly(assembly_path)

   # 2) Configure components (wordrate-only)
   extractor = FeatureExtractorFactory.create_extractor(
       modality="wordrate",
       model_name="wordrate",
       config={},
       cache_dir="cache",
   )

   downsampler = Downsampler()
   model = NestedCVModel(model_name="ridge_regression")

   # FIR, downsampling, and trimming match our LeBel defaults
   fir_delays = [1, 2, 3, 4]
   trimming_config = {
       "train_features_start": 10, "train_features_end": -5,
       "train_targets_start": 0,  "train_targets_end": None,
       "test_features_start": 50,  "test_features_end": -5,
       "test_targets_start": 40,   "test_targets_end": None,
   }

   downsample_config = {}

   # 3) Train
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
   print({
       "median_correlation": metrics.get("median_score", float("nan")),
       "n_significant": metrics.get("n_significant"),
   })

Prerequisites
-------------

Before running this example, you need to:

1. **Download the LeBel assembly**:

   .. code-block:: bash

      gdown 1q-XLPjvhd8doGFhYBmeOkcenS9Y59x64

2. **Install LITcoder**:

   .. code-block:: bash

      git clone git@github.com:GT-LIT-Lab/litcoder_core.git
      cd litcoder_core
      conda create -n litcoder -y python=3.12.8
      conda activate litcoder
      conda install pip
      pip install -e .

