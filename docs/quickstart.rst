Quickstart
==========

This minimal example wires an assembly, feature extractor, downsampler, model, and the trainer.

.. code-block:: python

   from encoding.assembly.assemblies import NarrativesAssembly
   from encoding.features.factory import FeatureExtractorFactory
   from encoding.downsample.downsampling import Downsampler
   from encoding.models.ridge_regression import RidgeRegressionModel
   from encoding.trainer import AbstractTrainer

   # 1) Load data (example: Narratives-style assembly)
   assembly = NarrativesAssembly(assembly_path="/path/to/narratives.h5")

   # 2) Configure features (e.g., language model embeddings)
   extractor = FeatureExtractorFactory.create_language_model(
       model_name="gpt2", context_type="fullcontext", last_token=True
   )

   # 3) Downsampler
   downsampler = Downsampler(method="linear")

   # 4) Model
   model = RidgeRegressionModel(n_alphas=20)

   # 5) Trainer
   trainer = AbstractTrainer(
       assembly=assembly,
       feature_extractors=[extractor],
       downsampler=downsampler,
       model=model,
       fir_delays=[0, 1, 2, 3, 4],
       trimming_config={"features_start": 5, "targets_start": 5},
       use_train_test_split=False,
       dataset_type="narratives",
       logger_backend="tensorboard",
       results_dir="results",
   )

   metrics = trainer.train()
   print("Median correlation:", metrics["median_score"]) 