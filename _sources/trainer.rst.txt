Trainer
=======

``encoding.trainer.AbstractTrainer`` orchestrates the full pipeline:

- Extract and downsample features per story
- Apply FIR delays
- Structure data (concat or train/test split)
- Fit a model and log/save results

Configuration
-------------

Key constructor arguments:

- ``assembly``: provides stories, brain/ timing arrays
- ``feature_extractors``: list of extractors from the factory
- ``downsampler``: aligns to TRs
- ``model``: must implement ``fit_predict``
- ``fir_delays``: list of sample delays
- ``trimming_config``: indices for trimming
- ``use_train_test_split``: bool, Lebel vs concatenated
- ``dataset_type``: e.g., ``narratives``, used for caching keys
- ``logger_backend``: ``wandb`` or ``tensorboard``

Outputs
-------

Returns a metrics dict with common fields: ``median_score``, ``mean_score``, ``std_score`` and optionally ``correlations``, ``significant_mask``. 