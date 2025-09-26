Caching
=======

Language model activations
--------------------------

- ``encoding.utils.ActivationCache`` persists multi-layer activations to a single ``.pkl`` per config
- ``encoding.utils.LazyLayerCache`` loads metadata fast and layers on demand
- Cache keys include story, model name, context type, lookback, dataset_type, etc.

Speech activations
------------------

- ``encoding.utils.SpeechActivationCache`` and ``SpeechLazyLayerCache`` mirror the API for speech features with optional time arrays

Model artifacts
---------------

- ``encoding.utils.ModelSaver`` records hyperparameters and metrics per run directory (hash + timestamp)
- Use ``list_runs()`` to enumerate previous runs 