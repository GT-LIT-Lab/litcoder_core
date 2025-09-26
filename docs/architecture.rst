Architecture
============

Data flow
---------

1. Assembly provides stories, brain data, timestamps, and split indices
2. Feature extractors produce time series per story (text or speech)
3. Downsampler aligns features to TRs
4. FIR expansion builds delayed feature matrices
5. Trainer structures data (concat or train/test split)
6. Model fits/predicts and metrics are logged/saved

Key components
--------------

- ``encoding.assembly``: dataset abstractions and loaders
- ``encoding.features``: text/speech feature factories and caches
- ``encoding.downsample``: resampling utilities
- ``encoding.features.FIR_expander.FIR``: delay expansion
- ``encoding.models``: ridge, nested CV, sklearn wrappers
- ``encoding.trainer.AbstractTrainer``: orchestration and logging
- ``encoding.plotting``: plotting and experiment logging
- ``encoding.utils``: caches, model saver, utilities 