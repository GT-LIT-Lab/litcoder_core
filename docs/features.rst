Features
========

Text and speech feature extractors share a factory-driven API and disk caches.

Text features
-------------

- ``encoding.features.language_model``: LM activations
- ``encoding.features.embeddings``: pooling and projection utilities
- ``encoding.features.factory.FeatureExtractorFactory``: creation + caching
- Multi-layer caching via ``encoding.utils.ActivationCache`` and ``LazyLayerCache``

Speech features
---------------

- ``encoding.features.speech_model``: speech encoders
- Caching via ``encoding.utils.SpeechActivationCache`` and ``SpeechLazyLayerCache``
- Returns (features, times) tuples for downsampling

FIR expansion
-------------

- ``encoding.features.FIR_expander.FIR``: ``make_delayed`` to build lagged design matrices

Example
-------

.. code-block:: python

   from encoding.features.factory import FeatureExtractorFactory
   extractor = FeatureExtractorFactory.create_language_model(
       model_name="gpt2", context_type="fullcontext", last_token=True
   ) 