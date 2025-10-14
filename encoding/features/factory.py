from __future__ import annotations
from typing import Dict, Any, Union, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    import numpy as np
    from litcoder_core.encoding.features.base import BaseFeatureExtractor
    from litcoder_core.encoding.features.language_model import LanguageModelFeatureExtractor
    from litcoder_core.encoding.features.speech_model import SpeechFeatureExtractor
    from litcoder_core.encoding.features.lowlevel_features import WordRateFeatureExtractor
    from litcoder_core.encoding.features.embeddings import StaticEmbeddingFeatureExtractor



class FeatureExtractorFactory:
    """Factory class for creating feature extractors with caching support."""

    _extractors = {
        "language_model": "litcoder_core.encoding.features.language_model.LanguageModelFeatureExtractor",
        "speech": "litcoder_core.encoding.features.speech_model.SpeechFeatureExtractor",
        "wordrate": "litcoder_core.encoding.features.lowlevel_features.WordRateFeatureExtractor",
        "embeddings": "litcoder_core.encoding.features.embeddings.StaticEmbeddingFeatureExtractor",
    }

    @classmethod
    def _lazy_import_extractor(cls, modality: str):
        """Lazy import the extractor class only when needed."""
        import_path = cls._extractors[modality]
        module_path, class_name = import_path.rsplit(".", 1)
        
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def create_extractor(
        cls,
        modality: str,
        model_name: str,
        config: Dict[str, Any],
        cache_dir: str = "cache",
    ) -> BaseFeatureExtractor:
        """Create a feature extractor based on modality and model name.

        Args:
            modality: The type of feature extractor ('language_model', 'speech', 'wordrate', 'embeddings')
            model_name: The specific model name (e.g., 'gpt2-small', 'word2vec', 'openai/whisper-tiny')
            config: Configuration dictionary for the extractor
            cache_dir: Directory for caching

        Returns:
            BaseFeatureExtractor: The appropriate feature extractor instance

        Raises:
            ValueError: If modality is not supported
        """
        if modality not in cls._extractors:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Supported modalities: {list(cls._extractors.keys())}"
            )

        extractor_class = cls._lazy_import_extractor(modality)

        # Add model_name to config if not present
        if "model_name" not in config:
            config["model_name"] = model_name

        # TODO: Change later to use **config for all extractors. But for now, only speech will use **config
        # ideally, they should all use a config, and that config should be a class.
        if modality == "language_model":
            extractor = extractor_class(config)
        elif modality == "speech":
            extractor = extractor_class(**config)
        else:
            extractor = extractor_class(config)

        print(f"this is the config: {config}")

        # Add caching capability
        if modality in ["language_model", "speech"]:
            extractor.cache_dir = cache_dir
            if modality == "speech":
                from ..utils import SpeechActivationCache
                extractor.speech_cache = SpeechActivationCache(cache_dir=cache_dir)
            else:
                from ..utils import ActivationCache
                extractor.activation_cache = ActivationCache(cache_dir=cache_dir)

        return extractor

    @classmethod
    def extract_features_with_caching(
        cls,
        extractor: BaseFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int = 9,
        lookback: int = 256,
        dataset_type: str = "narratives",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Extract features with caching support.

        Args:
            extractor: The feature extractor instance
            assembly: The assembly containing data
            story: Story name
            idx: Story index
            layer_idx: Layer index for multi-layer extractors
            lookback: Number of tokens to look back (for language models)
            dataset_type: Type of dataset (e.g., 'narratives', 'lebel', etc.)

        Returns:
            Features array, or (features, times) tuple for speech
        """
        modality = cls._get_modality_from_extractor(extractor)

        if modality == "language_model":
            return cls._extract_language_model_features(
                extractor, assembly, story, idx, layer_idx, lookback, dataset_type
            )
        elif modality == "speech":
            return cls._extract_speech_features(
                extractor, assembly, story, idx, layer_idx, dataset_type
            )
        elif modality == "wordrate":
            word_rates = assembly.get_word_rates()[idx]
            return extractor.extract_features(word_rates)
        elif modality == "embeddings":
            words = assembly.get_words()[idx]
            return extractor.extract_features(words)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    @classmethod
    def _get_modality_from_extractor(cls, extractor: BaseFeatureExtractor) -> str:
        """Get modality from extractor instance."""
        extractor_type = type(extractor).__name__
        
        for modality, import_path in cls._extractors.items():
            class_name = import_path.rsplit(".", 1)[1]
            if extractor_type == class_name:
                return modality
        
        raise ValueError(f"Unknown extractor type: {type(extractor)}")

    @classmethod
    def _extract_language_model_features(
        cls,
        extractor: LanguageModelFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int,
        lookback: int = 256,
        dataset_type: str = "narratives",
    ) -> np.ndarray:
        """Extract language model features with caching."""
        texts = assembly.get_stimuli()[idx]

        # Try to load cached activations
        cache_key = extractor.activation_cache._get_cache_key(
            story=story,
            lookback=lookback,  # You can make this configurable
            model_name=extractor.model_name,
            context_type=getattr(extractor, "context_type", "fullcontext"),
            last_token=getattr(extractor, "last_token", False),
            dataset_type=dataset_type,
            raw=True,
        )
        print(f"this is the last token: {getattr(extractor, 'last_token', False)}")
        print(f"this is the lookback: {lookback}")
        print(f'this is the layer: {layer_idx}')

        lazy_cache = extractor.activation_cache.load_multi_layer_activations(cache_key)

        if lazy_cache is not None:
            return lazy_cache.get_layer(layer_idx)
        else:
            # Compute and cache features
            all_features = extractor.extract_all_layers(texts)

            # Create metadata for caching
            metadata = {
                "model_name": extractor.model_name,
                "story": story,
                "lookback": lookback,
                "context_type": getattr(extractor, "context_type", "fullcontext"),
                "hook_type": extractor.hook_type,
                "last_token": getattr(extractor, "last_token", False),
                "dataset_type": dataset_type,
                "available_layers": list(all_features.keys()),
                "created_at": datetime.now().isoformat(),
            }

            # Save to cache
            extractor.activation_cache.save_multi_layer_activations(
                cache_key, all_features, metadata
            )

            return all_features[layer_idx]

    @classmethod
    def _extract_speech_features(
        cls,
        extractor: SpeechFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int,
        dataset_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract speech features with caching."""
        wav_path = assembly.get_audio_path()[idx]

        # Try to load from cache
        cache_key = extractor.speech_cache.get_cache_key(
            audio_id=wav_path,
            model_name=extractor.model_name,
            chunk_size=extractor.chunk_size,
            context_size=extractor.context_size,
            pool=extractor.pool,
            target_sample_rate=extractor.target_sample_rate,
            dataset_type=dataset_type,
            extra={"layer_mode": "all"},
        )

        lazy = extractor.speech_cache.load_multi_layer_activations(cache_key)

        if lazy is not None:
            # Validate cached data
            lazy.validate_params(
                expected={
                    "model_name": extractor.model_name,
                    "chunk_size": extractor.chunk_size,
                    "context_size": extractor.context_size,
                    "pool": extractor.pool,
                    "target_sample_rate": extractor.target_sample_rate,
                    "dataset_type": dataset_type,
                }
            )
            features = lazy.get_layer(layer_idx)
            times = lazy.get_times()
        else:
            # Compute and cache features
            layer_to_feats, times = extractor.extract_all_layers(wav_path)
            if len(layer_to_feats) == 0:
                raise RuntimeError(
                    "extract_all_layers returned no layers (audio too short?)."
                )

            # Save to cache
            metadata = {
                "modality": "speech",
                "audio_id": wav_path,
                "model_name": extractor.model_name,
                "chunk_size": extractor.chunk_size,
                "context_size": extractor.context_size,
                "pool": extractor.pool,
                "target_sample_rate": extractor.target_sample_rate,
                "dataset_type": dataset_type,
                "available_layers": sorted(layer_to_feats.keys()),
            }

            extractor.speech_cache.save_multi_layer_activations(
                cache_key,
                all_layer_activations=layer_to_feats,
                metadata=metadata,
                times=times,
            )

            features = layer_to_feats[layer_idx]

        return features, times

    @classmethod
    def get_supported_modalities(cls) -> list:
        """Get list of supported modalities."""
        return list(cls._extractors.keys())

    @classmethod
    def register_extractor(cls, modality: str, extractor_class: type):
        """Register a new feature extractor class.

        Args:
            modality: The modality name
            extractor_class: The extractor class to register
        """
        cls._extractors[modality] = extractor_class
