from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .language_model import LanguageModelFeatureExtractor
    from .speech_model import SpeechFeatureExtractor
    from .lowlevel_features import WordRateFeatureExtractor
    from .embeddings import StaticEmbeddingFeatureExtractor
    from .FIR_expander import FIR
    from .factory import FeatureExtractorFactory

__all__ = [
    "LanguageModelFeatureExtractor",
    "SpeechFeatureExtractor",
    "WordRateFeatureExtractor",
    "StaticEmbeddingFeatureExtractor",
    "FIR",
    "FeatureExtractorFactory",
]

def __getattr__(name: str):
    """Lazy load feature extractors on first access."""
    if name == "LanguageModelFeatureExtractor":
        from .language_model import LanguageModelFeatureExtractor
        return LanguageModelFeatureExtractor
    elif name == "SpeechFeatureExtractor":
        from .speech_model import SpeechFeatureExtractor
        return SpeechFeatureExtractor
    elif name == "WordRateFeatureExtractor":
        from .lowlevel_features import WordRateFeatureExtractor
        return WordRateFeatureExtractor
    elif name == "StaticEmbeddingFeatureExtractor":
        from .embeddings import StaticEmbeddingFeatureExtractor
        return StaticEmbeddingFeatureExtractor
    elif name == "FIR":
        from .FIR_expander import FIR
        return FIR
    elif name == "FeatureExtractorFactory":
        from .factory import FeatureExtractorFactory
        return FeatureExtractorFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")