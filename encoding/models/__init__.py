from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..features.base import BaseFeatureExtractor
    from .base import BasePredictivityModel
    from .linear import LinearPredictivityModel
    from .sklearn_model import SklearnPredictivityModel

__all__ = [
    "BaseFeatureExtractor",
    "BasePredictivityModel",
    "LinearPredictivityModel",
    "SklearnPredictivityModel",
]

def __getattr__(name: str):
    if name == "BaseFeatureExtractor":
        from ..features.base import BaseFeatureExtractor
        return BaseFeatureExtractor
    elif name == "BasePredictivityModel":
        from .base import BasePredictivityModel
        return BasePredictivityModel
    elif name == "LinearPredictivityModel":
        from .linear import LinearPredictivityModel
        return LinearPredictivityModel
    elif name == "SklearnPredictivityModel":
        from .sklearn_model import SklearnPredictivityModel
        return SklearnPredictivityModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
