from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

from .base import BaseFeatureExtractor

if TYPE_CHECKING:
    import numpy as np
    


class WordRateFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for pre-computed word rate features."""

    # ok so for all intents and purposes, this is not necessary, but I'm keeping it to follow,
    # a general pattern for feature extraction. don't judge me :)

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def extract_features(self, stimuli: np.ndarray, **kwargs) -> np.ndarray:
        """Return pre-computed word rate features.

        Args:
            stimuli: Pre-computed word rate array

        Returns:
            np.ndarray: Word rate features with shape (n_timepoints, 1)
        """
        if isinstance(stimuli, list):
            stimuli = np.array(stimuli)

        # Ensure it's 2D with shape (n_timepoints, 1)
        if stimuli.ndim == 1:
            stimuli = stimuli.reshape(-1, 1)
        elif stimuli.ndim == 2 and stimuli.shape[1] == 1:
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected stimuli shape: {stimuli.shape}")

        return stimuli
