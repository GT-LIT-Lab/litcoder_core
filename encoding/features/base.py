from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np



class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors.

    This class defines the interface that all feature extractor implementations must follow.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature extractor with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing extractor parameters
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def extract_features(self, stimuli: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Extract features from the input stimuli.

        Args:
            stimuli (Union[str, List[str]]): Input stimuli (text or list of texts)
            **kwargs: Additional arguments for feature extraction

        Returns:
            np.ndarray: Extracted features
        """
        pass

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        pass
