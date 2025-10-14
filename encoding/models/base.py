from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import numpy as np


class BasePredictivityModel(ABC):
    """Abstract base class for all predictivity models.

    This class defines the interface that all predictivity model implementations must follow.
    It provides common functionality and ensures consistent behavior across different models.
    """

    def __init__(self, model_name: str):
        """Initialize the predictivity model with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters
        """
        self.model_name = model_name

    @abstractmethod
    def fit_predict(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Fit the predictivity model to the data.

        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target matrix (brain responses)
            groups (Optional[np.ndarray]): Group labels for cross-validation
            **kwargs: Additional arguments for model fitting

        Returns:
            Dict[str, float]: Dictionary containing performance metrics
        """
        pass
