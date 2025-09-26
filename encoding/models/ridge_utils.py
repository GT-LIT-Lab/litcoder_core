import numpy as np
import torch
from typing import Union, Tuple


def z_score(x: Union[np.ndarray, torch.Tensor], dim=0, eps=1e-8):
    """Z-score a tensor/array along the specified dimension with numerical stability"""
    if isinstance(x, np.ndarray):
        mean = x.mean(axis=dim, keepdims=True)
        std = x.std(axis=dim, keepdims=True)
        return (x - mean) / (std + eps)
    else:  # torch.Tensor
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + eps)


def mult_diag(
    d: Union[np.ndarray, torch.Tensor], mtx: Union[np.ndarray, torch.Tensor], left=True
):
    """Efficient diagonal matrix multiplication"""
    if isinstance(d, np.ndarray):
        if left:
            return (d.reshape(-1, 1) * mtx.T).T
        else:
            return d.reshape(1, -1) * mtx
    else:  # torch.Tensor
        if left:
            return (d.unsqueeze(1) * mtx.T).T
        else:
            return d.unsqueeze(0) * mtx


def svd_wrapper(X: Union[np.ndarray, torch.Tensor], singcutoff: float = 1e-10) -> Tuple:
    """Perform SVD with fallbacks and truncation of small singular values"""
    if isinstance(X, np.ndarray):
        try:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            # Could implement a fallback here if needed
            raise

        # Truncate tiny singular values
        ngoodS = np.sum(S > singcutoff)
        U = U[:, :ngoodS]
        S = S[:ngoodS]
        Vh = Vh[:ngoodS]

    else:  # torch.Tensor
        device = X.device
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except:
            # Fall back to NumPy SVD if PyTorch SVD fails
            X_np = X.cpu().numpy()
            U_np, S_np, Vh_np = np.linalg.svd(X_np, full_matrices=False)
            U = torch.from_numpy(U_np).to(device)
            S = torch.from_numpy(S_np).to(device)
            Vh = torch.from_numpy(Vh_np).to(device)

        # Truncate tiny singular values
        ngoodS = torch.sum(S > singcutoff).item()
        U = U[:, :ngoodS]
        S = S[:ngoodS]
        Vh = Vh[:ngoodS]

    return U, S, Vh


class DataNormalizer:
    """Class to handle z-score normalization using training statistics"""

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        eps: float = 1e-8,
    ):
        """Initialize the normalizer

        Args:
            normalize_features: Whether to normalize input features
            normalize_targets: Whether to normalize target variables
            eps: Small constant for numerical stability
        """
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.eps = eps
        self.feature_means = None
        self.feature_stds = None
        self.target_means = None
        self.target_stds = None

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
    ) -> "DataNormalizer":
        """Compute normalization statistics from training data

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, n_targets)

        Returns:
            self: Returns self for method chaining
        """
        if self.normalize_features:
            if isinstance(X_train, np.ndarray):
                self.feature_means = X_train.mean(axis=0, keepdims=True)
                self.feature_stds = X_train.std(axis=0, keepdims=True)
            else:  # torch.Tensor
                self.feature_means = X_train.mean(dim=0, keepdim=True)
                self.feature_stds = X_train.std(dim=0, keepdim=True)

        if self.normalize_targets:
            if isinstance(y_train, np.ndarray):
                self.target_means = y_train.mean(axis=0, keepdims=True)
                self.target_stds = y_train.std(axis=0, keepdims=True)
            else:  # torch.Tensor
                self.target_means = y_train.mean(dim=0, keepdim=True)
                self.target_stds = y_train.std(dim=0, keepdim=True)

        return self

    def transform(
        self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Normalize data using pre-computed statistics

        Args:
            X: Features to normalize (n_samples, n_features)
            y: Targets to normalize (n_samples, n_targets)

        Returns:
            Tuple of (normalized_X, normalized_y)
        """
        if self.normalize_features and (
            self.feature_means is None or self.feature_stds is None
        ):
            raise ValueError(
                "Must call fit() before transform() when normalizing features"
            )
        if self.normalize_targets and (
            self.target_means is None or self.target_stds is None
        ):
            raise ValueError(
                "Must call fit() before transform() when normalizing targets"
            )

        X_norm = X
        y_norm = y

        if self.normalize_features:
            if isinstance(X, np.ndarray):
                X_norm = (X - self.feature_means) / (self.feature_stds + self.eps)
            else:  # torch.Tensor
                X_norm = (X - self.feature_means) / (self.feature_stds + self.eps)

        if self.normalize_targets:
            if isinstance(y, np.ndarray):
                y_norm = (y - self.target_means) / (self.target_stds + self.eps)
            else:  # torch.Tensor
                y_norm = (y - self.target_means) / (self.target_stds + self.eps)

        return X_norm, y_norm

    def fit_transform(
        self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Compute normalization statistics and transform data

        Args:
            X: Features to normalize (n_samples, n_features)
            y: Targets to normalize (n_samples, n_targets)

        Returns:
            Tuple of (normalized_X, normalized_y)
        """
        return self.fit(X, y).transform(X, y)
