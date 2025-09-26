from typing import Any, Dict, Optional
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from .base import BasePredictivityModel


class LinearPredictivityModel(BasePredictivityModel):
    """Linear predictivity model for predicting brain responses from features.

    This model uses linear regression to predict brain responses from stimulus features.
    It supports cross-validation and model persistence.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the linear predictivity model.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - n_folds (int): Number of cross-validation folds
                - output_dir (Optional[Path]): Directory to save model outputs
        """
        super().__init__(config)
        self.n_folds = config.get("n_folds", 1)
        self.output_dir = config.get("output_dir")
        self.best_model = None
        self.best_score = -np.inf
        self.scores = []
        self.models = []

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Fit the linear predictivity model to the data.

        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target matrix (brain responses)
            groups (Optional[np.ndarray]): Group labels for cross-validation
            **kwargs: Additional arguments for model fitting

        Returns:
            Dict[str, float]: Dictionary containing performance metrics
        """
        if groups is None:
            # If no groups provided, create dummy groups
            groups = np.zeros(len(features))

        group_kfold = GroupKFold(n_splits=self.n_folds)

        for fold_idx, (train_idx, test_idx) in enumerate(
            group_kfold.split(features, targets, groups=groups)
        ):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]

            # Ensure 2D arrays
            X_train = np.asarray(X_train).squeeze()
            X_test = np.asarray(X_test).squeeze()
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)

            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            test_predictions = model.predict(X_test)

            # Compute scores
            fold_scores = [
                stats.pearsonr(y_test[:, i], test_predictions[:, i])[0]
                for i in range(y_test.shape[1])
            ]
            median_score = np.median(fold_scores)

            print(
                f"Fold {fold_idx+1}/{self.n_folds} - Median score: {median_score:.3f}"
            )

            self.scores.append(fold_scores)
            self.models.append(model)

            # Update best model
            if median_score > self.best_score:
                self.best_score = median_score
                self.best_model = model

        # Compute final scores
        final_scores = np.array(self.scores).mean(axis=0)
        metrics = {
            "median_score": float(np.median(final_scores)),
            "mean_score": float(np.mean(final_scores)),
            "std_score": float(np.std(final_scores)),
            "correlations": final_scores.tolist(),
        }

        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using the best fitted model.

        Args:
            features (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Predicted brain responses
        """
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet")

        features = np.asarray(features).squeeze()
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        return self.best_model.predict(features)

    def save(self, path: Path) -> None:
        """Save the best model coefficients to disk.

        Args:
            path (Path): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "best_model_coefficients.npy", self.best_model.coef_)

        # Save intercept if it exists
        if hasattr(self.best_model, "intercept_"):
            np.save(path / "best_model_intercept.npy", self.best_model.intercept_)

    def load(self, path: Path) -> None:
        """Load model coefficients from disk.

        Args:
            path (Path): Path to load the model from
        """
        path = Path(path)
        coef_path = path / "best_model_coefficients.npy"

        if not coef_path.exists():
            raise FileNotFoundError(f"No model found at {coef_path}")

        coef = np.load(coef_path)
        self.best_model = LinearRegression()
        self.best_model.coef_ = coef

        # Load intercept if it exists
        intercept_path = path / "best_model_intercept.npy"
        if intercept_path.exists():
            self.best_model.intercept_ = np.load(intercept_path)
