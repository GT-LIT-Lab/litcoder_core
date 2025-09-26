from typing import Any, Dict, Optional
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
import warnings

from .base import BasePredictivityModel


class SklearnPredictivityModel(BasePredictivityModel):
    """Flexible predictivity model using scikit-learn regression models.

    This model can use any scikit-learn regression model (LinearRegression, Ridge, Lasso, etc.)
    and supports hyperparameter tuning via GridSearchCV. It can handle both grouped and
    non-grouped cross-validation.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the scikit-learn predictivity model.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model_type (str): Type of regression model ('linear', 'ridge', 'lasso', etc.)
                - model_kwargs (Dict): Keyword arguments for the model constructor
                - n_folds (int): Number of cross-validation folds
                - use_groups (bool): Whether to use group-based cross-validation
                - param_grid (Dict): Parameter grid for hyperparameter tuning (optional)
                - inner_cv (int): Number of inner CV folds for hyperparameter tuning (optional)
                - output_dir (Path): Directory to save model outputs (optional)
                - scoring (str): Scoring metric for hyperparameter tuning (default: 'r2')
        """
        super().__init__(config)

        # Model configuration
        self.model_type = config.get("model_type", "linear")
        self.model_kwargs = config.get("model_kwargs", {})
        self.estimator = self._get_estimator()

        # Cross-validation configuration
        self.n_folds = config.get("n_folds", 5)
        self.use_groups = config.get("use_groups", True)

        # Hyperparameter tuning configuration
        self.param_grid = config.get("param_grid", None)
        self.inner_cv = config.get("inner_cv", 3)
        self.scoring = config.get("scoring", "r2")

        # Output configuration
        self.output_dir = config.get("output_dir", None)

        # Results storage
        self.best_model = None
        self.best_score = -np.inf
        self.models = []
        self.scores = []
        self.feature_importances_ = None

    def _get_estimator(self) -> BaseEstimator:
        """Get the scikit-learn estimator based on model_type."""
        model_map = {"linear": LinearRegression, "ridge": Ridge, "lasso": Lasso}

        if self.model_type in model_map:
            return model_map[self.model_type](**self.model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(
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
        # Handle groups based on configuration
        use_groups = self.use_groups and groups is not None

        if use_groups:
            if groups is None:
                warnings.warn(
                    "Group-based CV requested but no groups provided. Using default groups."
                )
                groups = np.zeros(len(features))
            cv = GroupKFold(n_splits=self.n_folds)
            fold_split = cv.split(features, targets, groups=groups)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            fold_split = cv.split(features, targets)
        print("these are the groups", groups)
        # Store results for each fold
        fold_scores = []
        fold_models = []
        best_model = None
        best_score = -np.inf

        # Run cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(fold_split):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]

            # Ensure proper dimensions
            X_train = np.asarray(X_train).squeeze()
            X_test = np.asarray(X_test).squeeze()
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            # print the shape of the train and test sets
            print("the shape of the train set", X_train.shape)
            print("the shape of the test set", X_test.shape)
            # Hyperparameter tuning if param_grid is provided
            if self.param_grid is not None:
                print(
                    f"Fold {fold_idx+1}/{self.n_folds}: Running hyperparameter tuning..."
                )
                grid_search = GridSearchCV(
                    self._get_estimator(),
                    param_grid=self.param_grid,
                    cv=self.inner_cv,
                    scoring=self.scoring,
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                model = self._get_estimator()
                model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = model.predict(X_test)

            # Compute correlations for each feature
            correlations = []
            for i in range(y_test.shape[1]):
                corr = stats.pearsonr(y_test[:, i], y_pred[:, i])[0]
                if not np.isnan(corr):  # Skip NaN correlations
                    correlations.append(corr)

            median_corr = np.median(correlations)
            print(
                f"Fold {fold_idx+1}/{self.n_folds} - Median correlation: {median_corr:.3f}"
            )

            fold_scores.append(correlations)
            fold_models.append(model)

            # Update best model
            if median_corr > best_score:
                best_score = median_corr
                best_model = model

        # Store results
        self.scores = fold_scores
        self.models = fold_models
        self.best_model = best_model
        self.best_score = best_score

        # Extract feature importances if available
        if hasattr(best_model, "coef_"):
            self.feature_importances_ = best_model.coef_

        # Save best model if output directory is provided
        if self.output_dir is not None:
            self.save(Path(self.output_dir))

        # Compute average performance metrics across folds
        all_correlations = np.concatenate(fold_scores)
        metrics = {
            "median_score": float(np.median(all_correlations)),
            "mean_score": float(np.mean(all_correlations)),
            "std_score": float(np.std(all_correlations)),
            "min_score": float(np.min(all_correlations)),
            "max_score": float(np.max(all_correlations)),
            "best_fold_score": float(best_score),
            "correlations": all_correlations.tolist(),
        }

        # Add hyperparameters of the best model to the metrics
        if self.best_model is not None:
            if hasattr(self.best_model, "get_params"):
                best_params = self.best_model.get_params()
                # Print the best hyperparameters for visibility
                print("\nBest model hyperparameters:")
                for param_name, param_value in best_params.items():
                    if param_name in [
                        "alpha",
                        "fit_intercept",
                        "max_iter",
                        "tol",
                        "solver",
                    ]:
                        print(f"  {param_name}: {param_value}")

                # Store in metrics
                metrics["best_model_params"] = {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in best_params.items()
                }

                # Specifically extract alpha for ridge/lasso models
                if "alpha" in best_params:
                    metrics["alpha"] = float(best_params["alpha"])
                    print(f"  Selected alpha: {best_params['alpha']}")

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

        model_info = {
            "model_type": self.model_type,
            "model_kwargs": self.model_kwargs,
            "best_score": self.best_score,
        }

        # Save model info
        np.save(path / "model_info.npy", model_info)

        # Save coefficients if available
        if hasattr(self.best_model, "coef_"):
            np.save(path / "best_model_coefficients.npy", self.best_model.coef_)

        # Save intercept if available
        if hasattr(self.best_model, "intercept_"):
            np.save(path / "best_model_intercept.npy", self.best_model.intercept_)

    def load(self, path: Path) -> None:
        """Load model from disk.

        Args:
            path (Path): Path to load the model from
        """
        path = Path(path)

        # Load model info
        info_path = path / "model_info.npy"
        if not info_path.exists():
            raise FileNotFoundError(f"No model info found at {info_path}")

        model_info = np.load(info_path, allow_pickle=True).item()
        self.model_type = model_info["model_type"]
        self.model_kwargs = model_info["model_kwargs"]
        self.best_score = model_info["best_score"]

        # Initialize model
        self.best_model = self._get_estimator()

        # Load coefficients if available
        coef_path = path / "best_model_coefficients.npy"
        if coef_path.exists():
            self.best_model.coef_ = np.load(coef_path)
            self.feature_importances_ = self.best_model.coef_

        # Load intercept if available
        intercept_path = path / "best_model_intercept.npy"
        if intercept_path.exists():
            self.best_model.intercept_ = np.load(intercept_path)
