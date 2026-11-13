import numpy as np
import torch
from scipy.stats import pearsonr, combine_pvalues
from statsmodels.stats.multitest import fdrcorrection
import logging
from copy import deepcopy
from typing import Dict, Optional, List, Tuple, Union
from encoding.models.ridge_regression import ridge_torch, ridge_corr_torch
from encoding.models.folding import create_folds
from encoding.models.ridge_utils import DataNormalizer
from encoding.models.base import BasePredictivityModel
from typing import Any, Dict, List, Optional, Tuple, Union


class NestedCVModel(BasePredictivityModel):
    def __init__(self, model_name: str, *, estimator=None, param_grid=None,
                 target_mode="native"):
        super().__init__(model_name)
        self.estimator = estimator
        self.param_grid = param_grid
        self.target_mode = target_mode
        if estimator is None:
            if param_grid is not None or target_mode != "native":
                raise ValueError("param_grid and target_mode require a supplied estimator")
        else:
            from encoding.models.estimator import validate_estimator

            validate_estimator(estimator, param_grid, target_mode)

    def fit_predict(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        folding_type: str = "chunked",
        n_outer_folds: int = 5,
        n_inner_folds: int = 5,
        chunk_length: int = 20,
        alphas: Optional[List[float]] = None,
        alpha_fdr: float = 0.05,
        use_gpu: bool = True,
        single_alpha: bool = False,
        normalpha: bool = True,
        use_corr: bool = True,
        normalize_features: bool = False,
        normalize_targets: bool = False,
        singcutoff: float = 1e-10,
        trim_size: Optional[int] = None,
        result_metadata: Optional[Dict[str, Any]] = None,
        temporal_cv: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Dict[str, Union[float, List[float], List[bool]]],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Fit model with nested cross-validation, chunking, per-voxel or single alpha optimization, and FDR correction.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target matrix (n_samples, n_targets)
            X_test: Optional test features. If provided, skips outer CV
            y_test: Optional test targets. Must be provided if X_test is provided
            groups: Optional group labels for GroupKFold
            folding_type: Type of CV folding: "chunked", "kfold", "chunked_contiguous", "timeseries", "group"
            n_outer_folds: Number of outer CV folds
            n_inner_folds: Number of inner CV folds
            chunk_length: Length of chunks for respecting fMRI autocorrelation
            alphas: Ridge parameters to test (defaults to np.logspace(-1, 8, 10))
            alpha_fdr: Significance level for FDR correction
            use_gpu: Whether to use GPU acceleration if available
            single_alpha: If True, uses the same alpha for all voxels
            normalpha: Whether to normalize alpha by largest singular value
            use_corr: If True, use correlation as metric; if False, use R-squared
            normalize_features: If True, z-score normalize features using training statistics
            normalize_targets: If True, z-score normalize targets using training statistics
            singcutoff: Singularity cutoff for ridge_corr
            trim_size: Samples trimmed from each end of test chunks/folds for the
                "chunked_trimmed" and "kfold_trimmed" folding types
            result_metadata: Optional ordered target_metadata, sample_ids,
                test_sample_ids and provenance for persistence. Missing IDs are
                explicitly positional. Does not affect fitting or splitting.

        Normalization statistics are fit on training rows only, at both CV levels:
        each inner fold fits its own normalizer, and the final model refits on the
        full (outer) training set.

        Constant or non-finite targets in a test set get NaN correlation/p-value,
        are excluded from summary statistics and FDR, and are counted in
        ``n_invalid_targets``.

        Returns:
            Tuple of (metrics, weights, best_alphas)
            - metrics: Dictionary of evaluation metrics
            - weights: Model weights (n_features, n_targets)
            - best_alphas: Best alpha values for each target
            Supplied estimators return None for weights and best_alphas.
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)
        # A failed fit must not leave a previous successful result available.
        self.results_ = None

        # Set default alphas if not provided
        supplied = self.estimator is not None
        ecog = temporal_cv is not None
        diagnostic = ecog and temporal_cv["protocol"] == "diagnostic_in_sample"
        if diagnostic and len(features) < 2:
            raise ValueError("In-sample diagnostics require at least two eligible samples")
        if diagnostic and (supplied or alphas is None or len(alphas) != 1
                           or not np.isfinite(alphas[0]) or alphas[0] <= 0):
            raise ValueError("In-sample diagnostics require one fixed positive built-in ridge alpha")
        if supplied and (alphas is not None or not use_corr):
            raise ValueError("Supplied estimators use estimator parameters and Pearson selection")
        if alphas is None:
            alphas = np.logspace(-1, 8, 10)

        # Determine device - use GPU if available and requested

        if use_gpu and not supplied:
            if torch.backends.mps.is_available():
                device = "mps:0"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                # if no GPU is available, fall back to CPU
                device = "cpu"
        else:
             device = "cpu"
        logger.info(f"Using device: {device}")
        protocol = folding_type if temporal_cv is None else temporal_cv["protocol"]
        logger.info(f"Folding type: {protocol}")

        temporal_folds = None
        if temporal_cv is not None:
            from encoding.ecog_temporal import memory_preflight, temporal_splits

            if X_test is not None or y_test is not None or groups is not None:
                raise ValueError("Temporal CV requires one continuous prepared dataset")
            if len(temporal_cv["recordings"]) != len(features):
                raise ValueError("Temporal coordinates must match feature rows")
            memory_preflight(
                len(features), features.shape[1], targets.shape[1], features.dtype.itemsize,
                1 if diagnostic else n_outer_folds, len(alphas), temporal_cv["memory_budget_bytes"],
                resident_bytes=features.nbytes + targets.nbytes,
            )
            if not diagnostic:
                temporal_folds = temporal_splits(
                    temporal_cv["recordings"], temporal_cv["bounds"],
                    n_outer_folds, n_inner_folds, temporal_cv["protocol"], temporal_cv["embargo_s"],
                )

        # Convert inputs to PyTorch tensors
        features_torch = torch.tensor(features, dtype=torch.float32, device=device)
        targets_torch = torch.tensor(targets, dtype=torch.float32, device=device)

        # Check if we're in train-test mode or full CV mode
        train_test_mode = X_test is not None and y_test is not None
        metadata = deepcopy(result_metadata or {})
        metadata.setdefault("target_metadata", [
            {"target_index": i, "identity_kind": "positional"}
            for i in range(targets.shape[1])
        ])
        metadata.setdefault("sample_identity_kind", (
            "supplied" if "sample_ids" in metadata else "positional"
        ))
        metadata.setdefault("sample_ids", np.arange(len(features)))
        if len(metadata["target_metadata"]) != targets.shape[1]:
            raise ValueError("target_metadata must match target column order/count")
        if len(metadata["sample_ids"]) != len(features):
            raise ValueError("sample_ids must match training input rows")
        if train_test_mode:
            metadata.setdefault("test_sample_identity_kind", (
                "supplied" if "test_sample_ids" in metadata else "positional"
            ))
            metadata.setdefault("test_sample_ids", np.arange(len(X_test)))
            if len(metadata["test_sample_ids"]) != len(X_test):
                raise ValueError("test_sample_ids must match test input rows")
        result = {
            "schema_version": 2,
            "mode": "train_test" if train_test_mode else "nested_cv",
            "metadata": metadata,
            "configuration": {
                "model_name": self.model_name,
                "folding_type": protocol, "n_outer_folds": n_outer_folds,
                "n_inner_folds": n_inner_folds, "chunk_length": chunk_length,
                "trim_size": trim_size, "alphas": np.asarray(alphas).copy(),
                "single_alpha": single_alpha, "normalpha": normalpha,
                "use_corr": use_corr, "normalize_features": normalize_features,
                "normalize_targets": normalize_targets, "singcutoff": singcutoff,
                "alpha_fdr": alpha_fdr, "device": device,
                "dtype": "float32", "groups": deepcopy(groups),
            },
            "folds": [],
            "final_refit": None,
            "returned_weights_role": (
                "training_fit" if train_test_mode else "descriptive_fold_mean"
            ),
            "returned_alphas_role": (
                "selected_training_alphas" if train_test_mode else "descriptive_fold_mean"
            ),
            "prediction_input": "prepared feature design before fold normalization",
            "prediction_units": "model-input target units (before fold normalization)",
            "software": {"numpy": np.__version__, "torch": str(torch.__version__)},
        }

        if supplied:
            import sklearn

            result["configuration"].update(
                estimator_type=type(self.estimator).__qualname__,
                param_grid=deepcopy(self.param_grid),
                target_mode=self.target_mode,
                selection_metric="mean_inner_pearson",
                parameter_scope="shared_across_targets",
            )
            for key in ("alphas", "single_alpha", "normalpha", "singcutoff"):
                result["configuration"].pop(key)
            result["software"]["sklearn"] = sklearn.__version__
            result["returned_weights_role"] = "fitted_estimators_in_folds"
            result["returned_alphas_role"] = "not_applicable"

        def fit_fold(X_train, y_train, X_eval, y_eval, splits):
            if supplied:
                from encoding.models.estimator import fit_estimator_fold

                return fit_estimator_fold(
                    self.estimator, self.param_grid, self.target_mode,
                    X_train, y_train, X_eval, y_eval, splits,
                    normalize_features, normalize_targets,
                )

            if diagnostic:
                selected = torch.full((y_train.shape[1],), float(alphas[0]), device=device)
            else:
                selected = _find_best_alphas(
                    X_train, y_train, fold_splits=splits, alphas=alphas,
                    single_alpha=single_alpha, normalpha=normalpha, use_corr=use_corr,
                    logger=logger, singcutoff=singcutoff,
                    normalize_features=normalize_features, normalize_targets=normalize_targets,
                )
            valid = _valid_training_targets(y_train)
            normalizer = None
            if normalize_features or normalize_targets:
                normalizer = DataNormalizer(normalize_features, normalize_targets)
                X_train, y_train = normalizer.fit_transform(X_train, y_train)
                X_eval, y_eval = normalizer.transform(X_eval, y_eval)
            weights, info = ridge_torch(
                X_train, y_train, selected, normalpha=normalpha,
                singcutoff=singcutoff, return_info=True,
            )
            weights[:, ~valid] = float("nan")
            predictions = (X_eval @ weights).cpu().numpy()
            return weights, selected, normalizer, predictions, y_eval.cpu().numpy(), info

        if ecog:
            result["inference_status"] = "not_assessed"
        if diagnostic:
            rows = np.arange(len(features))
            wt, selected, normalizer, predictions, observed, info = fit_fold(
                features_torch, targets_torch, features_torch, targets_torch, []
            )
            fit = _fold_result(
                0, rows, np.empty(0, dtype=int), [], wt, selected, normalizer,
                predictions, observed, None, info,
            )
            fit.update(model_role="in_sample_diagnostic", prediction_indices=rows)
            fit.pop("test_indices")
            fit["index_space"] = {"train": "training_input", "prediction": "training_input"}
            result.update(mode="diagnostic_in_sample", returned_weights_role="in_sample_diagnostic",
                          returned_alphas_role="fixed_diagnostic_alpha")
            result["configuration"].update(n_outer_folds=0, n_inner_folds=0, selection="fixed")
            result["folds"].append(fit)
            self.results_ = result
            return {
                "evaluation": "in_sample_diagnostic",
                "inference_status": "not_assessed",
                "n_fit_samples": len(rows),
            }, wt.cpu().numpy(), selected.cpu().numpy()

        if train_test_mode:
            logger.info("Running in train-test mode with provided test set")
            X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)

            inner_splits = create_folds(
                len(features), folding_type, n_inner_folds,
                chunk_length=chunk_length, trim_size=trim_size, groups=groups,
            )
            wt, best_valphas, normalizer, y_pred, y_test_np, fit_info = fit_fold(
                features_torch, targets_torch, X_test_torch, y_test_torch, inner_splits
            )

            # Calculate correlations and p-values
            correlations, pvalues = _calculate_correlations_pvalues(y_test_np, y_pred)

            # Apply FDR correction
            significant, corrected_pvals = _fdr_valid(pvalues, alpha_fdr)
            n_significant = np.sum(significant)

            # Put results in a dictionary
            metrics = _create_metrics_dict(
                correlations,
                pvalues,
                corrected_pvals,
                significant,
                None if supplied else best_valphas.cpu().numpy(),
                n_significant,
            )

            result["folds"].append(_fold_result(
                0, np.arange(len(features)), np.arange(len(X_test)),
                inner_splits, wt, best_valphas, normalizer,
                y_pred, y_test_np, correlations, fit_info,
                separate_test=True,
            ))
            result["final_refit"] = {"fold_id": 0, "scope": "provided_training_set_only"}
            self.results_ = result

            if supplied:
                return metrics, None, None
            return metrics, wt.cpu().numpy(), best_valphas.cpu().numpy()

        else:
            logger.info("Running in full nested CV mode")

            # Set up CV splits
            if temporal_folds is not None:
                outer_splits = [(a, b) for a, b, _ in temporal_folds]
            elif groups is not None and folding_type == "group":
                # Use group-based folding
                outer_splits = create_folds(
                    len(features), "group", n_outer_folds, groups=groups
                )
            else:
                # Use specified folding type
                outer_splits = create_folds(
                    len(features),
                    folding_type,
                    n_outer_folds,
                    chunk_length=chunk_length,
                    trim_size=trim_size,
                    groups=groups,
                )

            # Store results from each fold
            fold_scores = []
            fold_pvalues = []
            fold_valphas = []
            fold_significant_masks = []
            fold_weights = []

            # Outer CV loop
            for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
                logger.info(f"Processing fold {fold_idx+1}/{n_outer_folds}")

                # Split data
                X_train, X_test = features_torch[train_idx], features_torch[test_idx]
                y_train, y_test = targets_torch[train_idx], targets_torch[test_idx]

                # Build inner splits within this outer training subset
                if temporal_folds is not None:
                    inner_splits = temporal_folds[fold_idx][2]
                elif groups is not None and folding_type == "group":
                    inner_groups = [groups[i] for i in train_idx]
                    inner_splits = create_folds(
                        len(train_idx), "group", n_inner_folds, groups=inner_groups
                    )
                else:
                    inner_splits = create_folds(
                        len(train_idx),
                        folding_type,
                        n_inner_folds,
                        chunk_length=chunk_length,
                        trim_size=trim_size,
                    )

                wt, best_valphas, normalizer, y_pred, y_test_np, fit_info = fit_fold(
                    X_train, y_train, X_test, y_test, inner_splits
                )
                if not supplied:
                    fold_valphas.append(best_valphas.cpu().numpy())
                    fold_weights.append(wt.cpu().numpy())

                # Calculate correlations and p-values
                correlations, pvalues = _calculate_correlations_pvalues(
                    y_test_np, y_pred, inference=not ecog
                )

                fold_scores.append(correlations)
                fold_pvalues.append(pvalues)
                result["folds"].append(_fold_result(
                    fold_idx, train_idx, test_idx, inner_splits, wt,
                    best_valphas, normalizer, y_pred, y_test_np,
                    correlations, fit_info,
                ))

                if temporal_folds is not None:
                    fold = result["folds"][-1]
                    fold["purged_indices"] = np.setdiff1d(
                        np.arange(len(features)), np.r_[train_idx, test_idx]
                    )
                    for split in fold["inner_splits"]:
                        split["purged_indices"] = np.setdiff1d(
                            train_idx, np.r_[split["train_indices"], split["validation_indices"]]
                        )

                median_corr = _nanmedian(correlations)
                logger.info("Fold %d/%d - Median correlation: %.3f", fold_idx + 1, n_outer_folds, median_corr)
                if not ecog:
                    significant, _ = _fdr_valid(pvalues, alpha_fdr)
                    fold_significant_masks.append(significant)
                    logger.info("Fold %d/%d - Significant voxels: %d/%d",
                                fold_idx + 1, n_outer_folds, np.sum(significant), len(significant))

            # Preserve target positions and fold-level exclusions.
            # Compute final metrics across folds: mean over the folds in which
            # each target was valid (NaN if it was never valid)
            fold_scores_arr = np.asarray(fold_scores, dtype=float)
            n_valid_folds = np.isfinite(fold_scores_arr).sum(axis=0)
            all_correlations = np.where(
                n_valid_folds > 0,
                np.nansum(fold_scores_arr, axis=0) / np.maximum(n_valid_folds, 1),
                np.nan,
            )

            mean_valphas = None if supplied else np.mean(fold_valphas, axis=0)
            mean_weights = None if supplied else np.mean(fold_weights, axis=0)
            if ecog:
                metrics = {
                    **_summary_scores(all_correlations),
                    "correlations": all_correlations.tolist(),
                    "best_alphas": None if supplied else mean_valphas.tolist(),
                    "inference_status": "not_assessed",
                }
                logger.info("ECoG inference not assessed. Median descriptive correlation: %.3f",
                            metrics["median_score"])
            else:
                all_pvalues = _combine_pvalues_across_folds(fold_pvalues, logger)
                significant_mask, corrected_pvalues = _fdr_valid(all_pvalues, alpha_fdr)
                n_significant = np.sum(significant_mask)
                significance_counts = np.sum(fold_significant_masks, axis=0)
                majority_significant_mask = significance_counts >= (n_outer_folds // 2 + 1)
                n_majority_significant = np.sum(majority_significant_mask)
                metrics = _create_full_cv_metrics_dict(
                    all_correlations, all_pvalues, corrected_pvalues, significant_mask,
                    majority_significant_mask, mean_valphas, n_significant, n_majority_significant,
                )

            metrics["fold_invalid_target_masks"] = (~np.isfinite(fold_scores_arr)).tolist()
            metrics["n_valid_folds_per_target"] = n_valid_folds.tolist()
            self.results_ = result

        if ecog:
            return metrics, mean_weights, mean_valphas

        # Print final results
        logger.info("\nFinal Results:")
        logger.info(f"Median correlation: {metrics['median_score']:.3f}")

        if train_test_mode:
            logger.info(
                f"Significant voxels: {n_significant}/{len(correlations)} ({metrics['percent_significant']:.1f}%)"
            )
        else:
            logger.info(
                f"Significant voxels (Fisher's method): {n_significant}/{len(all_correlations)} ({metrics['percent_significant']:.1f}%)"
            )
            logger.info(
                f"Significant voxels (majority vote): {n_majority_significant}/{len(all_correlations)} ({metrics['percent_majority_significant']:.1f}%)"
            )

        if "median_significant_score" in metrics:
            logger.info(
                f"Median correlation (significant voxels): {metrics['median_significant_score']:.3f}"
            )

        return metrics, mean_weights, mean_valphas

    @staticmethod
    def predict_fold(fold: Dict[str, Any], features: np.ndarray,
                     original_units: bool = True) -> np.ndarray:
        """Replay one persisted fold on a prepared design, never on mean weights.

        CPU float32 replay can differ slightly from GPU matrix multiplication.
        Feature extraction, FIR and legacy per-story preprocessing are upstream.
        original_units means model-input target units, not raw recording units
        when the supplied targets already underwent upstream preprocessing.
        """
        if "weights" not in fold and "estimator_state" not in fold:
            raise ValueError("Fold fitted state was omitted; prediction replay is unavailable")
        X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        state = fold["transform"]
        if state["normalize_features"]:
            X = (X - torch.as_tensor(state["feature_means"])) / (
                torch.as_tensor(state["feature_stds"]) + state["eps"]
            )
        if "estimator_state" in fold:
            from encoding.models.estimator import predict_estimator

            prediction = torch.from_numpy(predict_estimator(fold["estimator_state"], X.numpy()))
        else:
            prediction = X @ torch.as_tensor(fold["weights"])
        if original_units and state["normalize_targets"]:
            prediction = prediction * (
                torch.as_tensor(state["target_stds"]) + state["eps"]
            ) + torch.as_tensor(state["target_means"])
        return prediction.numpy()


def _fold_result(fold_id, train_idx, test_idx, inner_splits, weights, alphas,
                 normalizer, predictions, targets, correlations,
                 fit_info, separate_test=False):
    """CPU-only fitted state and coordinates, independent of live model objects."""
    state = {"normalize_features": False, "normalize_targets": False, "eps": 1e-8}
    if normalizer is not None:
        state.update(
            normalize_features=normalizer.normalize_features,
            normalize_targets=normalizer.normalize_targets, eps=normalizer.eps,
        )
        for key in ("feature_means", "feature_stds", "target_means", "target_stds"):
            value = getattr(normalizer, key)
            state[key] = None if value is None else value.detach().cpu().numpy().copy()
    pred_original, targets_original = predictions.copy(), targets.copy()
    if state["normalize_targets"]:
        scale = state["target_stds"] + state["eps"]
        pred_original = pred_original * scale + state["target_means"]
        targets_original = targets_original * scale + state["target_means"]
    train_idx = np.asarray(train_idx, dtype=int)
    fitted = fit_info if weights is None else {
        "weights": weights.detach().cpu().numpy().copy(),
        "selected_alphas": alphas.detach().cpu().numpy().copy(),
        "effective_penalties": fit_info["effective_penalties"],
    }
    scores = {} if correlations is None else {
        "correlations": np.asarray(correlations),
        "invalid_target_mask": ~np.isfinite(correlations),
    }
    return {
        **fitted,
        **scores,
        "fold_id": fold_id,
        "model_role": "training_fit" if separate_test else "outer_fold",
        "train_indices": train_idx,
        "test_indices": np.asarray(test_idx, dtype=int),
        "index_space": {"train": "training_input", "test": (
            "provided_test_input" if separate_test else "training_input"
        )},
        "inner_splits": [
            {"train_indices": train_idx[np.asarray(a, dtype=int)],
             "validation_indices": train_idx[np.asarray(b, dtype=int)]}
            for a, b in inner_splits
        ],
        "transform": state,
        "predictions": pred_original, "targets": targets_original,
        "predictions_scaled": predictions.copy(), "targets_scaled": targets.copy(),
    }


def _valid_training_targets(targets):
    return (
        torch.isfinite(targets).all(dim=0)
        & (targets.max(dim=0).values > targets.min(dim=0).values)
    )


def _find_best_alphas(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    fold_splits: List[Tuple[List[int], List[int]]],
    alphas: List[float],
    single_alpha: bool = False,
    normalpha: bool = False,
    use_corr: bool = True,
    logger: Optional[logging.Logger] = None,
    singcutoff: float = 1e-10,
    normalize_features: bool = False,
    normalize_targets: bool = False,
) -> torch.Tensor:
    """Find the best alpha(s) using inner cross-validation.

    If normalization is requested, a separate normalizer is fit on each inner
    training subset and applied to its validation subset, so validation rows never
    influence the statistics.

    Args:
        X_train: Training features tensor
        y_train: Training targets tensor
        fold_splits: List of (train_indices, val_indices) tuples for inner CV
        alphas: Ridge parameters to test
        single_alpha: If True, use a single alpha for all voxels
        normalpha: Whether to normalize alpha by largest singular value
        use_corr: If True, use correlation as metric; if False, use R-squared
        logger: Logger instance
        singcutoff: Singularity cutoff for ridge_corr

    Returns:
        Best alpha(s) - either a single value for all voxels or one per voxel
    """
    # Initialize storage for correlations from each inner fold
    inner_fold_corrs = []
    # alphas_torch = torch.tensor(alphas, dtype=torch.float32, device=X_train.device)

    # For each inner fold, compute correlations for each alpha
    for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(fold_splits):
        if logger:
            logger.info(f"  Inner fold {inner_fold_idx+1}/{len(fold_splits)}")

        # Split inner training data
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]

        valid_targets = (
            torch.isfinite(y_inner_train).all(dim=0)
            & torch.isfinite(y_inner_val).all(dim=0)
            & (y_inner_train.max(dim=0).values > y_inner_train.min(dim=0).values)
            & (y_inner_val.max(dim=0).values > y_inner_val.min(dim=0).values)
        )
        if normalize_features or normalize_targets:
            inner_normalizer = DataNormalizer(
                normalize_features=normalize_features,
                normalize_targets=normalize_targets,
            )
            X_inner_train, y_inner_train = inner_normalizer.fit_transform(
                X_inner_train, y_inner_train
            )
            X_inner_val, y_inner_val = inner_normalizer.transform(
                X_inner_val, y_inner_val
            )

        # Compute correlations for each alpha using ridge_corr
        corrs = ridge_corr_torch(
            X_inner_train,
            X_inner_val,
            y_inner_train,
            y_inner_val,
            alphas,
            singcutoff=singcutoff,
            use_corr=use_corr,
            normalpha=normalpha,
            logger=logger,
        )
        corrs[:, ~valid_targets] = float("nan")
        inner_fold_corrs.append(corrs)

    # Average correlations across inner folds
    mean_inner_corrs = torch.stack(inner_fold_corrs).nanmean(
        dim=0
    )  # Shape: (n_alphas, n_voxels)

    unselected = ~torch.isfinite(mean_inner_corrs).any(dim=0)
    if unselected.any() and logger:
        logger.warning(
            "Targets without valid inner folds (first alpha is a placeholder): %s",
            torch.where(unselected)[0].tolist(),
        )

    # Find the best alpha
    if single_alpha:
        # Find single best alpha across all voxels
        mean_across_voxels = mean_inner_corrs.nanmean(dim=1)  # Shape: (n_alphas,)
        if not torch.isfinite(mean_across_voxels).any():
            raise ValueError("No valid targets for inner-CV alpha selection")
        best_alpha_idx = torch.argmax(mean_across_voxels)
        best_alpha = alphas[best_alpha_idx]
        best_valphas = torch.tensor(
            [best_alpha] * y_train.shape[1], device=X_train.device
        )
        if logger:
            logger.info(f"  Found best single alpha = {best_alpha:.3f} for all voxels")
    else:
        # Find the best alpha for each voxel
        best_alpha_idx = torch.argmax(torch.nan_to_num(mean_inner_corrs, nan=-float("inf")), dim=0)  # Shape: (n_voxels,)
        best_valphas = torch.tensor(
            [alphas[i] for i in best_alpha_idx], device=X_train.device, dtype=torch.float32
        )
        if logger:
            logger.info("Found best alphas for each voxel")

    return best_valphas


def _calculate_correlations_pvalues(
    y_true: np.ndarray, y_pred: np.ndarray, *, inference: bool = True
) -> Tuple[List[float], List[float]]:
    """Calculate correlations and p-values between true and predicted values

    A target that is constant or non-finite in this test set is invalid: its
    correlation and p-value are NaN. A valid target with a constant prediction
    keeps the previous convention (r=0, p=1).

    Args:
        y_true: True target values (n_samples, n_targets)
        y_pred: Predicted target values (n_samples, n_targets)

    Returns:
        Tuple of (correlations, p-values) lists
    """
    invalid = (
        ~np.isfinite(y_true).all(axis=0)
        | ~np.isfinite(y_pred).all(axis=0)
        | (np.ptp(y_true, axis=0) == 0)
    )
    if invalid.any():
        logging.getLogger(__name__).warning(
            f"{int(invalid.sum())}/{y_true.shape[1]} targets are constant or non-finite "
            "or have non-finite predictions; reporting NaN and excluding them from summaries"
        )

    correlations = []
    pvalues = []

    for i in range(y_true.shape[1]):
        if invalid[i]:
            correlations.append(np.nan)
            pvalues.append(np.nan)
            continue
        if inference:
            corr, pval = pearsonr(y_true[:, i], y_pred[:, i])
        else:
            estimate = y_pred[:, i].astype(np.float64)
            actual = y_true[:, i].astype(np.float64)
            estimate -= estimate.mean()
            actual -= actual.mean()
            norm = np.linalg.norm(estimate) * np.linalg.norm(actual)
            corr = np.clip((estimate @ actual) / norm, -1, 1) if norm else 0.0
            pval = np.nan
        correlations.append(0.0 if np.isnan(corr) else corr)
        pvalues.append((1.0 if np.isnan(pval) else pval) if inference else np.nan)

    return correlations, pvalues


def _fdr_valid(pvalues, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """FDR over finite p-values only; invalid targets are never significant."""
    pvalues = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(pvalues)
    significant = np.zeros(pvalues.shape, dtype=bool)
    corrected = np.full(pvalues.shape, np.nan)
    if valid.any():
        significant[valid], corrected[valid] = fdrcorrection(pvalues[valid], alpha=alpha)
    return significant, corrected


def _nanmedian(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _combine_pvalues_across_folds(
    fold_pvalues: List[List[float]], logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Combine p-values across folds using Fisher's method

    Args:
        fold_pvalues: List of p-values from each fold
        logger: Logger instance

    Returns:
        Combined p-values array
    """
    all_pvalues = []

    for i in range(len(fold_pvalues[0])):  # For each voxel
        # Get p-values from the folds in which this voxel was a valid target
        voxel_pvals = [fold[i] for fold in fold_pvalues if np.isfinite(fold[i])]

        if not voxel_pvals:
            all_pvalues.append(np.nan)
        # Some voxels might have all p-values as 1.0, which causes a warning
        # Handle this special case
        elif all(p == 1.0 for p in voxel_pvals):
            all_pvalues.append(1.0)
        else:
            # Use Fisher's method to combine p-values across folds
            try:
                combined_stat, combined_p = combine_pvalues(
                    voxel_pvals, method="fisher"
                )
                all_pvalues.append(combined_p)
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Warning for voxel {i}: {e}. Using maximum p-value."
                    )
                all_pvalues.append(max(voxel_pvals))

    return np.array(all_pvalues)


def _summary_scores(correlations) -> Dict[str, Union[float, int, List[bool]]]:
    """Summary statistics over valid (finite) targets; NaN targets are counted."""
    correlations = np.asarray(correlations, dtype=float)
    valid = np.isfinite(correlations)
    if not valid.any():
        raise ValueError("No valid targets: every target is constant or non-finite")
    scores = correlations[valid]
    return {
        "median_score": float(np.median(scores)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "n_valid_targets": int(valid.sum()),
        "n_invalid_targets": int((~valid).sum()),
        "invalid_target_mask": (~valid).tolist(),
    }


def _create_metrics_dict(
    correlations: List[float],
    pvalues: List[float],
    corrected_pvalues: np.ndarray,
    significant_mask: np.ndarray,
    best_alphas: np.ndarray,
    n_significant: int,
) -> Dict[str, Union[float, List[float], List[bool]]]:
    """Create a dictionary of evaluation metrics for the train-test scenario

    Args:
        correlations: Correlation values
        pvalues: Raw p-values
        corrected_pvalues: FDR-corrected p-values
        significant_mask: Boolean mask of significant voxels
        best_alphas: Best alpha values
        n_significant: Number of significant voxels

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        **_summary_scores(correlations),
        "best_alphas": None if best_alphas is None else best_alphas.tolist(),
        "correlations": correlations,
        "p_values": pvalues,
        "corrected_p_values": corrected_pvalues.tolist(),
        "significant_mask": significant_mask.tolist(),
        "n_significant": int(n_significant),
    }
    metrics["percent_significant"] = float(
        n_significant / metrics["n_valid_targets"] * 100
    )

    # Add metrics for significant voxels if there are any
    sig_correlations = (
        np.array(correlations)[significant_mask] if n_significant > 0 else np.array([])
    )
    if n_significant > 0:
        metrics.update(
            {
                "median_significant_score": float(np.median(sig_correlations)),
                "mean_significant_score": float(np.mean(sig_correlations)),
                "min_significant_score": float(np.min(sig_correlations)),
                "max_significant_score": float(np.max(sig_correlations)),
            }
        )

    return metrics


def _create_full_cv_metrics_dict(
    all_correlations: np.ndarray,
    all_pvalues: np.ndarray,
    corrected_pvalues: np.ndarray,
    significant_mask: np.ndarray,
    majority_significant_mask: np.ndarray,
    mean_valphas: np.ndarray,
    n_significant: int,
    n_majority_significant: int,
) -> Dict[str, Union[float, List[float], List[bool]]]:
    """Create a dictionary of evaluation metrics for the full CV scenario

    Args:
        all_correlations: Correlation values averaged across folds
        all_pvalues: Combined p-values
        corrected_pvalues: FDR-corrected p-values
        significant_mask: Boolean mask of significant voxels (Fisher's method)
        majority_significant_mask: Boolean mask of majority-significant voxels
        mean_valphas: Mean best alpha values across folds
        n_significant: Number of significant voxels (Fisher's method)
        n_majority_significant: Number of majority-significant voxels

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        **_summary_scores(all_correlations),
        "best_alphas": None if mean_valphas is None else mean_valphas.tolist(),
        "correlations": all_correlations.tolist(),
        "p_values": all_pvalues.tolist(),
        "corrected_p_values": corrected_pvalues.tolist(),
        "significant_mask": significant_mask.tolist(),
        "majority_significant_mask": majority_significant_mask.tolist(),
        "n_significant": int(n_significant),
        "n_majority_significant": int(n_majority_significant),
    }
    n_valid = metrics["n_valid_targets"]
    metrics["percent_significant"] = float(n_significant / n_valid * 100)
    metrics["percent_majority_significant"] = float(
        n_majority_significant / n_valid * 100
    )

    # Add metrics for Fisher's method significant voxels if there are any
    sig_correlations = (
        all_correlations[significant_mask] if n_significant > 0 else np.array([])
    )
    if n_significant > 0:
        metrics.update(
            {
                "median_significant_score": float(np.median(sig_correlations)),
                "mean_significant_score": float(np.mean(sig_correlations)),
                "min_significant_score": float(np.min(sig_correlations)),
                "max_significant_score": float(np.max(sig_correlations)),
            }
        )

    # Add metrics for majority vote significant voxels if there are any
    majority_sig_correlations = (
        all_correlations[majority_significant_mask]
        if n_majority_significant > 0
        else np.array([])
    )
    if n_majority_significant > 0:
        metrics.update(
            {
                "median_majority_significant_score": float(
                    np.median(majority_sig_correlations)
                ),
                "mean_majority_significant_score": float(
                    np.mean(majority_sig_correlations)
                ),
                "min_majority_significant_score": float(
                    np.min(majority_sig_correlations)
                ),
                "max_majority_significant_score": float(
                    np.max(majority_sig_correlations)
                ),
            }
        )

    return metrics
