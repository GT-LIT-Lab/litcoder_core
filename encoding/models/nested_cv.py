import numpy as np
import torch
from scipy.stats import pearsonr, combine_pvalues
from statsmodels.stats.multitest import fdrcorrection
import logging
from typing import Dict, Optional, List, Tuple, Union
from encoding.models.ridge_regression import ridge_torch, ridge_corr_torch
from encoding.models.folding import create_folds
from encoding.models.ridge_utils import DataNormalizer
from encoding.models.base import BasePredictivityModel
from typing import Any, Dict, List, Optional, Tuple, Union


class NestedCVModel(BasePredictivityModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)

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
    ) -> Tuple[
        Dict[str, Union[float, List[float], List[bool]]],
        np.ndarray,
        np.ndarray,
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

        Returns:
            Tuple of (metrics, weights, best_alphas)
            - metrics: Dictionary of evaluation metrics
            - weights: Model weights (n_features, n_targets)
            - best_alphas: Best alpha values for each target
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)

        # Set default alphas if not provided
        if alphas is None:
            alphas = np.logspace(-1, 8, 10)

        # Determine device - use GPU if available and requested

        if use_gpu:
            if torch.backends.mps.is_available():
                device = "mps:0"
            elif torch.cuda.is_available():
                device = "cuda"
        else:
             device = "cpu"
        logger.info(f"Using device: {device}")
        logger.info(f"Folding type: {folding_type}")

        # Convert inputs to PyTorch tensors
        features_torch = torch.tensor(features, dtype=torch.float32, device=device)
        targets_torch = torch.tensor(targets, dtype=torch.float32, device=device)

        # Check if we're in train-test mode or full CV mode
        train_test_mode = X_test is not None and y_test is not None

        if train_test_mode:
            logger.info("Running in train-test mode with provided test set")
            X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)

            # Normalize data if requested
            if normalize_features or normalize_targets:
                logger.info(
                    f"Normalizing data using training statistics (features: {normalize_features}, targets: {normalize_targets})"
                )
                normalizer = DataNormalizer(
                    normalize_features=normalize_features,
                    normalize_targets=normalize_targets,
                )
                features_torch, targets_torch = normalizer.fit_transform(
                    features_torch, targets_torch
                )
                X_test_torch, y_test_torch = normalizer.transform(
                    X_test_torch, y_test_torch
                )

            # Find best alphas using inner CV on training data only
            best_valphas = _find_best_alphas(
                features_torch,
                targets_torch,
                fold_splits=create_folds(
                    len(features), folding_type, n_inner_folds, chunk_length, groups
                ),
                alphas=alphas,
                single_alpha=single_alpha,
                normalpha=normalpha,
                use_corr=use_corr,
                logger=logger,
                singcutoff=singcutoff,
            )

            # Compute weights with the best alphas
            wt = ridge_torch(
                features_torch,
                targets_torch,
                best_valphas,
                normalpha=normalpha,
                singcutoff=singcutoff,
            )

            # Predict on test set
            y_pred = torch.matmul(X_test_torch, wt).cpu().numpy()
            y_test_np = y_test_torch.cpu().numpy()

            # Calculate correlations and p-values
            correlations, pvalues = _calculate_correlations_pvalues(y_test_np, y_pred)

            # Apply FDR correction
            significant, corrected_pvals = fdrcorrection(pvalues, alpha=alpha_fdr)
            n_significant = np.sum(significant)

            # Put results in a dictionary
            metrics = _create_metrics_dict(
                correlations,
                pvalues,
                corrected_pvals,
                significant,
                best_valphas.cpu().numpy(),
                n_significant,
            )

            return metrics, wt.cpu().numpy(), best_valphas.cpu().numpy()

        else:
            logger.info("Running in full nested CV mode")

            # Set up CV splits
            if groups is not None and folding_type == "group":
                # Use group-based folding
                outer_splits = create_folds(
                    len(features), "group", n_outer_folds, groups=groups
                )
            else:
                # Use specified folding type
                outer_splits = create_folds(
                    len(features), folding_type, n_outer_folds, chunk_length, groups
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

                # Normalize data if requested
                if normalize_features or normalize_targets:
                    logger.info(
                        f"Normalizing data for fold {fold_idx+1} (features: {normalize_features}, targets: {normalize_targets})"
                    )
                    normalizer = DataNormalizer(
                        normalize_features=normalize_features,
                        normalize_targets=normalize_targets,
                    )
                    X_train, y_train = normalizer.fit_transform(X_train, y_train)
                    X_test, y_test = normalizer.transform(X_test, y_test)

                # Inner CV to find the best alpha for each voxel
                if groups is not None and folding_type == "group":
                    inner_groups = [groups[i] for i in train_idx]
                    inner_splits = create_folds(
                        len(train_idx), "group", n_inner_folds, groups=inner_groups
                    )
                else:
                    inner_splits = create_folds(
                        len(train_idx), folding_type, n_inner_folds, chunk_length
                    )

                # Find best alphas for this fold
                best_valphas = _find_best_alphas(
                    X_train,
                    y_train,
                    fold_splits=inner_splits,
                    alphas=alphas,
                    single_alpha=single_alpha,
                    normalpha=normalpha,
                    use_corr=use_corr,
                    logger=logger,
                    singcutoff=singcutoff,
                )

                fold_valphas.append(best_valphas.cpu().numpy())

                # Apply the best alphas to the entire training set and evaluate on test set
                wt = ridge_torch(
                    X_train,
                    y_train,
                    best_valphas,
                    normalpha=normalpha,
                    singcutoff=singcutoff,
                )
                fold_weights.append(wt.cpu().numpy())

                y_pred = torch.matmul(X_test, wt).cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                # Calculate correlations and p-values
                correlations, pvalues = _calculate_correlations_pvalues(
                    y_test_np, y_pred
                )

                fold_scores.append(correlations)
                fold_pvalues.append(pvalues)

                # Apply FDR correction for this fold
                significant, corrected_pvals = fdrcorrection(pvalues, alpha=alpha_fdr)
                fold_significant_masks.append(significant)
                n_significant = np.sum(significant)

                median_corr = np.median(correlations)
                logger.info(
                    f"Fold {fold_idx+1}/{n_outer_folds} - Median correlation: {median_corr:.3f}"
                )
                logger.info(
                    f"Fold {fold_idx+1}/{n_outer_folds} - Significant voxels: {n_significant}/{len(significant)}"
                )

            # Compute final metrics across folds
            all_correlations = np.mean(fold_scores, axis=0)  # average across folds

            # Use Fisher's method to combine p-values across folds for each voxel
            all_pvalues = _combine_pvalues_across_folds(fold_pvalues, logger)

            # Apply FDR correction to the combined p-values
            significant_mask, corrected_pvalues = fdrcorrection(
                all_pvalues, alpha=alpha_fdr
            )
            n_significant = np.sum(significant_mask)

            # Alternative: Count how many times each voxel was significant across folds
            significance_counts = np.sum(fold_significant_masks, axis=0)
            majority_significant_mask = significance_counts >= (n_outer_folds // 2 + 1)
            n_majority_significant = np.sum(majority_significant_mask)

            # Compute average best alpha for each voxel
            mean_valphas = np.mean(fold_valphas, axis=0)

            # Compute average weights across folds
            mean_weights = np.mean(fold_weights, axis=0)

            # Create metrics dictionary with all results
            metrics = _create_full_cv_metrics_dict(
                all_correlations,
                all_pvalues,
                corrected_pvalues,
                significant_mask,
                majority_significant_mask,
                mean_valphas,
                n_significant,
                n_majority_significant,
            )

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
) -> torch.Tensor:
    """Find the best alpha(s) using inner cross-validation

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
        inner_fold_corrs.append(corrs)

    # Average correlations across inner folds
    mean_inner_corrs = torch.stack(inner_fold_corrs).mean(
        dim=0
    )  # Shape: (n_alphas, n_voxels)

    # Find the best alpha
    if single_alpha:
        # Find single best alpha across all voxels
        mean_across_voxels = mean_inner_corrs.mean(dim=1)  # Shape: (n_alphas,)
        best_alpha_idx = torch.argmax(mean_across_voxels)
        best_alpha = alphas[best_alpha_idx]
        best_valphas = torch.tensor(
            [best_alpha] * y_train.shape[1], device=X_train.device
        )
        if logger:
            logger.info(f"  Found best single alpha = {best_alpha:.3f} for all voxels")
    else:
        # Find the best alpha for each voxel
        best_alpha_idx = torch.argmax(mean_inner_corrs, dim=0)  # Shape: (n_voxels,)
        best_valphas = torch.tensor(
            [alphas[i] for i in best_alpha_idx], device=X_train.device, dtype=torch.float32
        )
        if logger:
            logger.info("Found best alphas for each voxel")

    return best_valphas


def _calculate_correlations_pvalues(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Calculate correlations and p-values between true and predicted values

    Args:
        y_true: True target values (n_samples, n_targets)
        y_pred: Predicted target values (n_samples, n_targets)

    Returns:
        Tuple of (correlations, p-values) lists
    """
    correlations = []
    pvalues = []

    for i in range(y_true.shape[1]):
        corr, pval = pearsonr(y_true[:, i], y_pred[:, i])
        correlations.append(0.0 if np.isnan(corr) else corr)
        pvalues.append(1.0 if np.isnan(pval) else pval)

    return correlations, pvalues


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
        # Get p-values from all folds for this voxel
        voxel_pvals = [fold[i] for fold in fold_pvalues]

        # Some voxels might have all p-values as 1.0, which causes a warning
        # Handle this special case
        if all(p == 1.0 for p in voxel_pvals):
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
        "median_score": float(np.median(correlations)),
        "mean_score": float(np.mean(correlations)),
        "std_score": float(np.std(correlations)),
        "min_score": float(np.min(correlations)),
        "max_score": float(np.max(correlations)),
        "best_alphas": best_alphas.tolist(),
        "correlations": correlations,
        "p_values": pvalues,
        "corrected_p_values": corrected_pvalues.tolist(),
        "significant_mask": significant_mask.tolist(),
        "n_significant": int(n_significant),
        "percent_significant": float(n_significant / len(correlations) * 100),
    }

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
        "median_score": float(np.median(all_correlations)),
        "mean_score": float(np.mean(all_correlations)),
        "std_score": float(np.std(all_correlations)),
        "min_score": float(np.min(all_correlations)),
        "max_score": float(np.max(all_correlations)),
        "best_alphas": mean_valphas.tolist(),
        "correlations": all_correlations.tolist(),
        "p_values": all_pvalues.tolist(),
        "corrected_p_values": corrected_pvalues.tolist(),
        "significant_mask": significant_mask.tolist(),
        "majority_significant_mask": majority_significant_mask.tolist(),
        "n_significant": int(n_significant),
        "n_majority_significant": int(n_majority_significant),
        "percent_significant": float(n_significant / len(all_correlations) * 100),
        "percent_majority_significant": float(
            n_majority_significant / len(all_correlations) * 100
        ),
    }

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
