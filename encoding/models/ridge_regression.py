import torch
from typing import List, Union, Optional
import logging
from encoding.models.ridge_utils import svd_wrapper, mult_diag, z_score


def ridge_torch(
    Rstim: torch.Tensor,
    Rresp: torch.Tensor,
    alphas: Union[float, torch.Tensor],
    singcutoff: float = 1e-30,
    normalpha: bool = False,
) -> torch.Tensor:
    """PyTorch version of ridge function for computing weights

    Args:
        Rstim: Training stimulus matrix (n_samples, n_features)
        Rresp: Training response matrix (n_samples, n_voxels)
        alphas: Ridge parameters (single float or per-voxel tensor)
        singcutoff: Cutoff for small singular values
        normalpha: Whether to normalize alpha by largest singular value

    Returns:
        wt: Ridge regression weights (n_features, n_voxels)
    """
    # Calculate SVD of stimulus matrix
    U, S, Vh = svd_wrapper(Rstim, singcutoff=singcutoff)

    # Compute UR once for efficiency
    UR = torch.matmul(U.T, Rresp)

    # Expand alpha to a collection if it's just a single value
    if isinstance(alphas, (int, float)):
        alphas = torch.ones(Rresp.shape[1], device=Rstim.device) * alphas

    # Normalize alpha by the LSV norm if requested
    norm = S[0].item()
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    # Compute weights for each alpha
    ualphas = torch.unique(nalphas)
    wt = torch.zeros((Rstim.shape[1], Rresp.shape[1]), device=Rstim.device)

    for ua in ualphas:
        selvox = torch.nonzero(nalphas == ua).squeeze()

        # Handle case where only one voxel has this alpha
        if selvox.ndim == 0:
            selvox = selvox.unsqueeze(0)

        D = S / (S**2 + ua**2)

        # Compute weights for the voxels with this alpha
        Vh_D = torch.matmul(Vh.T, torch.diag(D))
        awt = torch.matmul(Vh_D, UR[:, selvox])
        wt[:, selvox] = awt

    return wt


def ridge_corr_torch(
    Rstim: torch.Tensor,
    Pstim: torch.Tensor,
    Rresp: torch.Tensor,
    Presp: torch.Tensor,
    alphas: List[float],
    singcutoff: float = 1e-30,
    use_corr: bool = True,
    normalpha: bool = False,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """PyTorch version of ridge_corr function to test multiple alphas

    Args:
        Rstim: Training stimulus matrix (n_train_samples, n_features)
        Pstim: Test stimulus matrix (n_test_samples, n_features)
        Rresp: Training response matrix (n_train_samples, n_voxels)
        Presp: Test response matrix (n_test_samples, n_voxels)
        alphas: List of ridge parameters to test
        singcutoff: Cutoff for small singular values
        use_corr: If True, use correlation as metric; if False, use R-squared
        normalpha: Whether to normalize alpha by largest singular value
        logger: Optional logger for progress updates

    Returns:
        Rcorrs: Correlations for each alpha and voxel (n_alphas, n_voxels)
    """
    # Calculate SVD of stimulus matrix
    U, S, Vh = svd_wrapper(Rstim, singcutoff=singcutoff)

    # Normalize alpha by the LSV norm if requested
    norm = S[0].item()
    if normalpha:
        nalphas = [alpha * norm for alpha in alphas]
    else:
        nalphas = alphas

    # Precompute products for speed
    UR = torch.matmul(U.T, Rresp)
    PVh = torch.matmul(Pstim, Vh.T)

    # Z-score the test responses for correlation calculation
    zPresp = z_score(Presp, dim=0)

    # Compute variance for explained variance calculation
    Prespvar = Presp.var(dim=0)

    # Compute correlations for each alpha
    Rcorrs = []
    for na, a in zip(nalphas, alphas):
        # Reweight singular vectors by the ridge parameter
        D = S / (S**2 + na**2)

        # Compute predictions
        pred = torch.matmul(mult_diag(D, PVh, left=False), UR)

        if use_corr:
            # Compute correlations
            zpred = z_score(pred, dim=0)
            Rcorr = (zPresp * zpred).mean(dim=0)
        else:
            # Compute variance explained
            resvar = (Presp - pred).var(dim=0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = torch.sqrt(torch.abs(Rsq)) * torch.sign(Rsq)

        # Replace NaN with zeros
        Rcorr = torch.nan_to_num(Rcorr)
        Rcorrs.append(Rcorr)

        if logger:
            logger.info(
                f"Alpha={a:.3f}, mean corr={Rcorr.mean().item():.5f}, max corr={Rcorr.max().item():.5f}"
            )

    return torch.stack(Rcorrs)


def ridge_corr_pred_torch(
    Rstim: torch.Tensor,
    Pstim: torch.Tensor,
    Rresp: torch.Tensor,
    Presp: torch.Tensor,
    valphas: torch.Tensor,
    singcutoff: float = 1e-30,
    use_corr: bool = True,
    normalpha: bool = True,
) -> torch.Tensor:
    """PyTorch version of ridge_corr_pred function using per-voxel alphas

    Args:
        Rstim: Training stimulus matrix (n_train_samples, n_features)
        Pstim: Test stimulus matrix (n_test_samples, n_features)
        Rresp: Training response matrix (n_train_samples, n_voxels)
        Presp: Test response matrix (n_test_samples, n_voxels)
        valphas: Ridge parameter for each voxel
        singcutoff: Cutoff for small singular values
        use_corr: If True, use correlation as metric; if False, use R-squared
        normalpha: Whether to normalize alpha by largest singular value

    Returns:
        corr: Correlation between predicted and actual responses (n_voxels,)
    """
    # Calculate SVD of stimulus matrix
    U, S, Vh = svd_wrapper(Rstim, singcutoff=singcutoff)

    # Normalize alpha by the LSV norm if requested
    norm = S[0].item()
    if normalpha:
        nalphas = valphas * norm
    else:
        nalphas = valphas

    # Precompute products for speed
    UR = torch.matmul(U.T, Rresp)
    PVh = torch.matmul(Pstim, Vh.T)

    # Z-score the test responses for correlation calculation
    zPresp = z_score(Presp, dim=0)

    # Compute variance for explained variance calculation
    Prespvar = Presp.var(dim=0)

    # Get unique alphas and compute predictions for each one
    ualphas = torch.unique(nalphas)
    corr = torch.zeros(Rresp.shape[1], device=Rstim.device)

    for ua in ualphas:
        selvox = torch.nonzero(nalphas == ua).squeeze()

        # Handle case where only one voxel has this alpha
        if selvox.ndim == 0:
            selvox = selvox.unsqueeze(0)

        # Reweight singular vectors by the ridge parameter
        D = S / (S**2 + ua**2)

        # Compute predictions for the voxels with this alpha
        alpha_pred = torch.matmul(torch.matmul(PVh, torch.diag(D)), UR[:, selvox])

        if use_corr:
            # Compute correlations
            zpred = z_score(alpha_pred, dim=0)
            corr[selvox] = (zPresp[:, selvox] * zpred).mean(dim=0)
        else:
            # Compute variance explained
            resvar = (Presp[:, selvox] - alpha_pred).var(dim=0)
            Rsq = 1 - (resvar / Prespvar[selvox])
            corr[selvox] = torch.sqrt(torch.abs(Rsq)) * torch.sign(Rsq)

    return corr
