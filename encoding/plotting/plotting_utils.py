from abc import ABC, abstractmethod
import numpy as np
from nilearn import plotting, datasets
from nilearn.plotting.cm import cold_hot
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import io
from PIL import Image


class Logger(ABC):
    """Abstract base class for different logging backends."""

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        """Log a matplotlib figure as an image."""
        pass

    @abstractmethod
    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        """Log a histogram of values."""
        pass


class WandBLogger(Logger):
    """Weights & Biases logger implementation."""

    def __init__(self):
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        log_dict = {name: value}
        if step is not None:
            log_dict["step"] = step
        self.wandb.log(log_dict)

    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        log_dict = {name: self.wandb.Image(figure)}
        if step is not None:
            log_dict["step"] = step
        self.wandb.log(log_dict)

    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        log_dict = {name: self.wandb.Histogram(values)}
        if step is not None:
            log_dict["step"] = step
        self.wandb.log(log_dict)


class TensorBoardLogger(Logger):
    """TensorBoard logger implementation."""

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError(
                "tensorboard not installed. Install with: pip install tensorboard torch"
            )

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)

    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        # Convert matplotlib figure to PIL Image, then to tensor
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf)

        # Convert PIL to numpy array (H, W, C)
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3:
            # Convert from HWC to CHW format for tensorboard
            img_array = img_array.transpose(2, 0, 1)

        self.writer.add_image(name, img_array, step, dataformats="CHW")
        buf.close()

    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        self.writer.add_histogram(name, values, step)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class BrainPlotter:
    """A class to handle brain surface visualization and correlation plots."""

    def __init__(self, logger: Logger):
        """Initialize with a specific logger backend.

        Args:
            logger: Logger instance (WandBLogger, TensorBoardLogger, etc.)
        """
        self.logger = logger

    @staticmethod
    def plot_surface_correlations(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Significant Prediction Correlations",
        only_significant: bool = True,
        is_volume: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot correlations on brain surface with ONE shared colorbar on the right.
        """
        if is_volume:
            print("Skipping surface plotting for volume data")
            return None

        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        N = 10242

        # Apply mask if requested
        masked_correlations = correlations.astype(float).copy()
        if only_significant:
            masked_correlations[~significant_mask.astype(bool)] = np.nan

        # Split hemispheres
        left_correlations = masked_correlations[:N]
        right_correlations = masked_correlations[N : 2 * N]

        # Symmetric color scale across panes
        vmax = np.nanmax(np.abs(masked_correlations))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cmap = cold_hot

        fig = plt.figure(figsize=(15, 10))

        # Left Lateral
        ax1 = fig.add_subplot(231, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            left_correlations,
            hemi="left",
            view="lateral",
            colorbar=False,
            axes=ax1,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Left Lateral",
        )

        # Left Medial
        ax2 = fig.add_subplot(232, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            left_correlations,
            hemi="left",
            view="medial",
            colorbar=False,
            axes=ax2,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Left Medial",
        )

        # Right Lateral
        ax3 = fig.add_subplot(234, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            right_correlations,
            hemi="right",
            view="lateral",
            colorbar=False,
            axes=ax3,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Right Lateral",
        )

        # Right Medial
        ax4 = fig.add_subplot(235, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            right_correlations,
            hemi="right",
            view="medial",
            colorbar=False,
            axes=ax4,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Right Medial",
        )

        # One shared colorbar on the right margin
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cax)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.97])
        return fig

    @staticmethod
    def plot_all_correlations_histogram(
        correlations: np.ndarray, title: str = "All Correlations Distribution"
    ) -> plt.Figure:
        """Plot histogram of all correlations."""
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        valid_correlations = correlations[~np.isnan(correlations)]
        sns.histplot(
            valid_correlations,
            bins=100,
            color="blue",
            label="All",
            kde=True,
            stat="density",
        )
        plt.legend()
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        return fig

    @staticmethod
    def plot_significant_correlations_histogram(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Significant Correlations Distribution",
    ) -> plt.Figure:
        """Plot histogram of significant correlations."""
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sig_correlations = correlations[significant_mask]
        valid_sig_correlations = sig_correlations[~np.isnan(sig_correlations)]

        sns.histplot(
            valid_sig_correlations,
            bins=100,
            color="green",
            label="Significant",
            kde=True,
            stat="density",
        )
        plt.legend()
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        return fig

    def log_plots(
        self,
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        prefix: str = "",
        step: Optional[int] = None,
        is_volume: bool = False,
        language_mask: Optional[np.ndarray] = None,
        roi_masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Log brain surface plots and correlation histograms using the configured logger.

        Args:
            correlations: (20484,) correlations for fsaverage5 surface (L then R)
            significant_mask: (20484,) boolean mask of significant vertices
            prefix: optional namespace for log keys
            step: optional step number for logging
            is_volume: skip surface plotting if True
            language_mask: (20484,) optional boolean mask for language network
            roi_masks: optional dict[str, np.ndarray] of additional ROI masks
        """

        def _sanitize(name: str) -> str:
            return "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name.strip()
            ).lower()

        N = 10242
        full_len = 2 * N

        # Sanity checks
        correlations = np.asarray(correlations)
        significant_mask = np.asarray(significant_mask, dtype=bool)

        if not is_volume:
            if correlations.shape[0] != full_len:
                raise ValueError(
                    f"`correlations` must be length {full_len}, got {correlations.shape}"
                )

        if significant_mask.shape[0] != correlations.shape[0]:
            raise ValueError(
                f"`significant_mask` must match correlations length, got {significant_mask.shape} vs {correlations.shape}"
            )

        # All correlations histogram
        fig_all = self.plot_all_correlations_histogram(
            correlations, title="All Correlations Distribution"
        )
        self.logger.log_image(f"{prefix}correlation_histogram_all", fig_all, step)
        plt.close(fig_all)

        valid_correlations = correlations[~np.isnan(correlations)]
        self.logger.log_histogram(
            f"{prefix}correlation_histogram_data_all", valid_correlations, step
        )

        # Surface plots (if not volume)
        if not is_volume:
            fig_significant = self.plot_surface_correlations(
                correlations,
                significant_mask,
                title="Significant Prediction Correlations",
                only_significant=True,
                is_volume=is_volume,
            )
            if fig_significant is not None:
                self.logger.log_image(
                    f"{prefix}brain_surface_significant", fig_significant, step
                )
                plt.close(fig_significant)

            fig_all_surface = self.plot_surface_correlations(
                correlations,
                significant_mask,
                title="All Prediction Correlations",
                only_significant=False,
                is_volume=is_volume,
            )
            if fig_all_surface is not None:
                self.logger.log_image(
                    f"{prefix}brain_surface_all", fig_all_surface, step
                )
                plt.close(fig_all_surface)

        # Significant correlations histogram
        fig_sig = self.plot_significant_correlations_histogram(
            correlations,
            significant_mask,
            title="Significant Correlations Distribution",
        )
        self.logger.log_image(
            f"{prefix}correlation_histogram_significant", fig_sig, step
        )
        plt.close(fig_sig)

        sig_correlations = correlations[significant_mask]
        valid_sig_correlations = sig_correlations[~np.isnan(sig_correlations)]
        self.logger.log_histogram(
            f"{prefix}correlation_histogram_data_significant",
            valid_sig_correlations,
            step,
        )

        # Language network analysis
        if language_mask is not None:
            language_mask = np.asarray(language_mask, dtype=bool)
            if language_mask.shape[0] != correlations.shape[0]:
                raise ValueError(
                    f"`language_mask` must match correlations length, got {language_mask.shape} vs {correlations.shape}"
                )

            lang_vals = correlations[language_mask]
            mean_v = float(np.nanmean(lang_vals)) if lang_vals.size else np.nan
            median_v = float(np.nanmedian(lang_vals)) if lang_vals.size else np.nan

            self.logger.log_scalar(f"{prefix}lanA_mean", mean_v, step)
            self.logger.log_scalar(f"{prefix}lanA_median", median_v, step)

            clean = lang_vals[~np.isnan(lang_vals)]
            if clean.size:
                self.logger.log_histogram(f"{prefix}lanA_hist", clean, step)

            if not is_volume:
                fig_lang = self.plot_surface_correlations(
                    correlations=correlations,
                    significant_mask=language_mask,
                    title="Language Network — Masked",
                    only_significant=True,
                    is_volume=is_volume,
                )
                if fig_lang is not None:
                    self.logger.log_image(f"{prefix}lanA_surface", fig_lang, step)
                    plt.close(fig_lang)

        # ROI analysis
        if roi_masks:
            if not isinstance(roi_masks, dict):
                raise TypeError(
                    "`roi_masks` must be a dict like {'V1': mask, 'AC1': mask, ...}"
                )

            for name, mask in roi_masks.items():
                arr = np.asarray(mask, dtype=bool)
                if arr.shape[0] != correlations.shape[0]:
                    raise ValueError(
                        f"ROI '{name}' mask must match correlations length, got {arr.shape} vs {correlations.shape}"
                    )

                key = _sanitize(name)
                roi_vals = correlations[arr]
                mean_v = float(np.nanmean(roi_vals)) if roi_vals.size else np.nan
                median_v = float(np.nanmedian(roi_vals)) if roi_vals.size else np.nan

                self.logger.log_scalar(f"{prefix}{key}_mean", mean_v, step)
                self.logger.log_scalar(f"{prefix}{key}_median", median_v, step)

                clean = roi_vals[~np.isnan(roi_vals)]
                if clean.size:
                    self.logger.log_histogram(f"{prefix}{key}_hist", clean, step)

                if not is_volume:
                    fig_roi = self.plot_surface_correlations(
                        correlations=correlations,
                        significant_mask=arr,
                        title=f"{name} — Masked",
                        only_significant=True,
                        is_volume=is_volume,
                    )
                    if fig_roi is not None:
                        self.logger.log_image(f"{prefix}{key}_surface", fig_roi, step)
                        plt.close(fig_roi)
