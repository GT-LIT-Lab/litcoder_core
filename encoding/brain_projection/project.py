import nibabel as nib
import numpy as np
from nilearn import surface, datasets
from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SurfaceData:
    """Data class to hold surface data information."""

    left_hemisphere: np.ndarray
    right_hemisphere: np.ndarray
    combined: np.ndarray


@dataclass
class VolumeData:
    """Data class to hold volume data information."""

    data: np.ndarray  # Shape: (n_timepoints, n_voxels)


class BaseBrainDataProcessor(ABC):
    """Abstract base class for brain data processing."""

    @abstractmethod
    def process_brain_data(
        self, volume_data: np.ndarray, affine: np.ndarray
    ) -> Union[SurfaceData, VolumeData]:
        """Process brain data (either surface or volume).

        Args:
            volume_data: 4D numpy array of shape (x, y, z, time)
            affine: Affine transformation matrix

        Returns:
            Either SurfaceData or VolumeData object
        """
        pass


class SurfaceProcessor(BaseBrainDataProcessor):
    """Processor for surface data.

    Args:
        mesh: Mesh to use for surface projection. Default is "fsaverage5".
    """

    def __init__(self, mesh: str = "fsaverage5"):
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
        self.mesh_left = surface.load_surf_mesh(self.fsaverage["pial_left"])
        self.mesh_right = surface.load_surf_mesh(self.fsaverage["pial_right"])

    def process_brain_data(
        self, volume_data: np.ndarray, affine: np.ndarray
    ) -> SurfaceData:
        """Project volumetric data to surface for both hemispheres."""
        n_timepoints = volume_data.shape[3]
        n_vertices_left = self.mesh_left[0].shape[0]
        n_vertices_right = self.mesh_right[0].shape[0]

        surface_data_left = np.zeros((n_timepoints, n_vertices_left))
        surface_data_right = np.zeros((n_timepoints, n_vertices_right))

        for t in range(n_timepoints):
            vol_t = volume_data[:, :, :, t]
            img_t = nib.Nifti1Image(vol_t, affine)

            data_left = surface.vol_to_surf(img_t, self.mesh_left)
            data_right = surface.vol_to_surf(img_t, self.mesh_right)

            surface_data_left[t, :] = data_left
            surface_data_right[t, :] = data_right

        combined = np.column_stack((surface_data_left, surface_data_right))
        return SurfaceData(surface_data_left, surface_data_right, combined)


class VolumeProcessor(BaseBrainDataProcessor):
    """Processor for volume data with optional masking."""

    def __init__(self, mask_path: Optional[str] = None):
        print(f"Initializing VolumeProcessor with mask_path: {mask_path}")
        self.mask = None
        if mask_path is not None:
            mask_img = nib.load(mask_path)
            self.mask = mask_img.get_fdata().astype(bool)  # 3D boolean mask

    def process_brain_data(
        self, volume_data: np.ndarray, affine: np.ndarray
    ) -> VolumeData:
        """Process volumetric data with optional masking and flattening.

        Args:
            volume_data: 4D numpy array of shape (x, y, z, time)
            affine: Affine transformation matrix

        Returns:
            VolumeData object containing masked and flattened data
        """
        n_timepoints = volume_data.shape[3]

        if self.mask is not None:
            assert (
                self.mask.shape == volume_data.shape[:3]
            ), f"Mask shape {self.mask.shape} does not match volume shape {volume_data.shape[:3]}"

            # Apply mask at each timepoint
            masked_data = np.zeros((n_timepoints, self.mask.sum()))
            for t in range(n_timepoints):
                vol_t = volume_data[:, :, :, t]
                masked_data[t, :] = vol_t[self.mask]

            return VolumeData(data=masked_data)

        else:
            # No mask, flatten full volume
            n_voxels = np.prod(volume_data.shape[:3])
            flattened_data = volume_data.reshape(n_voxels, n_timepoints).T
            return VolumeData(data=flattened_data)
