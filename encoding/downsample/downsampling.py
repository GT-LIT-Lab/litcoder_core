"""Downsampling module for temporal alignment of continuous data with discrete measurements.

This module provides classes and utilities for downsampling continuous data (e.g., word embeddings)
to align with discrete measurements (e.g., fMRI TRs).
"""

from typing import List
import numpy as np
from abc import ABC, abstractmethod
from . import interpdata


class BaseDownsampler(ABC):
    """Abstract base class for downsampling implementations."""

    @abstractmethod
    def downsample(
        self, data: np.ndarray, data_times: np.ndarray, tr_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Abstract method for downsampling implementation."""
        pass


class RectangularDownsampler(BaseDownsampler):
    """Implements rectangular (box) filter downsampling."""

    def downsample(
        self, data: np.ndarray, data_times: np.ndarray, tr_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Downsample using rectangular window."""
        output = np.zeros((len(tr_times), data.shape[1]))
        tr = np.mean(np.diff(tr_times))  # TR duration

        for i, t in enumerate(tr_times):
            mask = (data_times >= t - tr / 2) & (data_times < t + tr / 2)
            if np.any(mask):
                output[i] = np.mean(data[mask], axis=0)

        return output


class LastPointDownsampler(BaseDownsampler):
    """Implements last-point downsampling using split indices."""

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray = None,
        tr_times: np.ndarray = None,
        split_indices: list = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Take the last embedding for words that fall within the same TR.
        Args:
            data: Array of word embeddings (n_words, embedding_dim)
            data_times: Original timestamps (optional)
            tr_times: Target timestamps (optional)
            split_indices: List indicating which TR each word belongs to
            **kwargs: Additional arguments
        Returns:
            Array of last embeddings per TR (n_TRs, embedding_dim)
        """
        if split_indices is None:
            raise ValueError(
                "split_indices must be provided for last point downsampling"
            )

        # Get number of TRs (unique split indices)
        n_trs = max(split_indices) + 1
        embedding_dim = data.shape[1]

        # Initialize output matrix
        last_embeddings = np.zeros((n_trs, embedding_dim))

        # For each TR, take the last embedding of words in that TR
        for tr_idx in range(n_trs):
            # Get indices of words in this TR
            word_indices = [
                i for i, split_idx in enumerate(split_indices) if split_idx == tr_idx
            ]

            if word_indices:  # If there are words in this TR
                # Take the last embedding (highest index) for this TR
                last_word_idx = max(word_indices)
                last_embeddings[tr_idx] = data[last_word_idx]

        return last_embeddings


class AverageDownsampler(BaseDownsampler):
    """Implements averaging-based downsampling using split indices."""

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray = None,
        tr_times: np.ndarray = None,
        split_indices: list = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Average embeddings for words that fall within the same TR.

        Args:
            data: Array of word embeddings (n_words, embedding_dim)
            data_times: Original timestamps (optional)
            tr_times: Target timestamps (optional)
            split_indices: List indicating which TR each word belongs to
            **kwargs: Additional arguments

        Returns:
            Array of averaged embeddings per TR (n_TRs, embedding_dim)
        """
        if split_indices is None:
            raise ValueError("split_indices must be provided for average downsampling")

        # Get number of TRs (unique split indices)
        n_trs = max(split_indices) + 1
        embedding_dim = data.shape[1]

        # Initialize output matrix
        averaged_embeddings = np.zeros((n_trs, embedding_dim))

        # For each TR, average the embeddings of words in that TR
        for tr_idx in range(n_trs):
            # Get indices of words in this TR
            word_indices = [
                i for i, split_idx in enumerate(split_indices) if split_idx == tr_idx
            ]

            if word_indices:  # If there are words in this TR
                # Average the embeddings for these words
                averaged_embeddings[tr_idx] = np.mean(data[word_indices], axis=0)

        return averaged_embeddings


class SincDownsampler(BaseDownsampler):
    """Implements sinc filter downsampling."""

    def downsample(
        self, data: np.ndarray, data_times: np.ndarray, tr_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Downsample using sinc interpolation."""
        return interpdata.sincinterp2D(data, data_times, tr_times, **kwargs)


class LanczosDownsampler(BaseDownsampler):
    """Implements Lanczos filter downsampling."""

    def downsample(
        self, data: np.ndarray, data_times: np.ndarray, tr_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Downsample using Lanczos interpolation."""
        # log the kwargs
        return interpdata.lanczosinterp2D(data, data_times, tr_times, **kwargs)


class GaborDownsampler(BaseDownsampler):
    """Implements Gabor filter downsampling."""

    def downsample(
        self, data: np.ndarray, data_times: np.ndarray, tr_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Downsample using Gabor transform."""
        return np.abs(interpdata.gabor_xfm2D(data.T, data_times, tr_times, **kwargs)).T


class LegacyLastPointDownsampler(BaseDownsampler):
    """Implements Legacy-style downsampling by taking the last point in each chunk."""

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray,
        tr_times: np.ndarray,
        split_indices=None,
        **kwargs,
    ) -> np.ndarray:
        """Downsample by selecting the last point in each chunk.

        Args:
            data: Input data array of shape (time, features)
            data_times: Original timestamps (not used here but included for consistency)
            tr_times: Target timestamps (not used here but included for consistency)
            split_indices: Indices where data should be split into chunks
        """
        if split_indices is None:
            raise ValueError("split_indices must be provided for Legacy downsampling")

        dsize = data.shape[1]
        outmat = np.zeros((len(split_indices) + 1, dsize))

        chunks = np.split(data, split_indices)

        for ci, chunk in enumerate(chunks):
            if len(chunk):
                outmat[ci] = chunk[-1]  # take the last point

        return outmat


class LegacyAverageDownsampler(BaseDownsampler):
    """Implements Legacy-specific downsampling method using chunk averaging."""

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray,
        tr_times: np.ndarray,
        split_indices=None,
        **kwargs,
    ) -> np.ndarray:
        """Downsample using Legacy's chunk averaging method.

        Args:
            data: Input data array
            data_times: Original timestamps
            tr_times: Target timestamps
            split_indices: Indices where data should be split into chunks
        """
        if split_indices is None:
            raise ValueError("split_indices must be provided for Legacy downsampling")

        dsize = data.shape[1]
        outmat = np.zeros((len(split_indices) + 1, dsize))

        # Split data into chunks
        chunks = np.split(data, split_indices)

        for ci, chunk in enumerate(chunks):
            if len(chunk):
                outmat[ci] = np.mean(chunk, axis=0)

        return outmat


class SumDownsampler(BaseDownsampler):
    """Implements sum-based downsampling using split indices."""

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray = None,
        tr_times: np.ndarray = None,
        split_indices: list = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Sum embeddings for words that fall within the same TR.

        Args:
            data: Array of word embeddings (n_words, embedding_dim)
            data_times: Original timestamps (optional)
            tr_times: Target timestamps (optional)
            split_indices: List indicating which TR each word belongs to
            **kwargs: Additional arguments

        Returns:
            Array of summed embeddings per TR (n_TRs, embedding_dim)
        """
        if split_indices is None:
            raise ValueError("split_indices must be provided for sum downsampling")

        # Get number of TRs (unique split indices)
        n_trs = max(split_indices) + 1
        embedding_dim = data.shape[1]

        # Initialize output matrix
        summed_embeddings = np.zeros((n_trs, embedding_dim))

        # For each TR, sum the embeddings of words in that TR
        for tr_idx in range(n_trs):
            # Get indices of words in this TR
            word_indices = [
                i for i, split_idx in enumerate(split_indices) if split_idx == tr_idx
            ]

            if word_indices:  # If there are words in this TR
                # Sum the embeddings for these words
                summed_embeddings[tr_idx] = np.sum(data[word_indices], axis=0)

        return summed_embeddings


class LegacySumDownsampler(BaseDownsampler):
    """Implements Legacy-specific downsampling method using chunk summing."""

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray,
        tr_times: np.ndarray,
        split_indices=None,
        **kwargs,
    ) -> np.ndarray:
        """Downsample using Legacy's chunk summing method.

        Args:
            data: Input data array
            data_times: Original timestamps
            tr_times: Target timestamps
            split_indices: Indices where data should be split into chunks
        """
        if split_indices is None:
            raise ValueError("split_indices must be provided for Legacy downsampling")

        dsize = data.shape[1]
        outmat = np.zeros((len(split_indices) + 1, dsize))

        # Split data into chunks
        chunks = np.split(data, split_indices)

        for ci, chunk in enumerate(chunks):
            if len(chunk):
                outmat[ci] = np.sum(chunk, axis=0)

        return outmat


class Downsampler:
    """Main downsampling class that provides a unified interface for various downsampling methods.

    This class handles the temporal alignment of continuous data (e.g., word embeddings)
    with discrete measurements (e.g., fMRI TRs) using various downsampling methods.
    """

    # Define method-specific parameter requirements
    METHOD_PARAMS = {
        "lanczos": {"required": ["window", "cutoff_mult"], "optional": ["rectify"]},
        "sinc": {
            "required": ["window", "cutoff_mult"],
            "optional": ["causal", "renorm"],
        },
        "average": {"required": ["split_indices"], "optional": []},
        "sum": {"required": ["split_indices"], "optional": []},
        "last": {"required": ["split_indices"], "optional": []},
        "legacy_average": {"required": ["split_indices"], "optional": []},
        "legacy_sum": {"required": ["split_indices"], "optional": []},
        "legacy_last": {"required": ["split_indices"], "optional": []},
        "rect": {"required": [], "optional": []},
        "gabor": {"required": ["freqs", "sigma"], "optional": []},
    }

    def __init__(self):
        """Initialize the downsampler with various methods."""
        self._methods = {
            "rect": RectangularDownsampler(),
            "average": AverageDownsampler(),
            "sinc": SincDownsampler(),
            "lanczos": LanczosDownsampler(),
            "last": LastPointDownsampler(),
            "gabor": GaborDownsampler(),
            "legacy_average": LegacyAverageDownsampler(),
            "legacy_last": LegacyLastPointDownsampler(),
            "sum": SumDownsampler(),
            "legacy_sum": LegacySumDownsampler(),
        }

    def _validate_method_params(self, method: str, **kwargs) -> dict:
        """Validate and filter parameters for a specific downsampling method.

        Args:
            method: The downsampling method to use
            **kwargs: Parameters to validate

        Returns:
            dict: Filtered parameters for the method

        Raises:
            ValueError: If required parameters are missing or method is not supported
        """
        if method not in self._methods:
            raise ValueError(f"Unsupported downsampling method: {method}")

        method_params = self.METHOD_PARAMS.get(method, {"required": [], "optional": []})
        filtered_params = {}

        # Check required parameters
        for param in method_params["required"]:
            if param not in kwargs:
                raise ValueError(
                    f"Required parameter '{param}' missing for method '{method}'"
                )
            filtered_params[param] = kwargs[param]

        # Add optional parameters if provided
        for param in method_params["optional"]:
            if param in kwargs:
                filtered_params[param] = kwargs[param]
        # log the filtered params
        return filtered_params

    def downsample(
        self,
        data: np.ndarray,
        data_times: np.ndarray,
        tr_times: np.ndarray,
        method: str = "rect",
        **kwargs,
    ) -> np.ndarray:
        """Downsample data using the specified method.

        Args:
            data: Input data array of shape (n_samples, n_features)
            data_times: Timestamps for each sample in data
            tr_times: Target timestamps for downsampled data
            method: Downsampling method ('rect', 'average', 'sinc', 'lanczos', 'gabor', 'legacy_average', 'sum', 'legacy_sum')
            **kwargs: Additional arguments passed to specific downsampling method

        Returns:
            np.ndarray: Downsampled data aligned with tr_times

        Raises:
            ValueError: If method is not supported or required parameters are missing
        """
        # Validate and filter parameters for the specific method
        filtered_params = self._validate_method_params(method, **kwargs)

        # Call the appropriate downsampling method with filtered parameters
        return self._methods[method].downsample(
            data, data_times, tr_times, **filtered_params
        )

    @property
    def available_methods(self) -> List[str]:
        """Get list of available downsampling methods."""
        return list(self._methods.keys())

    def get_method_params(self, method: str) -> dict:
        """Get the required and optional parameters for a specific method.

        Args:
            method: The downsampling method to query

        Returns:
            dict: Dictionary containing required and optional parameters for the method

        Raises:
            ValueError: If method is not supported
        """
        if method not in self._methods:
            raise ValueError(f"Unsupported downsampling method: {method}")
        return self.METHOD_PARAMS.get(method, {"required": [], "optional": []})
