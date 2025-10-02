"""Base classes for dataset processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import surface, datasets
from transformers import GPT2Tokenizer
from ..brain_projection import (
    SurfaceProcessor,
    VolumeProcessor,
)
from .story_data import StoryData
from .assemblies import SimpleNeuroidAssembly
import pickle
import logging


class BaseAssemblyGenerator(ABC):
    """Abstract base class for assembly generation."""

    def __init__(
        self,
        data_dir: str,
        dataset_type: str,
        tr: float = 1.5,
        use_volume: bool = False,
        mask_path: Optional[str] = None,
        analysis_mask_path: Optional[str] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tr = tr
        self.analysis_mask = analysis_mask_path
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else GPT2Tokenizer.from_pretrained("gpt2")
        )
        self.dataset_type = dataset_type

        self.brain_processor = (
            VolumeProcessor(mask_path=mask_path) if use_volume else SurfaceProcessor()
        )
        self.use_volume = use_volume

    @abstractmethod
    def generate_assembly(
        self,
        subject: str,
        lookback: int = 256,
        context_type: str = "fullcontext",
        generate_temporal_baseline: bool = False,
        correlation_length: int = 100,
    ) -> SimpleNeuroidAssembly:
        """Generate an assembly for a subject.

        Args:
            subject: Subject identifier
            lookback: Number of tokens to look back (default 256)
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            generate_temporal_baseline: Whether to generate temporal baseline features
            correlation_length: How far temporal correlation extends (in stimulus units)
        """
        pass

    @abstractmethod
    def _discover_stories(self, subject_dir: Path) -> List[Dict[str, str]]:
        """Discover all stories/runs for a subject from the directory structure."""
        pass

    @abstractmethod
    def _process_single_story(
        self,
        story_name: str,
        volume_path: str,
        transcript_path: str,
        events_path: Optional[str],
        lookback: int,
        context_type: str = "fullcontext",
        correlation_length: int = 100,
        generate_temporal_baseline: bool = False,
        audio_path: str = None,
    ) -> StoryData:
        """Process a single story/run and return its data.

        Args:
            story_name: Name of the story/run
            volume_path: Path to the volume data file
            transcript_path: Path to the transcript data file
            events_path: Path to the events data file
            lookback: Number of tokens to look back
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)
            generate_temporal_baseline: Whether to generate temporal baseline features

        Returns:
            StoryData object containing processed story information
        """
        pass

    def generate_stimuli_with_context(
        self, transcript: pd.DataFrame, lookback: int
    ) -> List[str]:
        """Generate stimuli with token-based context window."""
        if self.context_type == "fullcontext":
            return self._process_fullcontext(transcript, lookback)
        elif self.context_type == "nocontext":
            return self._process_nocontext(transcript, lookback)
        elif self.context_type == "halfcontext":
            return self._process_halfcontext(transcript, lookback)
        else:
            raise ValueError(f"Invalid context type: {self.context_type}")

    def _process_fullcontext(
        self, transcript: pd.DataFrame, lookback: int
    ) -> List[str]:
        """Process story with full context window.

        Args:
            transcript: DataFrame containing transcript data
            lookback: Number of tokens to look back

        Returns:
            List of processed stimuli with full context
        """
        total_len = len(transcript["word_orig"])
        ds_data = transcript["word_orig"].astype(str)
        stimuli = []
        #print(f"this is the lookback: {lookback}")
        #print(f"heloo")

        for i, w in enumerate(ds_data):
            if w != "":
                text = " ".join(ds_data[max(0, i - lookback) : min(i + 1, total_len)])
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > lookback:
                    tokens = tokens[-lookback:]
                    text = self.tokenizer.decode(tokens)
                stimuli.append(text.strip())
            else:
                stimuli.append("")

        return stimuli

    def _process_nocontext(self, transcript: pd.DataFrame, lookback: int) -> List[str]:
        """Process story with no context window.

        Args:
            transcript: DataFrame containing transcript data
            lookback: Number of tokens to look back

        Returns:
            List of processed stimuli with no context
        """
        total_len = len(transcript["word_orig"])
        ds_data = transcript["word_orig"]
        stimuli = []
        start_idx = 0  # Track where we start accumulating from

        for i, w in enumerate(ds_data):
            if w != "":
                # Get text from start_idx to current point
                text = " ".join(ds_data[start_idx : i + 1])
                tokens = self.tokenizer.encode(text, add_special_tokens=False)

                if len(tokens) >= lookback:
                    # We've hit lookback, use current context and reset start_idx
                    stimuli.append(text.strip())
                    start_idx = i + 1  # Start fresh from next word
                else:
                    # Not enough tokens yet, use current context
                    stimuli.append(text.strip())
            else:
                stimuli.append("")

        return stimuli

    def _process_halfcontext(
        self, transcript: pd.DataFrame, lookback: int
    ) -> List[str]:
        """Process story with half context window using a sliding window approach.

        Args:
            transcript: DataFrame containing transcript data
            lookback: Number of tokens to look back

        Returns:
            List of processed stimuli with half context window
        """
        ds_data = transcript["word_orig"]
        stimuli = []
        start = 0
        i = 0
        half_lookback = lookback // 2

        while i < len(ds_data):
            if ds_data[i] != "":
                # Build text window from current start to i (inclusive)
                text = " ".join(ds_data[start : i + 1])
                tokens = self.tokenizer.encode(text, add_special_tokens=False)

                # If token count exceeds lookback, reset window start
                if len(tokens) > lookback:
                    start += half_lookback
                    continue

                stimuli.append(text.strip())
            else:
                stimuli.append("")
            i += 1

        return stimuli

    def apply_analysis_mask(
        self, brain_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply analysis mask to brain data and return masked data + indices."""
        return self._apply_analysis_mask(brain_data)

    def _apply_analysis_mask(
        self, brain_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply analysis mask to brain data and return masked data + indices.

        Args:
            brain_data: Full brain data (n_timepoints, n_voxels/vertices)

        Returns:
            tuple: (masked_brain_data, mask_indices)
                - masked_brain_data: (n_timepoints, n_masked_voxels)
                - mask_indices: indices of masked voxels in original space
        """
        if self.analysis_mask is None:
            return brain_data, np.arange(brain_data.shape[1])

        # Ensure mask is boolean and right size
        mask = np.asarray(self.analysis_mask, dtype=bool)
        if mask.shape[0] != brain_data.shape[1]:
            raise ValueError(
                f"Analysis mask size ({mask.shape[0]}) doesn't match brain data ({brain_data.shape[1]})"
            )

        # Apply mask
        masked_data = brain_data[:, mask]
        mask_indices = np.where(mask)[0]

        print(
            f"Applied analysis mask: {brain_data.shape[1]} -> {masked_data.shape[1]} voxels/vertices"
        )

        return masked_data, mask_indices

    def create_temporal_baseline(
        self, stimuli_list, d_model=128, correlation_length=75
    ):
        """
        Create temporal baseline features for a list of stimuli.
        This is completely content-agnostic we only care about temporal position!

        Args:
            stimuli_list: List of stimuli (content doesn't matter!)
            d_model: Dimension of the baseline features
            correlation_length: How far temporal correlation extends

        Returns:
            Array of shape (n_stimuli, d_model) with temporal baseline features
        """
        n_stimuli = len(stimuli_list)

        print(f"Creating temporal baseline for {n_stimuli} stimuli...")
        print(f"  d_model: {d_model}")
        print(f"  correlation_length: {correlation_length}")

        # Create the temporal baseline (content-agnostic!)
        temporal_features = self._create_temporal_baseline(
            n_stimuli=n_stimuli, d_model=d_model, correlation_length=correlation_length
        )
        return temporal_features

    def _create_temporal_baseline(self, n_stimuli, d_model=128, correlation_length=75):
        """
        Create temporal baseline features that are completely content-agnostic.
        Only depends on temporal position - stimuli close in time get similar representations.

        Args:
            n_stimuli: Number of stimuli in the sequence
            d_model: Dimension of the baseline features
            correlation_length: How far temporal correlation extends (in stimulus units)

        Returns:
            Array of shape (n_stimuli, d_model) with temporal baseline features
        """
        # Create autocorrelation matrix with exponential decay
        autocorr_matrix = np.zeros((n_stimuli, n_stimuli))

        for i in range(n_stimuli):
            for j in range(n_stimuli):
                distance = abs(i - j)
                autocorr_matrix[i, j] = np.exp(-distance / correlation_length)

        # Use SVD to create feature representation
        U, s, Vt = np.linalg.svd(autocorr_matrix)
        features = U[:, :d_model] * np.sqrt(s[:d_model])

        return features

    def compute_word_rate_features(
        self, transcript: pd.DataFrame, tr_times: np.ndarray
    ) -> np.ndarray:
        """Compute word rate by counting words per TR bin."""
        word_rates = []
        transcript["word_orig"] = transcript["word_orig"].astype(str)

        for i in range(len(tr_times)):
            # Define time window for this TR
            tr_start = tr_times[i]
            if i == len(tr_times) - 1:
                tr_end = tr_start + self.tr
            else:
                tr_end = tr_times[i + 1]

            # Find words that fall within this TR window
            words_in_tr = transcript[
                (transcript["word_times"] >= tr_start)
                & (transcript["word_times"] < tr_end)
            ]

            # Count non-empty words
            word_count = len([w for w in words_in_tr["word_orig"] if w.strip() != ""])
            word_rates.append([float(word_count)])
        return np.array(word_rates)  # Shape: (n_trs, 1)
    
    def process_transcript(
        self,
        data_dir: str,
        transcript_file: str,
        story_name: str,
    ) -> Tuple[pd.DataFrame, List[int], np.ndarray, np.ndarray]:
        """Process transcript data and generate split indices and timing information."""
        #TODO: Unify file structure for LITcoder
        #TODO: make it so stimulis data is in a dictionary instead of a list, or make file store modular
        # read pickle file
        with open(os.path.join(data_dir, f"transcripts/{transcript_file}"), "rb") as f:
            data = pickle.load(f)

        # this is a list, iterate over it and find the story_name
        #story = next((s for s in data if s.get("story_name") == story_name), None)
        story = data[story_name]

        if story is None:
            available = [s.get("story_name") for s in data]
            raise ValueError(
                f"Story '{story_name}' not found in {self.dataset_type}_data.pkl. Available stories: {available}"
            )

        words = story["words"]
        split_indices = story["split_indices"]
        tr_times = story["tr_times"]
        data_times = story["data_times"]
        words = pd.DataFrame({"word_orig": words, "word_times": data_times})
        # try to get TR_onset if it exists
        TR_onset = None

        if "TR_onset" in story:
            TR_onset = story["TR_onset"]

        return words, split_indices, tr_times, data_times, TR_onset
