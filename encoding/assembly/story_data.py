from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class StoryData:
    """Data class to hold story-specific data.


    Attributes:
        name (str): Name identifier for the story/run
        brain_data (np.ndarray): Brain activation data, shape (n_timepoints, n_voxels/vertices)
        is_volume (bool): true if brain data is volume data, false if surface
        stimuli (List[str]): List of text stimuli corresponding to each timepoint
        split_indices (List[int]): Indices marking TR boundaries in the data
        tr_times (np.ndarray): Array of TR timestamps
        data_times (np.ndarray): Array of precise timing for each datapoint
        temporal_baseline (Optional[np.ndarray]): Temporal baseline features if generated
        word_rates (Optional[np.ndarray]): Word presentation rates if calculated
        words (Optional[List[str]]): Individual words if stored separately
        mask_indices (Optional[np.ndarray]): Indices of masked voxels/vertices in original space
    """

    name: str
    brain_data: np.ndarray
    stimuli: List[str]
    split_indices: List[int]
    tr_times: np.ndarray
    data_times: np.ndarray
    temporal_baseline: Optional[np.ndarray] = None
    word_rates: Optional[np.ndarray] = None
    words: Optional[List[str]] = None
    mask_indices: Optional[np.ndarray] = None
    audio_path: Optional[str] = None
