"""Simple alternative to NeuroidAssembly that doesn't require brainio and Xarray."""

import numpy as np
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .assembly_generator import StoryData


class SimpleNeuroidAssembly:
    """Simple alternative to NeuroidAssembly that doesn't require brainio and Xarray."""

    def __init__(self, story_data_list: List["StoryData"], validation_method: str,is_volume:bool):
        """Initialize assembly with story-level separation.

        Args:
            story_data_list: List of StoryData objects, one for each story
        """
        self.stories = [story.name for story in story_data_list]
        self.story_data = {story.name: story for story in story_data_list}
        self.validation_method = validation_method
        self.is_volume = is_volume
        # Store combined data for backward compatibility
        self.data = np.vstack([story.brain_data for story in story_data_list])

        self.dims = ("presentation", "neuroid")
        self.shape = self.data.shape

        # Create coordinates dictionary
        self.coords = {
            "story_id": {
                "dim": "presentation",
                "values": np.repeat(
                    self.stories, [len(story.stimuli) for story in story_data_list]
                ),
            },
            "stimulus_id": {
                "dim": "presentation",
                "values": np.concatenate(
                    [np.arange(len(story.stimuli)) for story in story_data_list]
                ),
            },
        }

    def get_stimuli(self) -> List[List[str]]:
        """Get stimuli for each story.

        Returns:
            List of stimuli lists, one for each story
        """
        return [self.story_data[story].stimuli for story in self.stories]

    def get_split_indices(self) -> List[List[int]]:
        """Get split indices for each story.

        Returns:
            List of split indices lists, one for each story
        """
        return [self.story_data[story].split_indices for story in self.stories]

    def get_audio_path(self) -> List[str]:
        """Get audio path for each story.

        Returns:
            List of audio path lists, one for each story
        """
        return [self.story_data[story].audio_path for story in self.stories]

    def get_validation_method(self) -> str:
        """Get validation method for the assembly.
        For example, inner means inner cross-validation(narratives and lpp) and outer means a separate test set like lebel.
        """
        return self.validation_method

    def get_data_times(self) -> List[np.ndarray]:
        """Get data times for each story.

        Returns:
            List of data times arrays, one for each story
        """
        return [self.story_data[story].data_times for story in self.stories]

    def get_tr_times(self) -> List[np.ndarray]:
        """Get TR times for each story.

        Returns:
            List of TR times arrays, one for each story
        """
        return [self.story_data[story].tr_times for story in self.stories]

    def get_brain_data(self) -> List[np.ndarray]:
        """Get brain data for each story.

        Returns:
            List of brain data arrays, one for each story
        """
        return [self.story_data[story].brain_data for story in self.stories]

    def get_temporal_baseline(self, story_name: str) -> np.ndarray:
        """Get temporal baseline features for a specific story.

        Args:
            story_name: Name of the story to get features for

        Returns:
            np.ndarray: Temporal baseline features for the story
        """
        if story_name not in self.story_data:
            raise ValueError(f"Story {story_name} not found in assembly")
        return self.story_data[story_name].temporal_baseline

    def get_all_temporal_baselines(self) -> List[np.ndarray]:
        """Get temporal baseline features for all stories.

        Returns:
            List[np.ndarray]: List of temporal baseline features for each story
        """
        return [self.story_data[story].temporal_baseline for story in self.stories]

    def __getitem__(self, idx):
        return self.data[idx]

    def get_words(self) -> List[List[str]]:
        """Get words for each story.

        Returns:
            List of words lists, one for each story
        """
        return [self.story_data[story].words for story in self.stories]

    def get_word_rates(self) -> List[np.ndarray]:
        """Get word rate features for each story.

        Returns:
            List of word rate arrays, one for each story
        """
        return [self.story_data[story].word_rates for story in self.stories]

    def get_coord(self, name: str) -> np.ndarray:
        """Get coordinate values by name."""
        return self.coords[name]["values"]

    def coords_for_dim(self, dim_name: str) -> Dict[str, np.ndarray]:
        """Get all coordinates for a specific dimension."""
        return {
            name: info["values"]
            for name, info in self.coords.items()
            if info["dim"] == dim_name
        }

    def __repr__(self) -> str:
        lines = []
        lines.append(f"<SimpleNeuroidAssembly {self.shape}>")
        dims_str = (
            "("
            + ", ".join(f"{dim}: {size}" for dim, size in zip(self.dims, self.shape))
            + ")"
        )
        lines.append(dims_str)
        lines.append("")

        lines.append("Stories:")
        for story in self.stories:
            story_data = self.story_data[story]
            lines.append(f"  * {story}")
            lines.append(f"    - Stimuli: {len(story_data.stimuli)}")
            lines.append(f"    - Brain data shape: {story_data.brain_data.shape}")
            lines.append(f"    - Split indices: {len(story_data.split_indices)}")
            lines.append(f"    - TR times: {len(story_data.tr_times)}")
            lines.append(f"    - Data times: {len(story_data.data_times)}")

        lines.append("")
        lines.append("Attributes: (0)")

        return "\n".join(lines)
