from typing import Optional, List, Dict
from pathlib import Path
import glob
import nibabel as nib
from .base_processor import BaseAssemblyGenerator
from .assemblies import SimpleNeuroidAssembly
from transformers import GPT2Tokenizer
from .story_data import StoryData
from ..brain_projection.simple_cache import get_surface_cache
import logging


class NarrativesAssemblyGenerator(BaseAssemblyGenerator):
    """Generator for Narratives dataset assemblies."""

    def __init__(
        self,
        data_dir: str,
        dataset_type: str,
        tr: float = 1.5,
        use_volume: bool = False,
        mask_path: Optional[str] = None,
        analysis_mask_path: Optional[str] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
        **kwargs,
    ):
        super().__init__(data_dir, dataset_type, tr, use_volume, mask_path, tokenizer)
        self.analysis_mask = analysis_mask_path

    def generate_assembly(
        self,
        subject: str,
        lookback: int = 256,
        context_type: str = "fullcontext",
        correlation_length: int = 100,
        generate_temporal_baseline: bool = False,
    ) -> SimpleNeuroidAssembly:
        """Generate assembly for a subject by automatically discovering all stories.

        Args:
            subject: Subject identifier
            lookback: Number of tokens to look back (default 256)
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)
        """

        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        # Find all stories for this subject
        story_configs = self._discover_stories(subject_dir)
        if not story_configs:
            raise ValueError(f"No stories found for subject {subject}")

        story_data_list = []
        self.context_type = context_type
        self.correlation_length = correlation_length
        self.generate_temporal_baseline = generate_temporal_baseline
        self.lookback = lookback

        # Process each story
        for story_config in story_configs:
            story_data = self._process_single_story(
                story_name=story_config["name"],
                subject=subject,
                volume_path=story_config["volume_path"],
                correlation_length=self.correlation_length,
                generate_temporal_baseline=self.generate_temporal_baseline,
                audio_path=story_config["audio_path"],
            )
            story_data_list.append(story_data)

        # Create assembly with story-level separation
        return SimpleNeuroidAssembly(story_data_list, validation_method="inner",is_volume=self.use_volume)

    def _discover_stories(self, subject_dir: Path) -> List[Dict[str, str]]:
        """Discover all stories for a subject from the directory structure."""
        story_configs = []

        # Find all volume data files
        volume_files = glob.glob(
            str(
                subject_dir
                / "sub-*_task-21styear_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )
        )
        logging.info(f"this is the volume files: {volume_files}")
        # find audio files
        audio_files = glob.glob(str(self.data_dir / "21styear.wav"))
        logging.info(f"this is the audio files: {audio_files}")
        story_name = "21styear"  # We don't have to change this as this is the story we'll use!  # Move the story name to the class level
        # Look for corresponding transcript and events files
        events_file = (
            subject_dir / f"sub-{subject_dir.name[-3:]}_task-{story_name}_events.tsv"
        )
        if len(volume_files) > 0:
            story_configs.append(
                {
                    "name": story_name,
                    "volume_path": volume_files[0],
                    "audio_path": audio_files[0],
                }
            )

        return story_configs

    def _process_single_story(
        self,
        subject: str,
        story_name: str,
        volume_path: str,
        correlation_length: int = 100,
        generate_temporal_baseline: bool = False,
        audio_path: str = None,
    ) -> StoryData:
        """Process a single story and return its data using a specified context type.

        Args:
            subject: Subject identifier
            story_name: Name of the story being processed
            volume_path: Path to volume data file
            transcript_path: Path to transcript file
            events_path: Path to events file
            lookback: Number of tokens to look back
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)

        Returns:
            StoryData object containing processed story information
        """
        # Try to get cached brain data first
        surface_cache = get_surface_cache()
        cached_data = surface_cache.get(subject, volume_path)

        if cached_data is not None:
            logging.info(f"Using cached brain data for subject {subject}")
            brain_data = cached_data
        else:
            # Load volume data and process
            volume_data = nib.load(volume_path)

            # Process brain data using the appropriate processor
            processed_data = self.brain_processor.process_brain_data(
                volume_data.get_fdata(), volume_data.affine
            )

            # These are all optimzation steps to avoid loading the brain data into memory twice.
            # We will most likely remove this or make this optional.
            # Get the brain data array based on the processor type
            if hasattr(processed_data, "combined"):  # Surface data
                logging.info("using surface data")
                brain_data = processed_data.combined
                # Cache the surface data
                surface_cache.set(subject, volume_path, brain_data)
            else:  # Volume data
                brain_data = processed_data.data

        # Process transcript
        transcript, split_indices, tr_times, data_times, _ = self.process_transcript(
            self.data_dir,
            story_name,
        )

        if self.analysis_mask is not None:
            brain_data, mask_indices = self.apply_analysis_mask(brain_data)
            logging.info(f"this is the mask indices: {mask_indices}")
            sampled_brain_data = brain_data
        else:
            sampled_brain_data = brain_data
            mask_indices = None

        # Generate stimuli with specified context type
        stimuli = self.generate_stimuli_with_context(transcript, self.lookback)

        # Create temporal baseline features
        if generate_temporal_baseline:
            temporal_baseline = self.create_temporal_baseline(
                stimuli, correlation_length=correlation_length
            )
        else:
            temporal_baseline = None

        word_rates = self.compute_word_rate_features(transcript, tr_times)

        return StoryData(
            name=story_name,
            brain_data=sampled_brain_data,
            stimuli=stimuli,
            temporal_baseline=temporal_baseline,
            split_indices=split_indices,
            tr_times=tr_times,
            data_times=data_times,
            word_rates=word_rates,
            words=transcript["word_orig"].tolist(),
            mask_indices=mask_indices,
            audio_path=audio_path,
        )
