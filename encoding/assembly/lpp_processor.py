"""Processor for the little prince dataset."""

from __future__ import annotations
from typing import Optional, List, Dict, TYPE_CHECKING
from pathlib import Path

from .base_processor import BaseAssemblyGenerator

if TYPE_CHECKING:
    from transformers import GPT2Tokenizer
    from .assemblies import SimpleNeuroidAssembly
    from .story_data import StoryData


class LPPAssemblyGenerator(BaseAssemblyGenerator):
    """Generator for LPP dataset assemblies."""

    def __init__(
        self,
        data_dir: str,
        dataset_type: str,
        tr: float = 2.0,
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
        """Generate assembly for a subject by processing all runs.

        Args:
            subject: Subject identifier
            lookback: Number of tokens to look back (default 256)
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)
        """
        from .assemblies import SimpleNeuroidAssembly

        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        # Find all runs for this subject
        run_configs = self._discover_stories(subject_dir, subject)
        if not run_configs:
            raise ValueError(f"No runs found for subject {subject}")

        story_data_list = []
        self.context_type = context_type
        self.correlation_length = correlation_length
        self.generate_temporal_baseline = generate_temporal_baseline
        self.lookback = lookback
        # Process each run
        for run_config in run_configs:
            story_data = self._process_single_story(
                subject,
                run_config["name"],
                run_config["volume_path"],
                correlation_length,
                generate_temporal_baseline,
                audio_path=None,
            )
            story_data_list.append(story_data)

        # Create assembly with run-level separation
        return SimpleNeuroidAssembly(story_data_list, validation_method="inner",is_volume=self.use_volume)

    def _discover_stories(
        self, subject_dir: Path, subject: str
    ) -> List[Dict[str, str]]:
        """Discover all runs for a subject from the directory structure."""
        run_configs = []

        # Define run numbers and their corresponding sections
        runs = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
        sections = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Find all volume data files
        for run, section in zip(runs, sections):
            volume_file = (
                subject_dir
                / f"{subject}_task-lppEN_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold_fixed.nii.gz"
            )
            print(volume_file)
            if volume_file.exists():
                run_configs.append(
                    {
                        "name": f"run_{run}",
                        "volume_path": str(volume_file),
                        "section": section,
                    }
                )

        return run_configs

    def _process_single_story(
        self,
        subject: str,
        story_name: str,
        volume_path: str,
        correlation_length: int = 100,
        generate_temporal_baseline: bool = False,
        audio_path: str = None,
    ) -> StoryData:
        """Process a single run and return its data using a specified context type.

        Args:
            subject: Subject identifier
            story_name: Name of the run being processed
            volume_path: Path to volume data file
            events_path: Path to events file (None for LPP)
            lookback: Number of tokens to look back
            section: Section number for this run
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)

        Returns:
            StoryData object containing processed run information
        """
        from .story_data import StoryData
        from ..brain_projection.simple_cache import get_surface_cache

        # Try to get cached brain data first
        surface_cache = get_surface_cache()
        cached_data = surface_cache.get(subject, volume_path)

        if cached_data is not None:
            print(f"Using cached brain data for subject {subject}")
            brain_data = cached_data
        else:
            import nibabel as nib
            # Load volume data and process
            volume_data = nib.load(volume_path)

            # Process brain data using the appropriate processor
            processed_data = self.brain_processor.process_brain_data(
                volume_data.get_fdata(), volume_data.affine
            )

            if hasattr(processed_data, "combined"):  # Surface data
                print("using surface data")
                brain_data = processed_data.combined
                # Cache the surface data
                surface_cache.set(subject, volume_path, brain_data)
            else:  # Volume data
                brain_data = processed_data.data

        # Process transcript for this specific section
        print("am i here")
        transcript, split_indices, tr_times, data_times, TR_onset = (
            self.process_transcript(
                self.data_dir,
                story_name,
            )
        )

        brain_data = brain_data[4:, :]

        unique_trs = [int(tr) for tr in set(TR_onset)]
        sampled_brain_data = brain_data[unique_trs, :]

        if self.analysis_mask is not None:
            sampled_brain_data, mask_indices = self.apply_analysis_mask(
                sampled_brain_data
            )
            print(
                f"Applied analysis mask: {sampled_brain_data.shape[1]} voxels/vertices"
            )
        else:
            mask_indices = None

        # Generate stimuli with specified context type
        stimuli = self.generate_stimuli_with_context(transcript, self.lookback)
        if generate_temporal_baseline:
            temporal_baseline = self.create_temporal_baseline(
                stimuli, correlation_length=correlation_length
            )
        else:
            temporal_baseline = None

        word_rate_features = self.compute_word_rate_features(transcript, tr_times)
        return StoryData(
            name=story_name,
            brain_data=sampled_brain_data,
            stimuli=stimuli,
            temporal_baseline=temporal_baseline,
            split_indices=split_indices,
            tr_times=tr_times,
            data_times=data_times,
            words=transcript["word_orig"].tolist(),
            word_rates=word_rate_features,
            mask_indices=mask_indices,
            audio_path=audio_path,
        )
