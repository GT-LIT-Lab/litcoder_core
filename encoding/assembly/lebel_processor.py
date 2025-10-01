"""Processor for the Lebel dataset."""

import sys
import pickle

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

from .base_processor import BaseAssemblyGenerator, StoryData
from .assemblies import SimpleNeuroidAssembly
from transformers import GPT2Tokenizer
import logging



class LebelAssemblyGenerator(BaseAssemblyGenerator):
    """Generator for Lebel dataset assemblies."""

    def __init__(
        self,
        data_dir: str,
        dataset_type: str,
        tr: float = 1.5,
        use_volume: bool = False,
        mask_path: Optional[str] = None,
        analysis_mask_path: Optional[str] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
        stories: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(data_dir, dataset_type, tr, use_volume, mask_path, tokenizer)
        self.analysis_mask = analysis_mask_path
        self.stories = stories if stories is not None else [
            "adollshouse",
            "adventuresinsayingyes",
            "alternateithicatom",
            "avatar",
            "buck",
            "exorcism",
            "eyespy",
            "fromboyhoodtofatherhood",
            "hangtime",
            "haveyoumethimyet",
            "howtodraw",
            "inamoment",
            "itsabox",
            "legacy",
            "naked",
            "odetostepfather",
            "sloth",
            "souls",
            "stagefright",
            "swimmingwithastronauts",
            "thatthingonmyarm",
            "theclosetthatateeverything",
            "tildeath",
            "undertheinfluence",
            "wheretheressmoke",
        ]

    def generate_assembly(
        self,
        subject: str,
        lookback: int = 256,
        context_type: str = "fullcontext",
        correlation_length: int = 100,
        generate_temporal_baseline: bool = False,
        audio_path: Optional[str] = None,
        **kwargs,
    ) -> SimpleNeuroidAssembly:
        """Generate assembly for a subject by processing all stories.

        Args:
            subject: Subject identifier
            lookback: Number of tokens to look back (default 256)
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)
        """
        story_data_list = []
        self.lookback = lookback
        self.context_type = context_type
        self.correlation_length = correlation_length
        self.generate_temporal_baseline = generate_temporal_baseline

        # Process each story
        #TODO: fix this to load big files once outside loop
        for story in self.stories:
            story_data = self._process_single_story(
                subject=subject,
                story_name=story,
                correlation_length=correlation_length,
                generate_temporal_baseline=generate_temporal_baseline,
                audio_path=audio_path,
                brain_resp_file= kwargs.get('brain_resp_file', 'brain_resp_huge.pkl'),
            )
            story_data_list.append(story_data)

        # Create assembly with story-level separation
        return SimpleNeuroidAssembly(story_data_list, validation_method="outer")

    def _discover_stories(self, subject_dir: Path) -> List[Dict[str, str]]:
        """Discover all stories for a subject from the directory structure.

        For Lebel dataset, we don't need to discover stories as they are predefined.
        """
        return []

    def _process_single_story(
        self,
        subject: str,
        story_name: str,
        correlation_length: int = 100,
        generate_temporal_baseline: bool = False,
        audio_path: Optional[str] = None,
        brain_resp_file: Optional[str] = None,
    ) -> StoryData:
        """Process a single story and return its data using a specified context type.

        Args:
            subject: Subject identifier
            story_name: Name of the story being processed
            wordseq: Word sequence data for the story
            brain_data: Neural activity data for the story
            lookback: Number of tokens to look back
            context_type: Type of context to use ("fullcontext", "nocontext", or "halfcontext")
            correlation_length: How far temporal correlation extends (in stimulus units)

        Returns:
            StoryData object containing processed story information
        """
        #TODO: Unify file structure for LITcoder
        with open(f"{self.data_dir}/{subject}/{brain_resp_file}", "rb") as f:
            resp_dict = pickle.load(f)
        
        brain_data = resp_dict.get(story_name)

        transcript, split_indices, tr_times, data_times, _ = self.process_transcript(
            self.data_dir,
            story_name
        )
        stimuli = self.generate_stimuli_with_context(transcript, self.lookback)

        if self.analysis_mask is not None:
            brain_data, mask_indices = self.apply_analysis_mask(brain_data, self.analysis_mask)
            logging.info(f"this is the mask indices: {mask_indices}")

        if generate_temporal_baseline:
            temporal_baseline = self.create_temporal_baseline(
                stimuli, correlation_length=correlation_length
            )
        else:
            temporal_baseline = None
        
        # make a transcript that has the word_orig, word_times, and word_times_tr
        
        word_rates = self.compute_word_rate_features(transcript, tr_times)

        return StoryData(
            name=story_name,
            brain_data=brain_data,
            words=transcript["word_orig"].tolist(),
            stimuli=stimuli,
            temporal_baseline=temporal_baseline,
            split_indices=split_indices,
            tr_times=tr_times,
            data_times=data_times,
            word_rates=word_rates,
            audio_path=audio_path,
        )
