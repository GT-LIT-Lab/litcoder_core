"""Assembly generator for brain data processing and organization."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import surface, datasets
from .base_processor import BaseAssemblyGenerator
from .narratives_processor import NarrativesAssemblyGenerator

from .lpp_processor import LPPAssemblyGenerator
from .lebel_processor import LebelAssemblyGenerator
from .assemblies import SimpleNeuroidAssembly
from transformers import GPT2Tokenizer


class AssemblyGenerator:
    """Factory class for creating dataset-specific assembly generators."""

    @staticmethod
    def create(
        dataset_type: str,
        data_dir: str,
        tr: float = 1.5,
        use_volume: bool = False,
        mask_path: Optional[str] = None,
        analysis_mask_path: Optional[str] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
        **kwargs,
    ) -> BaseAssemblyGenerator:
        """Create a dataset-specific assembly generator.

        Args:
            dataset_type: Type of dataset ('narratives', 'lpp', or 'lebel')
            data_dir: Base directory containing subject data
            tr: TR value for timing calculations
            use_volume: Whether to use volume data instead of surface data

        Returns:
            BaseAssemblyGenerator: Dataset-specific assembly generator
        """
        generators = {
            "narratives": NarrativesAssemblyGenerator,
            "lpp": LPPAssemblyGenerator,
            "lebel": LebelAssemblyGenerator,
        }

        if dataset_type not in generators:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        return generators[dataset_type](
            data_dir,
            dataset_type,
            tr,
            use_volume,
            mask_path,
            analysis_mask_path,
            tokenizer,
            **kwargs
        )

    @staticmethod
    def generate_assembly(
        dataset_type: str,
        data_dir: str,
        subject: str,
        tr: float = 1.5,
        lookback: int = 256,
        context_type: str = "fullcontext",
        correlation_length: int = 100,
        use_volume: bool = False,
        mask_path: Optional[str] = None,
        generate_temporal_baseline: bool = False,
        analysis_mask_path: Optional[str] = None,
        tokenizer: Optional[GPT2Tokenizer] = None,
        **kwargs,
    ) -> SimpleNeuroidAssembly:
        """Generate assembly for a subject using the appropriate dataset processor.

        Args:
            dataset_type: Type of dataset ('narratives', 'lpp', or 'lebel')
            data_dir: Base directory containing subject data
            subject: Subject identifier
            tr: TR value for timing calculations
            lookback: Number of words to look back for context
            context_type: Type of context window to use
            correlation_length: How far temporal correlation extends (in stimulus units)
            use_volume: Whether to use volume data instead of surface data
            mask_path: Path to mask file for volume data
            tokenizer: Tokenizer for encoding and decoding text (default: GPT2Tokenizer)

        Returns:
            SimpleNeuroidAssembly: Generated assembly
        """
        generator = AssemblyGenerator.create(
            dataset_type,
            data_dir,
            tr,
            use_volume,
            mask_path,
            analysis_mask_path,
            tokenizer,
            **kwargs
        )
        return generator.generate_assembly(
            subject,
            lookback,
            context_type,
            correlation_length,
            generate_temporal_baseline,
            **kwargs
        )
