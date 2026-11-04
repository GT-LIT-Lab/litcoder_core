"""Assembly generator for brain data processing and organization."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from .assemblies import SimpleNeuroidAssembly

if TYPE_CHECKING:
    from transformers import GPT2Tokenizer
    from .base_processor import BaseAssemblyGenerator
    from .ecog_processor import ECoGAssemblyGenerator


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
    ) -> BaseAssemblyGenerator | ECoGAssemblyGenerator:
        """Create a dataset-specific assembly generator.

        Args:
            dataset_type: 'narratives', 'lpp', 'lebel', or 'ecog'
            data_dir: Base directory containing subject data
            tr: TR value for timing calculations
            use_volume: Whether to use volume data instead of surface data

        Returns:
            BaseAssemblyGenerator: Dataset-specific assembly generator
        """
        if dataset_type == "ecog":
            if tr != 1.5:
                raise ValueError("ECoG uses recording sampling_rate_hz, not the fMRI tr argument")
            if use_volume or mask_path is not None or analysis_mask_path is not None:
                raise ValueError("fMRI volume/mask options do not apply to ECoG")
            from .ecog_processor import ECoGAssemblyGenerator
            # tr is a legacy fMRI factory argument. ECoG always requires a
            # declared sampling_rate_hz and timestamps in each recording.
            return ECoGAssemblyGenerator(data_dir, tokenizer=tokenizer)

        if dataset_type not in {"narratives", "lpp", "lebel"}:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Do not force ECoG-only assembly loading through fMRI dependencies.
        from .narratives_processor import NarrativesAssemblyGenerator
        from .lpp_processor import LPPAssemblyGenerator
        from .lebel_processor import LebelAssemblyGenerator

        generators = {
            "narratives": NarrativesAssemblyGenerator,
            "lpp": LPPAssemblyGenerator,
            "lebel": LebelAssemblyGenerator,
        }

        return generators[dataset_type](
            data_dir,
            dataset_type,
            tr,
            use_volume,
            mask_path,
            analysis_mask_path,
            tokenizer,
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
        recordings=None,
    ) -> SimpleNeuroidAssembly:
        """Generate assembly for a subject using the appropriate dataset processor.

        Args:
            dataset_type: 'narratives', 'lpp', 'lebel', or 'ecog'
            data_dir: Base directory containing subject data
            subject: Subject identifier
            tr: TR value for timing calculations
            lookback: Number of words to look back for context
            context_type: Type of context window to use
            correlation_length: How far temporal correlation extends (in stimulus units)
            use_volume: Whether to use volume data instead of surface data
            mask_path: Path to mask file for volume data
            tokenizer: Tokenizer for encoding and decoding text (default: GPT2Tokenizer)
            recordings: Explicit preprocessed ECoG recording mappings, instead
                of a data_dir manifest. Not supported for fMRI datasets.

        Returns:
            SimpleNeuroidAssembly: Generated assembly
        """
        if recordings is not None and dataset_type != "ecog":
            raise ValueError("recordings is supported only for ECoG")
        generator = AssemblyGenerator.create(
            dataset_type,
            data_dir,
            tr,
            use_volume,
            mask_path,
            analysis_mask_path,
            tokenizer,
        )
        if dataset_type == "ecog":
            return generator.generate_assembly(
                subject, lookback, context_type, correlation_length,
                generate_temporal_baseline, recordings=recordings,
            )
        return generator.generate_assembly(
            subject,
            lookback,
            context_type,
            correlation_length,
            generate_temporal_baseline,
        )
