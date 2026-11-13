"""Shared stimulus context policies, without neural-data or model setup.

The policies intentionally retain existing fMRI semantics. Tokenizer and
context_type must be provided by the caller; no tokenizer is downloaded here.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class StimulusContextMixin:
    def generate_stimuli_with_context(
        self, transcript: pd.DataFrame, lookback: int, *, source_starts=None
    ) -> List[str]:
        """Generate stimuli with token-based context window."""
        if self.context_type == "fullcontext":
            return self._process_fullcontext(transcript, lookback, source_starts=source_starts)
        elif self.context_type == "nocontext":
            return self._process_nocontext(transcript, lookback)
        elif self.context_type == "halfcontext":
            return self._process_halfcontext(transcript, lookback)
        else:
            raise ValueError(f"Invalid context type: {self.context_type}")

    def _process_fullcontext(
        self, transcript: pd.DataFrame, lookback: int, *, source_starts=None
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
        print(f"this is the lookback: {lookback}")
        print(f"heloo")

        for i, w in enumerate(ds_data):
            if source_starts is not None:
                source_starts.append(max(0, i - lookback) if w else i)
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
