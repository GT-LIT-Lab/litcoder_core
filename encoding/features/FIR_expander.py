from __future__ import annotations
from typing import Iterable, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np

@dataclass
class FIR:
    """
    Finite Impulse Response (FIR) expander for creating delayed feature matrices.

    Usage options:
      - Static/class usage: FIR.make_delayed(stim, delays, circpad=False)
      - Instance usage: FIR(delays, circpad).expand(stim)
    """

    delays: Optional[Iterable[int]] = None
    circpad: bool = False

    def expand(self, stim: np.ndarray) -> np.ndarray:
        if self.delays is None:
            raise ValueError("delays must be provided for instance usage of FIR")
        return FIR.make_delayed(stim, self.delays, self.circpad)

    @staticmethod
    def make_delayed(
        stim: np.ndarray, delays: Iterable[int], circpad: bool = False
    ) -> np.ndarray:
        import numpy as np
        nt, ndim = stim.shape
        dstims = []
        for d in delays:
            dstim = np.zeros((nt, ndim))
            if d < 0:
                dstim[:d, :] = stim[-d:, :]
                if circpad:
                    dstim[d:, :] = stim[:-d, :]
            elif d > 0:
                dstim[d:, :] = stim[:-d, :]
                if circpad:
                    dstim[:d, :] = stim[-d:, :]
            else:
                dstim = stim.copy()
            dstims.append(dstim)
        return np.hstack(dstims)

    def n_delays(self) -> int:
        """Return the number of delays used."""
        return len(self.delays) if self.delays is not None else 0

    def output_dim(self, input_dim: int) -> int:
        """Return the output dimensionality after FIR expansion."""
        return input_dim * self.n_delays()

    def valid_length(self, nt: int) -> int:
        """
        Number of valid time points (non-padded).
        With circpad=True, always nt.
        Without circpad, depends on max shift.
        """
        if self.delays is None:
            raise ValueError("delays must be provided")
        if self.circpad:
            return nt
        max_shift = max(abs(d) for d in self.delays)
        return max(0, nt - max_shift)

    def summary(self, input_dim: Optional[int] = None, nt: Optional[int] = None) -> str:
        """Return a readable summary of FIR configuration."""
        msg = f"FIR(delays={list(self.delays)}, circpad={self.circpad})"
        if input_dim is not None:
            msg += f"\n- Output dim: {self.output_dim(input_dim)}"
        if nt is not None:
            msg += f"\n- Valid length: {self.valid_length(nt)}"
        return msg
