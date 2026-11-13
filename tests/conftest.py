"""Shared fixtures for the synthetic end-to-end pipeline tests.

Builds a small assembly with the same structural conventions as
``data/assembly_lebel_uts03.pkl`` and a signal planted at known FIR lags.
Conventions (verified against all 25 stories of that file):

* ``len(split_indices) + 1 == len(tr_times)`` -- split_indices are *cut points*
  into the word axis, consumed by ``np.split``, not a per-word TR label.
* ``brain_data.shape[0] == len(tr_times) - 15`` -- 10 TRs trimmed at the head and
  5 at the tail upstream of the assembly.
* ``len(words) == len(stimuli) == len(data_times) == n_words``
* ``word_rates.shape == (n_trs, 1)``
"""

import os

# Must happen before anything imports matplotlib/wandb (encoding.trainer pulls in
# encoding.plotting.plotting_utils, which imports matplotlib and nilearn at module
# scope).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DISABLED", "true")

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pytest

from encoding.assembly.assemblies import SimpleNeuroidAssembly
from encoding.assembly.story_data import StoryData
from encoding.features.base import BaseFeatureExtractor
from encoding.features.embeddings import StaticEmbeddingFeatureExtractor

N_VERTICES = 20484          # fsaverage5, both hemispheres
N_FEATURES = 4
N_SIGNAL_VERTICES = 10      # vertices 0..9 carry the planted signal
PLANTED_LAGS = (2, 3)
FIR_DELAYS = [1, 2, 3, 4]
TR = 1.5
BRAIN_TRIM_HEAD = 10
BRAIN_TRIM_TAIL = 5
BRAIN_TRIM_TOTAL = BRAIN_TRIM_HEAD + BRAIN_TRIM_TAIL

# Same trimming used in train_simple.py for the LeBel dataset.
TRIMMING_CONFIG = {
    "train_features_start": 10,
    "train_features_end": -5,
    "train_targets_start": 0,
    "train_targets_end": None,
    "test_features_start": 50,
    "test_features_end": -5,
    "test_targets_start": 40,
    "test_targets_end": None,
}


def token_vector(token: str, dim: int = N_FEATURES) -> np.ndarray:
    """Map a token to a fixed pseudo-random vector.

    Uses blake2b rather than ``hash()`` so the mapping is stable across processes
    (Python salts str hashes per interpreter run).
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little")
    return np.random.default_rng(seed).standard_normal(dim)


class SyntheticTokenFeatureExtractor(StaticEmbeddingFeatureExtractor):
    """Word-level extractor that plugs into the real factory dispatch.

    Subclasses ``StaticEmbeddingFeatureExtractor`` so the factory feeds it
    ``assembly.get_words()[idx]``; the parent ``__init__`` is bypassed because it
    requires a gensim vector file. The class name must not contain "wordrate", or
    ``AbstractTrainer._should_downsample`` would skip downsampling for it.
    """

    def __init__(self, dim: int = N_FEATURES):
        BaseFeatureExtractor.__init__(self, {"model_name": "synthetic-tokens"})
        self.dim = dim

    def extract_features(self, stimuli, **kwargs) -> np.ndarray:
        if isinstance(stimuli, str):
            stimuli = stimuli.split()
        return np.vstack([token_vector(tok, self.dim) for tok in stimuli])


def reference_legacy_average(word_features: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-TR mean of word features, from counts rather than np.split; empty TRs stay zero."""
    out = np.zeros((len(counts), word_features.shape[1]))
    pos = 0
    for i, c in enumerate(counts):
        if c:
            out[i] = word_features[pos:pos + c].mean(axis=0)
        pos += c
    return out


def reference_make_delayed(features: np.ndarray, delays) -> np.ndarray:
    """Causal FIR expansion: block d of row t holds features[t - d], else zeros."""
    n_t, n_d = features.shape
    blocks = []
    for d in delays:
        block = np.zeros((n_t, n_d))
        block[d:] = features[:-d]
        blocks.append(block)
    return np.hstack(blocks)


@dataclass
class StoryTruth:
    n_trs: int
    n_words: int
    counts: np.ndarray                 # words per TR, length n_trs
    word_features: np.ndarray          # (n_words, N_FEATURES)
    tr_features: np.ndarray            # (n_trs, N_FEATURES), the correct downsampling
    empty_trs: np.ndarray              # indices of TRs containing no words


@dataclass
class GroundTruth:
    stories: Dict[str, StoryTruth] = field(default_factory=dict)
    lag_weights: Dict[int, np.ndarray] = field(default_factory=dict)  # lag -> (N_FEATURES, N_SIGNAL)
    signal_vertices: np.ndarray = field(default_factory=lambda: np.arange(N_SIGNAL_VERTICES))
    noise_vertices: np.ndarray = field(default_factory=lambda: np.arange(N_SIGNAL_VERTICES, N_VERTICES))


def synthetic_assembly(
    n_stories: int = 3,
    n_trs: Tuple[int, ...] = (200, 185, 210),
    n_vertices: int = N_VERTICES,
    n_features: int = N_FEATURES,
    n_signal_vertices: int = N_SIGNAL_VERTICES,
    planted_lags: Tuple[int, ...] = PLANTED_LAGS,
    noise_ratio: float = 0.5,
    max_words_per_tr: int = 12,
    seed: int = 0,
) -> SimpleNeuroidAssembly:
    """Build a small assembly with a signal planted at known FIR lags.

    Ground truth is attached as ``assembly.truth``. Brain data for the first
    ``n_signal_vertices`` is

        brain[j, v] = sum_{d in planted_lags} F[j + BRAIN_TRIM_HEAD - d] @ lag_weights[d][:, v]

    where ``F`` is the legacy-averaged TR-level features; other vertices are noise.
    The ``+ BRAIN_TRIM_HEAD`` offset matches ``TRIMMING_CONFIG``: the trainer pairs
    ``features[10:-5]`` with ``targets[0:]``, so brain row ``j`` is aligned with
    feature row ``j + 10``, whose FIR block for delay ``d`` holds ``F[j + 10 - d]``.

    ``noise_ratio`` is noise std as a fraction of each signal vertex's std; 0.5
    gives a ceiling of r ~= 0.89.
    """
    if n_stories < 2:
        raise ValueError("need at least 2 stories for a train/test split")
    if max(planted_lags) >= BRAIN_TRIM_HEAD:
        raise ValueError("planted lags must be < BRAIN_TRIM_HEAD to stay in range")

    rng = np.random.default_rng(seed)
    lag_weights = {
        lag: rng.standard_normal((n_features, n_signal_vertices))
        for lag in planted_lags
    }

    truth = GroundTruth(
        lag_weights=lag_weights,
        signal_vertices=np.arange(n_signal_vertices),
        noise_vertices=np.arange(n_signal_vertices, n_vertices),
    )

    story_data_list: List[StoryData] = []

    for s in range(n_stories):
        story_rng = np.random.default_rng(seed + 1000 + s)
        name = f"synthstory{s}"
        n_tr = n_trs[s % len(n_trs)]

        counts = story_rng.integers(0, max_words_per_tr, size=n_tr)
        # Mirror the real data: silence at the very start and end of a run.
        counts[:3] = 0
        counts[-2:] = 0
        n_words = int(counts.sum())

        # n_tr - 1 cut points, so np.split yields n_tr chunks.
        split_indices = np.cumsum(counts)[:-1].astype(np.int64).tolist()

        words = [f"w{s}_{i}" for i in range(n_words)]
        stimuli = [" ".join(words[max(0, i - 5):i + 1]) for i in range(n_words)]

        tr_times = np.arange(n_tr, dtype=np.float64) * TR

        data_times = np.empty(n_words, dtype=np.float64)
        pos = 0
        for i, c in enumerate(counts):
            if c:
                offsets = np.sort(story_rng.uniform(0, TR, size=c))
                data_times[pos:pos + c] = tr_times[i] + offsets
                pos += c

        word_features = np.vstack([token_vector(w, n_features) for w in words])
        tr_features = reference_legacy_average(word_features, counts)

        n_brain = n_tr - BRAIN_TRIM_TOTAL
        j = np.arange(n_brain)
        signal = np.zeros((n_brain, n_signal_vertices))
        for lag in planted_lags:
            signal += tr_features[j + BRAIN_TRIM_HEAD - lag] @ lag_weights[lag]

        brain = np.empty((n_brain, n_vertices), dtype=np.float64)
        signal_std = signal.std(axis=0)
        brain[:, :n_signal_vertices] = signal + story_rng.standard_normal(
            (n_brain, n_signal_vertices)
        ) * (signal_std * noise_ratio)
        brain[:, n_signal_vertices:] = story_rng.standard_normal(
            (n_brain, n_vertices - n_signal_vertices)
        ) * float(np.median(signal_std))

        truth.stories[name] = StoryTruth(
            n_trs=n_tr,
            n_words=n_words,
            counts=counts,
            word_features=word_features,
            tr_features=tr_features,
            empty_trs=np.flatnonzero(counts == 0),
        )

        story_data_list.append(
            StoryData(
                name=name,
                brain_data=brain,
                stimuli=stimuli,
                split_indices=split_indices,
                tr_times=tr_times,
                data_times=data_times,
                temporal_baseline=None,
                word_rates=counts.reshape(-1, 1).astype(np.float64),
                words=words,
                mask_indices=None,
                audio_path=f"/nonexistent/audio/{name}.wav",
            )
        )

    assembly = SimpleNeuroidAssembly(story_data_list, validation_method="outer")
    assembly.truth = truth
    return assembly


class _NullLogger:
    """Stand-in for WandBLogger/TensorBoardLogger that records scalars in memory."""

    def __init__(self, *args, **kwargs):
        self.scalars: Dict[str, float] = {}

    def log_scalar(self, name, value, step=None):
        self.scalars[name] = value

    def log_image(self, name, figure, step=None):
        pass

    def log_histogram(self, name, values, step=None):
        pass


class _NullBrainPlotter:
    """Stand-in for BrainPlotter; the real one downloads fsaverage and renders 3D."""

    def __init__(self, logger=None):
        self.logger = logger
        self.calls = 0

    def log_plots(self, *args, **kwargs):
        self.calls += 1

    def log_descriptive_correlations(self, *args, **kwargs):
        self.calls += 1


@pytest.fixture(autouse=True)
def _no_external_side_effects(monkeypatch):
    """Keep tests offline: no wandb run, no nilearn surface download, no GUI backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)

    import encoding.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "WandBLogger", _NullLogger)
    monkeypatch.setattr(trainer_mod, "TensorBoardLogger", _NullLogger)
    monkeypatch.setattr(trainer_mod, "BrainPlotter", _NullBrainPlotter)

    # setup_logger calls wandb.init() directly before constructing WandBLogger.
    import wandb
    monkeypatch.setattr(wandb, "init", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(wandb, "log", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(wandb, "finish", lambda *a, **k: None, raising=False)

    import encoding.plotting.plotting_utils as plotting_utils
    monkeypatch.setattr(
        plotting_utils.datasets,
        "fetch_surf_fsaverage",
        lambda *a, **k: pytest.fail("test tried to download fsaverage"),
        raising=False,
    )


@pytest.fixture(scope="module")
def assembly() -> SimpleNeuroidAssembly:
    return synthetic_assembly()


@pytest.fixture(scope="module")
def truth(assembly) -> GroundTruth:
    return assembly.truth
