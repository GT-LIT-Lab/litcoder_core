import os
import json
import pickle
import h5py
import scipy.io
from typing import Optional, Dict, Any, Tuple, List, Union
from functools import wraps
import numpy as np
from pathlib import Path
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

## Demean -- remove the mean from each column
demean = lambda v: v - v.mean(0)
demean.__doc__ = """Removes the mean from each column of [v]."""
dm = demean


## Z-score -- z-score each column
def zscore(v):
    s = v.std(0)
    m = v - v.mean(0)
    for i in range(len(s)):
        if s[i] != 0.0:
            m[:, i] /= s[i]
    return m


# zscore = lambda v: (v-v.mean(0))/v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Rescale -- make each column have unit variance
rescale = lambda v: v / v.std(0)
rescale.__doc__ = """Rescales each column of [v] to have unit variance."""
rs = rescale

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1, c2: (zs(c1) * zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""

## Cross corr -- find corr. between each row of c1 and EACH row of c2
xcorr = lambda c1, c2: np.dot(zs(c1.T).T, zs(c2.T)) / (c1.shape[1])
xcorr.__doc__ = """Cross-column correlation. Finds the correlation between each row of [c1] and each row of [c2]."""


def validate_path(func):
    """Decorator to validate that assembly_path exists before loading."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.assembly_path.exists():
            raise FileNotFoundError(f"Assembly file not found: {self.assembly_path}")
        return func(self, *args, **kwargs)

    return wrapper


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  ## negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


class LazyLayerCache:
    """Lazy loading cache for multi-layer activations."""

    def __init__(self, cache_file_path: Union[str, Path]):
        """Initialize the lazy layer cache.

        Args:
            cache_file_path: Path to the cache file
        """
        self.cache_file_path = Path(cache_file_path)
        self._metadata = None
        self._loaded_layers = {}

    def get_metadata(self) -> Dict[str, Any]:
        """Load only metadata (fast).

        Returns:
            Dictionary containing cache metadata
        """
        if self._metadata is None:
            if not self.cache_file_path.exists():
                raise FileNotFoundError(f"Cache file not found: {self.cache_file_path}")

            with open(self.cache_file_path, "rb") as f:
                cache_data = pickle.load(f)
                self._metadata = cache_data["metadata"]
        return self._metadata

    def get_layer(self, layer_idx: int) -> np.ndarray:
        """Load specific layer on demand.

        Args:
            layer_idx: Index of the layer to load

        Returns:
            Layer activations as numpy array
        """
        if layer_idx not in self._loaded_layers:
            if not self.cache_file_path.exists():
                raise FileNotFoundError(f"Cache file not found: {self.cache_file_path}")

            with open(self.cache_file_path, "rb") as f:
                cache_data = pickle.load(f)

            if layer_idx not in cache_data["layers"]:
                available_layers = list(cache_data["layers"].keys())
                raise ValueError(
                    f"Layer {layer_idx} not found in cache. Available layers: {available_layers}"
                )

            self._loaded_layers[layer_idx] = cache_data["layers"][layer_idx]
        return self._loaded_layers[layer_idx]

    def get_layers(self, layer_indices: List[int]) -> List[np.ndarray]:
        """Load multiple specific layers.

        Args:
            layer_indices: List of layer indices to load

        Returns:
            List of layer activations
        """
        return [self.get_layer(idx) for idx in layer_indices]

    def clear_loaded_layers(self) -> None:
        """Clear loaded layers from memory."""
        self._loaded_layers.clear()

    def get_available_layers(self) -> List[int]:
        """Get list of available layers in the cache.

        Returns:
            List of available layer indices
        """
        metadata = self.get_metadata()
        return metadata.get("available_layers", [])

    def validate_context_type(self, expected_context_type: str) -> None:
        """Validate that the cache was created with the expected context type.

        Args:
            expected_context_type: Expected context type

        Raises:
            ValueError: If context type doesn't match
        """
        metadata = self.get_metadata()
        cached_context_type = metadata.get("context_type")
        if cached_context_type != expected_context_type:
            raise ValueError(
                f"Cache context_type mismatch: expected {expected_context_type}, "
                f"got {cached_context_type}"
            )


class ActivationCache:
    """Handles caching and loading of language model activations with multi-layer support."""

    def __init__(self, cache_dir: str = "cache"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached activations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(
        self,
        story: str,
        lookback: int,
        model_name: str,
        context_type: str,
        last_token: bool,
        dataset_type: str,
        raw: bool,
        hook_type: str,
        stimuli_sha256: str,
    ) -> str:
        """Generate a unique cache key for the given parameters.

        Args:
            story: Story identifier
            lookback: Number of words to look back
            model_name: Name of the language model
            context_type: Type of context (fullcontext, nocontext, halfcontext)
            last_token: Whether to use only the last token
            dataset_type: Type of dataset
            hook_type: TransformerLens hook the activations were read from
            stimuli_sha256: Hash of the exact stimulus strings fed to the model

        Returns:
            Cache key string
        """
        params = {
            "story": story,
            "lookback": lookback,
            "model_name": model_name,
            "context_type": context_type,
            "last_token": last_token,
            "dataset_type": dataset_type,
            "raw": raw,
            "hook_type": hook_type,
            "stimuli_sha256": stimuli_sha256,
        }
        # Create a deterministic hash of the parameters
        key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return key

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the path for a cached activation file."""
        return self.cache_dir / f"{cache_key}.pkl"

    def save_multi_layer_activations(
        self,
        cache_key: str,
        all_layer_activations: Dict[int, np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        """Save multi-layer activations to cache with lazy loading support.

        Args:
            cache_key: Cache key for the activations
            all_layer_activations: Dictionary mapping layer indices to activations
            metadata: Metadata about the cached activations
        """
        cache_path = self.get_cache_path(cache_key)

        cache_data = {"metadata": metadata, "layers": all_layer_activations}

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved multi-layer activations to {cache_path}")

    def load_multi_layer_activations(self, cache_key: str) -> Optional[LazyLayerCache]:
        """Load multi-layer activations from cache with lazy loading.

        Args:
            cache_key: Cache key for the activations

        Returns:
            LazyLayerCache object if cache exists, None otherwise
        """
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Loading multi-layer activations from {cache_path}")
            return LazyLayerCache(cache_path)
        return None

    # Backward compatibility methods
    def save_activations(self, cache_key: str, activations: np.ndarray):
        """Save single layer activations to cache (backward compatibility)."""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, "wb") as f:
            pickle.dump(activations, f)
        logger.info(f"Saved activations to {cache_path}")

    def load_activations(self, cache_key: str) -> Optional[np.ndarray]:
        """Load single layer activations from cache (backward compatibility)."""
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Loading activations from {cache_path}")
            with open(cache_path, "rb") as f:
                activations = pickle.load(f)
            return activations
        return None


class ModelSaver:
    """Class for saving and loading model weights and hyperparameters."""

    def __init__(self, base_dir: str = "results"):
        """Initialize the ModelSaver.

        Args:
            base_dir: Base directory for saving results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _json_value(value):
        """Allow scientific configuration values without silently stringifying them."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize configuration value {type(value).__name__}")

    def save_encoding_model(
        self,
        weights: Optional[np.ndarray],
        best_alphas: Optional[np.ndarray],
        hyperparams: Dict[str, Any],
        metrics: Dict[str, Any],
        save_weights: bool = True,
        results: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Publish a complete, uniquely named version-2 run.

        Returned ridge arrays and fitted folds are saved when available.
        With save_weights=False fitted state is omitted and stored predictions
        and metadata remain available.
        Pickle artifacts, including legacy metrics, must come from trusted sources.
        """
        import tempfile
        from uuid import uuid4

        # Validate configuration serialization before creating any output.
        hyperparams_json = json.dumps(hyperparams, sort_keys=True, indent=2,
                                      default=self._json_value)
        run_hash = hashlib.md5(hyperparams_json.encode()).hexdigest()[:8]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = self.base_dir / f"run_{stamp}_{run_hash}_{uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(prefix=".saving_", dir=self.base_dir) as tmp:
            staging = Path(tmp)
            (staging / "hyperparams.json").write_text(hyperparams_json)
            if best_alphas is not None:
                np.save(staging / "best_alphas.npy", best_alphas)
            weights_saved = save_weights and weights is not None
            if weights_saved:
                np.save(staging / "weights.npy", weights)
            with open(staging / "metrics.pkl", "wb") as f:
                pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
            payload = results
            if results is not None and not save_weights:
                payload = {
                    **results,
                    "folds": [
                        {key: value for key, value in fold.items()
                         if key not in ("weights", "estimator_state")}
                        for fold in results.get("folds", [])
                    ],
                }
            if payload is not None:
                with open(staging / "results.pkl", "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest = {
                "format_version": 2,
                "weights_saved": weights_saved,
                "alphas_saved": best_alphas is not None,
                "results_saved": payload is not None,
                "replay_available": any(
                    "weights" in fold or "estimator_state" in fold
                    for fold in (payload or {}).get("folds", [])
                ),
                "returned_weights_role": (
                    results["returned_weights_role"] if results else "caller_supplied_unspecified"
                ),
                "returned_alphas_role": (
                    results["returned_alphas_role"] if results else "caller_supplied_unspecified"
                ),
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
            # Rename only after all files have been written. A failure is raised,
            # never reported as a successful run or confused with plotting errors.
            staging.replace(destination)
        return destination

    def load_encoding_result(self, run_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load a trusted run, including fitted folds and metadata when available.

        Versionless legacy runs may lack weights/alphas. Return None for absent
        weights and recover alphas only from recorded metrics, with an explicit
        warning. Never invent a reproducible legacy model.
        """
        run_dir = Path(run_dir)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and manifest.get("format_version") != 2:
            raise ValueError(f"Unsupported encoding result format: {manifest.get('format_version')}")
        with open(run_dir / "metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        hyperparams = json.loads((run_dir / "hyperparams.json").read_text())
        if manifest is not None:
            weights = np.load(run_dir / "weights.npy", allow_pickle=False) if manifest["weights_saved"] else None
            alphas = (np.load(run_dir / "best_alphas.npy", allow_pickle=False)
                      if manifest.get("alphas_saved", True) else None)
            if manifest["results_saved"]:
                with open(run_dir / "results.pkl", "rb") as f:
                    results = pickle.load(f)
            else:
                results = None
        else:
            weights_path = run_dir / "weights.npy"
            alphas_path = run_dir / "best_alphas.npy"
            weights = np.load(weights_path, allow_pickle=False) if weights_path.exists() else None
            recorded = metrics.get("best_alphas")
            alphas = (np.load(alphas_path, allow_pickle=False) if alphas_path.exists()
                      else np.asarray(recorded) if recorded is not None else None)
            results = None
            logger.warning(
                "Legacy run has no fitted-fold/transform provenance; replay is unverified "
                "(weights present: %s, alphas present: %s)", weights is not None, alphas is not None,
            )
        return {"format_version": manifest["format_version"] if manifest else 1,
                "manifest": manifest, "weights": weights, "best_alphas": alphas,
                "hyperparams": hyperparams, "metrics": metrics, "results": results}

    def load_encoding_model(
        self, run_dir: Union[str, Path],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Compatibility four-tuple loader; use load_encoding_result for folds.

        In nested CV, returned weights/alphas are descriptive fold means, not a
        fitted full-data model. Replay an identified fold from results instead.
        """
        loaded = self.load_encoding_result(run_dir)
        return (loaded["weights"], loaded["best_alphas"],
                loaded["hyperparams"], loaded["metrics"])

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all saved runs with their hyperparameters and metrics.

        Returns:
            List of dictionaries containing run information
        """
        runs = []
        for run_dir in self.base_dir.glob("run_*"):
            if not run_dir.is_dir():
                continue

            try:
                # Load hyperparameters and metrics
                with open(run_dir / "hyperparams.json", "r") as f:
                    hyperparams = json.load(f)
                with open(run_dir / "metrics.pkl", "rb") as f:
                    metrics = pickle.load(f)

                runs.append(
                    {
                        "run_dir": str(run_dir),
                        "timestamp": run_dir.name.split("_")[1],
                        "hyperparams": hyperparams,
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                print(f"Error loading run {run_dir}: {e}")
                continue

        # Sort runs by timestamp
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return runs


class SpeechLazyLayerCache:
    """Lazy (API-compatible) loader for speech multi-layer activations (single .pkl file)."""

    def __init__(self, cache_file_path: Union[str, Path]):
        self.cache_file_path = Path(cache_file_path)
        self._metadata: Optional[Dict[str, Any]] = None
        self._loaded_layers: Dict[int, np.ndarray] = {}
        self._times: Optional[np.ndarray] = None
        self._layers_index: Optional[List[int]] = (
            None  # cached list of available layers
        )

    # ---- internal ----
    def _load_pickle_header(self) -> Dict[str, Any]:
        if not self.cache_file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {self.cache_file_path}")
        with open(self.cache_file_path, "rb") as f:
            cache_data = pickle.load(f)
        if (
            not isinstance(cache_data, dict)
            or "metadata" not in cache_data
            or "layers" not in cache_data
        ):
            raise ValueError(f"Corrupt speech cache at {self.cache_file_path}")
        return cache_data

    # ---- public API ----
    def get_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            cache_data = self._load_pickle_header()
            self._metadata = cache_data["metadata"]
            # Cache times index if present
            if "times" in cache_data and self._times is None:
                t = cache_data["times"]
                self._times = np.asarray(t) if t is not None else None
            # Cache available layers
            self._layers_index = sorted(int(k) for k in cache_data["layers"].keys())
        return self._metadata

    def get_times(self) -> Optional[np.ndarray]:
        if self._times is None:
            cache_data = self._load_pickle_header()
            t = cache_data.get("times", None)
            self._times = np.asarray(t) if t is not None else None
        return self._times

    def get_available_layers(self) -> List[int]:
        if self._layers_index is None:
            cache_data = self._load_pickle_header()
            self._layers_index = sorted(int(k) for k in cache_data["layers"].keys())
        return self._layers_index

    def get_layer(self, layer_idx: int) -> np.ndarray:
        if layer_idx in self._loaded_layers:
            return self._loaded_layers[layer_idx]
        cache_data = self._load_pickle_header()
        layers = cache_data["layers"]
        if layer_idx not in layers:
            raise ValueError(
                f"Layer {layer_idx} not found in cache. "
                f"Available layers: {sorted(int(k) for k in layers.keys())}"
            )
        arr = np.asarray(layers[layer_idx])
        self._loaded_layers[layer_idx] = arr
        return arr

    def get_layers(self, layer_indices: List[int]) -> List[np.ndarray]:
        return [self.get_layer(idx) for idx in layer_indices]

    def clear_loaded_layers(self) -> None:
        self._loaded_layers.clear()

    def validate_params(self, *, expected: Dict[str, Any]) -> None:
        """
        Validate core speech params match (e.g., model_name, chunk/context size, pool, sr).
        Raises ValueError on mismatch.
        """
        md = self.get_metadata()
        mismatches = []
        for k, v in expected.items():
            if md.get(k) != v:
                mismatches.append((k, md.get(k), v))
        if mismatches:
            msg = "Speech cache parameter mismatch:\n" + "\n".join(
                [
                    f"  - {k}: cached={got} vs expected={exp}"
                    for (k, got, exp) in mismatches
                ]
            )
            raise ValueError(msg)


class SpeechActivationCache:
    """Caching for speech model activations (multi-layer, single .pkl file)."""

    def __init__(self, cache_dir: str = "speech_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- keys & paths ----
    def _hash_from_params(self, params: Dict[str, Any]) -> str:
        # Deterministic hash (sorted keys; cast numpy/scalars to str safely)
        safe = json.loads(json.dumps(params, sort_keys=True, default=str))
        return hashlib.md5(json.dumps(safe, sort_keys=True).encode()).hexdigest()

    def get_cache_key(
        self,
        *,
        audio_id: str,  # stable identifier for the audio (path or your own hash)
        model_name: str,
        chunk_size: float,
        context_size: float,
        pool: str,  # 'last' or 'mean'
        target_sample_rate: int,
        dataset_type: str = "speech",
        extra: Optional[Dict[str, Any]] = None,  # any other disambiguating knobs
    ) -> str:
        params = {
            "modality": "speech",
            "audio_id": audio_id,
            "model_name": model_name,
            "chunk_size": float(chunk_size),
            "context_size": float(context_size),
            "pool": pool,
            "target_sample_rate": int(target_sample_rate),
            "dataset_type": dataset_type,
        }
        if extra:
            params["extra"] = extra
        return self._hash_from_params(params)

    def get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    # ---- save/load ----
    def save_multi_layer_activations(
        self,
        cache_key: str,
        all_layer_activations: Dict[int, np.ndarray],
        metadata: Dict[str, Any],
        times: Optional[np.ndarray] = None,
    ) -> None:
        """
        Persist to a single pickle:
          {
            "metadata": { ... speech params ... },
            "layers": { int: np.ndarray [n_chunks, D], ... },
            "times": np.ndarray [n_chunks] or None
          }
        """
        cache_path = self.get_cache_path(cache_key)
        # Normalize arrays for safety
        layers_clean = {int(k): np.asarray(v) for k, v in all_layer_activations.items()}
        cache_data = {
            "metadata": dict(metadata),
            "layers": layers_clean,
            "times": (np.asarray(times) if times is not None else None),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            f"[SpeechActivationCache] Saved multi-layer activations to {cache_path}"
        )

    def load_multi_layer_activations(
        self, cache_key: str
    ) -> Optional[SpeechLazyLayerCache]:
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            logger.info(
                f"[SpeechActivationCache] Loading multi-layer activations from {cache_path}"
            )
            return SpeechLazyLayerCache(cache_path)
        return None

    def save_activations(self, cache_key: str, activations: np.ndarray):
        path = self.get_cache_path(cache_key)
        with open(path, "wb") as f:
            pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[SpeechActivationCache] Saved single-layer activations to {path}")

    def load_activations(self, cache_key: str) -> Optional[np.ndarray]:
        path = self.get_cache_path(cache_key)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None


def unmask_correlations_for_plotting(
    masked_correlations: np.ndarray, mask_indices: np.ndarray, full_size: int
) -> np.ndarray:
    """Expand masked correlations back to full brain size for plotting.

    Args:
        masked_correlations: Correlations from masked analysis (n_masked_voxels,)
        mask_indices: Indices where mask was True (n_masked_voxels,)
        full_size: Size of full brain (e.g., 20484 for fsaverage5)

    Returns:
        full_correlations: Full-size array with NaNs in unmasked regions (full_size,)
    """
    # TODO: do the same for volume data. This has only been tested for surface data.
    full_correlations = np.full(full_size, np.nan)
    full_correlations[mask_indices] = masked_correlations
    return full_correlations
