"""Load preprocessed continuous ECoG into the existing assembly interface.

Input is a sequence of recording mappings, or ``ecog.json`` in data_dir (a JSON
manifest with schema_version=1 and recordings). Array fields in a manifest may
be JSON arrays or {"npy": "relative/path.npy"}; object/pickle arrays are not read.
There is no dataset-specific file discovery, preprocessing or resampling.

Required recording fields:
  recording_id, subject_id, brain_data [T,E], neural_times [T], sampling_rate_hz,
  neural_time_unit, stimulus_time_unit (s/ms/us), neural_clock, stimulus_clock,
  words, word_ids, stimulus_times [N], electrodes [E] (each has electrode_id),
  signal_units, valid_samples [T] (boolean), provenance (source/preprocessing).
Optional unknown_samples [T] is a boolean mask disjoint from valid_samples.
Rows in neither mask are known-invalid.
Provide stimuli [N] and provenance.stimulus_context, or supply a tokenizer to
reuse the existing configurable context policies. No tokenizer is downloaded.

Distinct clocks require synchronization={source_clock, target_clock, scale,
offset_s, provenance}: neural_seconds = scale * stimulus_seconds + offset_s.
Invalid intervals, if supplied, use start/stop in the declared neural clock/unit
and a reason; they must already agree with valid_samples. Invalid rows and
native times are retained. Gaps must be explicit invalid grid slots, not omitted
rows. The final bin is [last_time, last_time + 1/rate); all bins are half-open.
Only transcript bin boundaries and count/rate baselines are built here: no
feature alignment, delay expansion, exclusion dilation, or encoding integration.
"""
from copy import deepcopy
from pathlib import Path
import json
from collections.abc import Mapping

import numpy as np

from .assemblies import SimpleNeuroidAssembly
from .context import StimulusContextMixin
from .story_data import StoryData


_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
_ARRAY_FIELDS = ("brain_data", "neural_times", "stimulus_times", "valid_samples", "unknown_samples", "sample_ids")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _number(value, field, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite real number")
    if not np.isrealobj(value) or not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{field} must be finite" + (" and positive" if positive else ""))
    return float(value)


def _real_array(value, field, ndim):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a rectangular numeric array") from error
    if array.ndim != ndim or array.dtype.kind not in "iuf":
        raise ValueError(f"{field} must be a real numeric {ndim}-D array")
    return array.copy()


def _ids(value, count, field):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count:
        raise ValueError(f"{field} must contain one ID per row ({count})")
    # Do not stringify mixed IDs, merge numeric/string identities, or sort them.
    values = list(value)
    if any(not isinstance(x, (str, int, np.integer)) or isinstance(x, (bool, np.bool_))
           or (isinstance(x, str) and not x.strip()) for x in values):
        raise ValueError(f"{field} IDs must be non-empty strings or integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique IDs")
    return deepcopy(values)


class ECoGAssemblyGenerator(StimulusContextMixin):
    """Small explicit input adapter; does not inherit fMRI setup/constructors."""

    def __init__(self, data_dir=None, tokenizer=None):
        self.data_dir = None if data_dir is None else Path(data_dir)
        self.tokenizer = tokenizer

    def _read_manifest(self):
        if self.data_dir is None:
            raise ValueError("Supply recordings or an ECoG manifest data_dir")
        path = self.data_dir / "ecog.json" if self.data_dir.is_dir() else self.data_dir
        with path.open() as handle:
            manifest = json.load(handle)
        if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != 1):
            raise ValueError("ECoG manifest requires schema_version=1")
        records = manifest.get("recordings")
        if not isinstance(records, list):
            raise ValueError("ECoG manifest recordings must be a list")
        resolved = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each recording must be a mapping")
            record = deepcopy(record)
            for field in _ARRAY_FIELDS:
                value = record.get(field)
                if isinstance(value, dict):
                    if set(value) != {"npy"}:
                        raise ValueError(f"{field} file reference must contain only 'npy'")
                    filename = _text(value["npy"], f"{field}.npy")
                    if Path(filename).suffix.lower() != ".npy":
                        raise ValueError(f"{field} file reference must be a .npy array")
                    loaded = np.load(path.parent / filename, allow_pickle=False)
                    if not isinstance(loaded, np.ndarray):
                        loaded.close()
                        raise ValueError(f"{field} must reference one .npy array")
                    record[field] = loaded
            resolved.append(record)
        return resolved

    def generate_assembly(self, subject, lookback=256, context_type="fullcontext",
                          correlation_length=100, generate_temporal_baseline=False,
                          *, recordings=None):
        """Create one StoryData per recording; no rows are removed or reordered.

        Common factory arguments are retained; correlation_length is unused and
        generating synthetic temporal baselines is not supported for ECoG.
        Validation is not an assertion of safe temporal CV or preprocessing.
        """
        _text(subject, "subject")
        if generate_temporal_baseline:
            raise ValueError("ECoG temporal baseline generation is outside assembly loading")
        if context_type not in {"fullcontext", "nocontext", "halfcontext"}:
            raise ValueError(f"Invalid context type: {context_type}")
        if (isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer))
                or lookback < (2 if context_type == "halfcontext" else 1)):
            raise ValueError("lookback must be positive (at least 2 for halfcontext)")
        self.context_type = context_type
        records = self._read_manifest() if recordings is None else recordings
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("recordings must be a non-empty list or tuple")
        stories, names, electrode_order = [], set(), None
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("Each recording must be a mapping")
            name = record.get("recording_id", "<unknown>")
            try:
                story = self._recording(record, subject, lookback)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Recording {name!r}: {error}") from error
            if story.name in names:
                raise ValueError(f"Duplicate recording_id: {story.name}")
            names.add(story.name)
            current_order = [item["electrode_id"] for item in story.target_metadata]
            if electrode_order is not None and current_order != electrode_order:
                raise ValueError("Recordings must have identical electrode IDs/order; no implicit reordering")
            if stories and story.brain_data.dtype != stories[0].brain_data.dtype:
                raise ValueError("Recordings must use the same neural dtype; no implicit precision conversion")
            if stories and story.sample_metadata["signal_units"] != stories[0].sample_metadata["signal_units"]:
                raise ValueError("Recordings must use matching signal units")
            electrode_order = current_order
            stories.append(story)
        assembly = SimpleNeuroidAssembly(stories, validation_method="unconfigured")
        assembly.modality = "ecog"
        assembly.recording_slices = {}
        position = 0
        for story in stories:
            assembly.recording_slices[story.name] = (position, position + len(story.brain_data))
            position += len(story.brain_data)
        # ECoG sample coordinates are neural-grid rows, never transcript-word rows.
        recording_ids = np.concatenate([np.repeat(story.name, len(story.brain_data)) for story in stories])
        assembly.coords = {
            "story_id": {"dim": "presentation", "values": recording_ids},
            "recording_id": {"dim": "presentation", "values": recording_ids.copy()},
            "sample_index": {"dim": "presentation", "values": np.concatenate([
                np.arange(len(story.brain_data)) for story in stories])},
            "sample_id": {"dim": "presentation", "values": np.concatenate([
                np.asarray(story.sample_metadata["sample_ids"], dtype=object)
                for story in stories])},
            "sample_time_s": {"dim": "presentation", "values": np.concatenate([
                story.tr_times for story in stories])},
            "valid_sample": {"dim": "presentation", "values": np.concatenate([
                story.sample_metadata["valid_samples"] for story in stories])},
            "unknown_sample": {"dim": "presentation", "values": np.concatenate([
                story.sample_metadata["unknown_samples"] for story in stories])},
            "electrode_id": {"dim": "neuroid", "values": np.asarray(electrode_order)},
        }
        return assembly

    def _recording(self, record, subject, lookback):
        name = _text(record["recording_id"], "recording_id")
        if _text(record["subject_id"], "subject_id") != subject:
            raise ValueError("subject_id must match the requested subject")
        if "run_id" in record:
            _text(record["run_id"], "run_id")
        brain = _real_array(record["brain_data"], "brain_data", 2)
        if brain.shape[0] < 2 or brain.shape[1] == 0:
            raise ValueError("brain_data needs at least two samples and one electrode")
        rate = _number(record["sampling_rate_hz"], "sampling_rate_hz", positive=True)
        signal_units = _text(record["signal_units"], "signal_units")
        neural_unit, stimulus_unit = record["neural_time_unit"], record["stimulus_time_unit"]
        if neural_unit not in _TIME_UNITS or stimulus_unit not in _TIME_UNITS:
            raise ValueError("Time units must be declared as s, ms, or us")
        neural_clock = _text(record["neural_clock"], "neural_clock")
        stimulus_clock = _text(record["stimulus_clock"], "stimulus_clock")
        raw_grid = _real_array(record["neural_times"], "neural_times", 1)
        grid = raw_grid.astype(float) * _TIME_UNITS[neural_unit]
        if len(grid) != len(brain) or not np.isfinite(grid).all():
            raise ValueError("neural_times must be finite and match brain_data rows")
        period = 1.0 / rate
        grid_atol = min(1e-9, period * 1e-6)
        if (np.any(np.diff(grid) <= 0)
                or not np.isfinite(period)
                or not np.allclose(np.diff(grid), period, rtol=1e-6, atol=grid_atol)):
            raise ValueError("neural_times must be strictly increasing and uniform at sampling_rate_hz; retain missing slots")
        end = grid[-1] + period
        if not np.isfinite(end) or end <= grid[-1]:
            raise ValueError("Recording end must be finite and representable at the declared rate")
        valid = np.asarray(record["valid_samples"])
        if valid.dtype.kind != "b" or valid.shape != (len(brain),):
            raise ValueError("valid_samples must be a boolean vector matching neural rows")
        valid = valid.copy()
        unknown = np.asarray(record.get("unknown_samples", np.zeros(len(brain), dtype=bool)))
        if unknown.dtype.kind != "b" or unknown.shape != valid.shape or np.any(valid & unknown):
            raise ValueError("unknown_samples must be boolean, match neural rows and be disjoint from valid_samples")
        unknown = unknown.copy()
        if not np.isfinite(brain[valid]).all():
            raise ValueError("Non-finite neural values require explicitly invalid samples")
        reasons = deepcopy(record.get("invalid_sample_reasons"))
        if reasons is not None:
            if (not isinstance(reasons, (list, tuple)) or len(reasons) != len(brain)
                    or any(reason is not None and (not isinstance(reason, str) or not reason.strip())
                           for reason in reasons)):
                raise ValueError("invalid_sample_reasons must contain a reason or None for each neural row")
            if any(reason is not None and good for reason, good in zip(reasons, valid | unknown)):
                raise ValueError("invalid_sample_reasons must agree with valid_samples")
        recording_metadata = deepcopy(record.get("recording_metadata", {}))
        if not isinstance(recording_metadata, dict):
            raise ValueError("recording_metadata must be a mapping")
        provenance = deepcopy(record["provenance"])
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a mapping with source and preprocessing")
        _text(provenance.get("source"), "provenance.source")
        if not isinstance(provenance.get("preprocessing"), dict) or not provenance["preprocessing"]:
            raise ValueError("provenance.preprocessing must be a non-empty mapping")
        electrodes = deepcopy(record["electrodes"])
        if not isinstance(electrodes, list) or len(electrodes) != brain.shape[1]:
            raise ValueError("electrodes must match brain_data columns")
        if any(not isinstance(item, dict) for item in electrodes):
            raise ValueError("electrodes entries must be metadata mappings")
        electrode_ids = [_text(item.get("electrode_id"), "electrode_id") for item in electrodes]
        if len(set(electrode_ids)) != len(electrode_ids):
            raise ValueError("electrode_id values must be unique")
        for item in electrodes:
            if "coordinates" in item:
                xyz = _real_array(item["coordinates"], "electrode coordinates", 1)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError("electrode coordinates must contain three finite values")
                _text(item.get("coordinate_space"), "coordinate_space")
                _text(item.get("coordinate_units"), "coordinate_units")
        words = record["words"]
        if not isinstance(words, (list, tuple)) or any(not isinstance(w, str) for w in words):
            raise ValueError("words must be a sequence of strings (including explicit empty events)")
        words = list(words)
        word_ids = _ids(record["word_ids"], len(words), "word_ids")
        raw_onsets = _real_array(record["stimulus_times"], "stimulus_times", 1)
        if len(raw_onsets) != len(words) or not np.isfinite(raw_onsets).all():
            raise ValueError("stimulus_times must be finite and match words")
        onsets = raw_onsets.astype(float) * _TIME_UNITS[stimulus_unit]
        sync = deepcopy(record.get("synchronization"))
        if neural_clock == stimulus_clock:
            if sync is not None:
                raise ValueError("A common clock must not also specify synchronization")
        else:
            if not isinstance(sync, dict):
                raise ValueError("Distinct clocks require an explicit synchronization mapping")
            if sync.get("source_clock") != stimulus_clock or sync.get("target_clock") != neural_clock:
                raise ValueError("synchronization source_clock/target_clock do not match declared clocks")
            scale = _number(sync.get("scale"), "synchronization.scale", positive=True)
            offset = _number(sync.get("offset_s"), "synchronization.offset_s")
            if not isinstance(sync.get("provenance"), dict) or not sync["provenance"]:
                raise ValueError("synchronization.provenance must be a non-empty mapping")
            onsets = onsets * scale + offset
        if (not np.isfinite(onsets).all() or np.any(np.diff(onsets) < 0)
                or np.any(onsets < grid[0]) or np.any(onsets >= end)):
            raise ValueError("stimulus_times must be ordered and inside the half-open recording interval after synchronization")
        stimuli, context_metadata, context_sources = self._stimuli(record, words, provenance, lookback)
        if record.get("audio_path") is not None:
            _text(record["audio_path"], "audio_path")
        intervals = deepcopy(record.get("invalid_intervals", []))
        if not isinstance(intervals, list):
            raise ValueError("invalid_intervals must be a list")
        canonical_intervals = []
        for interval in intervals:
            if not isinstance(interval, dict):
                raise ValueError("invalid_intervals entries must be mappings")
            start = _number(interval.get("start"), "invalid interval start") * _TIME_UNITS[neural_unit]
            stop = _number(interval.get("stop"), "invalid interval stop") * _TIME_UNITS[neural_unit]
            _text(interval.get("reason"), "invalid interval reason")
            if not (grid[0] <= start < stop <= end):
                raise ValueError("invalid interval must lie inside the recording and have positive duration")
            if (valid | unknown)[(grid >= start) & (grid < stop)].any():
                raise ValueError("invalid_intervals must agree with valid_samples; no automatic masking")
            canonical_intervals.append({"start_s": start, "stop_s": stop, "reason": interval["reason"]})
        sample_ids = _ids(record.get("sample_ids", np.arange(len(brain))), len(brain), "sample_ids")
        # searchsorted yields T-1 cut points, including repeats for empty bins.
        cuts = np.searchsorted(onsets, grid[1:], side="left")
        counts = np.array([sum(bool(word.strip()) for word in chunk)
                           for chunk in np.split(np.asarray(words), cuts)], dtype=float).reshape(-1, 1)
        metadata = {
            "schema_version": 1, "recording_id": name, "subject_id": subject,
            "run_id": record.get("run_id"), "signal_units": signal_units,
            "sampling_rate_hz": rate, "timestamp_unit": "s", "clock": neural_clock,
            "neural_times_native": raw_grid, "neural_time_unit": neural_unit,
            "stimulus_times_native": raw_onsets, "stimulus_time_unit": stimulus_unit,
            "stimulus_clock": stimulus_clock, "synchronization": sync,
            "sample_ids": sample_ids,
            "sample_identity_kind": "supplied" if "sample_ids" in record else "positional_within_recording",
            "word_ids": word_ids, "valid_samples": valid, "unknown_samples": unknown,
            "invalid_sample_reasons": reasons, "recording_metadata": recording_metadata,
            "invalid_intervals_native": intervals, "invalid_intervals": canonical_intervals,
            "recording_interval_s": (float(grid[0]), float(end)),
            "word_rates_units": "words/bin", "words_per_second": counts * rate,
            "bin_convention": "[grid[i], grid[i+1]); last stop = grid[-1] + 1/rate",
            "grid_validation": {"rtol": 1e-6, "atol_s": grid_atol},
            "validity_applied": False,
            "stimulus_context": context_metadata, "context_source_indices": context_sources,
        }
        return StoryData(name=name, brain_data=brain, stimuli=stimuli,
                         split_indices=cuts.tolist(), tr_times=grid, data_times=onsets,
                         word_rates=counts, words=words, target_metadata=electrodes,
                         provenance=provenance, sample_metadata=metadata,
                         audio_path=record.get("audio_path"))

    def _stimuli(self, record, words, provenance, lookback):
        if "stimuli" in record:
            stimuli = record["stimuli"]
            if (not isinstance(stimuli, (list, tuple)) or len(stimuli) != len(words)
                    or any(not isinstance(s, str) for s in stimuli)):
                raise ValueError("stimuli must be strings matching transcript rows")
            if not isinstance(provenance.get("stimulus_context"), dict) or not provenance["stimulus_context"]:
                raise ValueError("Supplied stimuli require provenance.stimulus_context")
            return list(stimuli), deepcopy(provenance["stimulus_context"]), None
        if self.tokenizer is None:
            raise ValueError("Supply stimuli with context provenance or an explicit tokenizer")
        if not all(callable(getattr(self.tokenizer, method, None)) for method in ("encode", "decode")):
            raise ValueError("The explicit tokenizer must provide encode and decode methods")
        if self.context_type == "halfcontext" and any(
            len(self.tokenizer.encode(word, add_special_tokens=False)) > lookback for word in words
        ):
            raise ValueError("halfcontext cannot accept a word longer than lookback tokens")
        import pandas as pd
        source_starts = []
        stimuli = self.generate_stimuli_with_context(
            pd.DataFrame({"word_orig": words}), lookback, source_starts=source_starts,
        )
        if self.context_type != "fullcontext":
            # Window resets depend on earlier token counts in these stateful policies.
            source_starts = [0 if word else i for i, word in enumerate(words)]
        context_metadata = {
            "context_type": self.context_type, "lookback": int(lookback),
            "tokenizer": getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__),
            "implementation": "shared StimulusContextMixin",
        }
        return stimuli, context_metadata, np.asarray(source_starts, dtype=int)
