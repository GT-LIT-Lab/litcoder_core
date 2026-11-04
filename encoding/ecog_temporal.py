"""Recording-local ECoG preparation and support-aware nested evaluation."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
from numbers import Real

import numpy as np
from sklearn.model_selection import GroupKFold


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def delay_offsets(delays_s, rate):
    """Round decimal seconds × Hz to the nearest integer, ties away from zero."""
    if delays_s is None or not len(delays_s):
        raise ValueError("ECoG requires explicit nonempty delays_s")
    rate = _number(rate, "sampling rate")
    if rate <= 0:
        raise ValueError("sampling rate must be positive")
    requested = [_number(d, "delay") for d in delays_s]
    with localcontext() as context:
        context.prec = 40
        offsets = [
            int((Decimal(str(d)) * Decimal(str(rate))).to_integral_value(rounding=ROUND_HALF_UP))
            for d in requested
        ]
    if len(set(offsets)) != len(offsets):
        raise ValueError("Duplicate realized ECoG delay offsets")
    return dict(requested_s=requested, sample_offsets=offsets,
                realized_s=[offset / rate for offset in offsets])


def _support(declaration, name):
    if not isinstance(declaration, (list, tuple)) or len(declaration) != 2:
        raise ValueError(f"{name} requires known [before, after] seconds")
    return tuple(_number(v, name, 0) for v in declaration)


def _intervals(values):
    intervals = np.asarray(values, dtype=float)
    if intervals.shape == (0,):
        return intervals.reshape(0, 2)
    if (intervals.ndim != 2 or intervals.shape[1] != 2
            or not np.isfinite(intervals).all() or np.any(intervals[:, 0] >= intervals[:, 1])):
        raise ValueError("Missing intervals require finite start < stop")
    return intervals


def _overlaps(bounds, intervals, *, closed):
    """Query interval overlap using sorted starts and cumulative maximum ends."""
    if not len(intervals):
        return np.zeros(len(bounds), dtype=bool)
    ordered = intervals[np.argsort(intervals[:, 0])]
    indices = np.searchsorted(ordered[:, 0], bounds[:, 1], side="right") - 1
    ends = np.maximum.accumulate(ordered[:, 1])[np.maximum(indices, 0)]
    return (indices >= 0) & ((ends >= bounds[:, 0]) if closed else (ends > bounds[:, 0]))


def align_features(story, raw, spec, *, event_sources=None):
    """Return aligned values, validity and closed dependency bounds in neural seconds."""
    grid = story.tr_times
    meta = story.sample_metadata
    start, stop = meta["recording_interval_s"]
    edges = np.r_[grid, stop]
    method = spec["method"]
    before, after = _support(spec.get("support_s"), "feature support_s")
    if method == "speech_linear":
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise ValueError("speech_linear requires (features, frame_seconds)")
        values, times = map(np.asarray, raw)
        if values.dtype.kind != "f" or times.dtype.kind not in "iuf":
            raise ValueError("Speech features and frame timestamps must be real numeric arrays")
        origin = _number(spec.get("frame_origin_s"), "frame_origin_s")
        max_gap = _number(spec.get("max_gap_s"), "max_gap_s", 0)
        scale = _number(spec.get("frame_scale", 1), "frame_scale")
        if scale <= 0:
            raise ValueError("frame_scale must be positive")
        times = times * scale + origin
        if (values.ndim != 2 or times.ndim != 1 or len(times) != len(values)
                or not len(times) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Speech frames require finite strictly increasing timestamps and matching rows")
        right = np.searchsorted(times, grid)
        right = np.minimum(right, len(times) - 1)
        left = np.maximum(right - 1, 0)
        left[times[right] == grid] = right[times[right] == grid]
        gap = times[right] - times[left]
        fraction = np.divide(grid - times[left], gap, out=np.zeros_like(grid), where=gap > 0)
        valid = (grid >= times[0]) & (grid <= times[-1]) & (gap <= max_gap)
        valid &= np.isfinite(values[left]).all(axis=1) & np.isfinite(values[right]).all(axis=1)
        aligned = values[left] * (1 - fraction[:, None]) + values[right] * fraction[:, None]
        bounds = np.column_stack((times[left] - before, times[right] + after))
    else:
        bounds = np.column_stack((edges[:-1] - before,
                                  np.nextafter(edges[1:] + after, -np.inf)))
        valid = np.ones(len(grid), dtype=bool)
        if method in {"words_per_bin", "words_per_second"}:
            aligned = story.word_rates if method == "words_per_bin" else meta["words_per_second"]
        elif method in {"word_sum", "word_mean", "word_last"}:
            values = np.asarray(raw)
            if (values.ndim != 2 or values.dtype.kind != "f"
                    or len(values) != len(story.data_times)):
                raise ValueError("Word features must supply one row per original event")
            aligned = np.zeros((len(grid), values.shape[1]), dtype=values.dtype)
            if event_sources is not None:
                source_times = story.data_times[event_sources]
                source_bins = np.searchsorted(grid, source_times, side="right") - 1
                source_starts = np.split(grid[source_bins], story.split_indices)
            for i, chunk in enumerate(np.split(values, story.split_indices)):
                if not len(chunk):
                    continue
                if event_sources is not None:
                    # Include all inspected events, including rows checked for finiteness.
                    bounds[i, 0] = min(bounds[i, 0], source_starts[i].min() - before)
                valid[i] = np.isfinite(chunk).all()
                if valid[i]:
                    if method == "word_last":
                        aligned[i] = chunk[-1]
                    elif method == "word_sum":
                        aligned[i] = chunk.sum(axis=0)
                    else:
                        aligned[i] = chunk.mean(axis=0)
        else:
            raise ValueError(f"Unsupported ECoG alignment method: {method}")
    valid &= (bounds[:, 0] >= start) & (bounds[:, 1] <= stop)
    valid &= np.isfinite(aligned).all(axis=1)
    valid &= ~_overlaps(bounds, _intervals(spec.get("missing_intervals_s", [])), closed=False)
    return aligned, valid, bounds


def _shift(rows, offset):
    """Return matching destination/source slices for a signed delay."""
    if offset >= 0:
        return slice(offset, rows), slice(0, rows - offset)
    return slice(0, rows + offset), slice(-offset, rows)


def expand_recording(story, aligned, delays, preprocessing_support_s, *, diagnostic=False):
    """Expand on the full grid, then select rows with complete temporal support."""
    grid = story.tr_times
    meta = story.sample_metadata
    n = len(grid)
    start, stop = meta["recording_interval_s"]
    known_valid = meta["valid_samples"]
    unknown = meta.get("unknown_samples", np.zeros(n, dtype=bool))
    known_bad = ~(known_valid | unknown) | ~np.isfinite(story.brain_data).all(axis=1)
    valid = (known_valid | (unknown & diagnostic)) & ~known_bad
    if preprocessing_support_s is None:
        if not diagnostic:
            raise ValueError("Held-out evaluation requires known preprocessing support")
        bounds = np.column_stack((grid, grid))
    else:
        before, after = _support(preprocessing_support_s, "preprocessing support")
        bounds = np.column_stack((grid - before, grid + after))
        valid &= (bounds[:, 0] >= start) & (bounds[:, 1] < stop)
        bad_points = grid[known_bad if diagnostic else (known_bad | unknown)]
        valid &= ~_overlaps(bounds, np.column_stack((bad_points, bad_points)), closed=True)
        intervals = [(x["start_s"], x["stop_s"]) for x in meta["invalid_intervals"]]
        valid &= ~_overlaps(bounds, _intervals(intervals), closed=False)
    reasons = np.full(n, "", dtype=object)
    reasons[~valid] = "invalid neural row or preprocessing support"
    if not diagnostic:
        reasons[unknown & ~known_bad] = "unknown neural validity"
    reasons[~np.isfinite(story.brain_data).all(axis=1)] = "nonfinite neural row"
    width = sum(values.shape[1] for values, _, _ in aligned)
    offsets = delays["sample_offsets"]
    if any(abs(offset) >= n for offset in offsets):
        raise ValueError(f"{story.name}: delay leaves no supported rows")
    design = np.empty((n, width * len(offsets)),
                      dtype=np.result_type(*(values.dtype for values, _, _ in aligned)))
    column = 0
    for offset in offsets:
        dst, src = _shift(n, offset)
        supported = np.zeros(n, dtype=bool)
        supported[dst] = True
        for values, feature_valid, feature_bounds in aligned:
            supported[dst] &= feature_valid[src]
            design[dst, column:column + values.shape[1]] = values[src]
            column += values.shape[1]
            bounds[dst, 0] = np.minimum(bounds[dst, 0], feature_bounds[src, 0])
            bounds[dst, 1] = np.maximum(bounds[dst, 1], feature_bounds[src, 1])
        reasons[valid & ~supported] = "missing delayed feature support or recording edge"
        valid &= supported
    rows = np.flatnonzero(valid)
    if not len(rows):
        raise ValueError(f"{story.name}: no rows with complete temporal support")
    return design[rows], story.brain_data[rows], rows, bounds[rows], reasons


def memory_preflight(rows, columns, targets, design_itemsize, outer, alpha_count,
                     budget, *, resident_bytes=0):
    """Conservative array/workspace model, not an exact process peak guarantee."""
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget <= 0:
        raise ValueError("memory_budget_bytes must be a positive integer")
    rank = min(rows, columns)
    # NumPy full expansion, selected rows and concatenated design; Torch uses float32.
    design = 3 * rows * columns * design_itemsize
    # Input tensors, indexed and normalized inner/outer copies and predictions.
    tensors = 4 * (8 * rows * (columns + targets) + 8 * rows * targets)
    # Economy SVD factors, workspace/fallback copies and explicit diag(D).
    solver = 4 * (4 * rows * columns + 4 * rank * (rows + columns + rank)
                  + 4 * rank * targets + 4 * alpha_count * targets)
    # Fold weights, stacking/averaging copies, retained predictions and scores.
    retained = 8 * (4 * outer * columns * targets + 8 * rows * targets
                    + 8 * outer * targets)
    estimate = dict(resident=int(resident_bytes), design=int(design), tensors=int(tensors),
                    solver=int(solver), retained=int(retained))
    estimate["estimated_bytes"] = sum(estimate.values())
    estimate["budget_bytes"] = int(budget)
    if estimate["estimated_bytes"] > budget:
        raise MemoryError(f"ECoG memory preflight estimates {estimate['estimated_bytes']} bytes "
                          f"against budget {budget}; breakdown={estimate}")
    return estimate


def temporal_splits(recordings, bounds, outer, inner, protocol, embargo_s):
    """Return outer indices and inner positions into each purged training set."""
    embargo = _number(embargo_s, "embargo_s", 0)
    recordings = np.asarray(recordings)
    bounds = np.asarray(bounds, dtype=float)
    if (recordings.ndim != 1 or bounds.shape != (len(recordings), 2)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Temporal support bounds must match the original retained rows")

    def split(rows, count):
        if not isinstance(count, (int, np.integer)) or count < 2 or len(rows) < 2 * count:
            raise ValueError("Too few supported rows for requested temporal folds")
        if protocol == "recording":
            candidates = ((rows[a], rows[b]) for a, b in
                          GroupKFold(count).split(rows, groups=recordings[rows]))
        elif protocol == "blocked":
            # Balance retained rows in recording/grid order. Purge using original
            # seconds, so neither gaps nor nested selection compress support.
            candidates = ((np.setdiff1d(rows, test), test) for test in np.array_split(rows, count))
        else:
            raise ValueError("ECoG protocol must be blocked or recording")
        for train, test in candidates:
            keep = np.ones(len(train), dtype=bool)
            for recording in np.unique(recordings[test]):
                same = recordings[train] == recording
                test_bounds = bounds[test[recordings[test] == recording]] + [-embargo, embargo]
                keep[same] = ~_overlaps(bounds[train[same]], test_bounds, closed=True)
            train = train[keep]
            if min(len(train), len(test)) < 2:
                raise ValueError("Temporal exclusion leaves fewer than two train or test rows")
            yield train, test

    result = []
    for train, test in split(np.arange(len(recordings)), outer):
        nested = [(np.searchsorted(train, a), np.searchsorted(train, b))
                  for a, b in split(train, inner)]
        result.append((train, test, nested))
    return result


def prepare_ecog(trainer, config, fit_options):
    """Prepare the continuous ECoG inputs for the existing ridge trainer."""
    if trainer.fir_delays is not None or trainer.trimming_config or trainer.use_train_test_split:
        raise ValueError("ECoG requires delays_s and an explicit nested temporal protocol")
    if trainer.downsample_config:
        raise ValueError("ECoG alignment must be declared in feature_specs")
    if set(fit_options) & {"groups", "folding_type", "chunk_length", "trim_size", "temporal_cv"}:
        raise ValueError("ECoG uses its declared temporal protocol, not legacy fold options")
    unknown = set(config) - {
        "delays_s", "feature_specs", "protocol", "embargo_s", "memory_budget_bytes", "retrospective",
    }
    if unknown:
        raise ValueError(f"Unknown ECoG temporal settings: {sorted(unknown)}")
    specs = config.get("feature_specs", [])
    if not specs or len(specs) != len(trainer.feature_extractors):
        raise ValueError("ECoG requires one feature_specs declaration per extractor")
    from encoding.features.simple_features import WordRateFeatureExtractor

    rate_methods = {"words_per_bin", "words_per_second"}
    for extractor, spec in zip(trainer.feature_extractors, specs):
        method = spec.get("method")
        if method not in rate_methods | {"word_sum", "word_mean", "word_last", "speech_linear"}:
            raise ValueError(f"Unsupported ECoG alignment method: {method}")
        if (method in rate_methods) != isinstance(extractor, WordRateFeatureExtractor):
            raise ValueError("WordRateFeatureExtractor requires an explicit word-rate unit")
        allowed = {"method", "support_s", "missing_intervals_s"}
        if method == "speech_linear":
            allowed |= {"frame_origin_s", "frame_scale", "max_gap_s"}
        if set(spec) - allowed:
            raise ValueError(f"Unsupported options for ECoG alignment method {method}")
        _support(spec.get("support_s"), "feature support_s")
    names = trainer.stories_to_process
    if not names or len(set(names)) != len(names):
        raise ValueError("ECoG requires distinct selected recordings")
    protocol = config.get("protocol")
    diagnostic = protocol == "diagnostic_in_sample"
    if diagnostic and ({"n_outer_folds", "n_inner_folds"} & fit_options.keys()):
        raise ValueError("In-sample diagnostics do not use CV fold counts")
    outer = 1 if diagnostic else fit_options.get("n_outer_folds", 5)
    inner = fit_options.get("n_inner_folds", 5)
    for count in (() if diagnostic else (outer, inner)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError("ECoG requires at least two inner and outer folds")
    if protocol not in {"blocked", "recording", "diagnostic_in_sample"}:
        raise ValueError("ECoG requires protocol=blocked, recording or diagnostic_in_sample")
    embargo = _number(config.get("embargo_s", 0 if diagnostic else None), "embargo_s", 0)
    if diagnostic and embargo:
        raise ValueError("In-sample diagnostics do not use an embargo")
    budget = config.get("memory_budget_bytes")
    memory_preflight(0, 0, 0, 0, 0, 0, budget)
    stories = [trainer.assembly.story_data[name] for name in names]
    delays = [delay_offsets(config.get("delays_s"), s.sample_metadata["sampling_rate_hz"])
              for s in stories]
    retrospective = config.get("retrospective", False)
    if not isinstance(retrospective, bool):
        raise ValueError("retrospective must be boolean")
    if any(d < 0 for d in delays[0]["requested_s"]) and not retrospective:
        raise ValueError("Negative ECoG delays require retrospective=True")
    for story, delay in zip(stories, delays):
        if any(abs(offset) >= len(story.tr_times) for offset in delay["sample_offsets"]):
            raise ValueError(f"{story.name}: delay leaves no supported rows")
    supports = []
    for story in stories:
        preprocessing = story.provenance["preprocessing"]
        support = preprocessing.get("support_s")
        if preprocessing.get("support_status") in {"unknown", "unresolved"}:
            if support is not None:
                raise ValueError("Unknown preprocessing support cannot include a numeric bound")
        supports.append(None if diagnostic and support is None else
                        _support(support, f"{story.name} preprocessing support_s"))

    from encoding.features.embeddings import StaticEmbeddingFeatureExtractor
    from encoding.features.language_model import LanguageModelFeatureExtractor
    from encoding.features.speech_model import SpeechFeatureExtractor

    raw_by_story = []
    widths = []
    itemsize = 0
    resident = trainer.assembly.data.nbytes
    resident += sum(s.brain_data.nbytes for s in trainer.assembly.story_data.values())
    for story in stories:
        index = trainer.assembly.stories.index(story.name)
        raw_features = []
        dimensions = []
        for extractor, spec in zip(trainer.feature_extractors, specs):
            method = spec["method"]
            sources = None
            if isinstance(extractor, StaticEmbeddingFeatureExtractor):
                raw, sources = extractor.extract_features(story.words, return_source_indices=True)
            else:
                raw = trainer._extract_single_features(extractor, story.name, index)
                if isinstance(extractor, LanguageModelFeatureExtractor):
                    sources = story.sample_metadata.get("context_source_indices")
            values = np.asarray(raw[0] if isinstance(raw, tuple) else raw)
            if values.ndim != 2 or not values.shape[1] or values.dtype.kind != "f":
                raise ValueError("ECoG features must be nonempty floating-point columns")
            dimensions.append(values.shape[1])
            itemsize = max(itemsize, values.dtype.itemsize, 8 if method == "speech_linear" else 0)
            resident += values.nbytes
            if isinstance(raw, tuple):
                resident += np.asarray(raw[1]).nbytes
            raw_features.append((raw, sources))
        widths.append(tuple(dimensions))
        raw_by_story.append(raw_features)
    if len(set(widths)) != 1:
        raise ValueError("ECoG recordings must have matching feature widths")

    width = sum(widths[0])
    rows = sum(len(s.tr_times) for s in stories)
    alphas = fit_options.get("alphas")
    alpha_count = 10 if alphas is None else len(alphas)
    # Include aligned features, bounds, masks, sample metadata and nested indices.
    resident += 3 * sum(s.brain_data.nbytes for s in stories)
    resident += rows * (6 * width * itemsize + 512 + 32 * outer * (inner + 1))
    estimate = memory_preflight(rows, width * len(delays[0]["sample_offsets"]),
                                stories[0].brain_data.shape[1], itemsize, outer,
                                alpha_count, budget, resident_bytes=resident)
    features, targets, coordinates, support_bounds, exclusions = [], [], [], [], []
    records = {}
    for story, raw_features, delay, support in zip(stories, raw_by_story, delays, supports):
        local_specs = []
        for extractor, spec in zip(trainer.feature_extractors, specs):
            local = dict(spec)
            for field in ("frame_origin_s", "frame_scale", "missing_intervals_s"):
                if isinstance(local.get(field), dict):
                    if story.name not in local[field]:
                        raise ValueError(f"{field} must explicitly cover recording {story.name}")
                    local[field] = local[field][story.name]
            if isinstance(extractor, SpeechFeatureExtractor) and local["method"] == "speech_linear":
                rate = extractor.target_sample_rate
                window_s = int(extractor.context_size * rate) / rate
                scale = _number(local.get("frame_scale", 1), "frame_scale")
                before, after = local["support_s"]
                local["support_s"] = [max(before, window_s * scale), after]
            local_specs.append(local)
        aligned = [align_features(story, raw, spec, event_sources=sources)
                   for (raw, sources), spec in zip(raw_features, local_specs)]
        x, y, selected, bounds, reasons = expand_recording(story, aligned, delay, support, diagnostic=diagnostic)
        features.append(x)
        targets.append(y)
        support_bounds.append(bounds)
        meta = story.sample_metadata
        known_valid = meta["valid_samples"]
        unknown = meta.get("unknown_samples", np.zeros(len(known_valid), dtype=bool))
        records[story.name] = dict(delays=delay, sampling_rate_hz=meta["sampling_rate_hz"],
                                   clock=meta["clock"], provenance=story.provenance,
                                   target_metadata=story.target_metadata,
                                   preprocessing_support_s=support,
                                   recording_interval_s=meta["recording_interval_s"],
                                   invalid_intervals=meta["invalid_intervals"],
                                   grid_count=len(story.tr_times), effective_feature_specs=local_specs,
                                   **{key: meta[key] for key in (
                                       "signal_units", "run_id", "neural_time_unit", "stimulus_time_unit",
                                       "stimulus_clock", "synchronization", "stimulus_context",
                                       "recording_metadata", "bin_convention", "grid_validation",
                                   )})
        for i, (time, sample) in enumerate(zip(story.tr_times, meta["sample_ids"])):
            row = dict(recording_id=story.name, source_index=i, sample_id=sample,
                       time_s=float(time), validity=(
                           "unknown" if unknown[i] else "assessed_valid" if known_valid[i] else "known_invalid"
                       ))
            if reasons[i]:
                original_reasons = meta["invalid_sample_reasons"]
                exclusions.append(dict(row, reason=reasons[i], original_reason=(
                    None if original_reasons is None else original_reasons[i]
                )))
            else:
                coordinates.append(row)
    support_bounds = np.concatenate(support_bounds).tolist()
    recordings = [row["recording_id"] for row in coordinates]
    metadata = dict(target_metadata=stories[0].target_metadata, target_space="ecog",
                    sample_ids=[(r["recording_id"], r["sample_id"]) for r in coordinates],
                    temporal=dict(rows=coordinates, support_bounds_s=support_bounds,
                                  validity_applied=True,
                                  validity_policy=("allow_finite_unknown_in_sample" if diagnostic else
                                                   "assessed_valid_only"),
                                  support_bounds_scope=(
                                      "known dependencies only; neural preprocessing support unknown"
                                      if any(support is None for support in supports)
                                      else "known and declared feature and neural dependencies"
                                  ),
                                  exclusions=exclusions, recordings=records,
                                  protocol=protocol, embargo_s=embargo,
                                  retrospective=retrospective,
                                  feature_specs=specs, feature_widths=widths[0],
                                  design_order="delay, extractor, feature",
                                  memory_preflight=estimate))
    cv = dict(recordings=recordings, bounds=support_bounds, protocol=protocol,
              embargo_s=embargo, memory_budget_bytes=budget)
    return {"X": np.concatenate(features), "Y": np.concatenate(targets)}, metadata, cv
