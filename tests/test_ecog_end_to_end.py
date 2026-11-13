"""M5 public pipeline acceptance, with an independent synthetic oracle.

Frozen acceptance design: three 512-row recordings, AR(1) coefficient .65, two
electrodes, 12 null seeds 100–111, seed 29 for signal, and the gates below.
Null gates are on per-seed averages over electrodes and held-out blocks.
They are engineering checks for gross bias, not calibrated significance tests.
"""
import numpy as np
import pytest
from scipy.signal import lfilter

from encoding.assembly import ECoGAssemblyGenerator
from encoding.features.embeddings import StaticEmbeddingFeatureExtractor
from encoding.features.language_model import LanguageModelFeatureExtractor
from encoding.features.simple_features import WordRateFeatureExtractor
from encoding.features.speech_model import SpeechFeatureExtractor
from encoding.models import nested_cv
from encoding.trainer import AbstractTrainer
from encoding.utils import ActivationCache, SpeechActivationCache
from test_ecog_assembly import recording

N = 512
OFFSETS = (0, 1, 3)
REQUESTED = [0., .24, .74]
ALPHAS = [1e-4, .01, .1]
CUTOFF = 1e-5
KERNEL = np.array([
    [.5, -.4], [.2, .1], [-.3, .2], [.4, -.2],
    [-.6, .3], [.1, -.3], [.4, .1], [-.1, .5],
    [.2, -.2], [-.2, .6], [.1, -.3], [.3, .3],
], dtype=np.float32)
NULL_SEEDS = range(100, 112)
SIGNAL_R_MIN = .995
KERNEL_RELATIVE_MAX = .06
CONDITION_MAX = 12
NULL_BIAS_MAX = .06
NULL_MEAN_ABS_MAX = .08
NULL_MAX_ABS = .25
REFERENCE_ATOL = 3e-5
REPLAY_ATOL = 2e-6
INNER_SCORE_ATOL = 2e-6


def ar(rng, count):
    # Burn-in avoids a deterministic transient shared between streams.
    return lfilter([np.sqrt(1 - .65**2)], [1, -.65], rng.normal(size=count + 64))[64:]


def fixture(seed, signal):
    records, frames, activations, expected = [], {}, {}, {}
    for run, streams in enumerate(np.random.SeedSequence(seed).spawn(3)):
        srng, yrng = [np.random.default_rng(s) for s in streams.spawn(2)]
        name, origin = f"run-{run}", float(100 * run)
        rec = recording(name, origin)
        rec["neural_times"] = origin + np.arange(N) / 4
        rec["valid_samples"] = np.ones(N, dtype=bool)
        rec["valid_samples"][83:89] = False
        rec["invalid_intervals"] = [
            dict(start=origin + 83 / 4, stop=origin + 89 / 4, reason="neural gap")
        ]
        rec["sample_ids"] = [f"{name}:sample:{i}" for i in range(N)]
        rec["audio_path"] = f"{name}.wav"  # mock consumes the identifier, no file read
        rec["provenance"]["preprocessing"]["support_s"] = [0, 0]
        rec["words"], rec["stimuli"], rec["word_ids"], rec["stimulus_times"] = [], [], [], []
        counts = (ar(srng, N) > 0).astype(int) * 2
        signs = np.where(ar(srng, N * 2).reshape(N, 2) > 0, 1, -1)
        language = ar(srng, N).astype(np.float32)
        speech = ar(srng, N // 2).astype(np.float32)
        even = np.arange(0, N, 2)
        base = np.zeros((N, 4), dtype=np.float64)
        for i in range(N):
            for event in range(counts[i]):
                ident = f"{name}:{i}:{event}"
                rec["words"].append("plus" if signs[i, event] > 0 else "minus")
                rec["stimuli"].append(ident)
                rec["word_ids"].append(ident)
                rec["stimulus_times"].append(origin + i / 4 + event / 16)
                activations[ident] = language[i] / 2
            if counts[i]:
                base[i, 0] = language[i]
                base[i, 1] = signs[i].sum()
            base[i, 2] = 4 * counts[i]
            if i % 2 == 0:
                base[i, 3] = speech[i // 2]
            elif i < N - 1:
                base[i, 3] = (float(speech[i // 2]) + float(speech[i // 2 + 1])) / 2
        observed = speech.copy()
        observed[100] = np.nan
        frames[rec["audio_path"]] = ({0: observed[:, None]}, even / 4)
        noise = np.column_stack([ar(yrng, N), ar(yrng, N)])
        response = noise * (.015 if signal else 1.)
        for i in range(N):
            if signal:
                for lag_index, lag in enumerate(OFFSETS):
                    if i >= lag:
                        response[i] += base[i - lag] @ KERNEL[4 * lag_index:4 * (lag_index + 1)]
        rec["brain_data"] = (response + [2., -3.]).astype(np.float32)
        rec["brain_data"][~rec["valid_samples"]] = np.nan
        for i in range(N):
            source = [i - lag for lag in OFFSETS]
            if (not rec["valid_samples"][i]
                    or any(j < 2 or j >= N - 1 or 199 <= j <= 201 for j in source)):
                continue
            # Half-open event bins and speech brackets with one sample of past
            # support. This is the oracle, independent of production helpers.
            left = min(j - j % 2 - 1 for j in source)
            right = max(np.nextafter(origin + (j + 1) / 4, -np.inf) for j in source)
            right = max(right, max(origin + (j + j % 2) / 4 for j in source))
            expected[(name, i)] = (np.concatenate([base[j] for j in source]),
                                   rec["brain_data"][i], [origin + left / 4, right])
        records.append(rec)
    return records, frames, activations, expected


def run(path, data, monkeypatch, *, observe_normalizers=False, model=None):
    records, frames, activations, oracle = data
    path.mkdir()
    vectors = path / "tiny.w2v.txt"
    vectors.write_text("2 1\nplus 1\nminus -1\n")
    static = StaticEmbeddingFeatureExtractor(dict(
        vector_path=str(vectors), oov_handling="error", use_tqdm=False,
    ))
    language = object.__new__(LanguageModelFeatureExtractor)
    language.model_name, language.hook_type = "fixture-lm", "fixture-hook"
    language.activation_cache = ActivationCache(path / "language_cache")
    language.extract_all_layers = lambda texts: {
        0: np.array([activations[text] for text in texts], dtype=np.float32)[:, None]
    }
    speech = object.__new__(SpeechFeatureExtractor)
    speech.model_name, speech.chunk_size, speech.context_size = "fixture-speech", .5, .25
    speech.pool, speech.target_sample_rate = "mean", 4
    speech.speech_cache = SpeechActivationCache(path / "speech_cache")
    speech.extract_all_layers = frames.__getitem__
    assembly = ECoGAssemblyGenerator().generate_assembly("S1", recordings=records)
    if model is None:
        model = nested_cv.NestedCVModel("ridge")
    trainer = AbstractTrainer(
        assembly, [language, static, WordRateFeatureExtractor({}), speech], None, model,
        None, {}, layer_idx=0, dataset_type="ecog", logger_backend="tensorboard",
        results_dir=str(path / "results"),
        ecog_temporal_config=dict(
            delays_s=REQUESTED, protocol="blocked", embargo_s=.5,
            memory_budget_bytes=64_000_000,
            feature_specs=[
                dict(method="word_sum", support_s=[0, 0]),
                dict(method="word_sum", support_s=[0, 0]),
                dict(method="words_per_second", support_s=[0, 0]),
                dict(method="speech_linear", support_s=[0, 0], max_gap_s=.5,
                     frame_origin_s={r["recording_id"]: r["neural_times"][0] for r in records}),
            ],
        ),
    )
    observed = {}
    normalizer_inputs = []
    inner_scores = []
    original_score = nested_cv.ridge_corr_torch
    fit = model.fit_predict
    original_fit = nested_cv.DataNormalizer.fit

    def inspect_design(features, targets, **kwargs):
        observed.update(x=features, y=targets)
        return fit(features, targets, **kwargs)

    def inspect_normalizer(self, x, y):
        normalizer_inputs.append((x.numpy().copy(), y.numpy().copy()))
        return original_fit(self, x, y)

    def inspect_score(*args, **kwargs):
        scores = original_score(*args, **kwargs)
        inner_scores.append(scores.numpy().copy())
        return scores

    with monkeypatch.context() as patch:
        patch.setattr(model, "fit_predict", inspect_design)
        if observe_normalizers:
            patch.setattr(nested_cv.DataNormalizer, "fit", inspect_normalizer)
            patch.setattr(nested_cv, "ridge_corr_torch", inspect_score)
        trainer.train(alphas=ALPHAS if model.estimator is None else None, n_outer_folds=4, n_inner_folds=2, use_gpu=False,
                      normalpha=True, normalize_features=True, normalize_targets=True,
                      singcutoff=CUTOFF)
    result = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)
    metadata = result["results"]["metadata"]["temporal"]
    keys = [(r["recording_id"], r["source_index"]) for r in metadata["rows"]]
    assert keys == list(oracle)
    x, y, bounds = map(np.array, zip(*(oracle[key] for key in keys)))
    np.testing.assert_allclose(observed["x"], x, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(observed["y"], y)
    np.testing.assert_array_equal(metadata["support_bounds_s"], bounds)
    for row in metadata["rows"] + metadata["exclusions"]:
        run_index = int(row["recording_id"].split("-")[1])
        source = records[run_index]
        i = row["source_index"]
        assert row["sample_id"] == source["sample_ids"][i]
        assert row["time_s"] == source["neural_times"][i]
    excluded = {(r["recording_id"], r["source_index"]) for r in metadata["exclusions"]}
    assert excluded == {(r["recording_id"], i) for r in records for i in range(N)} - set(keys)
    return result, x.astype(np.float32), y, keys, normalizer_inputs, inner_scores


def reference(x_train, y_train, x_test, alphas):
    """Independent NumPy CPU float32 ridge in normalized coordinates."""
    x_mean, y_mean = x_train.mean(axis=0), y_train.mean(axis=0)
    x_scale = x_train.std(axis=0, ddof=1) + np.float32(1e-8)
    y_scale = y_train.std(axis=0, ddof=1) + np.float32(1e-8)
    x = (x_train - x_mean) / x_scale
    y = (y_train - y_mean) / y_scale
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    condition = s[0] / s[-1]
    keep = s > CUTOFF
    penalty = (np.broadcast_to(np.asarray(alphas, dtype=np.float32), y.shape[1]) * s[0]) ** 2
    s = s[keep]
    weights = np.column_stack([
        vt[keep].T @ ((s / (s**2 + penalty[i])) * (u[:, keep].T @ y[:, i]))
        for i in range(y.shape[1])
    ])
    prediction = ((x_test - x_mean) / x_scale) @ weights * y_scale + y_mean
    return prediction, weights, penalty, condition


def correlations(a, b):
    return np.array([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(a.shape[1])])


def check_purge(metadata, keys, eligible, train, held, purged):
    bounds = np.array(metadata["support_bounds_s"])
    rec = np.array([key[0] for key in keys])
    # Pairwise reference against every held-out interval on the same recording.
    candidates = np.setdiff1d(eligible, held)
    embargo = metadata["embargo_s"]
    overlap = ((rec[candidates, None] == rec[held])
               & (bounds[candidates, 0, None] <= bounds[held, 1] + embargo)
               & (bounds[candidates, 1, None] >= bounds[held, 0] - embargo)).any(axis=1)
    np.testing.assert_array_equal(train, candidates[~overlap])
    np.testing.assert_array_equal(purged, candidates[overlap])


def test_signal_recovery_isolation_ridge_parity_and_replay(tmp_path, monkeypatch):
    data = fixture(29, signal=True)
    result, x, y, keys, inputs, inner_scores = run(tmp_path / "signal", data, monkeypatch,
                                     observe_normalizers=True)
    saved = result["results"]
    metadata = saved["metadata"]["temporal"]
    assert [t["electrode_id"] for t in saved["metadata"]["target_metadata"]] == ["E9", "E1"]
    assert saved["metadata"]["sample_ids"] == [
        (name, f"{name}:sample:{i}") for name, i in keys
    ]
    assert saved["final_refit"] is None
    assert result["manifest"]["returned_weights_role"] == "descriptive_fold_mean"
    for recording_meta in metadata["recordings"].values():
        assert recording_meta["delays"] == dict(
            requested_s=REQUESTED, sample_offsets=list(OFFSETS), realized_s=[0., .25, .75],
        )
    scores, kernel_errors, conditions, parity, replay_errors = [], [], [], [], []
    fit_index = score_index = 0
    assert any(len(f["purged_indices"]) for f in saved["folds"])
    assert any(len(s["purged_indices"]) for f in saved["folds"] for s in f["inner_splits"])
    for fold_id, fold in enumerate(saved["folds"]):
        train, held = fold["train_indices"], fold["test_indices"]
        assert fold["fold_id"] == fold_id and fold["model_role"] == "outer_fold"
        check_purge(metadata, keys, np.arange(len(x)), train, held, fold["purged_indices"])
        selection = []
        actual_selection = []
        for split in fold["inner_splits"]:
            a, b = split["train_indices"], split["validation_indices"]
            check_purge(metadata, keys, train, a, b, split["purged_indices"])
            np.testing.assert_array_equal(inputs[fit_index][0], x[a])
            np.testing.assert_array_equal(inputs[fit_index][1], y[a])
            fit_index += 1
            selection.append([correlations(reference(x[a], y[a], x[b], alpha)[0], y[b])
                              * ((len(b) - 1) / len(b)) for alpha in ALPHAS])
            actual_selection.append(inner_scores[score_index])
            score_index += 1
            np.testing.assert_allclose(actual_selection[-1], selection[-1],
                                       rtol=0, atol=INNER_SCORE_ATOL)
        np.testing.assert_array_equal(inputs[fit_index][0], x[train])
        np.testing.assert_array_equal(inputs[fit_index][1], y[train])
        fit_index += 1
        best = np.array(ALPHAS, dtype=np.float32)[np.mean(actual_selection, axis=0).argmax(axis=0)]
        np.testing.assert_array_equal(fold["selected_alphas"], best)
        expected, weights, penalties, condition = reference(x[train], y[train], x[held], best)
        np.testing.assert_allclose(fold["effective_penalties"], penalties, rtol=3e-5, atol=1e-9)
        np.testing.assert_allclose(fold["weights"], weights, rtol=0, atol=REFERENCE_ATOL)
        np.testing.assert_allclose(fold["predictions"], expected, rtol=0, atol=REFERENCE_ATOL)
        replay = nested_cv.NestedCVModel.predict_fold(fold, x[held])
        np.testing.assert_allclose(replay, fold["predictions"], atol=REPLAY_ATOL, rtol=0)
        np.testing.assert_allclose(fold["targets"], y[held], atol=REPLAY_ATOL, rtol=0)
        state = fold["transform"]
        kernel = fold["weights"] * (state["target_stds"] + state["eps"]) / (
            state["feature_stds"].T + state["eps"]
        )
        kernel_errors.extend(np.linalg.norm(kernel - KERNEL, axis=0) / np.linalg.norm(KERNEL, axis=0))
        scores.extend(correlations(replay, y[held]))
        conditions.append(condition)
        parity.append(np.max(abs(expected - fold["predictions"])))
        replay_errors.append(np.max(abs(replay - fold["predictions"])))
    assert fit_index == len(inputs) and score_index == len(inner_scores)
    print(f"M5 signal scores={np.array(scores)} kernel_relative={np.array(kernel_errors)} "
          f"conditions={conditions} parity_max={max(parity):.8g} replay_max={max(replay_errors):.8g}")
    assert min(scores) >= SIGNAL_R_MIN
    assert max(kernel_errors) <= KERNEL_RELATIVE_MAX
    assert max(conditions) <= CONDITION_MAX

    # Intervene only on the first outer test set. Other folds may legitimately
    # train on these rows. The first fold's fit and selection must be invariant.
    first = saved["folds"][0]
    changed = fixture(29, signal=True)
    for i in first["test_indices"]:
        name, source_index = keys[i]
        rec = next(r for r in changed[0] if r["recording_id"] == name)
        rec["brain_data"][source_index] = -3 * rec["brain_data"][source_index] + 20
        design, _, bounds = changed[3][(name, source_index)]
        changed[3][(name, source_index)] = design, rec["brain_data"][source_index], bounds
    changed_result, *_ = run(tmp_path / "outer_intervention", changed, monkeypatch)
    after = changed_result["results"]["folds"][0]
    for field in ("train_indices", "test_indices", "selected_alphas", "weights", "predictions"):
        np.testing.assert_array_equal(after[field], first[field])
    for field in ("feature_means", "feature_stds", "target_means", "target_stds"):
        np.testing.assert_array_equal(after["transform"][field], first["transform"][field])
    assert not np.allclose(after["targets"], first["targets"])


def test_repeated_independent_autocorrelated_null(tmp_path, monkeypatch):
    averages = []
    for seed in NULL_SEEDS:
        result, _, y, *_ = run(tmp_path / str(seed), fixture(seed, signal=False), monkeypatch)
        scores = [correlations(fold["predictions"], y[fold["test_indices"]])
                  for fold in result["results"]["folds"]]
        averages.append(float(np.mean(scores)))
    print(f"M5 null seeds={list(NULL_SEEDS)} averages={np.round(averages, 6)} "
          f"mean={np.mean(averages):.6f} mean_abs={np.mean(np.abs(averages)):.6f} "
          f"sd={np.std(averages, ddof=1):.6f}")
    assert abs(np.mean(averages)) <= NULL_BIAS_MAX
    assert np.mean(np.abs(averages)) <= NULL_MEAN_ABS_MAX
    assert max(abs(np.array(averages))) <= NULL_MAX_ABS
