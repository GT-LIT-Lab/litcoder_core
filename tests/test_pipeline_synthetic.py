"""End-to-end tests for the encoding pipeline against a synthetic assembly.

Checks structure, recovery of a signal planted at FIR lags 2 and 3 in vertices
0..9, and that the held-out story cannot leak into training. Runs offline.
"""

import numpy as np
import pytest

from conftest import (
    BRAIN_TRIM_TAIL,
    BRAIN_TRIM_TOTAL,
    FIR_DELAYS,
    N_FEATURES,
    N_SIGNAL_VERTICES,
    N_VERTICES,
    PLANTED_LAGS,
    TRIMMING_CONFIG,
    SyntheticTokenFeatureExtractor,
    reference_legacy_average,
    reference_make_delayed,
    _NullBrainPlotter,
    _NullLogger,
)

from encoding.downsample.downsampling import Downsampler
from encoding.features.FIR_expander import FIR
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer

MODEL_KWARGS = dict(alphas=[1.0, 10.0, 100.0, 1000.0], n_inner_folds=2, use_gpu=False)


def make_trainer(assembly, results_dir, **overrides):
    """Build an AbstractTrainer wired the way train_simple.py wires it."""
    kwargs = dict(
        assembly=assembly,
        feature_extractors=[SyntheticTokenFeatureExtractor()],
        downsampler=Downsampler(),
        model=NestedCVModel(model_name="ridge_regression"),
        fir_delays=FIR_DELAYS,
        trimming_config=dict(TRIMMING_CONFIG),
        use_train_test_split=True,
        logger_backend="wandb",
        dataset_type="synthetic",
        results_dir=str(results_dir),
        downsample_config={"method": "legacy_average"},
    )
    kwargs.update(overrides)
    return AbstractTrainer(**kwargs)


@pytest.fixture(scope="module")
def pipeline_data(assembly, tmp_path_factory):
    """Module-scoped, so conftest's function-scoped autouse patching is not yet
    active; patch explicitly for the duration of trainer construction."""
    results_dir = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as mp:
        import encoding.trainer as trainer_mod
        import wandb

        mp.setattr(trainer_mod, "WandBLogger", _NullLogger)
        mp.setattr(trainer_mod, "BrainPlotter", _NullBrainPlotter)
        mp.setattr(wandb, "init", lambda *a, **k: None, raising=False)

        trainer = make_trainer(assembly, results_dir)
        features = trainer.extract_and_downsample_features()
        delayed = trainer.apply_fir_delays(features)
        data = trainer.structure_data(delayed)

    return {
        "trainer": trainer,
        "features": features,
        "delayed": delayed,
        "data": data,
        "results_dir": results_dir,
    }


@pytest.fixture(scope="module")
def fitted(pipeline_data):
    data = pipeline_data["data"]
    model = pipeline_data["trainer"].model
    metrics, weights, alphas = model.fit_predict(
        features=data["Rstim"],
        targets=data["Rresp"],
        X_test=data["Pstim"],
        y_test=data["Presp"],
        **MODEL_KWARGS,
    )
    return {"metrics": metrics, "weights": weights, "alphas": alphas}


def test_fixture_matches_lebel_conventions(assembly, truth):
    total_brain_rows = 0
    for name in assembly.stories:
        sd = assembly.story_data[name]
        st = truth.stories[name]

        assert len(sd.split_indices) + 1 == len(sd.tr_times)
        assert sd.brain_data.shape[0] == st.n_trs - BRAIN_TRIM_TOTAL
        assert len(sd.words) == st.n_words
        assert len(sd.stimuli) == st.n_words
        assert len(sd.data_times) == st.n_words
        assert sd.word_rates.shape == (st.n_trs, 1)
        total_brain_rows += sd.brain_data.shape[0]

    assert assembly.shape == (total_brain_rows, N_VERTICES)


def test_legacy_average_matches_independent_reference(assembly, truth):
    ds = Downsampler()
    for name in assembly.stories:
        sd = assembly.story_data[name]
        st = truth.stories[name]

        out = ds.downsample(
            data=st.word_features,
            data_times=sd.data_times,
            tr_times=sd.tr_times,
            method="legacy_average",
            split_indices=sd.split_indices,
        )
        assert out.shape == (st.n_trs, N_FEATURES)
        np.testing.assert_allclose(out, reference_legacy_average(st.word_features, st.counts))


def test_legacy_average_leaves_empty_trs_as_zero_rows(assembly, truth):
    """Empty TRs come back as exact zero rows, not NaN."""
    ds = Downsampler()
    name = assembly.stories[0]
    sd = assembly.story_data[name]
    st = truth.stories[name]

    out = ds.downsample(
        data=st.word_features,
        data_times=sd.data_times,
        tr_times=sd.tr_times,
        method="legacy_average",
        split_indices=sd.split_indices,
    )

    assert st.empty_trs.size > 0, "fixture should contain some silent TRs"
    assert np.all(out[st.empty_trs] == 0.0)
    assert np.isfinite(out).all()

    occupied = np.setdiff1d(np.arange(st.n_trs), st.empty_trs)
    assert np.any(out[occupied] != 0.0)


@pytest.mark.parametrize("method", ["legacy_average", "legacy_sum", "legacy_last"])
def test_legacy_downsamplers_return_one_row_per_tr(assembly, truth, method):
    ds = Downsampler()
    name = assembly.stories[0]
    sd = assembly.story_data[name]
    st = truth.stories[name]

    out = ds.downsample(
        data=st.word_features,
        data_times=sd.data_times,
        tr_times=sd.tr_times,
        method=method,
        split_indices=sd.split_indices,
    )
    assert out.shape == (st.n_trs, N_FEATURES)
    assert out.shape[0] == len(sd.split_indices) + 1


@pytest.mark.xfail(
    strict=True,
    reason="non-legacy downsamplers read split_indices as per-word TR labels "
    "and return ~n_words rows",
)
@pytest.mark.parametrize("method", ["average", "sum", "last"])
def test_nonlegacy_downsamplers_return_one_row_per_tr(assembly, truth, method):
    """Known bug: non-legacy methods read split_indices as per-word TR labels and
    return max(split_indices) + 1 rows (~n_words) instead of n_trs."""
    ds = Downsampler()
    name = assembly.stories[0]
    sd = assembly.story_data[name]
    st = truth.stories[name]

    out = ds.downsample(
        data=st.word_features,
        data_times=sd.data_times,
        tr_times=sd.tr_times,
        method=method,
        split_indices=sd.split_indices,
    )
    assert out.shape == (st.n_trs, N_FEATURES)


@pytest.mark.parametrize(
    "method", ["legacy_average", "legacy_sum", "legacy_last", "average", "sum", "last"]
)
def test_downsamplers_require_split_indices(assembly, truth, method):
    ds = Downsampler()
    sd = assembly.story_data[assembly.stories[0]]
    st = truth.stories[assembly.stories[0]]

    with pytest.raises(ValueError, match="split_indices"):
        ds.downsample(
            data=st.word_features,
            data_times=sd.data_times,
            tr_times=sd.tr_times,
            method=method,
        )


def test_fir_expansion_matches_reference(assembly, truth):
    st = truth.stories[assembly.stories[0]]
    delayed = FIR.make_delayed(st.tr_features, FIR_DELAYS)

    assert delayed.shape == (st.n_trs, N_FEATURES * len(FIR_DELAYS))
    np.testing.assert_allclose(delayed, reference_make_delayed(st.tr_features, FIR_DELAYS))

    for block, d in enumerate(FIR_DELAYS):
        cols = slice(block * N_FEATURES, (block + 1) * N_FEATURES)
        assert np.all(delayed[:d, cols] == 0.0)
        np.testing.assert_allclose(delayed[d:, cols], st.tr_features[:-d])


def test_structure_data_shapes_and_story_split(pipeline_data, assembly, truth):
    features = pipeline_data["features"]
    delayed = pipeline_data["delayed"]
    data = pipeline_data["data"]

    for name in assembly.stories:
        st = truth.stories[name]
        assert features[name].shape == (st.n_trs, N_FEATURES)
        assert delayed[name].shape == (st.n_trs, N_FEATURES * len(FIR_DELAYS))

    train_stories = assembly.stories[:-1]
    test_story = assembly.stories[-1]

    expected_train = sum(truth.stories[s].n_trs - BRAIN_TRIM_TOTAL for s in train_stories)
    expected_test = truth.stories[test_story].n_trs - (
        TRIMMING_CONFIG["test_features_start"] + BRAIN_TRIM_TAIL
    )

    assert data["Rstim"].shape == (expected_train, N_FEATURES * len(FIR_DELAYS))
    assert data["Rresp"].shape == (expected_train, N_VERTICES)
    assert data["Pstim"].shape == (expected_test, N_FEATURES * len(FIR_DELAYS))
    assert data["Presp"].shape == (expected_test, N_VERTICES)

    # _create_train_test_split does not check that features and targets line up.
    assert data["Rstim"].shape[0] == data["Rresp"].shape[0]
    assert data["Pstim"].shape[0] == data["Presp"].shape[0]
    assert np.isfinite(data["Rstim"]).all()
    assert np.isfinite(data["Pstim"]).all()


def test_recovers_planted_signal_vertices(fitted, truth):
    correlations = np.asarray(fitted["metrics"]["correlations"])
    assert correlations.shape == (N_VERTICES,)

    signal_r = correlations[truth.signal_vertices]
    # Ceiling is ~0.89 at the fixture's default noise_ratio of 0.5.
    assert np.median(signal_r) > 0.75
    assert signal_r.min() > 0.6


def test_noise_vertices_are_not_predicted(fitted, truth):
    correlations = np.asarray(fitted["metrics"]["correlations"])
    noise_r = np.abs(correlations[truth.noise_vertices])

    assert np.median(noise_r) < 0.12
    assert np.percentile(noise_r, 99) < 0.35
    signal_r = correlations[truth.signal_vertices]
    assert signal_r.min() > np.percentile(noise_r, 99.9)


def test_fdr_selects_the_planted_vertices(fitted, truth):
    significant = np.asarray(fitted["metrics"]["significant_mask"], dtype=bool)
    assert significant[truth.signal_vertices].all()
    assert significant.sum() < 5 * N_SIGNAL_VERTICES


def test_recovered_kernel_peaks_at_planted_lags(fitted, truth):
    weights = fitted["weights"]
    assert weights.shape == (N_FEATURES * len(FIR_DELAYS), N_VERTICES)

    # (n_delays, n_features, n_vertices) -- make_delayed hstacks blocks in delay order.
    by_lag = weights.reshape(len(FIR_DELAYS), N_FEATURES, N_VERTICES)
    signal_norms = {
        delay: float(np.linalg.norm(by_lag[i][:, truth.signal_vertices]))
        for i, delay in enumerate(FIR_DELAYS)
    }

    planted = [signal_norms[d] for d in PLANTED_LAGS]
    empty = [signal_norms[d] for d in FIR_DELAYS if d not in PLANTED_LAGS]

    assert min(planted) > 3 * max(empty), (
        f"kernel did not peak at lags {PLANTED_LAGS}: {signal_norms}"
    )

    noise_norm = float(np.linalg.norm(by_lag[:, :, truth.noise_vertices]))
    per_vertex_noise = noise_norm / np.sqrt(len(truth.noise_vertices))
    per_vertex_signal = min(planted) / np.sqrt(N_SIGNAL_VERTICES)
    assert per_vertex_signal > per_vertex_noise


def test_trainer_train_runs_offline_and_reports_metrics(assembly, tmp_path):
    trainer = make_trainer(assembly, tmp_path)
    assert isinstance(trainer.experiment_logger, _NullLogger)
    assert isinstance(trainer.brain_plotter, _NullBrainPlotter)

    metrics = trainer.train(**MODEL_KWARGS)

    correlations = np.asarray(metrics["correlations"])
    assert np.median(correlations[:N_SIGNAL_VERTICES]) > 0.75
    assert metrics["median_score"] == pytest.approx(float(np.median(correlations)))
    assert 0 < metrics["n_significant"] < 5 * N_SIGNAL_VERTICES

    assert "median_correlation" in trainer.experiment_logger.scalars
    assert trainer.brain_plotter.calls == 1
    assert any(tmp_path.rglob("hyperparams.json"))


def test_fir_is_causal(assembly, truth):
    """Perturbing TR t never changes delayed rows before t."""
    st = truth.stories[assembly.stories[0]]
    base = FIR.make_delayed(st.tr_features, FIR_DELAYS)

    perturb_at = st.n_trs // 2
    bumped = st.tr_features.copy()
    bumped[perturb_at] += 1000.0
    after = FIR.make_delayed(bumped, FIR_DELAYS)

    changed_rows = np.flatnonzero(np.any(base != after, axis=1))
    assert changed_rows.size > 0
    assert changed_rows.min() > perturb_at, (
        "a future timepoint leaked into an earlier delayed row"
    )
    assert set(changed_rows.tolist()) == {perturb_at + d for d in FIR_DELAYS}


def test_held_out_story_cannot_influence_training_matrix(pipeline_data, assembly):
    """NaN-poisoning the held-out story's features and brain data leaves the
    training matrices bit-identical."""
    trainer = pipeline_data["trainer"]
    baseline = pipeline_data["data"]
    test_story = assembly.stories[-1]

    poisoned = {k: v.copy() for k, v in pipeline_data["delayed"].items()}
    poisoned[test_story][:] = np.nan

    story = assembly.story_data[test_story]
    clean_brain = story.brain_data
    try:
        story.brain_data = np.full_like(clean_brain, np.nan)
        data = trainer.structure_data(poisoned)
    finally:
        story.brain_data = clean_brain

    assert np.isfinite(data["Rstim"]).all()
    assert np.isfinite(data["Rresp"]).all()
    np.testing.assert_array_equal(data["Rstim"], baseline["Rstim"])
    np.testing.assert_array_equal(data["Rresp"], baseline["Rresp"])

    assert not np.array_equal(data["Pstim"], baseline["Pstim"])
    assert np.isnan(data["Presp"]).all()


def test_nan_features_are_silently_zeroed(pipeline_data, assembly):
    """_create_train_test_split applies np.nan_to_num to features but not targets, so
    NaN features become zeros and NaN targets stay NaN; neither raises."""
    trainer = pipeline_data["trainer"]
    test_story = assembly.stories[-1]

    poisoned = {k: v.copy() for k, v in pipeline_data["delayed"].items()}
    poisoned[test_story][:] = np.nan

    data = trainer.structure_data(poisoned)

    assert np.all(data["Pstim"] == 0.0)
    assert np.isfinite(data["Presp"]).all()


def test_each_split_is_zscored_independently(pipeline_data, assembly):
    """Each story is z-scored within its split; the test split uses its own
    statistics, not the training set's."""
    trainer = pipeline_data["trainer"]
    delayed = pipeline_data["delayed"]
    baseline = pipeline_data["data"]

    scaled = {k: v.copy() for k, v in delayed.items()}
    for story in assembly.stories[:-1]:
        scaled[story] = scaled[story] * 50.0 + 7.0

    data = trainer.structure_data(scaled)

    np.testing.assert_allclose(data["Pstim"], baseline["Pstim"])
    np.testing.assert_allclose(data["Rstim"], baseline["Rstim"], atol=1e-8)

    assert np.allclose(baseline["Pstim"].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(baseline["Pstim"].std(axis=0), 1.0, atol=1e-6)


def test_determinism(pipeline_data, monkeypatch):
    """Compares correlation vectors because a different fold seed barely moves the
    median (~5e-6) and leaves n_significant unchanged on this fixture."""
    data = pipeline_data["data"]
    model = pipeline_data["trainer"].model

    def fit():
        metrics, _, alphas = model.fit_predict(
            features=data["Rstim"],
            targets=data["Rresp"],
            X_test=data["Pstim"],
            y_test=data["Presp"],
            **MODEL_KWARGS,
        )
        return np.asarray(metrics["correlations"]), np.asarray(alphas)

    first, first_alphas = fit()
    second, second_alphas = fit()

    assert np.allclose(first, second, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first_alphas, second_alphas)

    import encoding.models.nested_cv as nested_cv_mod
    from encoding.models.folding import create_folds as real_create_folds

    # fit_predict takes no seed; _find_best_alphas calls create_folds with the default.
    monkeypatch.setattr(
        nested_cv_mod,
        "create_folds",
        lambda *a, **k: real_create_folds(*a, **{**k, "seed": 7}),
    )
    other, other_alphas = fit()

    assert not np.allclose(first, other, rtol=0, atol=1e-12), (
        "fold seed 7 gave the same correlations as seed 0"
    )
    assert (first_alphas != other_alphas).any(), "no voxel changed its selected alpha"
