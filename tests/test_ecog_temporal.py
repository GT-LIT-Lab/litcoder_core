"""M4 hand-calculated temporal fixtures and continuous trainer regression."""
import numpy as np
import pytest

from encoding import ecog_temporal as temporal
from encoding.models.nested_cv import NestedCVModel
from encoding.features.simple_features import WordRateFeatureExtractor
from encoding.trainer import AbstractTrainer
from test_ecog_assembly import generate, recording


def test_signed_delays_round_decimal_ties_away_from_zero():
    assert temporal.delay_offsets([-.375, 0, .125, .625], 4) == {
        "requested_s": [-.375, 0., .125, .625],
        "sample_offsets": [-2, 0, 1, 3],
        "realized_s": [-.5, 0., .25, .75],
    }
    assert temporal.delay_offsets([.124999, -.125001], 4)["sample_offsets"] == [0, -1]
    assert temporal.delay_offsets([.15, -.15], 10)["sample_offsets"] == [2, -2]
    with pytest.raises(ValueError, match="Duplicate realized"):
        temporal.delay_offsets([0, .1249], 4)
    with pytest.raises(ValueError, match="explicit"):
        temporal.delay_offsets([], 4)


@pytest.mark.parametrize(("method", "expected"), [
    ("word_sum", [1, 5, 0, 0, 4, 0, 0, 5]),
    ("word_mean", [1, 2.5, 0, 0, 4, 0, 0, 5]),
    ("word_last", [1, 3, 0, 0, 4, 0, 0, 5]),
    ("words_per_bin", [1, 1, 0, 0, 1, 0, 0, 1]),
    ("words_per_second", [4, 4, 0, 0, 4, 0, 0, 4]),
])
def test_word_alignment_distinguishes_empty_and_missing_bins(method, expected):
    story = generate(recording()).story_data["run-A"]
    values, valid, _ = temporal.align_features(
        story, np.arange(1., 6.)[:, None],
        dict(method=method, support_s=[0, 0], missing_intervals_s=[[10.5, 10.75]]),
    )
    np.testing.assert_array_equal(values[:, 0], expected)
    np.testing.assert_array_equal(np.flatnonzero(~valid), [2])


def test_speech_linear_timestamp_coverage_and_support():
    story = generate(recording()).story_data["run-A"]
    spec = dict(method="speech_linear", support_s=[.25, 0], frame_origin_s=10,
                max_gap_s=.5)
    frames = np.array([[2.], [6.], [10.]])
    values, valid, bounds = temporal.align_features(
        story, (frames, np.array([.25, .75, 1.5])), spec,
    )
    np.testing.assert_array_equal(np.flatnonzero(valid), [1, 2, 3, 6])
    assert values[2, 0] == 4
    np.testing.assert_array_equal(bounds[2], [10., 10.75])
    _, valid, _ = temporal.align_features(
        story, (frames, np.array([.25, .75, 1.5])),
        dict(spec, missing_intervals_s=[[10., 10.125]]),
    )
    np.testing.assert_array_equal(np.flatnonzero(valid), [3, 6])
    frames[1, 0] = np.nan
    _, valid, _ = temporal.align_features(story, (frames, np.array([.25, .75, 1.5])), spec)
    np.testing.assert_array_equal(np.flatnonzero(valid), [1, 6])
    with pytest.raises(ValueError, match="strictly increasing"):
        temporal.align_features(story, (frames, np.array([.25, .25, 1.5])), spec)


def test_expansion_runs_on_original_grid_before_exclusion():
    rec = recording()
    rec["valid_samples"][5] = False
    story = generate(rec).story_data["run-A"]
    values = np.arange(8., dtype=np.float32)[:, None]
    good = np.ones(8, dtype=bool)
    good[3] = False
    bounds = np.column_stack((story.tr_times, story.tr_times))
    delays = temporal.delay_offsets([-.25, 0, .25], 4)
    design, _, rows, _, reasons = temporal.expand_recording(
        story, [(values, good, bounds)], delays, [0, 0],
    )
    np.testing.assert_array_equal(rows, [1, 6])
    np.testing.assert_array_equal(design, [[2, 1, 0], [7, 6, 5]])
    assert design.dtype == np.float32
    assert reasons[0] and reasons[7] and reasons[5]
    _, _, rows, _, _ = temporal.expand_recording(story, [(values, good, bounds)], delays, [.25, 0])
    np.testing.assert_array_equal(rows, [1])


def long_recording(name, origin):
    rec = recording(name, origin)
    n = 96
    rng = np.random.default_rng(54)
    rec["brain_data"] = rng.normal(size=(n, 2)).astype(np.float32)
    rec["neural_times"] = origin + np.arange(n) / 4
    rec["valid_samples"] = np.ones(n, dtype=bool)
    rec["valid_samples"][15:19] = False
    rec["sample_ids"] = [f"{name}-original-{i}" for i in range(n)]
    rec["stimulus_times"] = origin + np.arange(0, n, 2) / 4
    rec["words"] = rec["stimuli"] = ["event"] * (n // 2)
    rec["word_ids"] = [f"w{i}" for i in range(n // 2)]
    rec["provenance"]["preprocessing"]["support_s"] = [.25, 0]
    return rec


def trainer_fixture(tmp_path, budget=10_000_000):
    assembly = generate(long_recording("A", 10), long_recording("B", 100))
    return AbstractTrainer(
        assembly, [WordRateFeatureExtractor({})], None, NestedCVModel("ridge"), None, {},
        logger_backend="tensorboard", results_dir=str(tmp_path),
        ecog_temporal_config=dict(
            delays_s=[-.25, 0, .25], retrospective=True, protocol="blocked", embargo_s=.25,
            memory_budget_bytes=budget,
            feature_specs=[dict(method="words_per_bin", support_s=[0, 0],
                                missing_intervals_s={"A": [[14., 14.25]], "B": []})],
        ),
    )


def assert_disjoint(temporal_metadata, train, test):
    rows = temporal_metadata["rows"]
    bounds = np.array(temporal_metadata["support_bounds_s"])
    embargo = temporal_metadata["embargo_s"]
    for i in train:
        for j in test:
            if rows[i]["recording_id"] == rows[j]["recording_id"]:
                assert bounds[i, 1] < bounds[j, 0] - embargo or bounds[i, 0] > bounds[j, 1] + embargo


def test_saved_nested_splits_use_original_coordinates_and_support(tmp_path):
    trainer = trainer_fixture(tmp_path)
    trainer.train(alphas=[1.], use_gpu=False, n_outer_folds=3, n_inner_folds=2)
    result = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)["results"]
    metadata = result["metadata"]["temporal"]
    assert result["metadata"]["target_space"] == "ecog"
    assert metadata["validity_applied"]
    assert metadata["recordings"]["A"]["delays"]["sample_offsets"] == [-1, 0, 1]
    assert metadata["recordings"]["B"]["delays"]["realized_s"] == [-.25, 0, .25]
    assert metadata["rows"][0]["source_index"] == 1
    for row in metadata["rows"]:
        origin = {"A": 10, "B": 100}[row["recording_id"]]
        assert row["time_s"] == origin + row["source_index"] / 4
        assert row["source_index"] not in {0, 15, 16, 17, 18, 19, 95}
    assert any(row["sample_id"] == "A-original-15" for row in metadata["exclusions"])
    for fold in result["folds"]:
        assert_disjoint(metadata, fold["train_indices"], fold["test_indices"])
        assert set(fold["purged_indices"]) == (set(range(len(metadata["rows"])))
                - set(fold["train_indices"]) - set(fold["test_indices"]))
        for split in fold["inner_splits"]:
            assert set(split["train_indices"]) <= set(fold["train_indices"])
            assert set(split["validation_indices"]) <= set(fold["train_indices"])
            assert_disjoint(metadata, split["train_indices"], split["validation_indices"])
            assert set(split["purged_indices"]) == (set(fold["train_indices"])
                    - set(split["train_indices"]) - set(split["validation_indices"]))


def test_recording_holdout_and_impossible_purge():
    ids = np.repeat(["a", "b", "c", "d"], 12)
    bounds = np.column_stack((np.arange(48), np.arange(48)))
    for train, test, nested in temporal.temporal_splits(ids, bounds, 2, 2, "recording", 0):
        assert not set(ids[train]) & set(ids[test])
        for a, b in nested:
            assert not set(ids[train[a]]) & set(ids[train[b]])
    with pytest.raises(ValueError, match="Temporal exclusion"):
        temporal.temporal_splits(np.repeat("a", 48), bounds, 3, 2, "blocked", 100)


def test_budget_rejects_before_expansion_and_solver(tmp_path, monkeypatch):
    trainer = trainer_fixture(tmp_path, budget=1)
    monkeypatch.setattr(temporal, "expand_recording",
                        lambda *args: pytest.fail("expanded before memory check"))
    with pytest.raises(MemoryError, match="against budget 1"):
        trainer.train(n_outer_folds=3, n_inner_folds=2, use_gpu=False)

    import encoding.models.nested_cv as model_module
    monkeypatch.setattr(model_module.torch, "tensor",
                        lambda *args, **kwargs: pytest.fail("allocated solver input before check"))
    data = np.arange(20., dtype=np.float32)[:, None]
    with pytest.raises(MemoryError, match="against budget 1"):
        trainer.model.fit_predict(
            data, data, use_gpu=False, temporal_cv=dict(
                recordings=["a"] * 20, bounds=np.repeat(data, 2, axis=1),
                protocol="blocked", embargo_s=0, memory_budget_bytes=1,
            ),
        )


def test_unknown_support_is_rejected(tmp_path):
    trainer = trainer_fixture(tmp_path)
    trainer.assembly.story_data["A"].provenance["preprocessing"]["support_s"] = None
    with pytest.raises(ValueError, match="requires known"):
        trainer.train(use_gpu=False)


def test_negative_delays_require_retrospective_declaration(tmp_path):
    trainer = trainer_fixture(tmp_path)
    del trainer.ecog_temporal_config["retrospective"]
    with pytest.raises(ValueError, match="retrospective=True"):
        trainer.train(use_gpu=False)


def test_legacy_fmri_trainer_bypasses_ecog_preflight(tmp_path, monkeypatch):
    from test_results_saving import make_trainer

    monkeypatch.setattr(temporal, "prepare_ecog",
                        lambda *args: pytest.fail("fMRI entered ECoG preparation"))
    trainer = make_trainer(tmp_path)
    metrics = trainer.train(alphas=[1.], n_inner_folds=2, use_gpu=False)
    assert metrics["n_valid_targets"] == 2
    assert trainer.model.results_["mode"] == "train_test"


def test_static_copy_dependency_survives_delays_and_purges(tmp_path):
    from encoding.features.embeddings import StaticEmbeddingFeatureExtractor

    vectors = tmp_path / "tiny.w2v.txt"
    vectors.write_text("2 1\nknown 7\nother 2\n")
    extractor = StaticEmbeddingFeatureExtractor(dict(vector_path=str(vectors), use_tqdm=False))
    rec = long_recording("A", 10)
    rec["words"] = ["known", "other"] * 24
    rec["words"][8:11] = ["known", "oov", "oov"]
    rec["words"][19:25] = ["other"] + ["oov"] * 5
    trainer = trainer_fixture(tmp_path)
    trainer.assembly = generate(rec)
    trainer.stories_to_process = ["A"]
    trainer.feature_extractors = [extractor]
    trainer.ecog_temporal_config.update(
        delays_s=[.25], retrospective=False, embargo_s=0,
        feature_specs=[dict(method="word_sum", support_s=[0, 0],
                            missing_intervals_s=[[14., 14.25]])],
    )
    trainer.train(alphas=[1.], use_gpu=False, n_outer_folds=3, n_inner_folds=2)
    result = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)["results"]
    meta = result["metadata"]["temporal"]
    retained = {row["source_index"]: bounds for row, bounds in
                zip(meta["rows"], meta["support_bounds_s"])}
    # The event at 14 s supplies the copied vectors at 14.5 and 15 s.
    assert {17, 19, 21}.isdisjoint(retained)
    # The other copied span starts at event 19, grid row 38, then reaches row 49.
    assert retained[49][0] == 19.5
    for fold in result["folds"]:
        assert_disjoint(meta, fold["train_indices"], fold["test_indices"])
        for split in fold["inner_splits"]:
            assert_disjoint(meta, split["train_indices"], split["validation_indices"])
    assert any(len(fold["purged_indices"]) for fold in result["folds"])
    assert any(len(split["purged_indices"]) for fold in result["folds"] for split in fold["inner_splits"])


def test_diagnostic_policy_and_provenance_round_trip(tmp_path, monkeypatch, caplog):
    import encoding.models.nested_cv as model_module

    a, b = long_recording("A", 10), long_recording("B", 100)
    a["stimulus_clock"] = "transcript"
    a["stimulus_times"] -= 10
    a["synchronization"] = dict(source_clock="transcript", target_clock="recording", scale=1,
                                offset_s=10, provenance={"source": "fixture"})
    a["unknown_samples"] = np.zeros(96, dtype=bool)
    a["unknown_samples"][[15, 16]] = True
    a["brain_data"][16, 0] = np.nan
    a["provenance"]["preprocessing"] = {"support_status": "unknown"}
    trainer = trainer_fixture(tmp_path)
    trainer.assembly = generate(a, b)
    with pytest.raises(ValueError, match="requires known"):
        trainer.train(alphas=[1.], use_gpu=False)

    trainer.ecog_temporal_config.update(protocol="diagnostic_in_sample", embargo_s=0, delays_s=[0])
    with pytest.raises(ValueError, match="fixed"):
        trainer.train(alphas=[1., 2.], use_gpu=False)
    monkeypatch.setattr(model_module, "_find_best_alphas",
                        lambda *args, **kwargs: pytest.fail("diagnostic used selection"))
    monkeypatch.setattr(temporal, "temporal_splits",
                        lambda *args, **kwargs: pytest.fail("diagnostic used CV"))
    monkeypatch.setattr(trainer.brain_plotter, "log_descriptive_correlations",
                        lambda *args: pytest.fail("diagnostic plotted correlation"))
    with caplog.at_level("INFO"):
        metrics = trainer.train(alphas=[1.], use_gpu=False, normalize_features=True)
    assert "Inference not assessed" in caplog.text
    assert trainer.reporting_error_ is None
    loaded = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)
    result = loaded["results"]
    assert result["mode"] == "diagnostic_in_sample" and result["final_refit"] is None
    assert loaded["manifest"]["replay_available"]
    assert set(metrics) == {"evaluation", "inference_status", "n_fit_samples"}
    assert loaded["metrics"]["inference_status"] == result["inference_status"] == "not_assessed"
    fold, = result["folds"]
    assert fold["model_role"] == "in_sample_diagnostic"
    assert not fold["inner_splits"] and "test_indices" not in fold
    meta = result["metadata"]["temporal"]
    assert meta["validity_policy"] == "allow_finite_unknown_in_sample"
    rec = meta["recordings"]["A"]
    assert rec["signal_units"] == a["signal_units"] and rec["grid_count"] == 96
    assert rec["sampling_rate_hz"] == 4 and rec["synchronization"] == a["synchronization"]
    assert rec["preprocessing_support_s"] is None
    assert any(row["sample_id"] == "A-original-15" and row["validity"] == "unknown" for row in meta["rows"])
    excluded = {row["sample_id"]: row for row in meta["exclusions"]}
    assert excluded["A-original-16"]["reason"] == "nonfinite neural row"
    assert excluded["A-original-17"]["validity"] == "known_invalid"
    design, _, _ = temporal.prepare_ecog(trainer, trainer.ecog_temporal_config, {"alphas": [1.]})
    np.testing.assert_array_equal(trainer.model.predict_fold(fold, design["X"]), fold["predictions"])


def test_ecog_descriptive_reporting_has_no_inference(tmp_path, monkeypatch, caplog):
    import encoding.models.nested_cv as model_module

    trainer = trainer_fixture(tmp_path)
    monkeypatch.setattr(model_module, "pearsonr", lambda *args: pytest.fail("computed Pearson p-value"))
    monkeypatch.setattr(model_module, "fdrcorrection", lambda *args, **kwargs: pytest.fail("computed FDR"))
    monkeypatch.setattr(trainer.brain_plotter, "log_plots", lambda *args, **kwargs: pytest.fail("significance plot"))
    seen = []
    monkeypatch.setattr(trainer.brain_plotter, "log_descriptive_correlations", seen.append)
    with caplog.at_level("INFO"):
        metrics = trainer.train(alphas=[1.], use_gpu=False, n_outer_folds=3, n_inner_folds=2)
    assert trainer.reporting_error_ is None
    assert metrics["inference_status"] == "not_assessed" and "significant_mask" not in metrics
    assert "p_values" not in metrics
    assert "inference not assessed" in caplog.text and "Significant voxels" not in caplog.text
    np.testing.assert_array_equal(seen[0], metrics["correlations"])
