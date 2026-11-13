import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from encoding.models.nested_cv import NestedCVModel
from encoding.utils import ModelSaver
from test_ecog_end_to_end import fixture, run


OPTIONS = dict(folding_type="kfold", n_outer_folds=3, n_inner_folds=2, use_gpu=False)


class RecordingScaler(StandardScaler):
    fits = []

    def fit(self, X, y=None, **kwargs):
        self.fits.append((self, X.copy(), y.copy()))
        return super().fit(X, y, **kwargs)


class WrongShape(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros((len(X), 2))


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 4)).astype(np.float32)
    y = x[:, :2] + rng.normal(size=(60, 2)).astype(np.float32) * .1
    return x, y


def test_ridge_penalty_parity(data):
    ridge = NestedCVModel("optimized")
    sklearn = NestedCVModel("supplied", estimator=Ridge(alpha=4, fit_intercept=False, solver="svd"))
    options = dict(OPTIONS, normalize_features=True, normalize_targets=True)
    ridge.fit_predict(*data, alphas=[2.], normalpha=False, **options)
    _, weights, alphas = sklearn.fit_predict(*data, **options)
    assert weights is alphas is None
    for left, right in zip(ridge.results_["folds"], sklearn.results_["folds"]):
        np.testing.assert_array_equal(left["test_indices"], right["test_indices"])
        np.testing.assert_array_equal(left["effective_penalties"], [4, 4])
        np.testing.assert_allclose(left["predictions"], right["predictions"], atol=2e-6, rtol=0)


def test_pipeline_fits_only_on_training_rows_and_outer_targets_are_isolated(data):
    x, y = data
    pipeline = make_pipeline(RecordingScaler(), DecisionTreeRegressor(random_state=42))
    model = NestedCVModel("tree", estimator=pipeline,
                          param_grid={"decisiontreeregressor__max_depth": [1, 4]})
    RecordingScaler.fits.clear()
    model.fit_predict(x, y, **OPTIONS)
    before = model.results_["folds"][0]
    expected = []
    for fold in model.results_["folds"]:
        for _ in range(2):
            expected.extend(split["train_indices"] for split in fold["inner_splits"])
        expected.append(fold["train_indices"])
    assert len(RecordingScaler.fits) == len(expected)
    assert len({id(fitted) for fitted, _, _ in RecordingScaler.fits}) == len(expected)
    assert not hasattr(pipeline[0], "mean_")
    for (_, features, targets), indices in zip(RecordingScaler.fits, expected):
        np.testing.assert_array_equal(features, x[indices])
        np.testing.assert_array_equal(targets, y[indices])
    candidate_scores = {}
    for depth in (1, 4):
        scores = []
        for split in before["inner_splits"]:
            train, validation = split["train_indices"], split["validation_indices"]
            scaler = StandardScaler().fit(x[train])
            tree = DecisionTreeRegressor(random_state=42, max_depth=depth)
            tree.fit(scaler.transform(x[train]), y[train])
            predicted = tree.predict(scaler.transform(x[validation]))
            for actual, estimate in zip(y[validation].T, predicted.T):
                scores.append(np.corrcoef(actual, estimate)[0, 1] if np.ptp(estimate) else 0.)
        candidate_scores[depth] = np.mean(scores)
    assert before["selected_params"] == {
        "decisiontreeregressor__max_depth": max(candidate_scores, key=candidate_scores.get)
    }
    assert before["inner_selection_score"] == pytest.approx(max(candidate_scores.values()), abs=2e-6)
    changed = y.copy()
    changed[before["test_indices"]] *= -100
    model.fit_predict(x, changed, **OPTIONS)
    after = model.results_["folds"][0]
    assert before["selected_params"] == after["selected_params"]
    assert before["inner_selection_score"] == after["inner_selection_score"]
    np.testing.assert_array_equal(before["predictions"], after["predictions"])
    RecordingScaler.fits.clear()


def test_estimator_contract_and_explicit_adapter(data):
    x, y = data
    with pytest.raises(ValueError, match="regressor instance"):
        NestedCVModel("classifier", estimator=DecisionTreeClassifier())
    cached = make_pipeline(StandardScaler(), Ridge(), memory="/tmp")
    with pytest.raises(ValueError, match="caching must be disabled"):
        NestedCVModel("cached", estimator=cached)
    with pytest.raises(ValueError, match="native multi-output"):
        NestedCVModel("svr", estimator=SVR()).fit_predict(*data, **OPTIONS)
    with pytest.raises(ValueError, match="prediction shape"):
        NestedCVModel("shape", estimator=WrongShape()).fit_predict(x, y[:, :1], **OPTIONS)
    model = NestedCVModel("svr", estimator=SVR(), target_mode="per_target")
    model.fit_predict(*data, **OPTIONS)
    for fold in model.results_["folds"]:
        first, second = fold["estimator_state"]["models"]
        assert first is not second
        np.testing.assert_allclose(
            model.predict_fold(fold, x[fold["test_indices"]]), fold["predictions"]
        )


def test_ecog_pipeline_saved_replay(tmp_path, monkeypatch):
    model = NestedCVModel(
        "sklearn", estimator=make_pipeline(StandardScaler(), Ridge()),
        param_grid={"ridge__alpha": [.1, 1.]},
    )
    loaded, x, _, keys, *_ = run(tmp_path / "ecog", fixture(29, signal=True), monkeypatch, model=model)
    saved = loaded["results"]
    metadata = saved["metadata"]
    assert metadata["sample_ids"] == [(name, f"{name}:sample:{i}") for name, i in keys]
    assert [entry["electrode_id"] for entry in metadata["target_metadata"]] == ["E9", "E1"]
    assert loaded["weights"] is loaded["best_alphas"] is saved["final_refit"] is None
    assert loaded["manifest"]["replay_available"]
    for i, fold in enumerate(saved["folds"]):
        assert fold["fold_id"] == i and fold["model_role"] == "outer_fold"
        assert fold["selected_params"]["ridge__alpha"] in (.1, 1.)
        np.testing.assert_allclose(
            model.predict_fold(fold, x[fold["test_indices"]]), fold["predictions"], atol=2e-6
        )
    assert any(len(fold["purged_indices"]) for fold in saved["folds"])
    assert any(len(split["purged_indices"]) for fold in saved["folds"] for split in fold["inner_splits"])

    saver = ModelSaver(tmp_path / "without_models")
    path = saver.save_encoding_model(None, None, {}, loaded["metrics"], save_weights=False, results=saved)
    stripped = saver.load_encoding_result(path)
    assert not stripped["manifest"]["replay_available"]
    with pytest.raises(ValueError, match="omitted"):
        model.predict_fold(stripped["results"]["folds"][0], x)


def test_provided_test_partition_and_invalid_target_identity(data):
    x, y = data
    y[:, 0] = 0
    model = NestedCVModel("ridge", estimator=Ridge(alpha=2))
    metrics, _, _ = model.fit_predict(
        x[:40], y[:40], X_test=x[40:], y_test=y[40:],
        result_metadata={"test_sample_ids": list(range(40, 60))},
        **OPTIONS,
    )
    result = model.results_
    assert result["final_refit"] == {"fold_id": 0, "scope": "provided_training_set_only"}
    assert result["metadata"]["test_sample_ids"] == list(range(40, 60))
    fold = result["folds"][0]
    assert fold["model_role"] == "training_fit"
    assert metrics["invalid_target_mask"] == [True, False]
    np.testing.assert_array_equal(fold["estimator_state"]["target_indices"], [1])
    np.testing.assert_allclose(model.predict_fold(fold, x[40:]), fold["predictions"])
