"""Acceptance tests A1-A5 for the shared-correctness milestone (M1).

A1 fold arguments and fallbacks, A2 fold-local normalization, A3 LM cache identity,
A4 no silent truncation of feature families, A5 invalid targets excluded from scores.
"""

import pickle

import numpy as np
import pytest

import encoding.models.nested_cv as nested_cv
from encoding.features.factory import FeatureExtractorFactory
from encoding.features.language_model import LanguageModelFeatureExtractor
from encoding.models.folding import create_folds
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer
from encoding.utils import ActivationCache

FIT = dict(alphas=[1.0, 100.0], use_gpu=False)


def _data(n=120, f=3, t=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    Y = X @ rng.normal(size=(f, t)) + rng.normal(size=(n, t))
    return X, Y


# --- A1: fold arguments and fallbacks ---------------------------------------


def test_trimmed_chunks_use_trim_size_and_accept_groups():
    groups = np.repeat(np.arange(5), 20)
    splits = create_folds(100, "chunked_trimmed", 5, chunk_length=20, trim_size=3, groups=groups)
    for train, test in splits:
        assert len(test) == 20 - 2 * 3
        assert not set(train) & set(test)


def test_array_in_trim_size_slot_is_rejected():
    groups = np.repeat(np.arange(5), 20)
    with pytest.raises(TypeError, match="groups by keyword"):
        create_folds(100, "chunked_trimmed", 5, 20, groups)


@pytest.mark.parametrize(
    "fold_type, kwargs",
    [
        ("chunked", dict(chunk_length=20)),
        ("chunked_contiguous", dict(chunk_length=20)),
        ("chunked_trimmed", dict(chunk_length=20, trim_size=2)),
        ("kfold_trimmed", dict(trim_size=5)),
    ],
)
def test_too_few_samples_raise_instead_of_falling_back(fold_type, kwargs):
    with pytest.raises(ValueError):
        create_folds(30, fold_type, 5, **kwargs)


def test_trimming_that_empties_test_sets_raises():
    with pytest.raises(ValueError, match="0 test"):
        create_folds(100, "chunked_trimmed", 5, chunk_length=10, trim_size=5)


def test_fit_predict_passes_fold_arguments_by_keyword(monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return create_folds(*args, **kwargs)

    monkeypatch.setattr(nested_cv, "create_folds", spy)
    X, Y = _data()
    groups = np.repeat(np.arange(6), 20)
    NestedCVModel("ridge").fit_predict(
        X, Y, X_test=X[:30], y_test=Y[:30], groups=groups,
        folding_type="chunked_trimmed", n_inner_folds=3, chunk_length=10, trim_size=2, **FIT,
    )
    (args, kwargs), = calls
    assert len(args) == 3
    assert kwargs["trim_size"] == 2 and kwargs["chunk_length"] == 10
    assert kwargs["groups"] is groups


# --- A2: fold-local normalization ---------------------------------------------


def test_normalizers_are_fit_on_training_rows_only(monkeypatch):
    fitted_ids = []

    class RecordingNormalizer(nested_cv.DataNormalizer):
        def fit(self, X_train, y_train):
            fitted_ids.append(frozenset(X_train[:, 0].cpu().numpy().round().astype(int)))
            return super().fit(X_train, y_train)

    monkeypatch.setattr(nested_cv, "DataNormalizer", RecordingNormalizer)
    X, Y = _data()
    X[:, 0] = np.arange(len(X))  # row IDs, so each fit's rows can be identified

    NestedCVModel("ridge").fit_predict(
        X, Y, folding_type="kfold", n_outer_folds=3, n_inner_folds=2,
        normalize_features=True, normalize_targets=True, **FIT,
    )

    expected = []
    for outer_train, _ in create_folds(len(X), "kfold", 3):
        outer_train = np.asarray(outer_train)
        for inner_train, _ in create_folds(len(outer_train), "kfold", 2):
            expected.append(frozenset(outer_train[inner_train]))
        expected.append(frozenset(outer_train))
    assert fitted_ids == expected


def test_no_normalizer_without_normalization_flags(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("DataNormalizer constructed with normalization off")

    monkeypatch.setattr(nested_cv, "DataNormalizer", forbidden)
    X, Y = _data()
    NestedCVModel("ridge").fit_predict(X, Y, folding_type="kfold", n_outer_folds=3, n_inner_folds=2, **FIT)


# --- A3: LM cache identity ----------------------------------------------------


class _Assembly:
    def __init__(self, stimuli):
        self.stimuli = stimuli

    def get_stimuli(self):
        return [self.stimuli]


def _lm_extractor(tmp_path, hook_type="hook_resid_pre"):
    ext = object.__new__(LanguageModelFeatureExtractor)  # skip model download
    ext.model_name, ext.hook_type = "fake-lm", hook_type
    ext.last_token, ext.context_type = True, "fullcontext"
    ext.activation_cache = ActivationCache(cache_dir=str(tmp_path))
    ext.n_computed = 0

    def extract_all_layers(texts):
        ext.n_computed += 1
        return {0: np.full((len(texts), 2), float(ext.n_computed))}

    ext.extract_all_layers = extract_all_layers
    return ext


def _extract(ext, stimuli):
    return FeatureExtractorFactory.extract_features_with_caching(
        ext, _Assembly(stimuli), "story", 0, layer_idx=0, lookback=8, dataset_type="x"
    )


def test_cache_key_covers_hook_and_stimuli(tmp_path):
    ext = _lm_extractor(tmp_path)
    _extract(ext, ["a", "a b"])
    _extract(ext, ["a", "a b"])
    assert ext.n_computed == 1  # cache hit

    ext.hook_type = "hook_resid_post"
    _extract(ext, ["a", "a b"])
    assert ext.n_computed == 2  # hook changes the identity

    _extract(ext, ["a", "a c"])
    assert ext.n_computed == 3  # stimulus content changes the identity


def test_cache_metadata_mismatch_raises(tmp_path):
    ext = _lm_extractor(tmp_path)
    _extract(ext, ["a", "a b"])
    (path,) = tmp_path.glob("*.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    data["metadata"]["hook_type"] = "hook_resid_post"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    with pytest.raises(ValueError, match="do not match"):
        _extract(ext, ["a", "a b"])


# --- A4: feature families must align ------------------------------------------


def test_feature_families_with_different_row_counts_raise():
    trainer = object.__new__(AbstractTrainer)
    trainer.assembly = type("A", (), {"stories": ["s"]})()
    trainer.stories_to_process = ["s"]
    trainer.feature_extractors = ["short", "long"]
    rows = {"short": np.zeros((9, 2)), "long": np.zeros((10, 3))}
    trainer._extract_single_features = lambda ext, story, idx: rows[ext]
    trainer._should_downsample = lambda ext: False

    with pytest.raises(ValueError, match="refusing to truncate"):
        trainer.extract_and_downsample_features()


# --- A5: invalid targets are excluded, not scored as r=0 ----------------------


def test_constant_and_nonfinite_targets_are_nan():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(50, 4))
    y_true[:, 1] = 3.0
    y_true[5, 2] = np.nan
    corr, p = nested_cv._calculate_correlations_pvalues(y_true, rng.normal(size=(50, 4)))
    assert np.isnan(corr[1]) and np.isnan(p[1])
    assert np.isnan(corr[2]) and np.isnan(p[2])
    assert np.isfinite(corr[0]) and np.isfinite(corr[3])


@pytest.mark.parametrize("mode", ["train_test", "cv"])
def test_constant_target_is_excluded_from_summaries(mode):
    X, Y = _data()
    Y[:, 1] = 2.5  # constant everywhere, so invalid in every test set
    model = NestedCVModel("ridge")
    if mode == "train_test":
        metrics, _, _ = model.fit_predict(X[:90], Y[:90], X_test=X[90:], y_test=Y[90:],
                                          folding_type="kfold", n_inner_folds=3, **FIT)
    else:
        metrics, _, _ = model.fit_predict(X, Y, folding_type="kfold", n_outer_folds=3,
                                          n_inner_folds=2, **FIT)

    corr = np.asarray(metrics["correlations"], dtype=float)
    assert np.isnan(corr[1])
    assert metrics["n_invalid_targets"] == 1 and metrics["n_valid_targets"] == 3
    assert metrics["invalid_target_mask"] == [False, True, False, False]
    assert not metrics["significant_mask"][1]
    assert metrics["median_score"] == pytest.approx(np.median(corr[[0, 2, 3]]))


def test_inner_selection_excludes_invalid_targets_and_folds(monkeypatch):
    import torch

    X = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    y = torch.tensor([[0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1.], [5., 1.]])
    # Invalid column would dominate selection if included. The first column is
    # valid and favors alpha 2; the constant column must have no vote.
    monkeypatch.setattr(nested_cv, 'ridge_corr_torch',
                        lambda *a, **kw: torch.tensor([[0.1, 100.], [0.9, 0.]]))
    selected = nested_cv._find_best_alphas(
        X, y, [([0, 1, 2], [3, 4, 5])], [1., 2.], single_alpha=True,
    )
    assert selected.tolist() == [2., 2.]
    y[:, 0] = float('nan')
    with pytest.raises(ValueError, match='No valid targets'):
        nested_cv._find_best_alphas(
            X, y, [([0, 1, 2], [3, 4, 5])], [1., 2.], single_alpha=True,
        )


def test_nonfinite_training_target_retains_identity():
    X, Y = _data()
    Y[0, 1] = np.nan
    metrics, _, _ = NestedCVModel('ridge').fit_predict(
        X[:90], Y[:90], X_test=X[90:], y_test=Y[90:],
        folding_type='kfold', n_inner_folds=3, **FIT,
    )
    assert metrics['invalid_target_mask'] == [False, True, False, False]
    assert np.isnan(metrics['correlations'][1])


def test_zero_trim_keeps_test_rows():
    for train, test in create_folds(30, 'kfold_trimmed', 3, trim_size=0):
        assert len(train) == 20 and len(test) == 10


def test_constant_training_target_is_not_scored_on_variable_test():
    X, Y = _data()
    Y[:90, 1] = 2.5
    metrics, weights, _ = NestedCVModel('ridge').fit_predict(
        X[:90], Y[:90], X_test=X[90:], y_test=Y[90:],
        folding_type='kfold', n_inner_folds=3, **FIT,
    )
    assert metrics['invalid_target_mask'] == [False, True, False, False]
    assert np.isnan(weights[:, 1]).all()


@pytest.mark.parametrize('normalpha', [False, True])
def test_squared_alpha_matches_reference_ridge(normalpha):
    import torch
    from sklearn.linear_model import Ridge
    from encoding.models.ridge_regression import ridge_torch

    X, Y = _data()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    effective = 3. * (np.linalg.svd(X, compute_uv=False)[0] if normalpha else 1.)
    actual = ridge_torch(torch.tensor(X), torch.tensor(Y),
                         torch.full((Y.shape[1],), 3.), normalpha=normalpha)
    expected = Ridge(alpha=effective**2, fit_intercept=False, solver='svd').fit(X, Y)
    np.testing.assert_allclose(actual.numpy(), expected.coef_.T, rtol=2e-5, atol=2e-5)


def test_group_labels_propagate_to_nested_folds(monkeypatch):
    calls = []
    X, Y = _data()
    groups = np.repeat(np.arange(6), 20)

    def spy(n, kind, folds, **kwargs):
        labels = np.asarray(kwargs['groups'])
        splits = create_folds(n, kind, folds, **kwargs)
        for train, test in splits:
            assert not set(labels[train]) & set(labels[test])
        calls.append(labels)
        return splits

    monkeypatch.setattr(nested_cv, 'create_folds', spy)
    NestedCVModel('ridge').fit_predict(
        X, Y, groups=groups, folding_type='group', n_outer_folds=3,
        n_inner_folds=2, **FIT,
    )
    assert len(calls) == 4
    assert all(len(labels) == 80 for labels in calls[1:])
