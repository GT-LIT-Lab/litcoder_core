"""Fold-local fitting and replay for supplied scikit-learn regressors."""

import numpy as np
from sklearn.base import clone, is_regressor
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from encoding.models.ridge_utils import DataNormalizer


def validate_estimator(estimator, param_grid, target_mode):
    if target_mode not in ("native", "per_target"):
        raise ValueError("target_mode must be 'native' or 'per_target'")
    if (isinstance(estimator, type) or not is_regressor(estimator)
            or not callable(getattr(estimator, "predict", None))):
        raise ValueError("estimator must be a scikit-learn regressor instance or regression Pipeline")
    try:
        template = clone(estimator)
        grid = ParameterGrid(param_grid if param_grid is not None else {})
        if not len(grid):
            raise ValueError("param_grid must contain at least one candidate")
        for params in grid:
            candidate = clone(template).set_params(**params)
            if not is_regressor(candidate):
                raise ValueError("Each grid candidate must be a regressor")
            components = [candidate, *candidate.get_params(deep=True).values()]
            if any(isinstance(part, Pipeline) and part.memory is not None
                   for part in components):
                raise ValueError("Pipeline caching must be disabled for fold-local fitting")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot clone or configure supplied regressor: {exc}") from exc


def _supports_multi_output(estimator):
    # Pipeline's target tags do not always expose its final step's capabilities.
    while isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    if hasattr(estimator, "__sklearn_tags__"):
        return estimator.__sklearn_tags__().target_tags.multi_output
    return estimator._get_tags().get("multioutput", False)


def predict_estimator(state, features):
    columns = state["target_indices"]
    predictions = np.full((len(features), state["target_count"]), np.nan, dtype=np.float32)
    if not len(columns):
        return predictions
    models = state["models"]
    if state["target_mode"] == "per_target":
        outputs = [_checked_prediction(model, features, 1) for model in models]
        predictions[:, columns] = np.concatenate(outputs, axis=1)
    else:
        predictions[:, columns] = _checked_prediction(models[0], features, len(columns))
    return predictions


def _checked_prediction(model, features, width):
    output = np.asarray(model.predict(features.copy()))
    if width == 1 and output.shape == (len(features),):
        output = output[:, None]
    if output.shape != (len(features), width):
        raise ValueError(
            f"Regressor prediction shape {output.shape} must match {(len(features), width)}"
        )
    if output.dtype.kind not in "fiu":
        raise ValueError("Regressor predictions must be real numeric values")
    output = output.astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError("Regressor predictions must be finite in float32")
    return output


def _fit(estimator, params, target_mode, X, y, columns):
    models = []
    target_sets = [[column] for column in columns] if target_mode == "per_target" else [columns]
    for indices in target_sets:
        if not len(indices):
            continue
        # Clone after applying parameters too, including estimator-valued grid entries.
        model = clone(clone(estimator).set_params(**params))
        if not is_regressor(model):
            raise ValueError("Each grid candidate must be a regressor")
        if len(indices) > 1 and not _supports_multi_output(model):
            raise ValueError("Regressor lacks native multi-output support; explicitly choose target_mode='per_target'")
        values = y[:, indices]
        try:
            model.fit(X.copy(), values[:, 0] if len(indices) == 1 else values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Supplied regressor fit failed: {exc}") from exc
        models.append(model)
    return {
        "models": models, "target_indices": columns,
        "target_count": y.shape[1], "target_mode": target_mode,
    }


def fit_estimator_fold(estimator, param_grid, target_mode, X_train, y_train,
                       X_test, y_test, splits, normalize_features, normalize_targets):
    from encoding.models.nested_cv import _calculate_correlations_pvalues, _valid_training_targets

    def fit_and_predict(params, X, y, X_eval, y_eval):
        columns = np.flatnonzero(_valid_training_targets(y).numpy())
        normalizer = None
        if normalize_features or normalize_targets:
            normalizer = DataNormalizer(normalize_features, normalize_targets)
            X, y = normalizer.fit_transform(X, y)
            X_eval, y_eval = normalizer.transform(X_eval, y_eval)
        state = _fit(estimator, params, target_mode, X.numpy(), y.numpy(), columns)
        return state, normalizer, predict_estimator(state, X_eval.numpy()), y_eval.numpy()

    selected = {}
    best_score = None
    if param_grid is not None:
        for params in ParameterGrid(param_grid):
            scores = []
            for train, validation in splits:
                _, _, predictions, targets = fit_and_predict(
                    params, X_train[train], y_train[train], X_train[validation], y_train[validation]
                )
                correlations, _ = _calculate_correlations_pvalues(targets, predictions)
                scores.extend(value for value in correlations if np.isfinite(value))
            if scores:
                score = float(np.mean(scores))
                if best_score is None or score > best_score:
                    selected, best_score = params, score
        if best_score is None:
            raise ValueError("Estimator grid search found no valid inner-validation correlations")

    state, normalizer, predictions, targets = fit_and_predict(
        selected, X_train, y_train, X_test, y_test
    )
    return None, None, normalizer, predictions, targets, {
        "estimator_state": state,
        "selected_params": selected,
        "inner_selection_score": best_score,
    }
