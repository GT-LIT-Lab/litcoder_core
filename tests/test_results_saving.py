"""M2: target-generic reporting and reproducible, identified fitted results."""
import json
import pickle

import numpy as np
import pytest

from encoding.models.nested_cv import NestedCVModel
from encoding.plotting.plotting_utils import BrainPlotter
from encoding.trainer import AbstractTrainer
from encoding.utils import ModelSaver


FIT = dict(alphas=[1., 10.], use_gpu=False, folding_type='kfold',
           n_inner_folds=2, n_outer_folds=3)
TARGETS = [{'electrode_id': 'E9'}, {'electrode_id': 'E1'}, {'electrode_id': 'dead'}]


def fixture_data():
    rng = np.random.default_rng(51)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    Y = X @ rng.normal(size=(4, 3)).astype(np.float32) + 7
    Y[:, 2] = 2.5
    return X, Y


def make_trainer(tmp_path):
    from types import SimpleNamespace
    X, Y = fixture_data()
    assembly = SimpleNamespace(stories=['first', 'heldout'],
                               get_brain_data=lambda: [Y[:60], Y[60:]])
    trainer = AbstractTrainer(
        assembly, [], None, NestedCVModel('ridge'), [0], {},
        use_train_test_split=True, logger_backend='tensorboard',
        results_dir=str(tmp_path), result_metadata={
            'target_metadata': TARGETS,
            'provenance': {'signal': 'synthetic', 'units': 'fixture units'},
            'temporal_parameters': {'requested_delays_s': [0.], 'sample_offsets': [0]},
        },
    )
    trainer.extract_and_downsample_features = lambda: {'first': X[:60], 'heldout': X[60:]}
    # conftest replaces trainer's plotter globally; explicitly exercise the real
    # plotting implementation. Its fsaverage fetch is guarded against downloads.
    trainer.brain_plotter = BrainPlotter(trainer.experiment_logger)
    return trainer


def test_small_target_count_real_reporting_and_default_save(tmp_path):
    trainer = make_trainer(tmp_path)
    metrics = trainer.train(**FIT)
    assert metrics['n_valid_targets'] == 2
    assert trainer.reporting_error_ is None
    loaded = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)
    assert loaded['weights'].shape == (4, 3)
    assert loaded['results']['metadata']['target_metadata'] == TARGETS
    assert loaded['results']['metadata']['provenance']['signal'] == 'synthetic'
    assert loaded['results']['metadata']['temporal_parameters']['sample_offsets'] == [0]
    assert loaded['results']['metadata']['sample_ids'][0] == 'first:0'
    assert loaded['results']['metadata']['test_sample_ids'][-1] == 'heldout:29'
    assert loaded['results']['folds'][0]['invalid_target_mask'].tolist() == [False, False, True]
    assert loaded['results']['mode'] == 'train_test'
    assert loaded['results']['final_refit'] == {'fold_id': 0, 'scope': 'provided_training_set_only'}
    assert loaded['manifest']['returned_weights_role'] == 'training_fit'


def test_optional_plotting_failure_happens_after_save(tmp_path, monkeypatch, caplog):
    trainer = make_trainer(tmp_path)

    def broken_plot(*args, **kwargs):
        # The result must already be readable when optional plotting starts.
        assert trainer.model_saver.load_encoding_result(trainer.last_run_dir_)['results']
        raise RuntimeError('optional render failure')

    monkeypatch.setattr(trainer.brain_plotter, 'log_plots', broken_plot)
    metrics = trainer.train(**FIT)
    assert np.isfinite(metrics['median_score'])
    assert 'optional render failure' in trainer.reporting_error_
    assert 'Optional result reporting failed' in caplog.text
    assert (trainer.last_run_dir_ / 'results.pkl').is_file()


@pytest.mark.parametrize('stage', ['fit', 'save'])
def test_core_failures_propagate(tmp_path, monkeypatch, stage):
    trainer = make_trainer(tmp_path)

    def fail(*args, **kwargs):
        raise OSError(f'{stage} failed')

    if stage == 'fit':
        monkeypatch.setattr(trainer.model, 'fit_predict', fail)
    else:
        monkeypatch.setattr(trainer.model_saver, 'save_encoding_model', fail)
    with pytest.raises(OSError, match=f'{stage} failed'):
        trainer.train(**FIT)
    assert not list(tmp_path.glob('run_*'))


@pytest.mark.parametrize('mode', ['train_test', 'nested_cv'])
@pytest.mark.parametrize('normalize', [False, True])
def test_saved_folds_replay_predictions_and_retain_identity(tmp_path, mode, normalize):
    X, Y = fixture_data()
    # Scale/offset magnitudes make normalization state indispensable for replay.
    X = X * 5 + np.arange(4) * 3
    Y[:, :2] = Y[:, :2] * [3, 6] + [20, -40]
    metadata = {'target_metadata': TARGETS,
                'sample_ids': [f'recording:gap:{i*2}' for i in range(60 if mode == 'train_test' else 90)],
                'provenance': {'prior_normalization': 'none', 'clock': 'fixture'},
                'sample_time_s': np.arange(60 if mode == 'train_test' else 90) * 2.}
    kwargs = {}
    if mode == 'train_test':
        kwargs = dict(X_test=X[60:], y_test=Y[60:])
        metadata['test_sample_ids'] = [f'test:{i}' for i in range(30)]
    model = NestedCVModel('ridge')
    metrics, weights, alphas = model.fit_predict(
        X[:60] if mode == 'train_test' else X,
        Y[:60] if mode == 'train_test' else Y,
        result_metadata=metadata, normalize_features=normalize,
        normalize_targets=normalize, **kwargs, **FIT,
    )
    saver = ModelSaver(tmp_path)
    run = saver.save_encoding_model(weights, alphas, {'alpha_grid': np.array([1., 10.])},
                                    metrics, results=model.results_)
    loaded = saver.load_encoding_result(run)
    restored = loaded['results']
    assert restored['metadata']['target_metadata'] == TARGETS
    assert restored['metadata']['sample_ids'] == metadata['sample_ids']
    np.testing.assert_array_equal(restored['metadata']['sample_time_s'], metadata['sample_time_s'])
    assert restored['configuration']['normalpha'] is True
    assert len(restored['folds']) == (1 if mode == 'train_test' else 3)
    for i, fold in enumerate(restored['folds']):
        assert fold['fold_id'] == i
        indices = fold['test_indices']
        fixed_design = X[60:] if mode == 'train_test' else X[indices]
        prediction = model.predict_fold(fold, fixed_design)
        np.testing.assert_allclose(prediction, fold['predictions'], rtol=2e-5, atol=2e-5, equal_nan=True)
        np.testing.assert_allclose(model.predict_fold(fold, fixed_design, original_units=False),
                                   fold['predictions_scaled'], rtol=2e-5, atol=2e-5, equal_nan=True)
        np.testing.assert_allclose(fold['targets'][:, :2],
                                   (Y[60:] if mode == 'train_test' else Y[indices])[:, :2],
                                   rtol=2e-5, atol=2e-5)
        assert fold['transform']['normalize_features'] is normalize
        assert fold['transform']['normalize_targets'] is normalize
        for inner in fold['inner_splits']:
            assert set(inner['train_indices']) <= set(fold['train_indices'])
            assert set(inner['validation_indices']) <= set(fold['train_indices'])
        if mode == 'nested_cv':
            assert not set(fold['train_indices']) & set(indices)
            assert fold['model_role'] == 'outer_fold'
    if mode == 'nested_cv':
        assert restored['final_refit'] is None
        assert restored['returned_weights_role'] == 'descriptive_fold_mean'
        assert restored['returned_alphas_role'] == 'descriptive_fold_mean'
        assert loaded['manifest']['returned_alphas_role'] == 'descriptive_fold_mean'
        np.testing.assert_allclose(weights, np.mean([f['weights'] for f in restored['folds']], axis=0), equal_nan=True)
        assert not np.allclose(restored['folds'][0]['weights'][:, :2], restored['folds'][1]['weights'][:, :2])
    # Existing four-value loading remains usable, including default weight saving.
    w, a, config, m = saver.load_encoding_model(run)
    np.testing.assert_array_equal(w, weights)
    np.testing.assert_array_equal(a, alphas)
    assert config['alpha_grid'] == [1., 10.]
    assert m['invalid_target_mask'] == [False, False, True]


def test_legacy_preparation_stats_and_trim_coordinates_are_saved(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.trimming_config = dict(train_features_start=2, train_targets_start=2,
                                  test_features_start=3, test_targets_start=3)
    trainer.train(**FIT)
    result = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)['results']
    prepared = result['metadata']['preparation']
    assert prepared['partitions']['train']['features'][0]['source_index'] == 2
    assert prepared['partitions']['test']['targets'][0]['source_index'] == 3
    stats = prepared['upstream_transforms']
    X, _ = fixture_data()
    first = next(s for s in stats if s['domain'] == 'features' and s['partition'] == 'train')
    np.testing.assert_allclose(first['mean'], X[2:60].mean(axis=0))
    fold = result['folds'][0]
    test = next(s for s in stats if s['domain'] == 'features' and s['partition'] == 'test')
    fixed = (X[63:] - test['mean']) / test['std']
    np.testing.assert_allclose(NestedCVModel.predict_fold(fold, fixed), fold['predictions'], rtol=2e-5, atol=2e-5, equal_nan=True)


def test_versioned_omission_and_legacy_incomplete_load(tmp_path, caplog):
    saver = ModelSaver(tmp_path)
    run = saver.save_encoding_model(np.ones((2, 3)), np.ones(3), {}, {'best_alphas': [1, 1, 1]}, save_weights=False)
    loaded = saver.load_encoding_result(run)
    assert loaded['weights'] is None
    assert loaded['manifest']['replay_available'] is False
    assert loaded['best_alphas'].tolist() == [1, 1, 1]
    legacy = tmp_path / 'legacy'; legacy.mkdir()
    (legacy / 'hyperparams.json').write_text('{}')
    with open(legacy / 'metrics.pkl', 'wb') as f:
        pickle.dump({'best_alphas': [2., 3.]}, f)
    loaded = saver.load_encoding_result(legacy)
    assert loaded['format_version'] == 1
    assert loaded['results'] is None and loaded['weights'] is None
    assert loaded['best_alphas'].tolist() == [2., 3.]
    assert 'Legacy run' in caplog.text


def test_repeated_saves_do_not_overwrite_and_failure_is_not_published(tmp_path, monkeypatch):
    saver = ModelSaver(tmp_path)
    one = saver.save_encoding_model(np.ones((2, 3)), np.ones(3), {}, {})
    two = saver.save_encoding_model(np.zeros((2, 3)), np.ones(3), {}, {})
    assert one != two
    np.testing.assert_array_equal(saver.load_encoding_model(one)[0], np.ones((2, 3)))
    def broken_dump(*args, **kwargs):
        raise OSError('storage failure')
    monkeypatch.setattr(pickle, 'dump', broken_dump)
    with pytest.raises(OSError, match='storage failure'):
        saver.save_encoding_model(np.ones((2, 3)), np.ones(3), {}, {})
    assert len(list(tmp_path.glob('run_*'))) == 2
    assert not list(tmp_path.glob('.saving_*'))


def test_unknown_format_and_missing_required_v2_file_raise(tmp_path):
    saver = ModelSaver(tmp_path)
    run = saver.save_encoding_model(np.ones((2, 3)), np.ones(3), {}, {})
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['format_version'] = 999
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsupported'):
        saver.load_encoding_result(run)
    manifest['format_version'] = 2
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'weights.npy').unlink()
    with pytest.raises(FileNotFoundError):
        saver.load_encoding_result(run)


def test_existing_unified_entrypoint_saves_before_optional_reporting(tmp_path, caplog):
    import logging
    from unified import UnifiedTrainer

    X, Y = fixture_data()
    trainer = object.__new__(UnifiedTrainer)
    trainer.prepare_data = lambda: {'X': X, 'Y': Y}
    trainer.model = NestedCVModel('ridge')
    trainer.logger = logging.getLogger('test_unified_m2')
    trainer.model_saver = ModelSaver(tmp_path)
    trainer.config = dict(folding_type='kfold', n_outer_folds=3, n_inner_folds=2,
                          chunk_length=20, singcutoff=1e-10, use_gpu=False,
                          normalize_features=True, normalize_targets=True,
                          result_metadata={'target_metadata': TARGETS})

    def broken(metrics):
        assert trainer.model_saver.load_encoding_result(trainer.last_run_dir_)['results']
        raise RuntimeError('optional unified reporting failed')

    trainer.log_metrics = broken
    trainer.train()
    assert 'optional unified reporting failed' in trainer.reporting_error_
    assert 'Optional result reporting failed' in caplog.text
    loaded = trainer.model_saver.load_encoding_result(trainer.last_run_dir_)
    assert loaded['results']['final_refit'] is None
    assert loaded['results']['metadata']['target_metadata'] == TARGETS


@pytest.mark.parametrize('count', [1, 20484])
def test_generic_histograms_allow_single_target_and_explicit_surface_opt_out(count):
    from conftest import _NullLogger
    import matplotlib.pyplot as plt

    logger = _NullLogger()
    plotter = BrainPlotter(logger)
    scores = np.linspace(0.1, 0.9, count)
    plotter.log_plots(scores, np.zeros(count, dtype=bool), surface=False)
    assert not plt.get_fignums()


def test_figure_is_closed_when_optional_image_logger_fails():
    from conftest import _NullLogger
    import matplotlib.pyplot as plt

    class BrokenImageLogger(_NullLogger):
        def log_image(self, *args, **kwargs):
            raise RuntimeError('image logger failed')

    with pytest.raises(RuntimeError, match='image logger failed'):
        BrainPlotter(BrokenImageLogger()).log_plots(np.array([.1, .3]), np.array([False, True]))
    assert not plt.get_fignums()


def test_explicit_weight_omission_keeps_predictions_without_mutating_live_model(tmp_path):
    X, Y = fixture_data()
    model = NestedCVModel('ridge')
    metrics, weights, alphas = model.fit_predict(X, Y, **FIT)
    saver = ModelSaver(tmp_path)
    run = saver.save_encoding_model(weights, alphas, {}, metrics,
                                    results=model.results_, save_weights=False)
    restored = saver.load_encoding_result(run)
    assert restored['manifest']['replay_available'] is False
    assert restored['weights'] is None
    fold = restored['results']['folds'][0]
    assert 'weights' not in fold
    assert 'weights' in model.results_['folds'][0]
    np.testing.assert_array_equal(fold['predictions'], model.results_['folds'][0]['predictions'])
    with pytest.raises(ValueError, match='omitted'):
        NestedCVModel.predict_fold(fold, X[fold['test_indices']])
