"""M3 assembly-only acceptance: clocks, identity, validity and legacy loading."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from encoding.assembly import ECoGAssemblyGenerator, load_assembly, save_assembly
from encoding.assembly.assemblies import SimpleNeuroidAssembly
from encoding.assembly.assembly_generator import AssemblyGenerator
from encoding.assembly.story_data import StoryData


def recording(name='run-A', origin=10.):
    # A deliberately non-anatomical order must be kept exactly as supplied.
    return {
        'recording_id': name, 'subject_id': 'S1', 'run_id': name,
        'brain_data': np.arange(16, dtype=np.float32).reshape(8, 2),
        'neural_times': origin + np.arange(8) / 4,
        'sampling_rate_hz': 4., 'neural_time_unit': 's',
        'stimulus_time_unit': 's', 'neural_clock': 'recording',
        'stimulus_clock': 'recording',
        'stimulus_times': origin + np.array([0., .25, .25, 1., 1.999]),
        'words': ['one', '', 'two', 'three', 'four'],
        'word_ids': ['w0', 'empty', 'w2', 'w3', 'w4'],
        'stimuli': ['one', '', 'one two', 'two three', 'three four'],
        'electrodes': [{'electrode_id': 'E9', 'label': 'first supplied channel'},
                       {'electrode_id': 'E1', 'quality': 'synthetic'}],
        'signal_units': 'fixture units', 'valid_samples': np.ones(8, dtype=bool),
        'provenance': {'source': 'synthetic M3 fixture',
                       'preprocessing': {'description': 'already supplied synthetic trace',
                                         'filter_support': 'not applicable'},
                       'stimulus_context': {'policy': 'precomputed fixture strings'}},
    }


def generate(*records, **kwargs):
    return ECoGAssemblyGenerator().generate_assembly('S1', recordings=list(records), **kwargs)


def test_bins_empty_events_and_explicit_count_rate_units():
    r = recording()
    assembly = generate(r)
    story = assembly.story_data['run-A']
    assert len(story.split_indices) == len(story.tr_times) - 1
    assert story.split_indices == [1, 3, 3, 3, 4, 4, 4]
    np.testing.assert_array_equal(story.word_rates[:, 0], [1, 1, 0, 0, 1, 0, 0, 1])
    assert story.sample_metadata['word_rates_units'] == 'words/bin'
    np.testing.assert_array_equal(story.sample_metadata['words_per_second'], story.word_rates * 4)
    assert story.words == r['words'] and story.stimuli == r['stimuli']
    assert story.sample_metadata['word_ids'] == r['word_ids']
    # At an edge the event belongs to the next bin; even the immediately
    # preceding representable float remains in the previous bin (no snapping).
    r['stimulus_times'][0] = np.nextafter(r['neural_times'][1], -np.inf)
    edge = generate(r).story_data['run-A']
    assert edge.split_indices[0] == 1


def test_invalid_rows_and_recording_gaps_keep_positions_and_order():
    first, second = recording(), recording('run-B', origin=100.)
    first['valid_samples'][2:4] = False
    first['brain_data'][2:4] = np.nan
    first['sample_ids'] = [f'original-{i}' for i in range(8)]
    first['invalid_sample_reasons'] = [None, None, 'missing slot', 'artifact', None, None, None, None]
    first['invalid_intervals'] = [{'start': 10.5, 'stop': 11., 'reason': 'gap and artifact'}]
    first['recording_metadata'] = {'session': 'synthetic session'}
    original = deepcopy(first)
    assembly = generate(first, second)
    assert assembly.shape == (16, 2)
    assert assembly.stories == ['run-A', 'run-B']
    assert assembly.recording_slices == {'run-A': (0, 8), 'run-B': (8, 16)}
    assert assembly.validation_method == 'unconfigured'
    np.testing.assert_array_equal(assembly.get_tr_times()[0], original['neural_times'])
    np.testing.assert_array_equal(assembly.get_tr_times()[1], second['neural_times'])
    np.testing.assert_array_equal(assembly.get_brain_data()[0], original['brain_data'])
    assert assembly.get_brain_data()[0].dtype == np.float32
    assert len(assembly.get_coord('recording_id')) == 16  # neural, not word rows
    assert assembly.get_coord('sample_time_s')[8] == 100.
    assert assembly.get_coord('sample_id')[2] == 'original-2'
    assert assembly.get_coord('sample_index')[8] == 0
    assert assembly.get_coord('electrode_id').tolist() == ['E9', 'E1']
    assert not assembly.get_coord('valid_sample')[2:4].any()
    metadata = assembly.get_sample_metadata()[0]
    assert metadata['invalid_intervals_native'] == original['invalid_intervals']
    assert metadata['invalid_sample_reasons'][2] == 'missing slot'
    assert metadata['recording_metadata'] == first['recording_metadata']
    assert metadata['validity_applied'] is False
    np.testing.assert_array_equal(first['brain_data'], original['brain_data'])
    first['brain_data'][0] = 999
    assert assembly.get_brain_data()[0][0, 0] == 0  # owns its input snapshot


def test_units_common_clock_and_explicit_affine_synchronization():
    r = recording()
    r['neural_times'] *= 1000
    r['neural_time_unit'] = 'ms'
    r['stimulus_clock'] = 'audio'
    r['stimulus_time_unit'] = 'us'
    expected = r['stimulus_times'].copy()
    # Explicit mapping with a declared scale and offset, not a fitted alignment.
    r['stimulus_times'] = (expected - 10.) / 2 * 1e6
    r['synchronization'] = {'source_clock': 'audio', 'target_clock': 'recording',
                            'scale': 2., 'offset_s': 10.,
                            'provenance': {'source': 'synthetic fixed transform'}}
    assembly = generate(r)
    story = assembly.story_data['run-A']
    np.testing.assert_allclose(story.data_times, expected, atol=0, rtol=0)
    np.testing.assert_array_equal(story.tr_times, recording()['neural_times'])
    np.testing.assert_array_equal(story.sample_metadata['neural_times_native'], r['neural_times'])
    np.testing.assert_array_equal(story.sample_metadata['stimulus_times_native'], r['stimulus_times'])
    assert story.sample_metadata['synchronization'] == r['synchronization']
    common = recording()
    common['stimulus_times'] *= 1000
    common['stimulus_time_unit'] = 'ms'
    np.testing.assert_allclose(generate(common).get_data_times()[0], recording()['stimulus_times'])


@pytest.mark.parametrize('change,match', [
    ({'sampling_rate_hz': 0}, 'sampling_rate_hz'),
    ({'sampling_rate_hz': -4}, 'sampling_rate_hz'),
    ({'sampling_rate_hz': np.nan}, 'sampling_rate_hz'),
    ({'sampling_rate_hz': True}, 'sampling_rate_hz'),
    ({'sampling_rate_hz': 10}, 'uniform'),
    ({'neural_times': [10, 10.25, 10.25, 10.75, 11, 11.25, 11.5, 11.75]}, 'strictly increasing'),
    ({'neural_times': [10, 10.25, 10.75, 11, 11.25, 11.5, 11.75, 12]}, 'retain missing slots'),
    ({'neural_times': [10, 10.25]}, 'match brain_data'),
    ({'neural_times': [10, 10.25, 10.5, np.nan, 11, 11.25, 11.5, 11.75]}, 'finite'),
    ({'neural_time_unit': 'unknown'}, 'Time units'),
    ({'stimulus_time_unit': 'samples'}, 'Time units'),
    ({'stimulus_clock': 'different'}, 'synchronization'),
    ({'neural_clock': ''}, 'neural_clock'),
    ({'stimulus_times': [10, 10.5, 10.25, 11, 11.9]}, 'ordered'),
    ({'stimulus_times': [10, 10.25, 10.25, 11, 12.]}, 'half-open'),
    ({'stimulus_times': [9, 10.25, 10.25, 11, 11.9]}, 'inside'),
    ({'stimulus_times': [10, 10.25]}, 'match words'),
    ({'word_ids': ['a'] * 5}, 'unique IDs'),
    ({'word_ids': ['a']}, 'per row'),
    ({'sample_ids': ['s'] * 8}, 'unique IDs'),
    ({'words': ['one', 'two', 3, 'four', 'five']}, 'strings'),
    ({'stimuli': ['only one']}, 'matching transcript'),
    ({'electrodes': [{'electrode_id': 'E1'}, {'electrode_id': 'E1'}]}, 'unique'),
    ({'electrodes': [{'electrode_id': 'E1'}]}, 'columns'),
    ({'brain_data': np.zeros((8, 3))}, 'columns'),
    ({'brain_data': np.zeros((8, 2), dtype=complex)}, 'real numeric'),
    ({'brain_data': np.full((8, 2), np.nan)}, 'explicitly invalid'),
    ({'valid_samples': [1] * 8}, 'boolean'),
    ({'valid_samples': [True] * 7}, 'matching neural rows'),
    ({'provenance': {}}, 'provenance.source'),
    ({'provenance': {'source': 'x', 'preprocessing': {}}}, 'preprocessing'),
    ({'signal_units': ''}, 'signal_units'),
    ({'subject_id': 'other subject'}, 'requested subject'),
    ({'invalid_intervals': [{'start': 10.5, 'stop': 10.5, 'reason': 'x'}]}, 'positive duration'),
    ({'invalid_intervals': [{'start': 9, 'stop': 10.5, 'reason': 'x'}]}, 'inside'),
    ({'invalid_intervals': [{'start': 10, 'stop': 10.5, 'reason': 'x'}]}, 'agree'),
    ({'invalid_sample_reasons': ['artifact'] * 8}, 'agree'),
    ({'recording_metadata': []}, 'recording_metadata'),
])
def test_malformed_recordings_fail_without_repair(change, match):
    r = recording()
    r.update(change)
    with pytest.raises(ValueError, match=match):
        generate(r)


def test_duplicate_recordings_or_changed_electrode_order_are_not_merged():
    with pytest.raises(ValueError, match='Duplicate recording_id'):
        generate(recording(), recording())
    second = recording('run-B', origin=100.)
    second['electrodes'].reverse()
    with pytest.raises(ValueError, match='identical electrode IDs/order'):
        generate(recording(), second)
    second = recording('run-B', origin=100.)
    second['brain_data'] = second['brain_data'].astype(np.float64)
    with pytest.raises(ValueError, match='no implicit precision conversion'):
        generate(recording(), second)


@pytest.mark.parametrize('sync', [
    {'scale': -1., 'offset_s': 0., 'source_clock': 'audio', 'target_clock': 'recording', 'provenance': {'source': 'x'}},
    {'scale': 1., 'offset_s': np.inf, 'source_clock': 'audio', 'target_clock': 'recording', 'provenance': {'source': 'x'}},
    {'scale': 1., 'offset_s': 0., 'source_clock': 'wrong', 'target_clock': 'recording', 'provenance': {'source': 'x'}},
    {'scale': 1., 'offset_s': 0., 'source_clock': 'audio', 'target_clock': 'recording', 'provenance': {}},
])
def test_malformed_synchronization_is_rejected(sync):
    r = recording(); r['stimulus_clock'] = 'audio'; r['synchronization'] = sync
    with pytest.raises(ValueError, match='synchronization'):
        generate(r)


def test_metadata_round_trip_and_optional_coordinate_space(tmp_path):
    r = recording()
    r['valid_samples'][1] = False
    r['brain_data'][1] = np.nan
    r['sample_ids'] = [f'source-{i}' for i in range(8)]
    r['invalid_intervals'] = [{'start': 10.25, 'stop': 10.5, 'reason': 'synthetic missing slot'}]
    r['electrodes'][0].update(coordinates=[1., 2., 3.], coordinate_space='fixture space', coordinate_units='mm')
    assembly = generate(r)
    path = tmp_path / 'ecog.pkl'
    save_assembly(assembly, path)
    loaded = load_assembly(path)
    assert loaded.get_target_metadata() == assembly.get_target_metadata()
    assert loaded.get_provenance() == [r['provenance']]
    assert loaded.recording_slices == assembly.recording_slices
    np.testing.assert_array_equal(loaded.get_brain_data()[0], r['brain_data'])
    assert loaded.get_sample_metadata()[0]['sample_ids'] == r['sample_ids']
    assert loaded.get_sample_metadata()[0]['invalid_intervals_native'] == r['invalid_intervals']
    for key in ['valid_samples', 'neural_times_native', 'stimulus_times_native', 'words_per_second']:
        np.testing.assert_array_equal(loaded.get_sample_metadata()[0][key], assembly.get_sample_metadata()[0][key])
    del r['electrodes'][0]['coordinate_space']
    with pytest.raises(ValueError, match='coordinate_space'):
        generate(r)


def write_manifest(tmp_path):
    r = recording()
    for field in ['brain_data', 'neural_times', 'stimulus_times', 'valid_samples']:
        np.save(tmp_path / f'{field}.npy', r[field])
        r[field] = {'npy': f'{field}.npy'}
    path = tmp_path / 'ecog.json'
    path.write_text(json.dumps({'schema_version': 1, 'recordings': [r]}))
    return path


def test_manifest_factory_and_no_fmri_imports_or_downloads(tmp_path):
    manifest = write_manifest(tmp_path)
    assembly = AssemblyGenerator.generate_assembly('ecog', str(tmp_path), 'S1')
    np.testing.assert_array_equal(assembly.get_brain_data()[0], recording()['brain_data'])
    script = '''
import sys
class BlockSetupImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nibabel', 'nilearn', 'transformers', 'torch'} or fullname in {
            'encoding.assembly.base_processor', 'encoding.brain_projection'}:
            raise AssertionError('ECoG assembly imported setup dependency: ' + fullname)
sys.meta_path.insert(0, BlockSetupImports())
from encoding.assembly.assembly_generator import AssemblyGenerator
a = AssemblyGenerator.generate_assembly('ecog', sys.argv[1], 'S1')
assert a.shape == (8, 2)
assert a.validation_method == 'unconfigured'
'''
    completed = subprocess.run([sys.executable, '-c', script, str(manifest)],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with pytest.raises(ValueError, match='fMRI volume/mask'):
        AssemblyGenerator.create('ecog', None, use_volume=True)
    with pytest.raises(ValueError, match='sampling_rate_hz'):
        AssemblyGenerator.create('ecog', None, tr=2.)
    with pytest.raises(ValueError, match='only for ECoG'):
        AssemblyGenerator.generate_assembly('lebel', None, 'S1', recordings=[recording()])


def test_manifest_does_not_load_pickle_arrays_or_unknown_schema(tmp_path):
    manifest = write_manifest(tmp_path)
    obj = json.loads(manifest.read_text())
    obj['schema_version'] = 2
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='schema_version'):
        ECoGAssemblyGenerator(tmp_path).generate_assembly('S1')
    obj['schema_version'] = 1
    np.save(tmp_path / 'object.npy', np.array([{'not': 'numeric data'}], dtype=object))
    obj['recordings'][0]['brain_data'] = {'npy': 'object.npy'}
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='allow_pickle=False'):
        ECoGAssemblyGenerator(tmp_path).generate_assembly('S1')


class WordTokenizer:
    name_or_path = 'synthetic token fixture'

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return ' '.join(tokens)


@pytest.mark.parametrize('policy,expected', [
    ('fullcontext', ['one', '', 'one  two', 'two three', 'three four']),
    ('nocontext', ['one', '', 'one  two', 'three', 'three four']),
    ('halfcontext', ['one', '', 'one  two', 'two three', 'three four']),
])
def test_shared_context_policies_with_explicit_tokenizer(policy, expected):
    r = recording(); del r['stimuli']
    original_provenance = deepcopy(r['provenance'])
    assembly = ECoGAssemblyGenerator(tokenizer=WordTokenizer()).generate_assembly(
        'S1', recordings=[r], context_type=policy, lookback=2)
    story = assembly.story_data['run-A']
    assert story.stimuli == expected
    assert story.provenance == original_provenance
    assert story.sample_metadata['stimulus_context']['context_type'] == policy
    assert story.sample_metadata['stimulus_context']['lookback'] == 2
    expected_sources = [0, 1, 0, 1, 2] if policy == 'fullcontext' else [0, 1, 0, 0, 0]
    np.testing.assert_array_equal(story.sample_metadata['context_source_indices'], expected_sources)
    with pytest.raises(ValueError, match='explicit tokenizer'):
        generate(r)


def test_empty_transcript_and_legacy_fmri_state_remain_supported(tmp_path):
    r = recording(); r.update(words=[], word_ids=[], stimulus_times=[], stimuli=[])
    silent = generate(r).story_data['run-A']
    assert silent.stimuli == [] and silent.data_times.size == 0
    np.testing.assert_array_equal(silent.word_rates, np.zeros((8, 1)))
    legacy = StoryData('fmri', np.ones((3, 2)), ['word'], [1, 1], np.arange(3.), np.array([0.]))
    for key in ['target_metadata', 'sample_metadata', 'provenance']:
        del legacy.__dict__[key]  # emulate fields absent from an older pickle
    assembly = SimpleNeuroidAssembly([legacy], validation_method='outer')
    original_coords = deepcopy(assembly.coords)
    path = tmp_path / 'old.pkl'; save_assembly(assembly, path)
    loaded = load_assembly(path)
    assert loaded.get_target_metadata() == [None]
    assert loaded.get_sample_metadata() == [None]
    assert loaded.get_provenance() == [None]
    np.testing.assert_array_equal(loaded.get_brain_data()[0], assembly.get_brain_data()[0])
    assert loaded.get_validation_method() == 'outer'
    for key, coord in original_coords.items():
        np.testing.assert_array_equal(loaded.get_coord(key), coord['values'])
