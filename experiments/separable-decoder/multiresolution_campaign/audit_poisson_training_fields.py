"""Independent NumPy audit of Poisson training-factorial fields and membership.

Checks all declared output/configuration/repetition combinations, saved hashes,
physical errors, source-draw membership and fixed training endpoint accounting.
Does not regenerate PDE truth, prove a continuum bound or audit gradients.
"""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sample(seed, count):
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(.15, .85, count), rng.uniform(.15, .85, count),
        np.exp(rng.uniform(np.log(.02), np.log(.1), count)), rng.uniform(.5, 2., count)))


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def audit(path):
    result = json.loads(path.read_text())
    cfg, provenance = result['config'], result['provenance']
    assert result['complete'] and provenance['backend'] == 'gpu' and provenance['x64']
    assert provenance['matmul_precision'] == 'highest'
    original = sample(cfg['original_draw_seed'], cfg['original_draw_count'])
    training = np.concatenate((original[:cfg['original_training_prefix_count']],
        sample(cfg['additional_training_seed'], cfg['additional_training_count'])))
    validation = np.concatenate((sample(cfg['existing_development_seed'], cfg['existing_development_count']),
        sample(cfg['additional_development_seed'], cfg['additional_development_count'])))
    for actual, key in ((original, 'original_full_draw_parameters'), (training, 'union_parameters')):
        np.testing.assert_array_equal(actual, result['training'][key])
    np.testing.assert_array_equal(validation, result['cohort']['parameters'])
    assert array_hash(training) == result['training']['union_sha256']
    assert array_hash(validation) == result['cohort']['parameter_sha256']
    cohort_names = ['existing_development']*cfg['existing_development_count'] + ['fresh_development']*cfg['additional_development_count']
    assert result['cohort']['groups'] == cohort_names
    checkpoints = {r['model']: r for r in result['checkpoints']}
    assert set(checkpoints) == set(cfg['arms'])
    endpoint_records = []
    for r in result['training']['arms']:
        ck = checkpoints[r['tag']]
        valid = r['finite'] and r['steps_done'] == cfg['training_steps_each'] and not r['time_capped']
        assert ck['valid_endpoint'] == r['valid_endpoint'] == valid
        assert ck['steps_done'] == r['steps_done']
        assert ck['training_count'] == (cfg['original_training_prefix_count'] if r['tag'].startswith('original') else len(training))
        assert hashlib.sha256((path.parent/ck['path']).read_bytes()).hexdigest() == ck['sha256']
        endpoint_records.append(dict(model=r['tag'], complete=bool(valid), updates=r['steps_done'], training_count=ck['training_count']))
    assert len({r['initial_weights_sha256'] for r in result['training']['arms']}) == 1
    for prefix in ('original', 'expanded'):
        records = [r for r in result['training']['arms'] if r['tag'].startswith(prefix)]
        assert len(records) == 2 and len({r['initial_codes_sha256'] for r in records}) == 1
    subjects = [(None, 'dst', None), (None, 'dst_coarse128', None)] + [
        (model, arm, tau) for model in cfg['arms'] for arm in ('rom_modular', 'rom_fused', 'rom_gj') for tau in cfg['taus']]
    expected = {(n, c, model, arm, tau, rep) for n, c, (model, arm, tau), rep in
        itertools.product(cfg['intervals'], range(len(validation)), subjects, range(cfg['repetitions']))}
    observed = {(r['intervals'], r['case'], r['model'], r['arm'], r['tau'], r['repetition']) for r in result['rows']}
    assert expected == observed and len(observed) == len(result['rows'])
    discrete_hashes = {(r['intervals'], r['case']): r['field_sha256'] for r in result['rows'] if r['arm'] == 'dst'}

    @lru_cache(maxsize=4)
    def field(digest):
        with np.load(path.parent/'fields'/(digest+'.npz')) as saved:
            a = saved['field']
        assert a.dtype == np.float64 and array_hash(a) == digest and np.isfinite(a).all()
        return a

    @lru_cache(maxsize=4)
    def reference(case):
        with np.load(path.parent/f'reference_case{case}.npz') as saved:
            return saved['observation'], saved['coarser_observation']

    @lru_cache(maxsize=None)
    def metrics(digest, n, case):
        a = field(digest)
        assert a.shape == (n+1, n+1)
        assert not np.count_nonzero(a[[0, -1]]) and not np.count_nonzero(a[:, [0, -1]])
        fine, coarse = reference(case)
        relative = lambda x, y: float(np.linalg.norm(x-y)/np.linalg.norm(y))
        stride = n//cfg['observation_intervals']
        error, delta = relative(a[::stride, ::stride], fine), relative(coarse, fine)
        return dict(physical_error=error, same_grid_error=relative(a, field(discrete_hashes[n, case])),
            reference_delta=delta, conservative_physical_error=(error+delta)/(1-delta))

    maximum_difference = 0.
    for row in result['rows']:
        assert row['group'] == cohort_names[row['case']]
        for key, value in metrics(row['field_sha256'], row['intervals'], row['case']).items():
            np.testing.assert_allclose(value, row[key], rtol=3e-12, atol=2e-14)
            maximum_difference = max(maximum_difference, abs(value-row[key]))
        parts = ('input_seconds', 'fused_device_seconds', 'output_seconds') if row['arm'] in ('rom_fused', 'rom_gj') else (
            'input_seconds', 'projection_init_seconds', 'solver_seconds', 'output_seconds')
        assert all(np.isfinite(row[k]) and row[k] >= 0 for k in parts)
        np.testing.assert_allclose(sum(row[k] for k in parts), row['total_seconds'], rtol=1e-12, atol=1e-14)
        if row['model'] is not None:
            assert row['stationary'] == (row['stationarity'] <= cfg['stationarity_tolerance'])
            assert row['tau_reached'] == (row['reason'] == 2)
            if row['tau_reached']:
                assert row['tau'] > 0 and row['residual'] <= row['tau']*row['initial_residual']*(1+1e-12)
            assert row['solver_valid'] == bool(row['finite'] and row['parity_passed'] and (row['stationary'] or row['tau_reached']))
        if row['arm'] == 'rom_gj':
            assert 0 <= row['fallback_count'] <= row['attempts']
            if row['solver_valid']:
                assert row['max_linear_backward_error'] <= cfg['linear_backward_limit']
    return dict(scope=__doc__, source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job_id=provenance['job_id'], source_commit=provenance['commit'], verified_invocations=len(observed),
        distinct_field_hashes=len({r['field_sha256'] for r in result['rows']}),
        maximum_metric_disagreement=maximum_difference, declared_draws_and_repetitions_match=True,
        endpoints=endpoint_records, rigorous_reference_bound=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
