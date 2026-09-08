"""Root CPU audit of Poisson projection/initialization factorial outputs.

Checks every timed and accuracy-only output, physical errors, per-start stopping
thresholds and training-cache lookup from independently transformed source fields.
No production solver is imported; saved references are reused, not certified.
"""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn
from audit_poisson_training_fields import array_hash, sample


def audit(path):
    data = json.loads(path.read_text()); cfg = data['config']; meta = data['provenance']
    assert data['complete'] and meta['backend'] == 'gpu' and meta['x64'] and meta['matmul_precision'] == 'highest'
    pars = np.asarray(data['cohort']['parameters'])
    regenerated = np.concatenate((sample(cfg['existing_development_seed'], cfg['existing_development_count']),
        sample(cfg['fresh_development_seed'], cfg['fresh_development_count'])))
    np.testing.assert_array_equal(regenerated[:, [0, 1, 3]], pars[:, [0, 1, 3]])
    np.testing.assert_allclose(regenerated[:, 2], pars[:, 2], rtol=4*np.finfo(float).eps, atol=0)
    assert array_hash(pars) == cfg['parameter_sha256'] == data['cohort']['parameter_sha256']
    groups = ['existing_development']*cfg['existing_development_count']+['fresh_development']*cfg['fresh_development_count']
    assert groups == data['cohort']['groups']
    for ck in data['checkpoints']:
        assert ck['sha256'] == cfg['checkpoint_sha256'][ck['model']]
    roms = list(itertools.product(cfg['models'], cfg['projection'], cfg['initialization']))
    expected = set()
    for n, case in itertools.product(cfg['intervals'], range(len(pars))):
        for panel, reps, subjects in [('primary', cfg['repetitions'], [(None, 'dst', None), (None, 'dst_coarse128', None)]+roms), ('stationary', 1, roms)]:
            for (model, proj, init), rep in itertools.product(subjects, range(reps)):
                arm = 'rom_speed' if model else proj
                expected.add((panel, n, case, model, arm, proj if model else None, init, rep))
    all_rows = data['rows']+data['stationary_rows']
    key = lambda r: (r['panel'], r['intervals'], r['case'], r['model'], r['arm'], r['projection'], r['initialization'], r['repetition'])
    assert {key(r) for r in all_rows} == expected and len(all_rows) == len(expected)
    assert len(data['rows']) == cfg['expected_timed_invocations'] and len(data['stationary_rows']) == cfg['expected_stationary_invocations']
    assert all(r['panel'] == 'primary' for r in data['rows']) and all(r['panel'] == 'stationary' and r['tau'] == 0 for r in data['stationary_rows'])
    same = {(r['intervals'], r['case']): r['field_sha256'] for r in data['rows'] if r['arm'] == 'dst'}
    setups = {(s['intervals'], s['model']): s for s in data['setup']}
    relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))

    @lru_cache(maxsize=4)
    def field(digest):
        with np.load(path.parent/'fields'/(digest+'.npz')) as f:
            a = f['field']
        assert a.dtype == np.float64 and array_hash(a) == digest and np.isfinite(a).all()
        return a

    @lru_cache(maxsize=1)
    def reference(case):
        with np.load(path.parent/f'reference_case{case}.npz') as f:
            return f['observation'], f['coarser_observation']

    @lru_cache(maxsize=None)
    def metric(digest, n, case):
        a = field(digest); fine, coarse = reference(case)
        assert a.shape == (n+1, n+1) and not np.count_nonzero(a[[0,-1]]) and not np.count_nonzero(a[:,[0,-1]])
        error = relative(a[::n//cfg['observation_intervals'], ::n//cfg['observation_intervals']], fine)
        delta = relative(coarse, fine)
        return dict(physical_error=error, same_grid_error=relative(a, field(same[n, case])), reference_delta=delta,
            conservative_physical_error=(error+delta)/(1-delta))

    @lru_cache(maxsize=4)
    def cache(n, model):
        with np.load(path.parent/f'cache_{model}_{n}.npz') as f:
            codes, predictions, B = f['codes'], f['predictions'], f['B']
        setup = setups[n, model]
        assert array_hash(codes) == setup['cache']['training_codes_sha256']
        assert array_hash(predictions) == setup['cache']['prediction_sha256']
        assert array_hash(B) == setup['operator_sha256'] == setup['cache']['operator_sha256']
        return codes, predictions

    @lru_cache(maxsize=4)
    def initialization(n, model, case):
        x = np.arange(1, n, dtype=float)/n
        cx, cy, width, amplitude = pars[case]
        source = amplitude*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width**2))
        spectrum = dstn(source, type=1, norm='ortho')
        indices = np.asarray(setups[n, model]['mode_indices'])-1
        lam = 4*n*n*np.sin(np.pi*np.arange(1,n)/(2*n))**2
        fm = spectrum[indices[:,0], indices[:,1]]/(lam[indices[:,0]]+lam[indices[:,1]])
        codes, predictions = cache(n, model)
        dist2 = np.sum((predictions-fm)**2, axis=1)
        ordered = np.argsort(dist2, kind='stable'); index = int(ordered[0])
        return codes.mean(0), codes[index], index, float(np.sqrt(dist2[index])), float(dist2[ordered[1]]-dist2[index])

    maximum = 0.; lookup_maximum = 0.; lookups = set()
    for row in all_rows:
        n, case = row['intervals'], row['case']
        assert row['group'] == groups[case]
        for k, v in metric(row['field_sha256'], n, case).items():
            np.testing.assert_allclose(v, row[k], rtol=3e-12, atol=2e-14)
            maximum = max(maximum, abs(v-row[k]))
        parts = ('input_seconds', 'fused_device_seconds', 'output_seconds') if row['model'] else ('input_seconds', 'solver_seconds', 'output_seconds')
        assert all(np.isfinite(row[k]) and row[k] >= 0 for k in parts)
        np.testing.assert_allclose(sum(row[k] for k in parts), row['total_seconds'], rtol=1e-12, atol=1e-14)
        if row['model'] is None:
            continue
        assert row['tau'] == (cfg['primary_tau'] if row['panel'] == 'primary' else 0)
        np.testing.assert_allclose(row['absolute_tau_threshold'], row['tau']*row['initial_residual'], rtol=1e-14, atol=0)
        assert row['tau_reached'] == (row['reason'] == 2)
        if row['tau_reached']:
            assert row['tau'] > 0 and row['residual'] <= row['absolute_tau_threshold']*(1+1e-12)
        assert row['stationary'] == (row['stationarity'] <= cfg['stationarity_tolerance'])
        assert row['solver_valid'] == bool(row['finite'] and row['parity_passed'] and row['max_linear_backward_error'] <= cfg['linear_backward_error_limit'] and (row['stationary'] or row['tau_reached']))
        assert 0 <= row['fallback_count'] <= row['attempts']
        mean, nearest, index, distance, gap = initialization(n, row['model'], case)
        is_nearest = row['initialization'] == 'nearest_cached_scaled_weak_prediction'
        expected_code = nearest if is_nearest else mean
        np.testing.assert_allclose(expected_code, row['initial_latent'], rtol=2e-13, atol=2e-14)
        assert row['selected_training_code_index'] == (index if is_nearest else -1)
        if is_nearest:
            np.testing.assert_allclose([distance,gap], [row['cache_distance'],row['cache_squared_distance_gap']], rtol=2e-9, atol=2e-13)
            lookup_maximum = max(lookup_maximum, abs(distance-row['cache_distance']), abs(gap-row['cache_squared_distance_gap']))
            lookups.add((n, row['model'], case))
    return dict(scope=__doc__, source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job_id=meta['job_id'], source_commit=meta['commit'], verified_timed_invocations=len(data['rows']),
        verified_accuracy_controls=len(data['stationary_rows']), distinct_timed_and_control_fields=len({r['field_sha256'] for r in all_rows}),
        maximum_metric_disagreement=maximum, independent_nearest_lookup_panels=len(lookups),
        maximum_lookup_scalar_disagreement=lookup_maximum, per_start_thresholds_verified=True,
        timing_scope='Full host field and returned stop/counter/latent/guard/initialization values. Stationarity recomputation, physical-error evaluation and parity diagnostics occur afterward.',
        rigorous_reference_bound=None)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('input',type=Path); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); result=audit(args.input); args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
