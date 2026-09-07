"""Independent saved-field and replay-system audit of the Poisson kernel pilot."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(path):
    data = json.loads(path.read_text())
    cfg, provenance = data['config'], data['provenance']
    assert data['complete'] and provenance['backend'] == 'gpu' and provenance['x64']
    assert provenance['matmul_precision'] == 'highest'
    subjects = [('dst', None, None), ('dst_coarse128', None, None)] + [
        (arm, m, tau) for arm in ('rom_modular', 'rom_fused', 'rom_gj')
        for m in cfg['requested_modes_ladder'] for tau in cfg['taus']]
    expected = {(n, c, arm, m, tau, rep) for n in cfg['intervals'] for c in range(cfg['cohort_count'])
                for arm, m, tau in subjects for rep in range(cfg['repetitions'])}
    observed = {(r['intervals'], r['case'], r['arm'], r['requested_modes'], r['tau'], r['repetition']) for r in data['rows']}
    assert expected == observed and len(observed) == len(data['rows'])
    groups = defaultdict(list)
    for row in data['rows']:
        groups[row['intervals'], row['case'], row['arm'], row['requested_modes'], row['tau']].append(row)
    worst, field_hashes = 0., set()
    for (n, case, arm, m, tau), rows in groups.items():
        filename = f'field_n{n}_case{case}_{arm}_M{m}_tau{tau}.npz'
        with np.load(path.parent/filename) as saved:
            field = np.ascontiguousarray(saved['field'])
        digest = hashlib.sha256(field.tobytes()).hexdigest()
        field_hashes.add(digest)
        assert field.dtype == np.float64 and np.all(np.isfinite(field))
        with np.load(path.parent/f'field_n{n}_case{case}_dst_MNone_tauNone.npz') as saved:
            discrete = saved['field']
        with np.load(path.parent/f'reference_case{case}.npz') as saved:
            fine, coarse = saved['observation'], saved['coarser_observation']
        relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
        error = relative(field[::n//cfg['observation_intervals'], ::n//cfg['observation_intervals']], fine)
        delta = relative(coarse, fine)
        metrics = dict(physical_error=error, same_grid_error=relative(field, discrete),
                       reference_delta=delta, conservative_physical_error=(error+delta)/(1-delta))
        for row in rows:
            assert row['field_sha256'] == digest
            for name, value in metrics.items():
                np.testing.assert_allclose(value, row[name], rtol=3e-12, atol=2e-14)
                worst = max(worst, abs(value-row[name]))
            parts = ('input_seconds', 'fused_device_seconds', 'output_seconds') if arm in ('rom_fused', 'rom_gj') else (
                'input_seconds', 'projection_init_seconds', 'solver_seconds', 'output_seconds')
            assert all(np.isfinite(row[k]) and row[k] >= 0 for k in parts)
            np.testing.assert_allclose(sum(row[k] for k in parts), row['total_seconds'], rtol=1e-12, atol=1e-14)
            if arm.startswith('rom'):
                stationary = row['stationarity'] <= cfg['stationarity_tolerance']
                assert row['stationary'] == stationary
                assert row['tau_reached'] == (row['reason'] == 2)
                if row['tau_reached']:
                    assert tau > 0 and row['residual'] <= tau*row['initial_residual']*(1+1e-12)
                assert row['solver_valid'] == bool(row['finite'] and row['parity_passed'] and (stationary or row['tau_reached']))
            if arm == 'rom_gj':
                assert 0 <= row['fallback_count'] <= row['attempts']
                assert row['max_linear_backward_error'] <= cfg['linear_backward_limit']
    systems, maximum_backward, maximum_cholesky_difference = 0, 0., 0.
    replay_counter_mismatches = 0
    for row in data['trajectories']:
        with np.load(path.parent/row['artifact']) as saved:
            assert len(saved['A']) == row['attempts']
            np.testing.assert_array_equal(saved['final_z'], row['latent'])
            for a, b, x in zip(saved['A'], saved['b'], saved['step']):
                np.testing.assert_allclose(a, a.T, rtol=1e-12, atol=1e-20)
                chol = np.linalg.cholesky(a)
                independent = np.linalg.solve(chol.T, np.linalg.solve(chol, b))
                backward = np.linalg.norm(a@x-b)/(np.linalg.norm(a)*np.linalg.norm(x)+np.linalg.norm(b)+1e-300)
                diff = np.linalg.norm(x-independent)/(np.linalg.norm(independent)+1e-300)
                assert backward <= cfg['linear_backward_limit']
                maximum_backward = max(maximum_backward, float(backward))
                maximum_cholesky_difference = max(maximum_cholesky_difference, float(diff))
                systems += 1
        replay_counter_mismatches += int(not row['replay_counter_agreement'])
    return dict(scope='Independent saved output hashes/physical metrics/repetition accounting, plus SPD Cholesky replay checks. Replays are not timing evidence.',
        source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job_id=provenance['job_id'], source_commit=provenance['commit'], verified_invocations=len(data['rows']),
        preserved_fields=len(groups), distinct_field_hashes=len(field_hashes), maximum_metric_disagreement=worst,
        replayed_normal_systems=systems, maximum_linear_backward_error=maximum_backward,
        maximum_step_difference_from_cpu_cholesky=maximum_cholesky_difference,
        replay_counter_mismatches=replay_counter_mismatches, rigorous_reference_bound=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
