"""Root NumPy audit of preserved wave common-grid outputs and paired accounting.

No production solver import, training or PDE regeneration. The owner separately
audits full-grid ROM reconstruction, bank weights, affine dynamics and geometry.
This checks each declared query configuration and its preserved physical metrics.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np


def metrics(u, v, reference_u, reference_v, n, boundary, speed):
    weights = np.ones(u.shape[-1])
    if boundary == 'absorbing':
        weights[[0, -1]] = .5
    area = np.outer(weights, weights)/n**2

    def l2(a):
        return np.sqrt(np.einsum('tij,ij,tij->t', a, area, a))

    def phase(a, b):
        kinetic = l2(b)**2
        if boundary == 'dirichlet':
            a = np.pad(a, ((0, 0), (1, 1), (1, 1)))
            potential = (np.sum(np.diff(a, axis=1)**2, axis=(1, 2)) +
                         np.sum(np.diff(a, axis=2)**2, axis=(1, 2)))
        else:
            potential = (np.sum(np.diff(a, axis=1)**2*weights[None, None, :], axis=(1, 2)) +
                         np.sum(np.diff(a, axis=2)**2*weights[None, :, None], axis=(1, 2)))
        return np.sqrt(kinetic + speed**2*potential)

    phase_reference = phase(reference_u, reference_v)
    u_reference = l2(reference_u)
    definitions = (
        ('displacement', l2(u-reference_u), u_reference, u_reference[0]),
        ('velocity', l2(v-reference_v), l2(reference_v), phase_reference[0]),
        ('energy_state', phase(u-reference_u, v-reference_v), phase_reference, phase_reference[0]),
    )
    return {name: dict(absolute=error, reference_norm=norm, initial_scale=scale,
                      initial_normalized=error/scale) for name, error, norm, scale in definitions}


def audit(path):
    result = json.loads(path.read_text())
    cfg, provenance = result['config'], result['provenance']
    assert result['complete'] and not result['final_test_opened']
    assert provenance['jax_backend'] == 'gpu' and provenance['x64']
    assert provenance['matmul_precision'] == 'highest'
    larger_heads = 'new_latent_dimension' in cfg
    rows = result['invocations'] + result.get('accuracy_controls', [])
    groups = defaultdict(list)
    for row in rows:
        role = row.get('measurement_role', 'timed_comparison')
        if larger_heads:
            assert row['comparison_eligible'] == (role == 'timed_comparison')
            assert row['warmup_performed'] == (role == 'timed_comparison')
        groups[row['boundary'], row['intervals'], row['case'], row['method'], row['setting'], role].append(row)
    expected = set()
    for bc, n, case in itertools.product(cfg['boundaries'], cfg['meshes'], cfg['validation_indices']):
        if larger_heads:
            heads = [f'new_mlp32_seed{seed}' for seed in cfg['training']['optimizer_seeds']]
            choices = [('frozen_mlp16_seed691200', cfg['primary_dt']), ('affine32', 0.)]
            choices += [(name, dt) for name, dt in itertools.product(heads, cfg['nonlinear_dts'])]
            expected.update((bc, n, case, name, cfg['accuracy_only_dt'], 'accuracy_refinement_only') for name in heads)
        else:
            choices = [('rom', dt) for dt in cfg['rom_dts']]
            choices += [(name, 0.) for name in ('affine16', 'affine32', 'full64')]
        choices += [('dst', 0.)] if bc == 'dirichlet' else [('rk4', cfl) for cfl in cfg['fom_cfls']]
        expected.update((bc, n, case, method, setting, 'timed_comparison') for method, setting in choices)
    assert set(groups) == expected
    verified, accuracy_verified, discrepancies, summaries = 0, 0, [], []
    for (bc, n, case, method, setting, role), calls in groups.items():
        repetitions = cfg['repetitions'] if role == 'timed_comparison' else 1
        assert sorted(r['repetition'] for r in calls) == list(range(repetitions))
        first = next(r for r in calls if r['repetition'] == 0)
        assert all(r['completed'] and r['output_sha256'] == first['output_sha256'] for r in calls)
        file = path.parent/(first['invocation_id']+'.npz')
        assert hashlib.sha256(file.read_bytes()).hexdigest() == result['output_sha256'][file.name]
        with np.load(file) as saved, np.load(path.parent/f'reference_{bc}_{case}.npz') as ref:
            u, v, ut, vt = saved['u'], saved['v'], ref['u'], ref['v']
        assert u.shape == v.shape == ut.shape == vt.shape
        assert len(u) == cfg['query_observations'] and u.dtype == v.dtype == np.float64
        assert np.isfinite(u).all() and np.isfinite(v).all()
        actual = metrics(u, v, ut, vt, cfg['comparison_intervals'], bc, first['parameters'][5])
        for call in calls:
            for name, entries in actual.items():
                for key, value in entries.items():
                    reported = call['physical_reference_error'][name][key]
                    np.testing.assert_allclose(value, reported, rtol=2e-11, atol=2e-13)
                    discrepancies.append(float(np.max(np.abs(value-reported))))
            parts = call['seconds']
            assert abs(parts['complete_query']-sum(v for k, v in parts.items() if k != 'complete_query')) < 1e-12
            assert all(v > 0 for v in parts.values())
            size = n-1 if bc == 'dirichlet' else n+1
            assert call['output_bytes'] == 2*8*cfg['query_observations']*size**2
            if role == 'timed_comparison':
                verified += 1
            else:
                accuracy_verified += 1
        summaries.append(dict(boundary=bc, intervals=n, case=case, method=method, setting=setting,
            measurement_role=role, comparison_eligible=role == 'timed_comparison',
            query_median_seconds=statistics.median(r['seconds']['complete_query'] for r in calls),
            worst_initial_normalized=max(float(np.max(m['initial_normalized'])) for m in actual.values()),
            final_current_relative_energy=float(actual['energy_state']['absolute'][-1]/actual['energy_state']['reference_norm'][-1])))
    return dict(scope=__doc__, source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        job_id=provenance['job_id'], source_commit=provenance['source_commit'],
        configurations_verified=len(groups), verified_invocations=verified, verified_accuracy_controls=accuracy_verified,
        maximum_metric_disagreement=max(discrepancies), rows=summaries, rigorous_reference_bound=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('job_id', 'verified_invocations', 'maximum_metric_disagreement')}))
