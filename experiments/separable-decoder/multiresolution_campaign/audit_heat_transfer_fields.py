"""Independent CPU audit of frozen heat transfer outputs and full-query accounting.

Recomputes all saved physical/common/discrete errors, seeded memberships, frozen
checkpoint bytes, repeated-output hashes and initial-output policy. References
are retained spectral trajectories; this is not a rigorous continuum bound.
"""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import statistics

import numpy as np
from audit_heat_head_fields import metrics, restrict, compare_tree


def audit(path):
    native = json.loads(path.read_text())
    cfg, settings, meta = native['config'], native['settings'], native['metadata']
    assert native['complete'] and meta['backend'] == 'gpu' and meta['x64'] and meta['precision'] == 'highest'
    assert not native['final_cohort_opened'] and not native['validation_in_training_or_lookup']
    models = {r['name']: r for r in native['models']}
    assert models.keys() == {r['name'] for r in settings['checkpoints']}
    weights = []
    for name, r in models.items():
        assert r['frozen_weights'] and r['frozen_codes']
        blob = (path.parent/r['output_path']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == r['sha256']
        ck = pickle.loads(blob)
        assert ck['config'] == cfg
        assert ck['codes'].shape == (160*len(cfg['times']), cfg['k'])
        weights.append(ck['params'])
    for w in weights[1:]:
        for key in ('B', 'g', 'out_scale'):
            compare_tree(weights[0][key], w[key])
    cases = native['cases']
    offset = 0
    for cohort in settings['cohorts']:
        rng = np.random.default_rng(cohort['seed'])
        count = cohort['count']
        draws = np.column_stack((rng.uniform(*cfg['center_range'], (count, 2)),
            rng.uniform(*cfg['width_range'], count), rng.uniform(*cfg['amplitude_range'], count)))
        recorded = cases[offset:offset+count]
        assert all(r['cohort'] == cohort['name'] and r['seed'] == cohort['seed'] and r['case'] == offset+i and r['cohort_index'] == i for i, r in enumerate(recorded))
        np.testing.assert_array_equal(draws, [r['draw'] for r in recorded])
        offset += count
    assert len(cases) == offset and len({tuple(r['draw']) for r in cases}) == offset
    artifacts = {}

    @lru_cache(maxsize=5)
    def load_field(name, expected_hash):
        p = path.parent/name
        with np.load(p) as f:
            a = np.ascontiguousarray(f['field'])
        assert a.dtype == np.float64 and np.isfinite(a).all()
        actual = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()
        assert actual == expected_hash
        artifacts[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return a

    def field(record):
        a = load_field(record['path'], record['sha256_array'])
        assert list(a.shape) == record['shape'] and a.dtype.str == record['dtype']
        return a

    worst = 0.
    def check(a, b, n, reported):
        nonlocal worst
        actual = metrics(a, b, n)
        for key, value in actual.items():
            np.testing.assert_allclose(value, reported[key], rtol=3e-12, atol=3e-14)
            worst = max(worst, float(np.max(np.abs(value-reported[key]))))
        return actual

    references = {r['case']: r for r in native['reference_fields']}
    assert references.keys() == set(range(offset))
    for r in references.values():
        coarse, fine = field(r['coarse']), field(r['fine'])
        assert [r['coarse_intervals'], r['fine_intervals']] == settings['reference_intervals']
        check(coarse, restrict(fine, r['fine_intervals'], r['coarse_intervals']), r['coarse_intervals'], r['refinement'])
    expected = set()
    for n, c in itertools.product(settings['requested_intervals'], range(offset)):
        solvers = {n} | {s for s in settings['coarse_solver_intervals'] if s <= n}
        expected.update((n, c, name) for name in list(models)+[f'fom_dst_{s}' for s in solvers])
    actual_keys = {(r['intervals'], r['case'], r['method']) for r in native['rows']}
    assert actual_keys == expected and len(native['rows']) == len(expected)
    saved_cases = {(r['intervals'], r['case']): r for r in native['case_fields']}
    assert saved_cases.keys() == set(itertools.product(settings['requested_intervals'], range(offset)))
    count, summaries = 0, []
    for row in native['rows']:
        n, cid = row['intervals'], row['case']
        assert row['cohort'] == cases[cid]['cohort']
        saved, ref = saved_cases[n, cid], references[cid]
        initial, physical, discrete = field(saved['initial']), field(saved['physical']), field(saved['discrete'])
        np.testing.assert_array_equal(physical, restrict(field(ref['fine']), ref['fine_intervals'], n))
        np.testing.assert_array_equal(initial, physical[0])
        coarse = restrict(field(ref['coarse']), ref['coarse_intervals'], n)
        common = settings['observation_intervals']
        check(coarse, physical, n, saved['reference_refinement_full'])
        check(restrict(coarse, n, common), restrict(physical, n, common), common, saved['reference_refinement_common'])
        check(discrete, physical, n, saved['spatial_discrete_vs_physical'])
        assert sorted(r['repetition'] for r in row['repetitions']) == list(range(settings['timing_repetitions']))
        costs, full_errors, common_errors = [], [], []
        initial_bad, step_bad, outliers = 0, 0, 0
        for rep in row['repetitions']:
            a = field(rep['field'])
            assert a.shape == (len(cfg['times']), n-1, n-1)
            assert rep['full_field_sha256'] == rep['field']['sha256_array']
            check(a, discrete, n, rep['vs_same_grid'])
            full = check(a, physical, n, rep['vs_physical_per_grid'])
            shared = check(restrict(a, n, common), restrict(physical, n, common), common, rep['vs_physical_common_grid'])
            phases = rep['phases']
            assert all(np.isfinite(v) and v >= 0 for v in phases.values())
            assert sum(v for k,v in phases.items() if k != 'query_seconds') <= phases['query_seconds']+1e-12
            if row['model'] == 'fom':
                np.testing.assert_array_equal(a[0], initial)
                assert not rep['solver']
            else:
                fits, steps = np.asarray(rep['solver']['initial_fits']), np.asarray(rep['solver']['steps'])
                chosen = fits[np.argmin(fits[:, 3])]
                initial_bad += int(chosen[2] != 1 or chosen[4] > row['gradient_tolerance'])
                step_bad += int(np.sum((steps[:, 2] != 1) | (steps[:, 4] > row['gradient_tolerance'])))
            costs.append(phases['query_seconds'])
            full_errors.append(float(max(full['relative_current'])))
            common_errors.append(float(max(shared['relative_current'])))
            count += 1
        middle = statistics.median(costs)
        summaries.append(dict(intervals=n, case=cid, cohort=row['cohort'], method=row['method'], model=row['model'], solver_intervals=row['solver_intervals'],
            median_query_seconds=middle, timing_repetitions=costs, timing_outliers_above_twice_case_median=sum(t > 2*middle for t in costs),
            worst_full_grid_error=max(full_errors), worst_common_error=max(common_errors),
            initial_nonstationary_repetitions=initial_bad, nonstationary_steps_all_repetitions=step_bad))
    assert count == native['timed_invocations'] == settings['estimated_timed_calls']
    return dict(scope=__doc__, source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_commit=native['source_manifest']['source_commit'], job_id=meta['job_id'],
        verified_timed_invocations=count, maximum_metric_disagreement=worst, artifact_hashes=artifacts,
        declared_checkpoint_case_and_repetition_counts_match=True, frozen_spatial_parameters_match=True,
        fom_initial_passthrough_verified=True, rows=summaries, rigorous_reference_bound=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('job_id', 'maximum_metric_disagreement', 'verified_timed_invocations')}))
