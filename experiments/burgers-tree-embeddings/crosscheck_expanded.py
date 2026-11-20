"""Independent FD/CN qualification of separately seeded B8 reference chunks."""
import argparse
from common import *
import data, fd_reference as F
from neural import expanded_cases as E
from neural.named_references import initial
from neural.build_data import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--source-jobs', required=True, help='Comma-separated owned expanded-reference job names')
    ap.add_argument('--start', type=int, default=0, help='First case in source-job order')
    ap.add_argument('--count', type=int)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ns = Path(os.environ['STUDY_NAMESPACE'])
    jobs = a.source_jobs.split(',')
    assert jobs and len(set(jobs)) == len(jobs)
    assert all(job and job.replace('-', '').isalnum() for job in jobs)
    sources, cases = [], []
    for job in jobs:
        src = resolve(ns/job/'output/result.json', ns)
        source = json.loads(src.read_text()); source_sha = digest(src)
        assert source['status'] == 'complete_expanded_ladders_pending_independent_crosscheck'
        assert source['all_ladders_pass'] and not source['final_test_opened']
        cohort = source['cohort']; spec = source['spec']
        assert cohort in ('train', 'dev') and spec['cohort'] == cohort and spec['nu'] == E.NU
        expected = E.definitions(cohort, spec['start'], spec['count'])
        assert spec['definitions'] == expected
        assert [row['case'] for row in source['cases']] == [row['case'] for row in expected]
        cache = src.parent/source['coordinate_data']['file']
        assert cache.parent == src.parent and digest(cache) == source['coordinate_data']['sha256']
        for row, original in zip(source['cases'], expected):
            assert all(row[key] == value for key, value in original.items())
            assert row['complete'] and row['provisional_ladder_pass']
            assert not row['admitted_reference'] and row['nu'] == E.NU
            assert max(row['allowance']) <= .001 and max(row['scoring_change']) <= .0002
            for level in row['levels']:
                path = src.parent/level['file']
                assert path.parent == src.parent and digest(path) == level['sha256']
            cases.append((row, src.parent, source_sha, job))
        sources.append(dict(job=job, file=str(src), source_sha256=source_sha, cohort=cohort,
                            cache_sha256=source['coordinate_data']['sha256']))
    assert len({row['case'] for row, _, _, _ in cases}) == len(cases), 'Duplicate cases across source jobs'
    count = len(cases)-a.start if a.count is None else a.count
    assert 0 <= a.start < len(cases) and 0 < count <= len(cases)-a.start
    selected = cases[a.start:a.start+count]
    result = dict(meta=preflight(), status='running', sources=sources, cases=[],
                  selected_cases=[row['case'] for row, _, _, _ in selected],
                  final_test_opened=False, normalization='current-time reference physical L2',
                  reference_error_bound=.001, temporal_change_bound=.0002, solver_residual_bound=1e-11)
    save(out/'result.json', result)
    solvers = {level: F.make_solver(2, level) for level in (256, 512)}
    for row, folder, source_sha, job in selected:
        record = row['levels'][-1]
        assert record['N'] == 512 and record['dt'] == .00015625
        with np.load(folder/record['file']) as payload:
            b = payload['coefficients']
            assert np.array_equal(payload['times'], data.TIMES) and float(payload['nu']) == row['nu']
        records, states = [], []
        for level, dt in [(256, .0003125), (512, .0003125), (512, .00015625)]:
            axis = np.arange(1, level)/level
            xy = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
            values, seconds = timer(lambda: solvers[level](
                initial(jnp.asarray(xy), row['definition']), row['nu'], dt, 'cnr', 1e-11, 1e-11))
            state = np.asarray(values['out']).reshape(11, level-1, level-1)
            assert np.isfinite(state).all()
            truth = np.asarray(data.synth_grid(jnp.asarray(b), jnp.asarray(axis)))
            norm = np.linalg.norm(truth.reshape(11, -1), axis=1)
            assert np.isfinite(norm).all() and np.all(norm > 0)
            error = np.linalg.norm((state-truth).reshape(11, -1), axis=1)/norm
            used = int(values['stages']); residuals = np.asarray(values['res'])
            assert 0 < used <= len(residuals)
            residual = float(np.max(residuals[:used])); assert np.isfinite(residual)
            path = out/f"{row['case']}_L{level}_dt{dt:.8f}.npz"
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **values, times=data.TIMES, nu=row['nu'],
                                case=row['case'], cohort=row['cohort'])
            tmp.replace(path)
            records.append(dict(L=level, dt=dt, file=path.name, sha256=digest(path),
                                relative_error=error, max_solver_residual=residual, seconds=seconds))
            states.append(state)
            print('expanded_crosscheck', row['case'], level, dt, float(error.max()), residual, flush=True)
        timechange = np.linalg.norm((states[-1]-states[-2]).reshape(11, -1), axis=1)/norm
        passed = bool(max(records[-1]['relative_error']) <= .001 and max(timechange) <= .0002
                      and all(level['max_solver_residual'] <= 1e-11 for level in records))
        result['cases'].append(dict(case=row['case'], sourcecohort=row['cohort'], cohort=row['cohort'],
                                    family=row['family'], definition=row['definition'], nu=row['nu'],
                                    source_job=job, source_sha256=source_sha,
                                    reference_file=str(folder/record['file']), reference_sha256=record['sha256'],
                                    levels=records, temporal_change=timechange, passed=passed,
                                    admitted_reference=passed, reference_allowance=row['allowance']))
        save(out/'result.json', result)
    result.update(status='complete', passed=all(row['passed'] for row in result['cases']))
    save(out/'result.json', result)


if __name__ == '__main__':
    main()
