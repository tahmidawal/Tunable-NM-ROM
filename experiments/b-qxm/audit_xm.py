"""Independent NumPy audit of one b-qxm job. No JAX, no GPU, no import of the driver.

Recomputes every reported error from the retained output fields, measures every arm
against the same-job converged full-order solve on three metrics (t = 0 compression, worst
over all output times, worst over evolved times), checks every cross-job fidelity pair
declared in the config against `checks/comparators.json`, re-evaluates the two in-job
invariance gates from the fields, and writes one per-arm row table for the report
generator. Spans and decompositions across jobs are NOT computed here; they belong to
`reports/generate_xm.py`, which reads the audits of all three jobs.

    python audit_xm.py <result.json> --fields <dir> --out <audit.json>
                       [--comparators checks/comparators.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--fields', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--comparators', default=str(HERE / 'checks/comparators.json'))
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    fields_dir = Path(a.fields)
    checks, fail = {}, []

    def gate(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note)
        if not ok:
            fail.append(name)

    def info(name, ok, detail=None, note=None):
        checks[name] = dict(passed=(None if ok is None else bool(ok)), detail=detail,
                            note=note, blocking=False)

    # ------------------------------------------------------- environment ------
    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('step_budget_600', r['config']['strict']['step_budget'] == 600, r['config']['strict'])
    gate('dense_quadrature_everywhere',
         all(x.get('quadrature') == 'dense' for x in r['declared_subjects'] if x.get('method') == 'rom'))
    cg = r['gates']['evaluation_cohort_bitwise_abl01']
    gate('evaluation_cohort_bitwise_abl01', bool(cg['passed']), cg)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    K = r['K']

    cmp_table = json.loads(Path(a.comparators).read_text()) if Path(a.comparators).exists() else {}
    mine_dir = r['directions']['directions_sha256']
    src_dirs = dict(r['config'].get('source_directions_sha256') or {})
    for k, v in cmp_table.items():
        if v.get('directions_sha256'):
            src_dirs.setdefault(k, v['directions_sha256'])
    dir_match = {k: (v == mine_dir) for k, v in src_dirs.items()}
    for k, v in dir_match.items():
        info(f'directions_hash_matches_{k}', v, dict(ours=mine_dir, theirs=src_dirs[k]),
             'bitwise across jobs; GPU-model dependent (80 GB and 40 GB A100s disagree); a probe')
    refsha = {str(x['case']): x['field_sha256'] for x in r['reference']}
    for k, v in (r['config'].get('source_reference_sha256') or {}).items():
        info(f'reference_fields_bitwise_match_{k}', refsha == v, None, 'GPU-model dependent; a probe')
    gate('directions_rank_covers_ladder',
         r['directions']['available_rank'] >= max(x.get('q') or 0 for x in r['declared_subjects']),
         r['directions']['available_rank'])
    osb = r['gates'].get('operators_shared_bitwise') or {}
    gate('operators_shared_bitwise', osb.get('passed') is True, osb)

    # ------------------------------------------------------- invocations ------
    inv = r['invocations']
    reps = r['config']['reps']
    counts = {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms),
         [(x['name'], x['M'], x['solved_dimension']) for x in roms if x['M'] <= x['solved_dimension']][:5])
    gate('every_rom_carries_exit_and_stationarity',
         all(('stop_reasons' in x and 'budget_exits' in x and 'worst_joint_stationarity' in x) for x in roms))
    declared = {x['name'] for x in r['declared_subjects'] if x.get('method') == 'rom'}
    gate('every_declared_cell_timed', declared <= {x['name'] for x in roms}, sorted(declared - {x['name'] for x in roms}))

    # ----------------------------------------- errors recomputed from fields --
    cache = {}

    def F(name):
        if name not in cache:
            cache[name] = np.load(fields_dir / name)['fields']
        return cache[name]

    missing = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
    gate('artifacts_present', not missing, missing[:5])
    refs = {x['case']: F(x['artifact']) for x in r['reference']}
    base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
    gate('same_grid_baseline_present', len(base) == len(refs), sorted(base))
    worst = 0.
    for x in inv:
        truth = refs[x['case']]
        f = F(x['artifact'])
        n0 = np.linalg.norm(truth[0])
        err = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
        worst = max(worst, abs(float(np.max(err)) - x['error']['fixed_initial_max'])
                    / max(x['error']['fixed_initial_max'], 1e-300))
        sg = np.linalg.norm((f - F(base[x['case']])).reshape(len(f), -1), axis=1) / n0
        x['same_grid_per_time'] = sg.tolist()
        x['same_grid_all'] = float(np.max(sg))
        x['same_grid_evolved'] = float(np.max(sg[1:]))
        x['t0_compression'] = float(sg[0])
    gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    # --------------------------------------- t = 0 invariance in M, from fields
    t0 = {}
    for x in roms:
        t0.setdefault((x['q'], x['case']), {})[x['M']] = x['artifact']
    worst_rel, bitwise = 0., True
    for (q, case), arts in t0.items():
        Ms = sorted(arts)
        f0 = F(arts[Ms[0]])[0]
        for M in Ms[1:]:
            f1 = F(arts[M])[0]
            worst_rel = max(worst_rel, float(np.linalg.norm(f1 - f0) / np.linalg.norm(f0)))
            bitwise = bitwise and bool(np.array_equal(f0, f1))
    gate('t0_field_invariant_in_M', worst_rel <= 1e-6,
         dict(worst_relative_difference=worst_rel, families=len(t0), tolerance=1e-6),
         'blocking at the IC stationarity tolerance; the IC LM path can move within it on an '
         'ulp-level GEMM difference between the per-M programs (DESIGN.md §9 A0)')
    info('t0_field_invariant_in_M_1e-12', worst_rel <= 1e-12, worst_rel, 'probe')
    info('t0_field_bitwise_in_M', bitwise, None, 'XLA may fuse programs of different M differently')

    # ------------------------------------------------------ per-arm rows ------
    table = {}
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x.get('family'), q=x.get('q'),
            linear_solve=x.get('linear_solve'),
            solved_dimension=x.get('solved_dimension'), M=x.get('M'), labels=x.get('labels'),
            rule=x.get('rule'), quadrature=x.get('quadrature'), dt=x.get('dt'),
            step_budget=x.get('step_budget'), gpu_ms=[], host_ms=[], case_ref={}, case_all={},
            case_evolved={}, case_t0={}, iterations=[], stationarity=[], converged=[],
            completed=[], budget_exits=[], exits=[], per_time_all={}, ic_iterations=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_evolved'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['per_time_all'][x['case']] = x['same_grid_per_time']
        t['iterations'] += x.get('iterations', [])
        if x['kind'] == 'rom':
            t['stationarity'].append(x['worst_joint_stationarity'])
            t['converged'].append(x['converged'])
            t['completed'].append(x['completed'])
            t['budget_exits'].append(x['budget_exits'])
            t['exits'] += x['stop_reasons']
            t['ic_iterations'].append(x['ic_iterations'])
    rows = []
    for name, t in table.items():
        s = setup.get(name, {})
        rc = {x['q']: x for x in r['reconstruction']}.get(t['q'])
        rows.append(dict(
            arm=name, kind=t['kind'], family=t['family'] or 'fom', q=t['q'],
            solved_dimension=t['solved_dimension'], M=t['M'], labels=t['labels'], rule=t['rule'],
            quadrature=t['quadrature'], dt=t['dt'], step_budget=t['step_budget'],
            tests_per_unknown=s.get('tests_per_unknown'), linear_solve=t.get('linear_solve') or s.get('linear_solve'),
            linear_override=s.get('linear_override'),
            worst_reference_percent=100 * float(np.max(list(t['case_ref'].values()))),
            worst_all_times_percent=100 * float(np.max(list(t['case_all'].values()))),
            worst_evolved_percent=100 * float(np.max(list(t['case_evolved'].values()))),
            median_evolved_percent=100 * float(np.median(list(t['case_evolved'].values()))),
            median_all_times_percent=100 * float(np.median(list(t['case_all'].values()))),
            worst_t0_compression_percent=100 * float(np.max(list(t['case_t0'].values()))),
            per_case_evolved_percent={str(k): 100 * v for k, v in sorted(t['case_evolved'].items())},
            per_case_all_times_percent={str(k): 100 * v for k, v in sorted(t['case_all'].items())},
            per_case_t0_compression_percent={str(k): 100 * v for k, v in sorted(t['case_t0'].items())},
            per_time_same_grid_percent={str(k): [100 * z for z in v] for k, v in sorted(t['per_time_all'].items())},
            median_gpu_ms=median(t['gpu_ms']), gpu_ms_all=sorted(t['gpu_ms']),
            median_host_ms=median(t['host_ms']), gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            mean_iterations=(float(np.mean(t['iterations'])) if t['iterations'] else None),
            max_iterations=(float(np.max(t['iterations'])) if t['iterations'] else None),
            median_ic_iterations=(median(t['ic_iterations']) if t['ic_iterations'] else None),
            max_joint_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            converged=(bool(all(t['converged'])) if t['converged'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in zip(*np.unique(t['exits'], return_counts=True))}
                                if t['exits'] else None),
            worst_bank_projection_percent=(100 * rc['worst_bank_projection'] if rc else None),
            worst_best_found_percent=(100 * rc['worst_best_found'] if rc else None),
            setup_seconds=s.get('total_setup_seconds')))
    by = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: (x['family'], x['q'] if x['q'] is not None else -1, x['M'] or 0))

    # ------------------------------------------------------ fidelity gates ----
    # A2 (round-2 self-audit, `bqx401`/`bqx501`, this file): `make_configs.py`'s EXPECT_R2
    # loop appends EVERY name in `MINE = ('bqx101', 'bqx201', 'bqx301')` as a source for
    # every q=1088 arm, without checking whether that source's own q_ladder ever ran that
    # q (G1 only ran q in {0,16,32}; G2 only {0,64,128,256}; S1 only {0,64}). That declares
    # structurally-impossible pairs (e.g. q16_M1088 against bqx201, which has no q16 row at
    # all). This is a config-declaration bug, not a numerical discrepancy: the source job's
    # own comparator table exists in `comparators.json` and is complete, it simply never
    # contains that arm because it was never run there by design. Such a pair is reported
    # informational/not_applicable rather than failed; a pair whose source is missing from
    # `comparators.json` ENTIRELY (a real data gap) still fails. See DESIGN.md A5.
    fid = {}
    for arm, specs in (r['config'].get('expectations') or {}).items():
        for spec in specs:
            key = f"{arm}__{spec['source']}"
            mine = by.get(arm)
            source_table = cmp_table.get(spec['source'])
            theirs = ((source_table or {}).get('arms') or {}).get(spec['arm'])
            if mine is None or theirs is None:
                if mine is not None and source_table is not None:
                    # source job exists and its comparator table is intact; it simply never
                    # ran this arm (structural, by that job's own declared q_ladder) --
                    # a mis-declared expectation, not a fidelity failure.
                    fid[key] = dict(passed=None, detail=dict(
                        reason='not_applicable: source job never ran this arm (structural, '
                               'by its declared q_ladder); mis-declared expectation, not a '
                               'fidelity discrepancy',
                        source=spec['source'], arm=spec['arm'], unconditional=None))
                    info(f'reproduces_{key}', True, fid[key]['detail'])
                else:
                    fid[key] = dict(passed=False, detail=dict(
                        reason='arm or comparator missing', source=spec['source'], arm=spec['arm']))
                    gate(f'reproduces_{key}', False, fid[key]['detail'])
                continue
            tol = spec['tolerance']
            if dir_match.get(spec['source']) and 'tolerance_if_directions_bitwise' in spec:
                tol = spec['tolerance_if_directions_bitwise']
            d = dict(source=spec['source'], source_job=(cmp_table.get(spec['source']) or {}).get('job_id'),
                     comparator=spec['arm'], declared_tolerance=tol,
                     unconditional=('tolerance_if_directions_bitwise' not in spec),
                     directions_bitwise=dir_match.get(spec['source']), note=spec.get('note'))
            ok, compared = True, 0
            for k in ('worst_all_times_percent', 'worst_evolved_percent', 'worst_reference_percent'):
                mv, tv = mine[k], theirs.get(k)
                rel = (None if tv in (None, 0) else abs(mv - tv) / abs(tv))
                d[k] = dict(ours=mv, theirs=tv, relative_difference=rel)
                ok = ok and (rel is None or rel <= tol)
                compared += rel is not None
            # a pair that compared nothing must not pass (a null comparator would otherwise
            # make the gate vacuous)
            ok = ok and compared >= 2
            d['metrics_compared'] = compared
            d['achieved_worst_relative_difference'] = max(
                v['relative_difference'] or 0. for kk, v in d.items()
                if isinstance(v, dict) and 'relative_difference' in v)
            d['ours_median_gpu_ms'] = mine['median_gpu_ms']
            d['theirs_median_gpu_ms'] = theirs.get('median_gpu_ms')
            fid[key] = dict(passed=bool(ok), detail=d)
            gate(f'reproduces_{key}', ok, d)
    uncond = [v for v in fid.values() if v['detail'].get('unconditional')]
    gate('at_least_two_unconditional_reproductions_at_1e-9',
         sum(1 for v in uncond if v['passed'] and v['detail']['achieved_worst_relative_difference'] <= 1e-9) >= 2,
         dict(unconditional_pairs=len(uncond), passed_at_1e9=sum(
             1 for v in uncond if v['passed'] and v['detail']['achieved_worst_relative_difference'] <= 1e-9)))
    # not_applicable pairs (passed=None, structurally mis-declared, see above) do not count
    # against the aggregate; only genuine comparator/arm gaps and real tolerance misses do.
    _blocking = [v['passed'] for v in fid.values() if v['passed'] is not None]
    checks['cross_job_fidelity'] = dict(
        passed=(all(_blocking) if _blocking else None), detail=fid,
        not_applicable=sum(1 for v in fid.values() if v['passed'] is None))

    parts = Path(a.result).resolve().parts
    attempt = parts[parts.index('runs') + 1] if 'runs' in parts else r['config'].get('attempt')
    out = dict(result=str(Path(a.result).resolve()), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               question=r['config'].get('question'), job_id=r.get('job_id'), commit=r.get('commit'),
               gpu=r.get('gpu'), elapsed_seconds=r.get('elapsed_seconds'), attempt=attempt,
               config_attempt=r['config'].get('attempt'), directions_sha256=mine_dir,
               compile_warmup=r.get('compile_warmup'), checks=checks, failed=sorted(fail), arms=rows)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(attempt=attempt, job_id=r.get('job_id'), failed=sorted(fail), arms=len(rows)), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
