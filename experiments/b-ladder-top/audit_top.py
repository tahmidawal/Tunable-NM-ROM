"""Independent NumPy audit of the b-ladder-top jobs. No JAX, no GPU.

Recomputes every reported error from the retained output fields, measures each arm
against the same-job converged full-order solve on THREE error metrics (t = 0
compression, worst over all output times, worst over evolved times only), checks the
cross-job fidelity gates against the cheap-corrections job (`cclad01`), and derives the
non-dominated sets on each metric. Works on both the Q1 convergence sweep and the Q2
envelope; `--mode` only selects which extra gates apply.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CCLAD = ROOT / 'experiments/cheap-corrections/artifacts/cclad01/result.json'
CCLAD_AUDIT = ROOT / 'experiments/cheap-corrections/checks/cclad01-audit.json'


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def nondominated(rows, cost, err):
    pts = [r for r in rows if r.get(cost) is not None and r.get(err) is not None]
    out = []
    for r in pts:
        if not any((o[cost] <= r[cost] and o[err] <= r[err] and
                    (o[cost] < r[cost] or o[err] < r[err])) for o in pts):
            out.append(r['arm'])
    return sorted(set(out))


def cclad_rows():
    """Per-arm aggregates of the cheap-corrections job, recomputed here from its JSON."""
    if not CCLAD.exists():
        return {}
    r = json.loads(CCLAD.read_text())
    table = {}
    for x in r['invocations']:
        t = table.setdefault(x['name'], dict(case_ref={}, gpu_ms=[], t0={}, evolved={}))
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['t0'][x['case']] = x['error']['fixed_initial_per_time'][0]
        t['evolved'][x['case']] = max(x['error']['fixed_initial_per_time'][1:])
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
    return {k: dict(worst_reference=float(max(v['case_ref'].values())),
                    worst_t0_vs_reference=float(max(v['t0'].values())),
                    worst_evolved_vs_reference=float(max(v['evolved'].values())),
                    median_gpu_ms=median(v['gpu_ms'])) for k, v in table.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', required=True)
    p.add_argument('--mode', choices=('q1', 'q2'), required=True)
    p.add_argument('--fno-timing', default=None)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    fields_dir = Path(a.fields)
    checks, fail = {}, []

    def gate(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note)
        if not ok:
            fail.append(name)

    def informational(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note, blocking=False)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    informational('reference_fields_bitwise_match_cclad01',
                  all(bool(x.get('bitwise_matches_cclad01')) for x in r['reference']),
                  [x.get('bitwise_matches_cclad01') for x in r['reference']],
                  'a cross-job bitwise probe; informative, not required')
    dg = dict(r['gates'].get('directions_hash', {}))
    # The config's expected hash is `qlad01`'s, produced on an A100-PCIE-40GB. The
    # cheap-corrections job ran on an A100 80GB PCIe and produced a different one; both
    # are recorded, so the probe says WHICH job this one reproduces, if either.
    cclad_dir = None
    if CCLAD.exists():
        cclad_dir = json.loads(CCLAD.read_text())['directions']['directions_sha256']
    dg['cclad01_sha256'] = cclad_dir
    dg['matches_cclad01'] = bool(cclad_dir is not None and dg.get('got') == cclad_dir)
    dir_bitwise = bool(dg.get('passed')) or dg['matches_cclad01']
    informational('directions_hash_matches_cclad01', dg['matches_cclad01'], dg,
                  'bitwise across jobs; qlad01 (A100-PCIE-40GB) and cclad01 (A100 80GB PCIe) '
                  'already disagree with each other, so this is a probe, not a requirement. The '
                  'substantive check is the q = 0 / q = 128 reproduction below')
    gate('directions_rank_covers_ladder',
         r['directions']['available_rank'] >= max(
             [x.get('q') or 0 for x in r['declared_subjects'] if x.get('method') == 'rom'] or [0]),
         r['directions']['available_rank'])

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
         all(('stop_reasons' in x and 'budget_exits' in x and 'worst_joint_stationarity' in x)
             for x in roms))
    eq = [s for s in r['arm_setup'] if s.get('quadrature') == 'eq']
    gate('every_eq_rule_reports_validity',
         all(('eq_rule_valid' in s or s.get('eq_fit', {}).get('fitter') == 'retained_nnls_capped')
             for s in eq),
         [s['arm'] for s in eq if 'eq_rule_valid' not in s][:5])
    if a.mode == 'q1':
        g0 = r['gates'].get('q0_arms_bitwise', {})
        if g0.get('arms'):
            gate('q0_arms_bitwise', bool(g0.get('passed')), g0)
        else:
            # A sweep with no q = 0 dense arms has nothing to compare; the gate is not
            # applicable rather than failed.
            checks['q0_arms_bitwise'] = dict(
                passed=None, detail=g0, note='not applicable: this sweep has no q = 0 dense arms')

    # ------------------------------------------- errors recomputed from fields
    cache = {}

    def fields_of(name):
        if name not in cache:
            cache[name] = np.load(fields_dir / name)['fields']
        return cache[name]

    bad = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
    gate('artifacts_present', not bad, bad[:5])
    refs = {e['case']: fields_of(e['artifact']) for e in r['reference']}
    base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
    gate('same_grid_baseline_present', len(base) == len(refs), sorted(base))
    worst = 0.
    for x in inv:
        truth = refs[x['case']]
        f = fields_of(x['artifact'])
        n0 = np.linalg.norm(truth[0])
        err = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
        worst = max(worst, abs(float(np.max(err)) - x['error']['fixed_initial_max'])
                    / max(x['error']['fixed_initial_max'], 1e-300))
        g = fields_of(base[x['case']])
        sg = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
        x['same_grid_per_time'] = sg.tolist()
        x['same_grid_all'] = float(np.max(sg))
        x['same_grid_evolved'] = float(np.max(sg[1:]))
        x['t0_compression'] = float(sg[0])
    gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    # ------------------------------------------------------- the FNO subject --
    fno = None
    if a.fno_timing and Path(a.fno_timing).exists():
        fno = json.loads(Path(a.fno_timing).read_text())
        gate('fno_returns_supplied_field_at_t0',
             all(c['t0_returned_exactly'] for c in fno['cases']),
             [c['case_id'] for c in fno['cases'] if not c['t0_returned_exactly']])
        for c in fno['cases']:
            case = int(c['case_index'])
            f = fields_of(c['artifact'])
            truth = refs[case]
            n0 = np.linalg.norm(truth[0])
            g = fields_of(base[case])
            sg = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
            ref = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
            for rep, (dsec, hsec) in enumerate(zip(c['device_seconds'], c['host_seconds'])):
                inv.append(dict(intervals=r['intervals'], case=case, rep=rep, kind='operator',
                                family='fno', name=fno['model'], gpu_seconds=dsec,
                                host_seconds=dsec + hsec, field_sha256=c['field_sha256'],
                                artifact=c['artifact'], finite=True,
                                error=dict(fixed_initial_per_time=ref.tolist(),
                                           fixed_initial_max=float(np.max(ref))),
                                same_grid_per_time=sg.tolist(), same_grid_all=float(np.max(sg)),
                                same_grid_evolved=float(np.max(sg[1:])), t0_compression=float(sg[0])))
        cohort = r['gates'].get('fno_cohort_disjoint_from_training')
        if cohort is not None:
            gate('fno_cohort_disjoint_from_training', bool(cohort['passed']), cohort)

    # ------------------------------------------------------ per-arm aggregates
    default_budget = (r['config'].get('strict') or {}).get('step_budget')
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    recon_rom = {x['q']: x for x in r['reconstruction'] if x.get('family', 'rom') == 'rom'}
    recon_pod = {x.get('k'): x for x in r['reconstruction'] if x.get('family') == 'pod'}
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x.get('family'), q=x.get('q'), k=x.get('k'),
            solved_dimension=x.get('solved_dimension'), M=x.get('M'), m=x.get('m'),
            rule=x.get('rule'), fix=x.get('fix'), quadrature=x.get('quadrature'),
            gtol=x.get('gtol'), dt=x.get('dt'), cascade_source=x.get('cascade_source'),
            step_budget=x.get('step_budget', default_budget),
            trust_scale=x.get('trust_scale', 1.0),
            gpu_ms=[], host_ms=[], case_ref={}, case_all={}, case_evolved={}, case_t0={},
            iterations=[], stationarity=[], stationary=[], completed=[], converged=[],
            stationary_1e6=[], budget_exits=[], exit_reasons=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_evolved'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['iterations'] += x.get('iterations', [])
        if x['kind'] == 'rom':
            t['stationarity'].append(x['worst_joint_stationarity'])
            t['stationary'].append(x.get('stationary', x.get('stationary_at_arm_tolerance')))
            t['stationary_1e6'].append(x.get('stationary_1e6', x.get('stationary')))
            t['completed'].append(x['completed'])
            t['converged'].append(x['converged'])
            t['budget_exits'].append(x['budget_exits'])
            t['exit_reasons'] += x['stop_reasons']
            t.setdefault('ic_rel', []).append(x.get('ic_relative_residual'))
            t.setdefault('ic_reasons', []).append(x.get('ic_reason'))
            t.setdefault('ic_gj', []).append(x.get('ic_joint_stationarity'))
            t.setdefault('step_gj', []).append(max(x['step_joint_stationarity'])
                                               if x.get('step_joint_stationarity') else None)
            bad = [i for i, s in enumerate(x['stop_reasons']) if s == 0]
            if bad:
                t.setdefault('failing', {}).setdefault(x['case'], sorted(bad))

    rows = []
    for name, t in table.items():
        st = setup.get(name, {})
        eqf = (st.get('eq_fit') or {})
        rc = recon_rom.get(t['q']) if t.get('family') != 'pod' else recon_pod.get(t['k'])
        row = dict(
            arm=name, kind=t['kind'], family=t['family'] or ('fom' if t['kind'] == 'fom' else 'rom'),
            q=t['q'], k=t['k'], solved_dimension=t['solved_dimension'], M=t['M'], m=t['m'],
            rule=t['rule'], fix=t['fix'], quadrature=t['quadrature'], gtol=t['gtol'], dt=t['dt'],
            cascade_source=t['cascade_source'], step_budget=t['step_budget'],
            trust_scale=t['trust_scale'],
            worst_reference_percent=float(np.max(list(t['case_ref'].values())) * 100),
            median_reference_percent=float(np.median(list(t['case_ref'].values())) * 100),
            worst_all_times_percent=float(np.max(list(t['case_all'].values())) * 100),
            worst_evolved_percent=float(np.max(list(t['case_evolved'].values())) * 100),
            median_evolved_percent=float(np.median(list(t['case_evolved'].values())) * 100),
            worst_t0_compression_percent=float(np.max(list(t['case_t0'].values())) * 100),
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            max_iterations=(float(np.max(t['iterations'])) if t['iterations'] else None),
            max_joint_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
            stationary_1e6=(bool(all(t['stationary_1e6'])) if t['stationary_1e6'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            converged=(bool(all(t['converged'])) if t['converged'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in
                                 zip(*np.unique(t['exit_reasons'], return_counts=True))}
                                if t['exit_reasons'] else None),
            quadrature_fit_seconds=eqf.get('seconds'), quadrature_fit_relative=st.get('eq_relative_fit'),
            quadrature_support=eqf.get('support'), quadrature_truncated=eqf.get('truncated'),
            eq_rule_valid=st.get('eq_rule_valid'),
            worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100)
                                           if rc and 'worst_bank_projection' in rc else None),
            worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
            setup_seconds=st.get('total_setup_seconds'))
        # Where the non-convergence lives: the supplied-field fit, or the time stepping.
        if t.get('ic_rel'):
            icr = [v for v in t['ic_rel'] if v is not None]
            icg = [v for v in t['ic_gj'] if v is not None]
            sgj = [v for v in t['step_gj'] if v is not None]
            row.update(
                ic_relative_residual_max=(float(np.max(icr)) if icr else None),
                ic_relative_residual_min=(float(np.min(icr)) if icr else None),
                ic_joint_stationarity_max=(float(np.max(icg)) if icg else None),
                step_joint_stationarity_max=(float(np.max(sgj)) if sgj else None),
                ic_reason_counts={str(k2): int(v2) for k2, v2 in
                                  zip(*np.unique([v for v in t['ic_reasons'] if v is not None],
                                                 return_counts=True))},
                failing_cases={str(c): v for c, v in sorted((t.get('failing') or {}).items())},
                failing_case_count=len(t.get('failing') or {}),
                failure_located_in=(
                    None if (row['converged'] is None or row['converged']) else
                    ('initial fit only' if (row['total_budget_exits'] == 0
                                            and (icg and np.max(icg) > 1e-6)
                                            and (not sgj or np.max(sgj) <= 1e-6))
                     else ('time stepping only' if (icg and np.max(icg) <= 1e-6) else 'both'))))
        rows.append(row)
    by_arm = {x['arm']: x for x in rows}
    for x in rows:
        src = x.get('cascade_source')
        x['cascade_total_gpu_ms'] = (x['median_gpu_ms'] + by_arm[src]['median_gpu_ms']
                                     if src and src in by_arm else None)
        x['effective_gpu_ms'] = x['cascade_total_gpu_ms'] or x['median_gpu_ms']
    rows.sort(key=lambda x: (x['family'] or '', x['q'] if x['q'] is not None else -1,
                             x['k'] if x['k'] is not None else -1, x['arm']))

    # ------------------------------------------------------- fidelity gates ---
    old = cclad_rows()
    cclad_same_grid = {}
    if CCLAD_AUDIT.exists():
        cclad_same_grid = {x['arm']: x.get('worst_same_grid_percent') for x in
                           json.loads(CCLAD_AUDIT.read_text())['checks']['arm_table']}
    exp = r['config'].get('cclad01_expectations', {})
    fid = {}
    for arm, spec in exp.items():
        mine = by_arm.get(arm)
        theirs = old.get(spec['reproduces'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail='arm or comparator missing')
            continue
        rel_ref = abs(mine['worst_reference_percent'] / 100 - theirs['worst_reference']) / max(
            theirs['worst_reference'], 1e-300)
        sg = cclad_same_grid.get(spec['reproduces'])
        rel_sg = (abs(mine['worst_all_times_percent'] - sg) / max(sg, 1e-300)
                  if sg is not None else None)
        tol = spec['tolerance']
        if dir_bitwise and 'tolerance_if_directions_bitwise' in spec:
            tol = spec['tolerance_if_directions_bitwise']
        d = dict(comparator=spec['reproduces'], declared_tolerance=tol,
                 relative_worst_reference_difference=rel_ref,
                 relative_worst_same_grid_difference=rel_sg,
                 ours_worst_reference_percent=mine['worst_reference_percent'],
                 theirs_worst_reference_percent=theirs['worst_reference'] * 100,
                 ours_worst_same_grid_percent=mine['worst_all_times_percent'],
                 theirs_worst_same_grid_percent=sg,
                 ours_median_gpu_ms=mine['median_gpu_ms'], theirs_median_gpu_ms=theirs['median_gpu_ms'])
        ok = rel_ref <= tol and (rel_sg is None or rel_sg <= tol)
        fid[arm] = dict(passed=bool(ok), detail=d)
        gate(f'reproduces_{arm}', ok, d)
    checks['cclad01_fidelity'] = dict(passed=all(v['passed'] for v in fid.values()) if fid else None,
                                      detail=fid)

    # ------------------------------------------------------- frontier sets ----
    conv = [x for x in rows if x['family'] == 'rom' and x['converged']]
    frontier = dict(
        all_subjects_all_times=nondominated(rows, 'effective_gpu_ms', 'worst_all_times_percent'),
        all_subjects_evolved=nondominated(rows, 'effective_gpu_ms', 'worst_evolved_percent'),
        converged_rom_all_times=nondominated(conv, 'effective_gpu_ms', 'worst_all_times_percent'),
        converged_rom_evolved=nondominated(conv, 'effective_gpu_ms', 'worst_evolved_percent'))

    def spans(names, err):
        sel = [by_arm[n] for n in names]
        if len(sel) < 2:
            return dict(points=len(sel), cost_span=None, error_span=None)
        c = [x['effective_gpu_ms'] for x in sel]
        v = [x[err] for x in sel]
        return dict(points=len(sel), cost_span=max(c) / max(min(c), 1e-300),
                    error_span=max(v) / max(min(v), 1e-300))

    knob = dict(
        evolved=spans(frontier['converged_rom_evolved'], 'worst_evolved_percent'),
        all_times=spans(frontier['converged_rom_all_times'], 'worst_all_times_percent'))
    def mono(vals):
        return bool(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])))

    for metric, key in (('evolved', 'worst_evolved_percent'), ('all_times', 'worst_all_times_percent')):
        ladder = sorted([x for x in rows if x['family'] == 'rom' and x['quadrature'] == 'eq'
                         and x['gtol'] == 1e-6 and x['q'] is not None], key=lambda x: x['q'])
        # The pre-registered criterion says "within the retained configuration", which is the
        # FIXED test count. Rungs above q = 239 cannot use M = 256 and carry their own M, so
        # monotonicity is reported both over the fixed-M rungs alone and over every rung.
        counts = [x['M'] for x in ladder]
        fixed = max(set(counts), key=counts.count) if counts else None
        same = [x for x in ladder if x['M'] == fixed]
        knob[metric]['test_count_of_retained_configuration'] = fixed
        knob[metric]['monotone_fixed_test_count'] = mono([x[key] for x in same])
        knob[metric]['monotone_all_rungs'] = mono([x[key] for x in ladder])
        knob[metric]['monotone'] = knob[metric]['monotone_fixed_test_count']
        knob[metric]['ladder'] = [(x['q'], x['M'], x[key], x['effective_gpu_ms']) for x in ladder]
        knob[metric]['passes'] = bool(knob[metric]['monotone']
                                      and (knob[metric].get('points') or 0) >= 3
                                      and (knob[metric].get('cost_span') or 0) >= 2
                                      and (knob[metric].get('error_span') or 0) >= 2)

    out = dict(result=str(Path(a.result).resolve()), mode=a.mode, job_id=r.get('job_id'),
               commit=r.get('commit'), gpu=r.get('gpu'), elapsed_seconds=r.get('elapsed_seconds'),
               checks=checks, failed=sorted(fail), arms=rows, frontier=frontier,
               knob_criterion=knob, cclad01_comparators=old,
               conditioning=r.get('conditioning', []), cascade_sources=r.get('cascade_sources', {}),
               fno=(dict(model=fno['model'], environment=fno['environment'],
                         checkpoint_sha256=fno['checkpoint_sha256'],
                         device_query_pooled=fno['device_query_pooled'],
                         real_parameter_count=fno['real_parameter_count']) if fno else None))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows),
                          frontier={k: len(v) for k, v in frontier.items()}), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
