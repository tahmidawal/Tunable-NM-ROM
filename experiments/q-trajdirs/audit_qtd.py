"""Independent NumPy audit of the q-trajdirs job. No JAX, no GPU, no driver import.

Recomputes every reported error from the retained output fields, measures every subject
against the same-job converged full-order solve on three metrics (t = 0 compression,
worst over all output times, worst over evolved times only), checks the cross-job
fidelity gates against `cclad01` and the `b-ladder-top` jobs, walks each declared ladder
for monotonicity on both metrics, derives the non-dominated sets, and evaluates the
pre-registered pass of `DESIGN.md` section 6.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPARATORS = {
    'cclad01': ROOT / 'experiments/cheap-corrections/artifacts/cclad01/result.json',
    'btq101': ROOT / 'experiments/b-ladder-top/artifacts/btq101/result.json',
    'btq201': ROOT / 'experiments/b-ladder-top/artifacts/btq201/result.json',
}
COMPARATOR_AUDITS = {
    'cclad01': ROOT / 'experiments/cheap-corrections/checks/cclad01-audit.json',
    'btq101': ROOT / 'experiments/b-ladder-top/checks/btq101-audit.json',
    'btq201': ROOT / 'experiments/b-ladder-top/checks/btq201-audit.json',
}


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


def mono(vals):
    return bool(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])))


def comparator_rows(path):
    """Per-arm aggregates of a comparator job, recomputed here from its own JSON."""
    if not path.exists():
        return {}
    r = json.loads(path.read_text())
    table = {}
    for x in r['invocations']:
        t = table.setdefault(x['name'], dict(ref={}, t0={}, evolved={}, gpu_ms=[]))
        t['ref'][x['case']] = x['error']['fixed_initial_max']
        t['t0'][x['case']] = x['error']['fixed_initial_per_time'][0]
        t['evolved'][x['case']] = max(x['error']['fixed_initial_per_time'][1:])
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
    return {k: dict(worst_reference=float(max(v['ref'].values())),
                    worst_t0_vs_reference=float(max(v['t0'].values())),
                    worst_evolved_vs_reference=float(max(v['evolved'].values())),
                    median_gpu_ms=median(v['gpu_ms'])) for k, v in table.items()}


def comparator_same_grid(path):
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    rows = d.get('arms') or d.get('checks', {}).get('arm_table') or []
    out = {}
    for x in rows:
        out[x['arm']] = dict(
            same_grid_all=x.get('worst_all_times_percent', x.get('worst_same_grid_percent')),
            same_grid_evolved=x.get('worst_evolved_percent'),
            t0=x.get('worst_t0_compression_percent'),
            median_gpu_ms=x.get('median_gpu_ms'), converged=x.get('converged'))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', required=True)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    fields_dir = Path(a.fields)
    checks, fail = {}, []

    def gate(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note)
        if not ok:
            fail.append(name)

    def info(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note, blocking=False)

    cfg = r['config']
    K = r['K']
    qlad = list(cfg['q_ladder'])

    # ------------------------------------------------------- environment gates
    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    info('reference_fields_bitwise_match_comparator',
         all(bool(x.get('bitwise_matches_comparator')) for x in r['reference']),
         [x.get('bitwise_matches_comparator') for x in r['reference']],
         'a cross-job bitwise probe; informative, not required')

    # ------------------------------------------------------------ cohort gates
    cg = r['gates']['evaluation_cohort_bitwise_abl01']
    abl = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'
    recomputed = None
    if abl.exists():
        pc = np.array(json.loads(abl.read_text())['physical_cases'])
        recomputed = hashlib.sha256(np.ascontiguousarray(pc).tobytes()).hexdigest()
        cg = dict(cg, abl01_recomputed_sha256=recomputed,
                  bitwise_equal=bool(np.array_equal(np.array(r['physical_cases']), pc)))
    gate('evaluation_cohort_bitwise_abl01',
         bool(cg.get('bitwise_equal', cg.get('passed'))), cg)
    dc = r['gates']['direction_cohorts_disjoint_from_evaluation']
    gate('direction_cohorts_disjoint_from_evaluation', bool(dc['passed']), dc['detail'])

    # -------------------------------------------------------- direction gates
    ds = r['direction_sets']
    gate('directions_hashed_and_saved',
         all((fields_dir / v['artifact']).exists() for v in ds.values()),
         {k: v['artifact'] for k, v in ds.items()})
    gate('directions_rank_covers_ladder',
         all(v['available_rank'] >= max(qlad) for v in ds.values()),
         {k: v['available_rank'] for k, v in ds.items()})
    prefix_ok, prefix_detail = True, {}
    for name, v in ds.items():
        C = np.load(fields_dir / v['artifact'])['C']
        got = {str(q): hashlib.sha256(
            np.ascontiguousarray(C[:, :int(q)]).tobytes()).hexdigest() for q in qlad}
        same = got == {str(q): v['prefix_sha256'][str(q)] for q in qlad}
        prefix_detail[name] = dict(matches=bool(same),
                                   whole_matrix_sha256=hashlib.sha256(
                                       np.ascontiguousarray(C).tobytes()).hexdigest(),
                                   reported=v['directions_sha256'])
        prefix_ok = prefix_ok and same
    gate('direction_prefixes_recomputed_from_artifact', prefix_ok, prefix_detail,
         'the hash of the first q columns of each SAVED matrix equals the hash the job '
         'recorded for the slice it handed rung q')
    gate('nested_prefix_consistent', bool(r['gates']['nested_prefix_consistent']['passed']),
         r['gates']['nested_prefix_consistent'])
    info('old_directions_hash_matches_comparator',
         bool(ds['old'].get('bitwise_matches_comparator')),
         dict(expected=ds['old'].get('expected_sha256'), got=ds['old']['directions_sha256']),
         'bitwise across jobs; qlad01, cclad01 and the b-ladder-top jobs already disagree '
         'with each other, so this is a probe, not a requirement')
    dir_bitwise = bool(ds['old'].get('bitwise_matches_comparator'))

    # --------------------------------------------------------- bookkeeping ---
    inv = r['invocations']
    reps = cfg['reps']
    counts = {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps},
         sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired',
         all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms),
         [(x['name'], x['M'], x['solved_dimension']) for x in roms
          if x['M'] <= x['solved_dimension']][:5])
    gate('every_rom_carries_exit_and_stationarity',
         all(('stop_reasons' in x and 'budget_exits' in x and 'worst_joint_stationarity' in x)
             for x in roms))
    gate('step_budget_is_600',
         all(x.get('step_budget') == 600 for x in roms), cfg['strict'])
    eq = [s for s in r['arm_setup'] if s.get('quadrature') == 'eq']
    gate('every_eq_rule_reports_validity',
         all(('eq_rule_valid' in s or s.get('eq_fit', {}).get('fitter') == 'retained_nnls_capped')
             for s in eq),
         [s['arm'] for s in eq if 'eq_rule_valid' not in s][:5])
    gate('no_eq_rule_truncated',
         all(not (s.get('eq_fit') or {}).get('truncated', False) for s in eq),
         [s['arm'] for s in eq if (s.get('eq_fit') or {}).get('truncated')])

    # ------------------------------------------- errors recomputed from fields
    cache = {}

    def fields_of(name):
        if name not in cache:
            cache[name] = np.load(fields_dir / name)['fields']
        return cache[name]

    bad = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
    gate('artifacts_present', not bad, bad[:5])
    refs = {x['case']: fields_of(x['artifact']) for x in r['reference']}
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

    # ------------------------------------------------------ per-arm aggregates
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    recon = {(x['dirset'], x['q']): x for x in r['reconstruction']}
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x.get('family'), dirset=x.get('dirset'),
            q=x.get('q'), solved_dimension=x.get('solved_dimension'), M=x.get('M'),
            m=x.get('m'), rule=x.get('rule'), quadrature=x.get('quadrature'),
            fitter=x.get('fitter'), dt=x.get('dt'), gpu_ms=[], host_ms=[],
            case_ref={}, case_all={}, case_evolved={}, case_t0={}, case_per_time={},
            iterations=[], stationarity=[], stationary=[], completed=[], converged=[],
            budget_exits=[], exit_reasons=[], ic_reasons=[], failing={}))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_evolved'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['case_per_time'][x['case']] = x['same_grid_per_time']
        t['iterations'] += x.get('iterations', [])
        if x['kind'] == 'rom':
            t['stationarity'].append(x['worst_joint_stationarity'])
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['converged'].append(x['converged'])
            t['budget_exits'].append(x['budget_exits'])
            t['exit_reasons'] += x['stop_reasons']
            t['ic_reasons'].append(x['ic_reason'])
            bd = [i for i, s in enumerate(x['stop_reasons']) if s == 0]
            if bd:
                t['failing'].setdefault(x['case'], sorted(bd))

    rows = []
    for name, t in table.items():
        st = setup.get(name, {})
        eqf = st.get('eq_fit') or {}
        rc = recon.get((t['dirset'], t['q']))
        per_time = np.array([t['case_per_time'][c] for c in sorted(t['case_per_time'])])
        row = dict(
            arm=name, kind=t['kind'], family=t['family'] or 'fom', dirset=t['dirset'],
            q=t['q'], solved_dimension=t['solved_dimension'], M=t['M'], m=t['m'],
            rule=t['rule'], quadrature=t['quadrature'], fitter=t['fitter'], dt=t['dt'],
            worst_reference_percent=float(np.max(list(t['case_ref'].values())) * 100),
            worst_all_times_percent=float(np.max(list(t['case_all'].values())) * 100),
            worst_evolved_percent=float(np.max(list(t['case_evolved'].values())) * 100),
            median_evolved_percent=float(np.median(list(t['case_evolved'].values())) * 100),
            worst_t0_compression_percent=float(np.max(list(t['case_t0'].values())) * 100),
            worst_percent_per_time=(np.max(per_time, axis=0) * 100).tolist(),
            per_case_evolved_percent={str(c): v * 100 for c, v in sorted(t['case_evolved'].items())},
            per_case_all_times_percent={str(c): v * 100 for c, v in sorted(t['case_all'].items())},
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']), gpu_ms_all=sorted(t['gpu_ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            max_iterations=(float(np.max(t['iterations'])) if t['iterations'] else None),
            max_joint_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            converged=(bool(all(t['converged'])) if t['converged'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in
                                 zip(*np.unique(t['exit_reasons'], return_counts=True))}
                                if t['exit_reasons'] else None),
            failing_cases={str(c): v for c, v in sorted(t['failing'].items())},
            quadrature_fit_seconds=eqf.get('seconds'),
            quadrature_fit_relative=st.get('eq_relative_fit'),
            quadrature_support=eqf.get('support'), quadrature_truncated=eqf.get('truncated'),
            eq_rule_valid=st.get('eq_rule_valid'),
            worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100)
                                           if rc else None),
            worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
            setup_seconds=st.get('total_setup_seconds'))
        rows.append(row)
    by_arm = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: (x['family'], x['dirset'] or '', x['q'] if x['q'] is not None else -1,
                             x['M'] or 0, x['arm']))

    # -------------------------------------------------------- fidelity gates -
    comp = {k: comparator_rows(v) for k, v in COMPARATORS.items()}
    comp_sg = {k: comparator_same_grid(v) for k, v in COMPARATOR_AUDITS.items()}
    fid = {}
    for arm, spec in cfg.get('fidelity_expectations', {}).items():
        mine = by_arm.get(arm)
        theirs = comp.get(spec['job'], {}).get(spec['reproduces'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail='arm or comparator missing')
            gate(f'reproduces_{arm}', False, fid[arm])
            continue
        rel_ref = abs(mine['worst_reference_percent'] / 100 - theirs['worst_reference']) / max(
            theirs['worst_reference'], 1e-300)
        sg = (comp_sg.get(spec['job'], {}).get(spec['reproduces']) or {}).get('same_grid_all')
        rel_sg = (abs(mine['worst_all_times_percent'] - sg) / max(sg, 1e-300)
                  if sg is not None else None)
        tol = spec['tolerance']
        second = spec.get('second_tier_tolerance')
        d = dict(job=spec['job'], comparator=spec['reproduces'], declared_tolerance=tol,
                 second_tier_tolerance=second, directions_bitwise=dir_bitwise,
                 relative_worst_reference_difference=rel_ref,
                 relative_worst_same_grid_difference=rel_sg,
                 ours_worst_reference_percent=mine['worst_reference_percent'],
                 theirs_worst_reference_percent=theirs['worst_reference'] * 100,
                 ours_worst_same_grid_percent=mine['worst_all_times_percent'],
                 theirs_worst_same_grid_percent=sg,
                 ours_median_gpu_ms=mine['median_gpu_ms'],
                 theirs_median_gpu_ms=theirs['median_gpu_ms'])
        worst_rel = max([rel_ref] + ([rel_sg] if rel_sg is not None else []))
        d['worst_relative_difference'] = worst_rel
        d['passes_declared'] = bool(worst_rel <= tol)
        d['passes_second_tier'] = (None if second is None else bool(worst_rel <= second))
        fid[arm] = dict(passed=d['passes_declared'], detail=d)
        gate(f'reproduces_{arm}', d['passes_declared'], d)
    checks['cross_job_fidelity'] = dict(
        passed=(all(v['passed'] for v in fid.values()) if fid else None), detail=fid)

    # A probe, not a gate: the same rows against the b-ladder-top envelope job.
    btq = {}
    for arm, other in cfg.get('btq201_expectations', {}).items():
        mine = by_arm.get(arm)
        theirs = comp_sg.get('btq201', {}).get(other)
        if mine is None or theirs is None:
            continue
        btq[arm] = dict(comparator=other, ours_all=mine['worst_all_times_percent'],
                        theirs_all=theirs['same_grid_all'],
                        ours_evolved=mine['worst_evolved_percent'],
                        theirs_evolved=theirs['same_grid_evolved'],
                        ours_gpu_ms=mine['median_gpu_ms'], theirs_gpu_ms=theirs['median_gpu_ms'])
    info('btq201_envelope_probe', True, btq,
         'same-grid comparison against the b-ladder-top envelope job; the EQ rules are '
         'refitted per job so these are not expected to be bitwise')

    # ------------------------------------------------------------- ladders ---
    ladders = {}
    for lid, lad in r['ladders'].items():
        rungs = []
        for rung in lad['rungs']:
            x = by_arm.get(rung['arm'])
            if x is None:
                continue
            rungs.append(dict(q=rung['q'], arm=rung['arm'], M=x['M'], m=x['m'],
                              all_times=x['worst_all_times_percent'],
                              evolved=x['worst_evolved_percent'],
                              t0=x['worst_t0_compression_percent'],
                              reference=x['worst_reference_percent'],
                              best_found=x['worst_best_found_percent'],
                              gpu_ms=x['median_gpu_ms'], converged=x['converged'],
                              budget_exits=x['total_budget_exits'],
                              max_joint_stationarity=x['max_joint_stationarity'],
                              per_time=x['worst_percent_per_time']))
        conv = [u for u in rungs if u['converged']]
        ladders[lid] = dict(
            id=lid, label=lad['label'], dirset=lad['dirset'], rule=lad['rule'],
            quadrature=lad['quadrature'], rungs=rungs,
            all_converged=bool(rungs and all(u['converged'] for u in rungs)),
            monotone_evolved=mono([u['evolved'] for u in rungs]),
            monotone_all_times=mono([u['all_times'] for u in rungs]),
            monotone_evolved_converged_only=mono([u['evolved'] for u in conv]),
            monotone_t0=mono([u['t0'] for u in rungs]),
            regressions_evolved=[dict(from_q=a['q'], to_q=b['q'],
                                      from_percent=a['evolved'], to_percent=b['evolved'])
                                 for a, b in zip(rungs, rungs[1:]) if b['evolved'] > a['evolved']],
            regressions_all_times=[dict(from_q=a['q'], to_q=b['q'],
                                        from_percent=a['all_times'], to_percent=b['all_times'])
                                   for a, b in zip(rungs, rungs[1:])
                                   if b['all_times'] > a['all_times']],
            error_span_evolved=(max(u['evolved'] for u in rungs) / max(
                min(u['evolved'] for u in rungs), 1e-300) if rungs else None),
            cost_span=(max(u['gpu_ms'] for u in rungs) / max(
                min(u['gpu_ms'] for u in rungs), 1e-300) if rungs else None))

    # ------------------------------------------------------- frontier sets ---
    conv_rom = [x for x in rows if x['family'] == 'rom' and x['converged']]
    frontier = dict(
        all_subjects_all_times=nondominated(rows, 'median_gpu_ms', 'worst_all_times_percent'),
        all_subjects_evolved=nondominated(rows, 'median_gpu_ms', 'worst_evolved_percent'),
        converged_rom_all_times=nondominated(conv_rom, 'median_gpu_ms', 'worst_all_times_percent'),
        converged_rom_evolved=nondominated(conv_rom, 'median_gpu_ms', 'worst_evolved_percent'))

    def spans(names, key):
        sel = [by_arm[n] for n in names]
        if len(sel) < 2:
            return dict(points=len(sel), cost_span=None, error_span=None)
        c = [x['median_gpu_ms'] for x in sel]
        v = [x[key] for x in sel]
        return dict(points=len(sel), cost_span=max(c) / max(min(c), 1e-300),
                    error_span=max(v) / max(min(v), 1e-300))

    # Restricted to the pre-registered subject: that config's own primary ladder.
    pc = dict(cfg.get('pass_criteria')
              or {'ladder': 'traj_primary', 'error_span': 2.0, 'cost_span': None})
    primary = ladders.get(pc['ladder'], {})
    prim_names = [u['arm'] for u in primary.get('rungs', []) if u['converged']]
    prim_front = nondominated([by_arm[n] for n in prim_names],
                              'median_gpu_ms', 'worst_evolved_percent')
    verdict = dict(
        ladder=pc['ladder'], criteria=pc,
        criterion_1_evolved_monotone_every_rung_converged=bool(
            primary.get('monotone_evolved') and primary.get('all_converged')),
        criterion_2_all_times_monotone=bool(primary.get('monotone_all_times')),
        criterion_3_error_span=spans(prim_front, 'worst_evolved_percent'),
        converged_nondominated_evolved=prim_front,
        monotone_evolved=primary.get('monotone_evolved'),
        monotone_all_times=primary.get('monotone_all_times'),
        all_converged=primary.get('all_converged'),
        regressions_evolved=primary.get('regressions_evolved'),
        regressions_all_times=primary.get('regressions_all_times'))
    sp = verdict['criterion_3_error_span']
    verdict['criterion_3_passes'] = bool(
        (sp.get('error_span') or 0) >= pc['error_span']
        and (pc.get('cost_span') is None or (sp.get('cost_span') or 0) >= pc['cost_span']))
    verdict['passes'] = bool(verdict['criterion_1_evolved_monotone_every_rung_converged']
                             and verdict['criterion_2_all_times_monotone']
                             and verdict['criterion_3_passes'])
    # Falsification F1: the q = 16 regression on the evolved metric.
    verdict['q16_regression'] = {}
    for lid, lad in ladders.items():
        rg = {u['q']: u['evolved'] for u in lad['rungs']}
        if 0 in rg and 16 in rg:
            verdict['q16_regression'][lid] = dict(
                q0=rg[0], q16=rg[16], regression=bool(rg[16] > rg[0]),
                delta_percentage_points=rg[16] - rg[0])

    out = dict(result=str(Path(a.result).resolve()),
               result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               elapsed_seconds=r.get('elapsed_seconds'), output_times=r['output_times'],
               checks=checks, failed=sorted(fail), arms=rows, ladders=ladders,
               frontier=frontier, verdict=verdict,
               direction_sets={k: {kk: vv for kk, vv in v.items()
                                   if kk not in ('singular_values',)}
                               for k, v in r['direction_sets'].items()},
               direction_comparison=r['direction_comparison'],
               direction_cohorts=r['direction_cohorts'],
               comparators=comp_sg)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows),
                          verdict=verdict['passes'],
                          frontier={k: len(v) for k, v in frontier.items()}), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
