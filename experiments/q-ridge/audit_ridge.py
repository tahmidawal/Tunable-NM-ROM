"""Independent NumPy audit of a q-ridge job. No JAX, no GPU.

Recomputes every reported error from the retained output fields, measures each arm against
the same-job converged full-order solve on three metrics ($t=0$ compression, worst over all
output times, worst over evolved times only), checks the cross-job fidelity gates against
`btq101`, `btq102` and `cclad01`, computes the R3 held-out weak residual from the retained
per-step bank coefficients, and evaluates the pre-registered verdict per quadrature.

    python audit_ridge.py <result.json> --fields <dir> --out <audit.json> --mode r1|r2
                          [--bank <bank_G.npz>] [--comparators <comparators.json>]
                          [--r3-arms a,b,c] [--no-r3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import r3 as R3

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LADDER_Q = {'r1': [0, 16, 64, 256], 'r2': [0, 16, 64]}
# One common held-out block for the whole lane: modes ranked 1537..2048 in the shared
# ordering, strictly beyond the largest M any arm of either job solves against (1280).
COMMON_SKIP, COMMON_COUNT = 1536, 512
PER_ARM_CAP = 2048


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


def nondominated(rows, cost, err):
    pts = [r for r in rows if r.get(cost) is not None and r.get(err) is not None]
    return sorted({r['arm'] for r in pts
                   if not any((o[cost] <= r[cost] and o[err] <= r[err]
                               and (o[cost] < r[cost] or o[err] < r[err])) for o in pts)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--fields', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=('r1', 'r2'), required=True)
    p.add_argument('--bank', default=None)
    p.add_argument('--comparators', default=str(HERE / 'checks/comparators.json'))
    p.add_argument('--r3-arms', default=None)
    p.add_argument('--no-r3', action='store_true')
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
    gate('step_budget_600', (r['config']['strict']['step_budget'] == 600),
         r['config']['strict'])
    cg = r['gates']['evaluation_cohort_bitwise_abl01']
    gate('evaluation_cohort_bitwise_abl01', bool(cg['passed']), cg)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))

    cmp_table = {}
    if Path(a.comparators).exists():
        cmp_table = json.loads(Path(a.comparators).read_text())
    mine_dir = r['directions']['directions_sha256']
    dir_match = {k: (v == mine_dir) for k, v in
                 (r['config'].get('source_directions_sha256') or {}).items()}
    info('directions_hash_matches_cclad01', dir_match.get('cclad01'),
         dict(ours=mine_dir, sources=r['config'].get('source_directions_sha256'),
              matches=dir_match),
         'bitwise across jobs; btq101/cclad01 (A100 80GB PCIe) and btq201 (A100-PCIE-40GB) '
         'already disagree with each other, so this is a probe. The substantive statement is '
         'the reproduction of the named rows below.')
    refsha = {str(x['case']): x['field_sha256'] for x in r['reference']}
    info('reference_fields_bitwise_match_a_source',
         any(refsha == v for v in (r['config'].get('source_reference_sha256') or {}).values()),
         {k: (refsha == v) for k, v in
          (r['config'].get('source_reference_sha256') or {}).items()},
         'GPU-model dependent; a probe, not a requirement')
    gate('directions_rank_covers_ladder',
         r['directions']['available_rank'] >= max(
             [x.get('q') or 0 for x in r['declared_subjects'] if x.get('method') == 'rom'] or [0]),
         r['directions']['available_rank'])

    # ------------------------------------------------------- invocations ------
    inv = r['invocations']
    reps = r['config']['reps']
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
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    eq = [s for s in setup.values() if s.get('quadrature') == 'eq']
    gate('every_eq_rule_reports_validity',
         all(('eq_rule_valid' in s or s.get('eq_fit', {}).get('fitter') == 'retained_nnls_capped')
             for s in eq),
         [s['arm'] for s in eq if 'eq_rule_valid' not in s][:5])
    bad_eq = [s['arm'] for s in eq if s.get('eq_fit', {}).get('truncated')]
    gate('eq_rules_untruncated', not bad_eq, bad_eq)
    t0g = r['gates']['t0_field_invariant_in_lambda']
    checks['t0_field_invariant_in_lambda'] = dict(passed=t0g['passed'], detail=t0g)
    if t0g['passed'] is False:
        fail.append('t0_field_invariant_in_lambda')

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

    # ------------------------------------------------------ per-arm rows ------
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x.get('family'), q=x.get('q'),
            solved_dimension=x.get('solved_dimension'), M=x.get('M'), m=x.get('m'),
            rule=x.get('rule'), quadrature=x.get('quadrature'), lam_rel=x.get('lam_rel'),
            lam_abs=x.get('lam_abs'), sigma_q=x.get('sigma_q'), fitter=x.get('fitter'),
            dt=x.get('dt'), step_budget=x.get('step_budget'), gpu_ms=[], host_ms=[],
            case_ref={}, case_all={}, case_evolved={}, case_t0={}, iterations=[],
            stationarity=[], converged=[], completed=[], budget_exits=[], exits=[],
            per_time_all={}, per_time_ref={}))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_evolved'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['per_time_all'][x['case']] = x['same_grid_per_time']
        t['per_time_ref'][x['case']] = x['error']['fixed_initial_per_time']
        t['iterations'] += x.get('iterations', [])
        if x['kind'] == 'rom':
            t['stationarity'].append(x['worst_joint_stationarity'])
            t['converged'].append(x['converged'])
            t['completed'].append(x['completed'])
            t['budget_exits'].append(x['budget_exits'])
            t['exits'] += x['stop_reasons']

    rows = []
    for name, t in table.items():
        s = setup.get(name, {})
        eqf = s.get('eq_fit') or {}
        rc = {x['q']: x for x in r['reconstruction']}.get(t['q'])
        rows.append(dict(
            arm=name, kind=t['kind'], family=t['family'] or 'fom', q=t['q'],
            solved_dimension=t['solved_dimension'], M=t['M'], m=t['m'], rule=t['rule'],
            quadrature=t['quadrature'], lam_rel=t['lam_rel'], lam_abs=t['lam_abs'],
            sigma_q=t['sigma_q'], fitter=t['fitter'], dt=t['dt'], step_budget=t['step_budget'],
            tests_per_unknown=s.get('tests_per_unknown'),
            quadrature_points_per_test=s.get('quadrature_points_per_test'),
            m_target=s.get('m_target'),
            worst_reference_percent=100 * float(np.max(list(t['case_ref'].values()))),
            worst_all_times_percent=100 * float(np.max(list(t['case_all'].values()))),
            worst_evolved_percent=100 * float(np.max(list(t['case_evolved'].values()))),
            median_evolved_percent=100 * float(np.median(list(t['case_evolved'].values()))),
            worst_t0_compression_percent=100 * float(np.max(list(t['case_t0'].values()))),
            per_case_evolved_percent={str(k): 100 * v for k, v in sorted(t['case_evolved'].items())},
            per_case_all_times_percent={str(k): 100 * v for k, v in sorted(t['case_all'].items())},
            per_time_same_grid_percent={str(k): [100 * z for z in v]
                                        for k, v in sorted(t['per_time_all'].items())},
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            max_iterations=(float(np.max(t['iterations'])) if t['iterations'] else None),
            max_joint_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            converged=(bool(all(t['converged'])) if t['converged'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in
                                 zip(*np.unique(t['exits'], return_counts=True))}
                                if t['exits'] else None),
            quadrature_fit_seconds=eqf.get('seconds'),
            quadrature_fit_relative=s.get('eq_relative_fit'),
            quadrature_support=eqf.get('support'), quadrature_truncated=eqf.get('truncated'),
            eq_rule_valid=s.get('eq_rule_valid'),
            worst_bank_projection_percent=(100 * rc['worst_bank_projection'] if rc else None),
            worst_best_found_percent=(100 * rc['worst_best_found'] if rc else None),
            setup_seconds=s.get('total_setup_seconds')))
    by = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: (x['family'], x['q'] if x['q'] is not None else -1,
                             x['M'] or 0, x['lam_rel'] if x['lam_rel'] is not None else -1))

    # ------------------------------------------------------ fidelity gates ----
    fid = {}
    for arm, spec in (r['config'].get('expectations') or {}).items():
        mine = by.get(arm)
        src = (cmp_table.get(spec['source']) or {}).get('arms', {})
        theirs = src.get(spec['arm'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail=dict(reason='arm or comparator missing',
                                                      source=spec['source'], arm=spec['arm']))
            gate(f'reproduces_{arm}', False, fid[arm]['detail'])
            continue
        tol = spec['tolerance']
        if dir_match.get(spec['source']) and 'tolerance_if_directions_bitwise' in spec:
            tol = spec['tolerance_if_directions_bitwise']
        d = dict(source=spec['source'], comparator=spec['arm'], declared_tolerance=tol,
                 directions_bitwise=dir_match.get(spec['source']))
        ok = True
        for key in ('worst_all_times_percent', 'worst_evolved_percent',
                    'worst_reference_percent'):
            mv, tv = mine[key], theirs.get(key)
            rel = (None if tv in (None, 0) else abs(mv - tv) / abs(tv))
            d[key] = dict(ours=mv, theirs=tv, relative_difference=rel)
            ok = ok and (rel is None or rel <= tol)
        d['ours_median_gpu_ms'] = mine['median_gpu_ms']
        d['theirs_median_gpu_ms'] = theirs.get('median_gpu_ms')
        fid[arm] = dict(passed=bool(ok), detail=d)
        gate(f'reproduces_{arm}', ok, d)
    checks['cross_job_fidelity'] = dict(
        passed=(all(v['passed'] for v in fid.values()) if fid else None), detail=fid)

    # ------------------------------------------------------------- R3 --------
    r3out = None
    if not a.no_r3 and a.bank and Path(a.bank).exists():
        L = r['intervals']
        dt = r['dt']
        Gm = np.load(a.bank)['G']
        gate('bank_sha256_consistent',
             R3_sha(Gm) == r['bank_G']['sha256'], dict(recorded=r['bank_G']['sha256']))
        phys = np.asarray(r['physical_cases'])
        if a.r3_arms:
            want = [s for s in a.r3_arms.split(',') if s]
        else:
            want = sorted({x['arm'] for x in rows if x['family'] == 'rom'
                           and (x['lam_rel'] == 0. or x['q'] == 16)})
        blocks_cache = {}

        def block(skip, count):
            if (skip, count) not in blocks_cache:
                phi, lam, _ = R3.modes(L, count, skip)
                blocks_cache[(skip, count)] = (phi, lam)
            return blocks_cache[(skip, count)]

        common = block(COMMON_SKIP, COMMON_COUNT)
        art = {}
        for x in inv:
            if x['kind'] == 'rom':
                art.setdefault(x['name'], {})[x['case']] = x['artifact']
        decoded_dev = 0.
        per_arm = {}
        for name in want:
            row = by.get(name)
            if row is None:
                continue
            M = int(row['M'])
            own = block(0, M)
            nxt = block(M, min(4 * M, PER_ARM_CAP))
            acc = {}
            for case, fn in sorted(art.get(name, {}).items()):
                z = np.load(fields_dir / fn)
                coef = z['coefficients']
                # gate: the retained coefficients decode to the retained output fields
                U = coef @ Gm.T
                keep = max(1, (len(U) - 1) // 5)
                dec = np.stack([np.pad(u.reshape(L - 1, L - 1), 1) for u in U[::keep]])
                decoded_dev = max(decoded_dev, float(
                    np.linalg.norm(dec - z['fields']) / np.linalg.norm(z['fields'])))
                res = R3.trajectory_residuals(
                    Gm, coef, {'in_space': own, 'held_out_next_4M': nxt,
                               'held_out_common': common},
                    float(phys[case, 4]), dt, L)
                for k in ('in_space', 'held_out_next_4M', 'held_out_common'):
                    acc.setdefault(k, []).append(res[k])
            per_arm[name] = {
                k: dict(modes=v[0]['modes'],
                        worst_normalised_per_mode_rms=float(max(z['normalised_max'] for z in v)),
                        mean_normalised_per_mode_rms=float(np.mean([z['normalised_mean'] for z in v])),
                        worst_raw=float(max(z['raw_max'] for z in v)),
                        per_case_normalised_max=[z['normalised_max'] for z in v])
                for k, v in acc.items()}
            per_arm[name]['held_over_in_common'] = (
                per_arm[name]['held_out_common']['worst_normalised_per_mode_rms']
                / max(per_arm[name]['in_space']['worst_normalised_per_mode_rms'], 1e-300))
            print('R3', name, per_arm[name]['held_out_common']['worst_normalised_per_mode_rms'],
                  flush=True)
        gate('decoded_fields_match_saved_outputs', decoded_dev < 1e-12, decoded_dev)

        def cmp_pair(quad, rule, q_lo, q_hi, lam=0.):
            lo = next((x['arm'] for x in rows if x['family'] == 'rom' and x['q'] == q_lo
                       and x['quadrature'] == quad and x['rule'] == rule
                       and x['fitter'] != 'retained'), None)
            hi = next((x['arm'] for x in rows if x['family'] == 'rom' and x['q'] == q_hi
                       and x['quadrature'] == quad and x['rule'] == rule
                       and x['lam_rel'] == lam and x['fitter'] != 'retained'), None)
            if lo not in per_arm or hi not in per_arm:
                return None
            a_, b_ = per_arm[lo], per_arm[hi]
            return dict(
                low=lo, high=hi, block='held_out_common',
                low_held=a_['held_out_common']['worst_normalised_per_mode_rms'],
                high_held=b_['held_out_common']['worst_normalised_per_mode_rms'],
                low_in_space=a_['in_space']['worst_normalised_per_mode_rms'],
                high_in_space=b_['in_space']['worst_normalised_per_mode_rms'],
                held_out_higher_at_high_q=bool(
                    b_['held_out_common']['worst_normalised_per_mode_rms']
                    > a_['held_out_common']['worst_normalised_per_mode_rms']))
        pairs = {q: {r_: cmp_pair(q, 'm4', 0, 16) for r_ in ('m4',)} for q in ('dense', 'eq')}
        r3out = dict(common_block=dict(skip=COMMON_SKIP, count=COMMON_COUNT,
                                       note='modes ranked 1537..2048, beyond every arm\'s M '
                                            'in either job (max M = 1280)'),
                     per_arm_block_cap=PER_ARM_CAP, arms=per_arm,
                     falsification_pairs={k: v['m4'] for k, v in pairs.items()})

    # -------------------------------------------------- ladders and verdict ---
    qs = LADDER_Q[a.mode]
    ladders, verdict = {}, {}
    for quad in ('dense', 'eq'):
        def pick(q, rule, lam):
            use = 0. if q == 0 else lam
            c = [x for x in rows if x['family'] == 'rom' and x['q'] == q and x['rule'] == rule
                 and x['quadrature'] == quad and x['lam_rel'] == use
                 and x['fitter'] != 'retained']
            return c[0] if c else None

        if a.mode == 'r1':
            keys = sorted({x['lam_rel'] for x in rows if x['family'] == 'rom'
                           and x['quadrature'] == quad and x['rule'] == 'm4'
                           and x['lam_rel'] is not None})
            variants = [('lam', k, 'm4', k) for k in keys]
        else:
            keys = [k for k in ('m4', 'm8', 'm16')
                    if any(x['rule'] == k and x['quadrature'] == quad for x in rows)]
            variants = [('rule', k, k, 0.) for k in keys]
        out = {}
        for kind, label, rule, lam in variants:
            lad = [pick(q, rule, lam) for q in qs]
            if any(v is None for v in lad):
                continue
            base_lad = [pick(q, 'm4', 0.) for q in qs]
            ev = [x['worst_evolved_percent'] for x in lad]
            al = [x['worst_all_times_percent'] for x in lad]
            cost = [x['median_gpu_ms'] for x in lad]
            bcost = [x['median_gpu_ms'] for x in base_lad]
            ball = [x['worst_all_times_percent'] for x in base_lad]
            cost_ratio = [c / max(b, 1e-300) for c, b in zip(cost, bcost)]
            all_raised = [(v - b) / max(b, 1e-300) for v, b in zip(al, ball)]
            entry = dict(
                kind=kind, label=(f'{label:g}' if kind == 'lam' else label), rule=rule,
                lam_rel=lam, q=qs, arms=[x['arm'] for x in lad], M=[x['M'] for x in lad],
                m=[x['m'] for x in lad],
                worst_evolved_percent=ev, worst_all_times_percent=al,
                worst_t0_compression_percent=[x['worst_t0_compression_percent'] for x in lad],
                median_gpu_ms=cost, cost_ratio_to_incumbent=cost_ratio,
                all_times_relative_to_incumbent=all_raised,
                converged=[x['converged'] for x in lad],
                eq_rule_valid=[x['eq_rule_valid'] for x in lad],
                monotone_evolved=mono(ev), monotone_all_times=mono(al),
                all_converged=bool(all(x['converged'] for x in lad)),
                cost_within_1p5x=bool(all(c <= 1.5 + 1e-12 for c in cost_ratio)),
                all_times_not_raised=bool(all(v <= 1e-4 for v in all_raised)),
                top_rung_gain_kept=bool(ev[-1] <= .9 * ev[0]),
                top_rung_evolved_percent=ev[-1], base_rung_evolved_percent=ev[0])
            entry['passes'] = bool(entry['monotone_evolved'] and entry['all_converged']
                                   and entry['cost_within_1p5x'] and entry['all_times_not_raised'])
            entry['passes_without_losing_the_gain'] = bool(entry['passes']
                                                           and entry['top_rung_gain_kept'])
            out[entry['label']] = entry
        ladders[quad] = out
        winners = [k for k, v in out.items() if v['passes']]
        clean = [k for k, v in out.items() if v['passes_without_losing_the_gain']]
        incumbent = out.get('0' if a.mode == 'r1' else 'm4')
        verdict[quad] = dict(
            regression_present_at_incumbent=(None if incumbent is None
                                             else not incumbent['monotone_evolved']),
            incumbent_evolved=(None if incumbent is None
                               else incumbent['worst_evolved_percent']),
            passing=winners, passing_without_losing_the_gain=clean,
            passes=bool(winners), passes_cleanly=bool(clean))

    frontier = dict(
        all_subjects_all_times=nondominated(rows, 'median_gpu_ms', 'worst_all_times_percent'),
        all_subjects_evolved=nondominated(rows, 'median_gpu_ms', 'worst_evolved_percent'))

    out = dict(result=str(Path(a.result).resolve()), mode=a.mode, job_id=r.get('job_id'),
               commit=r.get('commit'), gpu=r.get('gpu'), elapsed_seconds=r.get('elapsed_seconds'),
               attempt=r['config'].get('attempt'), checks=checks, failed=sorted(fail),
               arms=rows, ladders=ladders, verdict=verdict, frontier=frontier, r3=r3out)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows),
                          verdict={k: {kk: v[kk] for kk in ('passes', 'passes_cleanly',
                                                            'regression_present_at_incumbent')}
                                   for k, v in verdict.items()}), indent=2))
    print('AUDIT WROTE', a.out)


def R3_sha(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


if __name__ == '__main__':
    main()
