"""Independent NumPy audit of the EQ-certification job. No JAX, no GPU.

Recomputes every reported error from the retained output fields, tabulates every rule with
its held-out rho against the bar declared in DESIGN.md §A3.4, checks the cross-job fidelity
gates against `b-ladder-top`, computes the R3 held-out weak residual from the retained
per-step bank coefficients, and evaluates the pre-registered verdict.

    python audit_eqcert.py <result.json> --fields <dir> --out <audit.json>
                           [--bank <bank_G.npz>] [--comparators <comparators.json>] [--no-r3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import r3 as R3

HERE = Path(__file__).resolve().parent
COMMON_SKIP, COMMON_COUNT = 1536, 512
PER_ARM_CAP = 2048


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--fields', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bank', default=None)
    p.add_argument('--comparators', default=str(HERE / 'checks/comparators.json'))
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

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('step_budget_600', r['config']['strict']['step_budget'] == 600, r['config']['strict'])
    cg = r['gates']['evaluation_cohort_bitwise_abl01']
    gate('evaluation_cohort_bitwise_abl01', bool(cg['passed']), cg)
    gate('fit_and_certification_trajectories_disjoint',
         bool(r['gates']['fit_and_certification_trajectories_disjoint']['passed']),
         r['trajectory_split'])
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))

    cmp_table = {}
    if Path(a.comparators).exists():
        cmp_table = json.loads(Path(a.comparators).read_text())
    mine_dir = r['directions']['directions_sha256']
    dir_match = {k: (v == mine_dir) for k, v in
                 (r['config'].get('source_directions_sha256') or {}).items()}
    info('directions_hash_matches_cclad01', dir_match.get('cclad01'),
         dict(ours=mine_dir, matches=dir_match), 'a cross-job probe, not a requirement')

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
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()))
    gate('every_invocation_paired',
         all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms))
    gate('every_rom_carries_exit_and_stationarity',
         all(('stop_reasons' in x and 'budget_exits' in x) for x in roms))

    # ---------------------------------------------------- the rule table -----
    bar = float(r['rho_bar'])
    rules = r['rules']
    gate('every_rule_reports_rho', all('certification' in x for x in rules),
         [x.get('q') for x in rules if 'certification' not in x])
    gate('every_rule_archived',
         all(all(k in x for k in ('nodes_sha256', 'weights_sha256', 'm', 'relative_fit'))
             for x in rules))
    used_m = {(x['q'], x['population'], x['m_target']) for x in rules}
    gate('rule_grid_complete',
         all((q, 'reachable', m) in {(a_, b_, c_) for a_, b_, c_ in used_m}
             for q in r['config']['q_ladder'] for m in r['config']['m_grid']),
         sorted(used_m))
    trunc = [f"q{x['q']}/{x['population']}/m{x['m_target']}" for x in rules if x['truncated']]
    info('no_rule_truncated', not trunc, trunc,
         'a truncated rule is disqualified by the certification rule, not by this gate')

    # ------------------------------------------ errors recomputed from fields
    cache = {}

    def F(name):
        if name not in cache:
            cache[name] = np.load(fields_dir / name)['fields']
        return cache[name]

    missing = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
    gate('artifacts_present', not missing, missing[:5])
    refs = {x['case']: F(x['artifact']) for x in r['reference']}
    base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
    gate('same_grid_baseline_present', len(base) == len(refs))
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

    # ------------------------------------------------------ per-arm rows -----
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x.get('family'), q=x.get('q'),
            M=x.get('M'), m=x.get('m'), rule_kind=x.get('rule_kind'),
            quadrature=x.get('quadrature'), solved_dimension=x.get('solved_dimension'),
            dt=x.get('dt'), gpu_ms=[], host_ms=[], case_ref={}, case_all={}, case_ev={},
            case_t0={}, per_time={}, iters=[], stat=[], conv=[], comp=[], be=[], exits=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['error']['fixed_initial_max']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_ev'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['per_time'][x['case']] = x['same_grid_per_time']
        t['iters'] += x.get('iterations', [])
        if x['kind'] == 'rom':
            t['stat'].append(x['worst_joint_stationarity'])
            t['conv'].append(x['converged'])
            t['comp'].append(x['completed'])
            t['be'].append(x['budget_exits'])
            t['exits'] += x['stop_reasons']
    rows = []
    for name, t in table.items():
        s = setup.get(name, {})
        cert = s.get('certification') or {}
        rows.append(dict(
            arm=name, kind=t['kind'], family=t['family'] or 'fom', q=t['q'], M=t['M'],
            m=t['m'], rule_kind=t['rule_kind'], quadrature=t['quadrature'], dt=t['dt'],
            solved_dimension=t['solved_dimension'],
            rho_max=s.get('rho_max', cert.get('rho_max')),
            rho_p95=s.get('rho_p95', cert.get('rho_p95')),
            certified_primary=s.get('certified_primary'),
            certified_secondary=s.get('certified_secondary'),
            relative_fit=s.get('relative_fit'),
            worst_reference_percent=100 * float(np.max(list(t['case_ref'].values()))),
            worst_all_times_percent=100 * float(np.max(list(t['case_all'].values()))),
            worst_evolved_percent=100 * float(np.max(list(t['case_ev'].values()))),
            worst_t0_compression_percent=100 * float(np.max(list(t['case_t0'].values()))),
            per_case_evolved_percent={str(k): 100 * v for k, v in sorted(t['case_ev'].items())},
            per_time_same_grid_percent={str(k): [100 * z for z in v]
                                        for k, v in sorted(t['per_time'].items())},
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=(median(t['iters']) if t['iters'] else None),
            max_joint_stationarity=(float(np.max(t['stat'])) if t['stat'] else None),
            converged=(bool(all(t['conv'])) if t['conv'] else None),
            total_budget_exits=(int(np.sum(t['be'])) if t['be'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in
                                 zip(*np.unique(t['exits'], return_counts=True))}
                                if t['exits'] else None)))
    by = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: (x['family'], x['q'] if x['q'] is not None else -1, x['arm']))

    # ------------------------------------------------------ fidelity gates ---
    fid = {}
    for arm, spec in (r['config'].get('expectations') or {}).items():
        mine = by.get(arm)
        theirs = (cmp_table.get(spec['source']) or {}).get('arms', {}).get(spec['arm'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail=dict(reason='arm or comparator missing'))
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
        fid[arm] = dict(passed=bool(ok), detail=d)
        gate(f'reproduces_{arm}', ok, d)
    checks['cross_job_fidelity'] = dict(
        passed=(all(v['passed'] for v in fid.values()) if fid else None), detail=fid)

    # ------------------------------------------------------------- R3 -------
    r3out = None
    if not a.no_r3 and a.bank and Path(a.bank).exists():
        L, dt = r['intervals'], r['dt']
        Gm = np.load(a.bank)['G']
        gate('bank_sha256_consistent', sha(Gm) == r['bank_G']['sha256'])
        phys = np.asarray(r['physical_cases'])
        blocks_cache, short = {}, {}

        def block(skip, count):
            if (skip, count) not in blocks_cache:
                phi, lam, _ = R3.modes(L, count, skip)
                if phi.shape[1] != count:
                    short[f'{skip}+{count}'] = int(phi.shape[1])
                blocks_cache[(skip, count)] = (phi, lam)
            return blocks_cache[(skip, count)]

        common = block(COMMON_SKIP, COMMON_COUNT)
        art = {}
        for x in inv:
            if x['kind'] == 'rom':
                art.setdefault(x['name'], {})[x['case']] = x['artifact']
        dev, per_arm = 0., {}
        for name in sorted(art):
            row = by[name]
            M = int(row['M'])
            own = block(0, M)
            nxt = block(M, min(4 * M, PER_ARM_CAP))
            acc = {}
            for case, fn in sorted(art[name].items()):
                z = np.load(fields_dir / fn)
                coef = z['coefficients']
                U = coef @ Gm.T
                keep = max(1, (len(U) - 1) // 5)
                dec = np.stack([np.pad(u.reshape(L - 1, L - 1), 1) for u in U[::keep]])
                dev = max(dev, float(np.linalg.norm(dec - z['fields'])
                                     / np.linalg.norm(z['fields'])))
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
                        worst_raw=float(max(z['raw_max'] for z in v)))
                for k, v in acc.items()}
            per_arm[name]['held_over_in_common'] = (
                per_arm[name]['held_out_common']['worst_normalised_per_mode_rms']
                / max(per_arm[name]['in_space']['worst_normalised_per_mode_rms'], 1e-300))
            print('R3', name, per_arm[name]['held_out_common']['worst_normalised_per_mode_rms'],
                  flush=True)
        gate('decoded_fields_match_saved_outputs', dev < 1e-12, dev)
        gate('r3_mode_blocks_complete', not short, short)
        r3out = dict(common_block=dict(skip=COMMON_SKIP, count=COMMON_COUNT),
                     per_arm_block_cap=PER_ARM_CAP, arms=per_arm)
    else:
        checks['bank_sha256_consistent'] = dict(passed=None, detail=None, note='R3 not computed')

    # ------------------------------------------------- the rebuilt ladder ----
    QS = list(r['config']['q_ladder'])

    def ladder(prefix, quad):
        out = []
        for q in QS:
            nm = f'q{q}_m4_{prefix}'
            if nm in by:
                out.append(by[nm])
        return out if len(out) == len(QS) else []

    cert_l = ladder('eqcert', 'eq')
    old_l = ladder('eqstatic', 'eq')
    dense_l = [by[f'q{q}_m4_dense'] for q in r['config']['dense_twins']
               if f'q{q}_m4_dense' in by]

    def summarise(lad, baseline=None, name=''):
        if not lad:
            return None
        ev = [x['worst_evolved_percent'] for x in lad]
        al = [x['worst_all_times_percent'] for x in lad]
        cost = [x['median_gpu_ms'] for x in lad]
        d = dict(name=name, q=[x['q'] for x in lad], arms=[x['arm'] for x in lad],
                 M=[x['M'] for x in lad], m=[x['m'] for x in lad],
                 rho_max=[x['rho_max'] for x in lad],
                 certified_primary=[x['certified_primary'] for x in lad],
                 worst_evolved_percent=ev, worst_all_times_percent=al,
                 worst_t0_compression_percent=[x['worst_t0_compression_percent'] for x in lad],
                 median_gpu_ms=cost, converged=[x['converged'] for x in lad],
                 monotone_evolved=mono(ev), monotone_all_times=mono(al),
                 all_converged=bool(all(x['converged'] for x in lad)))
        if baseline:
            bc = [x['median_gpu_ms'] for x in baseline]
            ba = [x['worst_all_times_percent'] for x in baseline]
            d['cost_ratio_to_old_rule'] = [c / max(b, 1e-300) for c, b in zip(cost, bc)]
            d['all_times_relative_to_old_rule'] = [(v - b) / max(b, 1e-300)
                                                   for v, b in zip(al, ba)]
            d['cost_within_2x'] = bool(all(c <= 2. + 1e-12 for c in d['cost_ratio_to_old_rule']))
            d['all_times_not_raised'] = bool(all(v <= 1e-4 for v in
                                                 d['all_times_relative_to_old_rule']))
            d['passes'] = bool(d['monotone_evolved'] and d['all_converged']
                               and d['cost_within_2x'] and d['all_times_not_raised'])
        return d

    ladders = dict(certified=summarise(cert_l, old_l, 'EQ, cheapest certified rule per rung'),
                   old_rule=summarise(old_l, None, 'EQ, incumbent static-population rule'),
                   dense=summarise(dense_l, None, 'dense (exact) quadrature'))
    verdict = dict(
        passes=bool((ladders['certified'] or {}).get('passes')),
        regression_present_in_old_rule=(None if not ladders['old_rule']
                                        else not ladders['old_rule']['monotone_evolved']),
        every_rung_certified=bool(ladders['certified']
                                 and all(ladders['certified']['certified_primary'])),
        note=('pass requires the certified EQ ladder to be non-increasing in q on the evolved '
              'metric, every rung converged, at most 2x the old-rule cost per rung, and no '
              'rung worse on the all-times metric'))
    # falsification: does any m reach the bar at the top rung?
    top = max(QS)
    topr = [x for x in rules if x['q'] == top and x['population'] == 'reachable']
    topr.sort(key=lambda x: x['m_target'])
    verdict['top_rung_rules'] = [dict(m=x['m'], m_target=x['m_target'],
                                      rho_max=x['certification']['rho_max'],
                                      rho_p95=x['certification']['rho_p95'],
                                      certified=x['certified_primary']) for x in topr]
    verdict['top_rung_certified'] = any(x['certified_primary'] for x in topr)
    if topr and not verdict['top_rung_certified'] and len(topr) >= 2:
        # log-log extrapolation of rho against m, reported rather than acted on
        lm = np.log(np.array([x['m'] for x in topr], dtype=float))
        lr = np.log(np.array([x['certification']['rho_max'] for x in topr], dtype=float))
        sl, ic = np.polyfit(lm, lr, 1)
        verdict['extrapolated_m_for_bar'] = (
            float(np.exp((np.log(bar) - ic) / sl)) if sl < 0 else None)
        verdict['extrapolation_slope'] = float(sl)

    out = dict(result=str(Path(a.result).resolve()), mode='eqcert', job_id=r.get('job_id'),
               commit=r.get('commit'), gpu=r.get('gpu'), attempt=r['config'].get('attempt'),
               elapsed_seconds=r.get('elapsed_seconds'), rho_bar=bar, checks=checks,
               failed=sorted(fail), arms=rows, rules=rules, rule_choice=r['rule_choice'],
               collection=r['collection'], ladders=ladders, verdict=verdict, r3=r3out)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows), rules=len(rules),
                          verdict={k: verdict[k] for k in
                                   ('passes', 'regression_present_in_old_rule',
                                    'every_rung_certified', 'top_rung_certified')}), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
