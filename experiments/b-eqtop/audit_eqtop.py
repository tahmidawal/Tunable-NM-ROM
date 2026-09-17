"""Independent NumPy audit of a b-eqtop job. No JAX, no GPU, no driver import.

Recomputes every reported error from the retained output fields against the job's own
converged full-order solve, checks the gates, tabulates every rule (archived and new) with
its held-out rho against both bars, evaluates the cross-job fidelity gates against qrg304,
fits the rho-vs-m law per (rung, arm), builds the primary / tight / hybrid ladders with
their monotonicity verdicts, and writes the machine-readable audit.

    python audit_eqtop.py <result.json> --fields <dir> --out <audit.json>
                          [--comparators checks/comparators.json] [--bank bank_G.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def loglog_fit(ms, rhos):
    ms, rhos = np.asarray(ms, float), np.asarray(rhos, float)
    ok = (ms > 0) & (rhos > 0)
    if ok.sum() < 2:
        return None
    sl, ic = np.polyfit(np.log(ms[ok]), np.log(rhos[ok]), 1)
    return dict(alpha=float(-sl), intercept=float(ic), points=int(ok.sum()))


def m_for_bar(fit, bar):
    if fit is None or fit['alpha'] <= 0:
        return None
    return float(np.exp((np.log(bar) - fit['intercept']) / (-fit['alpha'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--fields', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--comparators', default=str(HERE / 'checks/comparators.json'))
    p.add_argument('--bank', default=None)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    cfg = r['config']
    fields_dir = Path(a.fields)
    checks, fail = {}, []
    timed = bool(cfg.get('timed_phase'))

    def gate(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note)
        if not ok:
            fail.append(name)

    def info(name, ok, detail=None, note=None):
        checks[name] = dict(passed=(None if ok is None else bool(ok)), detail=detail,
                            note=note, blocking=False)

    gate('complete', r.get('complete') is True)
    gate('fit_phase_complete', r.get('fit_phase_complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('step_budget_600', cfg['strict']['step_budget'] == 600, cfg['strict'])
    cg = r['gates']['evaluation_cohort_bitwise_abl01']
    gate('evaluation_cohort_bitwise_abl01', bool(cg['passed']), cg)
    gate('fit_and_certification_trajectories_disjoint',
         bool(r['gates']['fit_and_certification_trajectories_disjoint']['passed']))
    rc = r['gates']['archived_rules_recertify']
    gate('archived_rules_recertify', bool(rc['passed']), rc,
         'the qrg304 rules re-certified on this job\'s held-out states reproduce their rho_max')
    info('collection_pools_bitwise_qrg304', r['gates']['collection_pools_bitwise_qrg304']['passed'],
         [x['matches_qrg304'] for x in r['collection']], 'GPU-model dependent')
    dh = r['gates']['directions_hash_matches_qrg304']
    info('directions_hash_matches_qrg304', dh['passed'], dh, 'sets the fidelity tolerance tier')
    bars = {k: float(v) for k, v in r['bars'].items()}

    # ---------------------------------------------------- the rule table -----
    rules = []
    for x in r['archived_rules']:
        c = x['certification']
        rules.append(dict(source='qrg304', q=x['q'], M=x['M'], arm=x['population'],
                          population=x['population'], m=x['m'], m_target=x['m_target'],
                          fit_states=x['fit_states'], candidates=x['candidates'],
                          design_rows=x['design_rows'], relative_fit=x['relative_fit'],
                          rho_max=c['rho_max'], rho_p95=c['rho_p95'], rho_median=c['rho_median'],
                          rho_max_qrg304=x['rho_max_qrg304'],
                          certified_primary=bool(c['rho_max'] <= bars['primary']),
                          certified_tight=bool(c['rho_max'] <= bars['tight']),
                          certified_secondary=bool(c['rho_p95'] <= bars['primary']),
                          truncated=False, fit_seconds=x.get('fit_seconds'), scaling='row',
                          compressed=False))
    failed_fits = [x for x in r['rules'] if x.get('failed')]
    for x in r['rules']:
        if x.get('failed'):
            continue
        c = x['certification']
        f = x['fit']
        d = next(dd for dd in r['designs'] if dd['key'] == x['key'])
        rules.append(dict(source='this_job', q=x['q'], M=x['M'], arm=x['arm'],
                          population='reachable', m=x['m'], m_target=x['m_target'],
                          fit_states=d['fit_states'], candidates=d['candidates'],
                          design_rows=d['design']['rows'], relative_fit=f['relative_fit'],
                          rho_max=c['rho_max'], rho_p95=c['rho_p95'], rho_median=c['rho_median'],
                          rho_max_qrg304=None,
                          certified_primary=bool(c['rho_max'] <= bars['primary'] and not f['truncated']),
                          certified_tight=bool(c['rho_max'] <= bars['tight'] and not f['truncated']),
                          certified_secondary=bool(c['rho_p95'] <= bars['primary'] and not f['truncated']),
                          truncated=bool(f['truncated']), fit_seconds=f['seconds'],
                          stop_reason=f['stop_reason'], scaling=d['scaling'],
                          compressed=bool(d['compression']['compressed']),
                          wall_seconds=x.get('wall_seconds')))
    # the driver's own flags must agree with the audit's recomputation from rho
    agree = all((x['certified_primary'] == (x['certification']['rho_max'] <= bars['primary']
                                            and not x['fit']['truncated']))
                for x in r['rules'] if not x.get('failed'))
    gate('certification_flags_recomputed', agree)
    gate('every_rule_archived', all('nodes_sha256' in x and 'weights_sha256' in x
                                    for x in r['rules'] if not x.get('failed')))
    info('no_worker_failed', not failed_fits, failed_fits)
    rules.sort(key=lambda x: (x['q'], x['source'], x['arm'], x['m_target']))

    # rho-vs-m law per (q, arm) over this job's rules, and the m needed for each bar
    laws = []
    for q in sorted({x['q'] for x in rules}):
        for arm in sorted({x['arm'] for x in rules if x['q'] == q and x['source'] == 'this_job'}):
            pts = [x for x in rules if x['q'] == q and x['arm'] == arm and x['source'] == 'this_job'
                   and not x['truncated']]
            pts.sort(key=lambda x: x['m'])
            fit = loglog_fit([x['m'] for x in pts], [x['rho_max'] for x in pts])
            fit95 = loglog_fit([x['m'] for x in pts], [x['rho_p95'] for x in pts])
            cheapest = {b: next((x['m'] for x in pts if x['rho_max'] <= v), None)
                        for b, v in bars.items()}
            laws.append(dict(q=q, arm=arm, m=[x['m'] for x in pts],
                             rho_max=[x['rho_max'] for x in pts],
                             rho_p95=[x['rho_p95'] for x in pts], fit_rho_max=fit,
                             fit_rho_p95=fit95,
                             m_for_bar={b: m_for_bar(fit, v) for b, v in bars.items()},
                             cheapest_certified_m=cheapest))

    # ------------------------------------------ per-rung choice (from the job)
    choice = r['rule_choice']

    out = dict(result=str(Path(a.result).resolve()), job_id=r.get('job_id'),
               commit=r.get('commit'), gpu=r.get('gpu'), attempt=cfg.get('attempt'),
               question=cfg.get('question'), elapsed_seconds=r.get('elapsed_seconds'),
               fit_phase_seconds=r.get('fit_phase_seconds'), bars=bars, rules=rules,
               laws=laws, rule_choice=choice, chains=r.get('chains'), designs=r.get('designs'),
               collection=r['collection'])

    if not timed:
        out.update(checks=checks, failed=sorted(fail), arms=[], ladders=None, verdict=None)
        Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(dict(failed=sorted(fail), rules=len(rules)), indent=2))
        print('AUDIT WROTE', a.out)
        return

    # ------------------------------------------ errors recomputed from fields
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    inv = r['invocations']
    reps = cfg['reps']
    counts = {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()))
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms))
    gate('every_rom_carries_exit_and_stationarity',
         all(('stop_reasons' in x and 'budget_exits' in x) for x in roms))
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
    if a.bank and Path(a.bank).exists():
        Gm = np.load(a.bank)['G']
        gate('bank_sha256_consistent', sha(Gm) == r['bank_G']['sha256'])
        L = r['intervals']
        dev = 0.
        for x in inv:
            if x['kind'] != 'rom' or x['rep'] != 0:
                continue
            z = np.load(fields_dir / x['artifact'])
            U = z['coefficients'] @ Gm.T
            keep = max(1, (len(U) - 1) // 5)
            dec = np.stack([np.pad(u.reshape(L - 1, L - 1), 1) for u in U[::keep]])
            dev = max(dev, float(np.linalg.norm(dec - z['fields']) / np.linalg.norm(z['fields'])))
        gate('decoded_fields_match_saved_outputs', dev < 1e-12, dev)

    # ------------------------------------------------------ per-arm rows -----
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], q=x.get('q'), M=x.get('M'), m=x.get('m'),
            rule_kind=x.get('rule_kind'), quadrature=x.get('quadrature'),
            solved_dimension=x.get('solved_dimension'), dt=x.get('dt'), gpu_ms=[], host_ms=[],
            case_ref={}, case_all={}, case_ev={}, case_t0={}, per_time={}, iters=[], stat=[],
            conv=[], be=[], exits=[]))
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
            t['be'].append(x['budget_exits'])
            t['exits'] += x['stop_reasons']
    rows = []
    for name, t in table.items():
        s = setup.get(name, {})
        rows.append(dict(
            arm=name, kind=t['kind'], family=('rom' if t['kind'] == 'rom' else 'fom'), q=t['q'],
            M=t['M'], m=t['m'], rule_kind=t['rule_kind'], quadrature=t['quadrature'],
            solved_dimension=t['solved_dimension'], rho_max=s.get('rho_max'),
            rho_p95=s.get('rho_p95'), certified_primary=s.get('certified_primary'),
            certified_tight=s.get('certified_tight'), certified_secondary=s.get('certified_secondary'),
            basis=s.get('basis'), rule_source=s.get('rule_source'), rule_arm=s.get('rule_arm'),
            relative_fit=s.get('relative_fit'),
            worst_reference_percent=100 * float(np.max(list(t['case_ref'].values()))),
            worst_all_times_percent=100 * float(np.max(list(t['case_all'].values()))),
            worst_evolved_percent=100 * float(np.max(list(t['case_ev'].values()))),
            median_evolved_percent=100 * median(list(t['case_ev'].values())),
            worst_t0_compression_percent=100 * float(np.max(list(t['case_t0'].values()))),
            per_case_evolved_percent={str(k): 100 * v for k, v in sorted(t['case_ev'].items())},
            per_time_same_grid_percent={str(k): [100 * z for z in v]
                                        for k, v in sorted(t['per_time'].items())},
            median_gpu_ms=median(t['gpu_ms']), gpu_ms_all=t['gpu_ms'],
            median_host_ms=median(t['host_ms']), gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=(median(t['iters']) if t['iters'] else None),
            max_joint_stationarity=(float(np.max(t['stat'])) if t['stat'] else None),
            converged=(bool(all(t['conv'])) if t['conv'] else None),
            total_budget_exits=(int(np.sum(t['be'])) if t['be'] else None)))
    by = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: (x['family'], x['q'] if x['q'] is not None else -1, x['arm']))

    # ------------------------------------------------------ fidelity gates ---
    cmp_table = json.loads(Path(a.comparators).read_text()) if Path(a.comparators).exists() else {}
    dir_ok = bool(dh['passed'])
    fid = {}
    for arm, spec in (cfg.get('expectations') or {}).items():
        mine = by.get(arm)
        theirs = (cmp_table.get(spec['source']) or {}).get('arms', {}).get(spec['arm'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail=dict(reason='arm or comparator missing'))
            gate(f'reproduces_{arm}', False, fid[arm]['detail'])
            continue
        tol = spec['tolerance']
        if dir_ok and 'tolerance_if_directions_bitwise' in spec:
            tol = spec['tolerance_if_directions_bitwise']
        d = dict(source=spec['source'], comparator=spec['arm'], declared_tolerance=tol,
                 directions_bitwise=dir_ok)
        ok = True
        for key in ('worst_all_times_percent', 'worst_evolved_percent', 'worst_reference_percent'):
            mv, tv = mine[key], theirs.get(key)
            rel = (None if tv in (None, 0) else abs(mv - tv) / abs(tv))
            d[key] = dict(ours=mv, theirs=tv, relative_difference=rel)
            ok = ok and (rel is None or rel <= tol)
        fid[arm] = dict(passed=bool(ok), detail=d)
        gate(f'reproduces_{arm}', ok, d)
    checks['cross_job_fidelity'] = dict(passed=(all(v['passed'] for v in fid.values()) if fid else None),
                                        detail=fid)

    # ------------------------------------------------- the ladders ------------
    QS = list(cfg['q_ladder'])

    def ladder(name, arms_by_q, allow_missing=False):
        lad = [arms_by_q.get(q) for q in QS]
        if any(x is None for x in lad):
            if not allow_missing:
                return None
            lad = [x for x in lad if x is not None]
        ev = [x['worst_evolved_percent'] for x in lad]
        al = [x['worst_all_times_percent'] for x in lad]
        cost = [x['median_gpu_ms'] for x in lad]
        d = dict(name=name, q=[x['q'] for x in lad], arms=[x['arm'] for x in lad],
                 M=[x['M'] for x in lad], m=[x['m'] for x in lad],
                 quadrature=[x['quadrature'] for x in lad], rho_max=[x['rho_max'] for x in lad],
                 certified_primary=[x['certified_primary'] for x in lad],
                 certified_tight=[x['certified_tight'] for x in lad],
                 worst_evolved_percent=ev, worst_all_times_percent=al,
                 worst_t0_compression_percent=[x['worst_t0_compression_percent'] for x in lad],
                 median_gpu_ms=cost, converged=[x['converged'] for x in lad],
                 monotone_evolved=mono(ev), monotone_all_times=mono(al),
                 all_converged=bool(all(x['converged'] for x in lad)),
                 every_rung_eq_certified_primary=bool(all(
                     x['quadrature'] == 'eq' and x['certified_primary'] for x in lad)))
        dense = {q: by.get(f'q{q}_dense') for q in QS}
        d['cost_vs_dense_twin'] = [(c / dense[q]['median_gpu_ms'] if dense.get(q) else None)
                                   for q, c in zip(d['q'], cost)]
        d['cheaper_than_dense_where_measured'] = bool(all(
            v is None or v < 1. for v in d['cost_vs_dense_twin']))
        d['evolved_vs_dense_twin_pp'] = [(e_ - dense[q]['worst_evolved_percent'] if dense.get(q) else None)
                                         for q, e_ in zip(d['q'], ev)]
        d['passes'] = bool(d['monotone_evolved'] and d['all_converged']
                           and d['cheaper_than_dense_where_measured'])
        d['violations'] = [dict(from_q=q1, to_q=q2, from_percent=e1, to_percent=e2)
                           for (q1, e1), (q2, e2) in zip(zip(d['q'], ev), zip(d['q'][1:], ev[1:]))
                           if e2 > e1 + 1e-12]
        return d

    primary = {q: by.get(f'q{q}_eq_primary') for q in QS}
    primary_cert = {q: x for q, x in primary.items() if x is not None and x['certified_primary']}
    tight = {}
    for q in QS:
        x = by.get(f'q{q}_eq_tight')
        if x is None and primary.get(q) is not None and primary[q]['certified_tight']:
            x = primary[q]
        if x is not None and x['certified_tight']:
            tight[q] = x
    hybrid = {q: (primary_cert.get(q) or by.get(f'q{q}_dense')) for q in QS}
    hybrid = {q: x for q, x in hybrid.items() if x is not None}
    ladders = dict(
        primary=ladder('EQ, cheapest primary-certified rule per rung (secondary fallback where none)',
                       {q: x for q, x in primary.items() if x is not None}),
        primary_certified_only=ladder('EQ, primary-certified rungs only', primary_cert, allow_missing=True),
        tight=ladder('EQ, cheapest tight-certified rule per rung', tight, allow_missing=True),
        hybrid=ladder('hybrid: primary-certified EQ where it exists, dense elsewhere', hybrid,
                      allow_missing=True),
        dense=ladder('dense (exact) quadrature twins', {q: by[f'q{q}_dense'] for q in QS
                                                          if f'q{q}_dense' in by}, allow_missing=True))
    top = max(QS)
    verdict = dict(
        every_rung_primary_certified=bool(all(q in primary_cert for q in QS)),
        rungs_without_primary_rule=[q for q in QS if q not in primary_cert],
        primary_ladder_monotone_evolved=(ladders['primary'] or {}).get('monotone_evolved'),
        primary_ladder_passes=(ladders['primary'] or {}).get('passes'),
        tight_ladder_monotone_evolved=(ladders['tight'] or {}).get('monotone_evolved'),
        hybrid_ladder_monotone_evolved=(ladders['hybrid'] or {}).get('monotone_evolved'),
        top_rung=dict(q=top, primary_rule=(primary.get(top) or {}).get('m'),
                      primary_rho_max=(primary.get(top) or {}).get('rho_max'),
                      primary_certified=(primary.get(top) or {}).get('certified_primary'),
                      evolved_percent=(primary.get(top) or {}).get('worst_evolved_percent'),
                      dense_evolved_percent=(by.get(f'q{top}_dense') or {}).get('worst_evolved_percent'),
                      m_for_bar_by_arm={f"{l['arm']}": l['m_for_bar'] for l in laws if l['q'] == top}),
        note=('deliverable: a primary-certified EQ rule at every rung AND the primary ladder '
              'monotone on the evolved metric with every rung converged and cheaper than its '
              'dense twin; otherwise the m at which the bar would be reached by the log-log law, '
              'and the hybrid ladder with dense at the uncertified rung, are reported'))
    # rho against evolved error at the top rung, over every EQ arm run there
    verdict['top_rung_rho_vs_evolved'] = sorted(
        [dict(arm=x['arm'], m=x['m'], rho_max=x['rho_max'], rho_p95=x['rho_p95'],
              evolved_percent=x['worst_evolved_percent'], certified_primary=x['certified_primary'],
              certified_tight=x['certified_tight'])
         for x in rows if x['q'] == top and x['quadrature'] == 'eq'],
        key=lambda d: (d['rho_max'] if d['rho_max'] is not None else 1e9))
    out.update(checks=checks, failed=sorted(fail), arms=rows, ladders=ladders, verdict=verdict,
               fidelity=fid)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows), rules=len(rules),
                          verdict={k: verdict[k] for k in ('every_rung_primary_certified',
                                                           'primary_ladder_monotone_evolved',
                                                           'primary_ladder_passes',
                                                           'tight_ladder_monotone_evolved',
                                                           'hybrid_ladder_monotone_evolved')}),
                     indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
