"""Independent NumPy audit of one panel job. No JAX, no GPU, no driver import.

    python audit_panel.py <result.json> --fields <dir> --out <audit.json> [--comparators <json>]

Recomputes every reported error from the retained output fields against the job's own
converged `fft_tight` solve and against the finer independent reference, re-derives the
convergence flags from the raw per-step quantities the driver recorded, evaluates every gate,
the cross-job fidelity gates, the ladders, and the non-dominated sets of DESIGN.md section 7.
The trained FNO's fields and timings, written by `fno_panel.py` in the same allocation, are
scored with the same code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IC_RESIDUAL_FLOOR = 1e-10
PAIRS = {'gpu_all': ('median_gpu_ms', 'worst_all_times_percent'),
         'gpu_evolved': ('median_gpu_ms', 'worst_evolved_percent'),
         'complete_all': ('median_host_ms', 'worst_all_times_percent'),
         'complete_evolved': ('median_host_ms', 'worst_evolved_percent')}


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def nondominated(rows, cost_key, err_key):
    pts = [(r['arm'], r[cost_key], r[err_key]) for r in rows if r.get(cost_key) is not None and r.get(err_key) is not None]
    keep = []
    for a, c, er in pts:
        dominated = any((c2 <= c and e2 <= er and (c2 < c or e2 < er)) for b, c2, e2 in pts if b != a)
        if not dominated:
            keep.append(a)
    order = {a: (c, er) for a, c, er in pts}
    return sorted(keep, key=lambda a: order[a][0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--fields', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--comparators', default=str(HERE / 'checks/comparators.json'))
    p.add_argument('--fno-name', default='fno-large')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    cfg = r['config']
    fields_dir = Path(a.fields)
    checks, fail = {}, []

    def gate(name, ok, detail=None, note=None):
        checks[name] = dict(passed=bool(ok), detail=detail, note=note)
        if not ok:
            fail.append(name)

    def info(name, ok, detail=None, note=None):
        checks[name] = dict(passed=(None if ok is None else bool(ok)), detail=detail, note=note, blocking=False)

    L, dt = r['intervals'], r['dt']
    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('step_budget_600', cfg['strict']['step_budget'] == 600, cfg['strict'])
    cg = r['gates'].get('evaluation_cohort_bitwise_abl01', {})
    gate('evaluation_cohort_bitwise_abl01', cg.get('passed') in (True, None), cg)
    for g in ('directions_file_sha256', 'directions_prefix_hashes', 'repetition_output_identical',
              'fft_tight_converged_everywhere'):
        gate(g, bool(r['gates'].get(g, {}).get('passed')), r['gates'].get(g))
    for g in ('direct_reproduces_fft_tight', 'fast_parity', 'fno_cohort_disjoint_from_training', 'transfer_fit_cert_disjoint'):
        if g in r['gates']:
            gate(g, bool(r['gates'][g].get('passed')), r['gates'][g])
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    gate('every_rule_hash_matches_provenance',
         all(x['nodes_sha256_matches'] and x['weights_sha256_matches'] for x in r['rules']))
    info('no_subject_dropped', not r['dropped'], r['dropped'], 'a drop is a reported finding, not a failure of the job')

    inv = r['invocations']
    reps = cfg['reps']
    counts = {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms))
    gate('every_rom_carries_exit_and_stationarity', all('stop_reasons' in x and 'budget_exits' in x for x in roms))
    gate('host_time_covers_gpu_time', all(x['host_seconds'] >= x['gpu_seconds'] for x in inv))

    # ------------------------------------------------ errors from fields ------
    cache = {}

    def F(name):
        if name not in cache:
            cache[name] = np.load(fields_dir / name)['fields']
        return cache[name]

    missing = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
    gate('artifacts_present', not missing, missing[:5])
    refs = {x['case']: F(x['artifact']) for x in r['reference']}
    tight = cfg.get('same_grid_reference', 'fft_tight')
    base = {x['case']: x['artifact'] for x in inv if x['name'] == tight}
    gate('same_grid_baseline_present', len(base) == len(refs))

    def score(x, f):
        truth = refs[x['case']]
        n0 = np.linalg.norm(truth[0])
        err = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
        sg = np.linalg.norm((f - F(base[x['case']])).reshape(len(f), -1), axis=1) / n0
        x['reference_per_time'] = err.tolist()
        x['reference_all'] = float(np.max(err))
        x['same_grid_per_time'] = sg.tolist()
        x['same_grid_all'] = float(np.max(sg))
        x['same_grid_evolved'] = float(np.max(sg[1:]))
        x['t0_compression'] = float(sg[0])

    worst = 0.
    for x in inv:
        f = F(x['artifact'])
        score(x, f)
        worst = max(worst, abs(x['reference_all'] - x['error']['fixed_initial_max']) / max(x['error']['fixed_initial_max'], 1e-300))
    gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    # ------------------------------------------------ the FNO, same code ------
    fno_rows, fno_meta = [], None
    tj = fields_dir / f'{a.fno_name}-timing.json'
    if tj.exists():
        fno_meta = json.loads(tj.read_text())
        for c in fno_meta['cases']:
            idx = int(c['case_index'])
            f = F(c['artifact'])
            for k, (dsec, hsec) in enumerate(zip(c['device_seconds'], c['host_seconds'])):
                x = dict(kind='fno', family='fno', name=a.fno_name, case=idx, rep=k, gpu_seconds=float(dsec),
                         host_seconds=float(dsec + hsec), artifact=c['artifact'], iterations=[])
                score(x, f)
                fno_rows.append(x)
        gate('fno_returns_supplied_field_at_t0', all(c['t0_returned_exactly'] for c in fno_meta['cases']))
        info('fno_timed_in_same_allocation', True, dict(deviation=fno_meta.get('deviation'), status=fno_meta.get('scientific_status')))
    else:
        info('fno_phase_present', False, str(tj), 'no FNO timing file: the FNO phase did not run or failed')

    # ------------------------------------------------ convergence re-derived --
    dev = []
    for x in roms:
        reasons = x['stop_reasons']
        gj = np.asarray(x['step_joint_stationarity'], dtype=float)
        gtol = x['gtol']
        step_ok = all(rr in (1, 2, 4) for rr in reasons)
        step_grad_ok = all((g <= gtol * (1 + 1e-7)) or (rr == 1) for g, rr in zip(gj, reasons))
        ic_ok = (x['ic_reason'] in (1, 2, 4)) and ((x['ic_joint_stationarity'] <= gtol * (1 + 1e-7))
                                                   or (x['ic_relative_residual'] <= IC_RESIDUAL_FLOOR))
        conv = bool(step_ok and step_grad_ok and ic_ok)
        strict = bool(x['worst_joint_stationarity'] <= gtol * (1 + 1e-7) and step_ok and x['ic_reason'] in (1, 2, 4))
        if conv != x['converged'] or strict != x['converged_strict']:
            dev.append(dict(name=x['name'], case=x['case'], rep=x['rep']))
        x['converged_audit'], x['converged_strict_audit'] = conv, strict
    gate('convergence_flags_reproduced', not dev, dev[:5])

    # ------------------------------------------------ per-subject rows --------
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    table = {}
    for x in inv + fno_rows:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x['family'], q=x.get('q'), k=x.get('k'), M=x.get('M'), m=x.get('m'),
            quadrature=x.get('quadrature'), gtol=x.get('gtol'), dt=x.get('dt'), solved_dimension=x.get('solved_dimension'),
            rule_kind=x.get('rule_kind'), gpu_ms=[], host_ms=[], case_ref={}, case_all={}, case_ev={}, case_t0={},
            per_time={}, iters=[], maxit=[], stat=[], icstat=[], icrel=[], conv=[], strict=[], comp=[], be=[], exits=[],
            ntol=x.get('ntol'), ltol=x.get('ltol'), preconditioner=x.get('preconditioner'), newton=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_ref'][x['case']] = x['reference_all']
        t['case_all'][x['case']] = x['same_grid_all']
        t['case_ev'][x['case']] = x['same_grid_evolved']
        t['case_t0'][x['case']] = x['t0_compression']
        t['per_time'][x['case']] = x['same_grid_per_time']
        t['iters'] += list(x.get('iterations', []))
        if x['kind'] == 'rom':
            t['maxit'].append(x['max_iterations'])
            t['stat'].append(x['worst_joint_stationarity'])
            t['icstat'].append(x['ic_joint_stationarity'])
            t['icrel'].append(x['ic_relative_residual'])
            t['conv'].append(x['converged_audit'])
            t['strict'].append(x['converged_strict_audit'])
            t['comp'].append(x['completed'])
            t['be'].append(x['budget_exits'])
            t['exits'] += x['stop_reasons']
        if x['kind'] == 'fom':
            t['newton'].append(x['newton_iterations_total'])
    fast_ok = bool(r['gates'].get('fast_parity', {}).get('passed'))
    rows = []
    for name, t in table.items():
        s = setup.get(name, {})
        rule = s.get('rule') or {}
        conv = bool(all(t['conv'])) if t['conv'] else None
        basis = rule.get('basis')
        certified = (basis in ('primary', 'secondary')) if t['quadrature'] == 'eq' else None
        if t['kind'] in ('fom', 'fno'):
            admissible = True
        elif t['family'] == 'fast':
            admissible = bool(conv) and fast_ok and (certified is not False)
        else:
            admissible = bool(conv) and (certified is not False)
        rows.append(dict(
            arm=name, kind=t['kind'], family=t['family'], q=t['q'], k=t['k'], M=t['M'], m=t['m'],
            quadrature=t['quadrature'], gtol=t['gtol'], dt=t['dt'], solved_dimension=t['solved_dimension'],
            rule_kind=t['rule_kind'], rule_basis=basis, rule_m=rule.get('m'), rho_max=rule.get('rho_max', rule.get('source_rho_max')),
            rho_p95=rule.get('rho_p95', rule.get('source_rho_p95')), certified_primary=rule.get('certified_primary'),
            certified_secondary=rule.get('certified_secondary'), rule_source_job=rule.get('source_job'),
            ntol=t['ntol'], ltol=t['ltol'], preconditioner=t['preconditioner'],
            worst_reference_percent=100 * float(np.max(list(t['case_ref'].values()))),
            median_reference_percent=100 * median(list(t['case_ref'].values())),
            worst_all_times_percent=100 * float(np.max(list(t['case_all'].values()))),
            median_all_times_percent=100 * median(list(t['case_all'].values())),
            worst_evolved_percent=100 * float(np.max(list(t['case_ev'].values()))),
            median_evolved_percent=100 * median(list(t['case_ev'].values())),
            worst_t0_compression_percent=100 * float(np.max(list(t['case_t0'].values()))),
            per_case_evolved_percent={str(k): 100 * v for k, v in sorted(t['case_ev'].items())},
            per_case_all_times_percent={str(k): 100 * v for k, v in sorted(t['case_all'].items())},
            per_time_same_grid_percent={str(k): [100 * z for z in v] for k, v in sorted(t['per_time'].items())},
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            median_of_case_median_gpu_ms=median([median([g for g, x in zip(t['gpu_ms'], inv + fno_rows) if False] or [0])]) if False else None,
            gpu_ms_repetitions=len(t['gpu_ms']), gpu_ms_all=t['gpu_ms'],
            median_iterations=(median(t['iters']) if t['iters'] else None),
            max_iterations=(int(max(t['maxit'])) if t['maxit'] else None),
            newton_iterations_median=(median(t['newton']) if t['newton'] else None),
            max_joint_stationarity=(float(np.max(t['stat'])) if t['stat'] else None),
            max_ic_stationarity=(float(np.max(t['icstat'])) if t['icstat'] else None),
            max_ic_relative_residual=(float(np.max(t['icrel'])) if t['icrel'] else None),
            converged=conv, converged_strict=(bool(all(t['strict'])) if t['strict'] else None),
            completed=(bool(all(t['comp'])) if t['comp'] else None),
            total_budget_exits=(int(np.sum(t['be'])) if t['be'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in zip(*np.unique(t['exits'], return_counts=True))} if t['exits'] else None),
            admissible=bool(admissible)))
    for row in rows:
        row.pop('median_of_case_median_gpu_ms', None)
    by = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: ({'rom': 0, 'fast': 1, 'pod': 2, 'free': 3, 'fno': 4, 'fom': 5}.get(x['family'], 9),
                             x['q'] if x['q'] is not None else (x['k'] or -1), x['arm']))

    # ------------------------------------------------ fidelity gates ----------
    cmp_table = json.loads(Path(a.comparators).read_text()) if Path(a.comparators).exists() else {}
    fid = {}
    for arm, spec in (cfg.get('fidelity_expectations') or {}).items():
        mine = by.get(arm)
        theirs = (cmp_table.get(spec['source']) or {}).get('arms', {}).get(spec['arm'])
        if mine is None or theirs is None:
            fid[arm] = dict(passed=False, detail=dict(reason='arm or comparator missing', dropped=[d['name'] for d in r['dropped']]))
            gate(f'reproduces_{arm}', False, fid[arm]['detail'])
            continue
        d = dict(source=spec['source'], comparator=spec['arm'], job=cmp_table[spec['source']].get('job_id'),
                 first_tier=spec['tolerance'], second_tier=spec.get('second_tier_tolerance'))
        worst_rel = 0.
        for key in ('worst_all_times_percent', 'worst_evolved_percent', 'worst_reference_percent'):
            mv, tv = mine[key], theirs.get(key)
            relv = (None if tv in (None, 0) else abs(mv - tv) / abs(tv))
            d[key] = dict(ours=mv, theirs=tv, relative_difference=relv)
            worst_rel = max(worst_rel, relv or 0.)
        first = worst_rel <= spec['tolerance']
        second = spec.get('second_tier_tolerance') is not None and worst_rel <= spec['second_tier_tolerance']
        d.update(worst_relative_difference=worst_rel, passed_first_tier=bool(first), passed_second_tier=bool(second),
                 tier=('first' if first else 'second' if second else 'none'))
        fid[arm] = dict(passed=bool(first or second), detail=d)
        gate(f'reproduces_{arm}', first or second, d)
    checks['cross_job_fidelity'] = dict(passed=(all(v['passed'] for v in fid.values()) if fid else None), detail=fid)

    # ------------------------------------------------ ladders -----------------
    K = r['K']

    def ladder(pred, key):
        lad = sorted([x for x in rows if pred(x)], key=key)
        if not lad:
            return None
        ev = [x['worst_evolved_percent'] for x in lad]
        al = [x['worst_all_times_percent'] for x in lad]
        cost = [x['median_gpu_ms'] for x in lad]
        adm = [x for x in lad if x['admissible']]
        nd = nondominated(adm, 'median_gpu_ms', 'worst_evolved_percent') if adm else []
        ndr = [x for x in adm if x['arm'] in nd]
        return dict(arms=[x['arm'] for x in lad], q_or_k=[x['q'] if x['q'] is not None else x['k'] for x in lad],
                    worst_evolved_percent=ev, worst_all_times_percent=al, median_gpu_ms=cost,
                    converged=[x['converged'] for x in lad], admissible=[x['admissible'] for x in lad],
                    monotone_evolved=mono(ev), monotone_all_times=mono(al), all_converged=bool(all(x['converged'] for x in lad)),
                    nondominated_converged_points=len(ndr),
                    error_span=(max(x['worst_evolved_percent'] for x in ndr) / max(min(x['worst_evolved_percent'] for x in ndr), 1e-300) if ndr else None),
                    cost_span=(max(x['median_gpu_ms'] for x in ndr) / max(min(x['median_gpu_ms'] for x in ndr), 1e-300) if ndr else None))

    g6 = f"g{cfg['strict']['gtol']:g}".replace('-', 'm').replace('.', 'p')
    ladders = dict(
        dense=ladder(lambda x: x['family'] == 'rom' and x['quadrature'] == 'dense' and x['M'] == 4 * (K + x['q']) and x['arm'].endswith(g6), lambda x: x['q']),
        eq_g1em06=ladder(lambda x: x['family'] == 'rom' and x['quadrature'] == 'eq' and x['arm'].endswith(g6), lambda x: x['q']),
        eq_g0p001=ladder(lambda x: x['family'] == 'rom' and x['quadrature'] == 'eq' and x['arm'].endswith('g0p001'), lambda x: x['q']),
        pod=ladder(lambda x: x['family'] == 'pod', lambda x: x['k']))

    # ------------------------------------------------ non-dominated sets ------
    nd = {}
    reduced_fams = {'rom', 'fast', 'pod', 'free'}
    for tag, (ck_, ek) in PAIRS.items():
        adm = [x for x in rows if x['admissible']]
        nd[tag] = dict(cost=ck_, error=ek, all=nondominated(rows, ck_, ek),
                       admissible=nondominated(adm, ck_, ek),
                       # DESIGN.md A4, added after seeing job 1: the frontier among REDUCED
                       # subjects only, which is what "does the nonlinear manifold beat POD"
                       # asks. It changes no pre-registered criterion and is labelled post-hoc.
                       reduced_only=nondominated([x for x in adm if x['family'] in reduced_fams], ck_, ek))
    # the FOM controls' own reference error (the mesh's discretisation error) for context
    disc = {x['arm']: x['worst_reference_percent'] for x in rows if x['family'] == 'fom'}

    parts = Path(a.result).resolve().parts
    attempt = parts[parts.index('runs') + 1] if 'runs' in parts else cfg.get('attempt')
    out = dict(result=str(Path(a.result).resolve()), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'), attempt=attempt, intervals=L, dt=dt,
               mem_fraction=r.get('mem_fraction'), elapsed_seconds=r.get('elapsed_seconds'), K=K, R=r['R'],
               output_times=r['output_times'], checks=checks, failed=sorted(fail), arms=rows, ladders=ladders,
               nondominated=nd, fom_discretisation_error_percent=disc, dropped=r['dropped'], rules=r['rules'],
               transfer=r.get('transfer', []), directions=r['directions'], reconstruction=r['reconstruction'],
               fno=fno_meta and dict(model=fno_meta['model'], checkpoint_sha256=fno_meta['checkpoint_sha256'],
                                     real_parameter_count=fno_meta['real_parameter_count'],
                                     device_query_pooled=fno_meta['device_query_pooled'],
                                     host_transfer_pooled=fno_meta['host_transfer_pooled'], deviation=fno_meta.get('deviation')),
               compile_warmup=r.get('compile_warmup'), timed_subjects=r.get('timed_subjects'))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows), dropped=len(r['dropped']),
                          nondominated_gpu_evolved_admissible=nd['gpu_evolved']['admissible']), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
