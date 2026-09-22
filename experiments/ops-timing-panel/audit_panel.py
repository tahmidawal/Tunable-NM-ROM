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
# DESIGN.md §5 fixes the gradient threshold of the convergence rule at 1e-6 for EVERY reduced
# subject, whatever tolerance the query was run at. Until 2026-09-19 (DESIGN §A13) this file
# evaluated the rule against each invocation's own `gtol`, so the 1e-3 arms were flagged
# converged at 1e-3; the Codex report audit caught it. `converged_design5` is now the primary
# flag and defines `admissible`; the as-implemented flag is kept as `converged_own_gtol`.
DESIGN5_GTOL = 1e-6
# name -> family, read from the lane's own checkpoint manifest.
try:
    OPERATOR_FAMILY = {c['name']: c['family']
                       for c in json.loads((HERE / 'operators.json').read_text())['checkpoints']}
except (OSError, KeyError, ValueError):       # an audit run outside the lane directory
    OPERATOR_FAMILY = {}
ADMISSIBILITY_RULE = dict(
    primary='converged_design5: DESIGN.md §5 as written — every time step g <= 1e-6 or residual-rule exit, '
            'initial fit g <= 1e-6 or relative residual <= 1e-10, every exit regular; plus rule certification / fast parity',
    secondary='converged_own_gtol: the same rule against each invocation\'s own gtol (1e-6 or 1e-3) — the flag the '
              'report carried before DESIGN §A13 (2026-09-19); kept as a labelled secondary column, defines nothing')
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
    # ops-timing-panel: the panel now carries SEVERAL operator arms (FNO, U-Net and
    # Transolver capacities), each written by its own `fno_panel.py` process in the one
    # allocation. `--fno-name` therefore takes a list; every name is scored by the same code
    # against the same same-job `fft_tight` solve. `fno_meta` keeps the first arm present so
    # the existing single-model summary block is unchanged; `operators` carries them all.
    p.add_argument('--fno-name', nargs='+', default=['fno-large'])
    p.add_argument('--operator-reps', type=int, default=5,
                   help='repetitions each operator arm must have retained per case')
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
    mr = r['gates'].get('matched_rule_files_bitwise')
    if mr is not None and mr.get('passed') is not None:
        gate('matched_rule_files_bitwise', bool(mr['passed']), mr,
             'two rule sets carrying the same rule file must give bitwise identical arms (DESIGN A8)')
    elif mr is not None:
        info('matched_rule_files_bitwise', None, mr, 'no two rule sets share a rule file in this job')
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
    if mr is not None and mr.get('pairs'):
        # the driver's matched-rule gate, recomputed from the SAVED fields of both arms
        art = {(x['name'], x['case']): x['artifact'] for x in inv}
        pairs = []
        for t in mr['pairs']:
            na, nb = t['arms']
            same = all(np.array_equal(F(art[(na, c)]), F(art[(nb, c)])) for c in sorted({x['case'] for x in inv}))
            pairs.append(dict(arms=t['arms'], q=t['q'], gtol=t['gtol'], saved_fields_identical=bool(same)))
        gate('matched_rule_files_bitwise_recomputed', all(t['saved_fields_identical'] for t in pairs), pairs)

    # ------------------------------------------------ the FNO, same code ------
    fno_rows, fno_meta, operator_meta = [], None, {}
    reps_ops = a.operator_reps
    for name in a.fno_name:
        tj = fields_dir / f'{name}-timing.json'
        if not tj.exists():
            info(f'operator_phase_present_{name}', False, str(tj),
                 'no timing file: this operator arm did not run or failed')
            continue
        meta = json.loads(tj.read_text())
        assert meta['model'] == name, (meta['model'], name)
        # The family is declared data, not a guess from the arm name: `don-*` fell through a
        # prefix rule to 'fno' and mislabelled every DeepONet row in the first opt201 report.
        fam = OPERATOR_FAMILY.get(name, 'fno')
        for c in meta['cases']:
            idx = int(c['case_index'])
            f = F(c['artifact'])
            # `kind` stays 'fno' -- it is this file's label for a direct multi-time operator
            # query, and the admissibility and scoring branches key on it. `family` carries the
            # actual architecture so the table does not call a U-Net an FNO (Codex audit #2c).
            for k, (dsec, hsec) in enumerate(zip(c['device_seconds'], c['host_seconds'])):
                x = dict(kind='fno', family=fam, name=name, case=idx, rep=k, gpu_seconds=float(dsec),
                         host_seconds=float(dsec + hsec), artifact=c['artifact'], iterations=[])
                score(x, f)
                fno_rows.append(x)
        # --- the operator-arm integrity gates (Codex design audit, findings 2a/2b/5a) --------
        # The `every_subject_case_has_all_reps` gate above covers the JAX subjects only, and
        # `zip` would silently truncate unequal device/host arrays, so the shape of every
        # operator arm is asserted here instead of assumed.
        gate(f'operator_cohort_and_reps_{name}',
             ({int(c['case_index']) for c in meta['cases']} == set(refs)) and
             all(len(c['device_seconds']) == len(c['host_seconds']) == reps_ops
                 for c in meta['cases']),
             dict(cases=sorted(int(c['case_index']) for c in meta['cases']), expected=sorted(refs),
                  reps=sorted({len(c['device_seconds']) for c in meta['cases']} |
                              {len(c['host_seconds']) for c in meta['cases']}), expected_reps=reps_ops),
             'every panel case present, device and host arrays the same length, and that length '
             'is the declared repetition count')
        # Not the recorded boolean: the operator's own saved t0 is compared, bitwise, with the
        # initial state of THIS panel's converged fft_tight solve. A stale field file, or a file
        # from a different cohort, fails here.
        gate(f'operator_returns_supplied_field_at_t0_{name}',
             all(np.array_equal(F(c['artifact'])[0], F(base[int(c['case_index'])])[0])
                 for c in meta['cases']) and all(c['t0_returned_exactly'] for c in meta['cases']))
        gate(f'operator_saved_fields_match_recorded_hash_{name}',
             all(hashlib.sha256(np.ascontiguousarray(F(c['artifact'])).tobytes()).hexdigest()
                 == c['field_sha256'] for c in meta['cases']))
        # The timing is admissible only if it happened on the GPU the JAX phase ran on.
        gate(f'operator_gpu_matches_panel_{name}',
             meta['environment'].get('gpu') == r.get('gpu'),
             dict(operator=meta['environment'].get('gpu'), panel=r.get('gpu')))
        info(f'operator_timed_in_same_allocation_{name}', True,
             dict(deviation=meta.get('deviation'), status=meta.get('scientific_status')))
        operator_meta[name] = dict(model=meta['model'], checkpoint_sha256=meta['checkpoint_sha256'],
                                   config=meta['config'], best_epoch=meta['best_epoch'],
                                   real_parameter_count=meta['real_parameter_count'],
                                   repetitions_per_case=meta['repetitions_per_case'],
                                   burn_in_per_block=meta['burn_in_per_block'],
                                   device_query_pooled=meta['device_query_pooled'],
                                   host_transfer_pooled=meta['host_transfer_pooled'])
        if fno_meta is None:
            fno_meta = meta
    if not operator_meta:
        info('operator_phase_present', False, None, 'no operator timing file at all')

    # ------------------------------------------------ convergence re-derived --
    def design5(x, gtol):
        """DESIGN.md §5 (i)-(iii) with gradient threshold `gtol`."""
        reasons = x['stop_reasons']
        gj = np.asarray(x['step_joint_stationarity'], dtype=float)
        step_ok = all(rr in (1, 2, 4) for rr in reasons)
        step_grad_ok = all((g <= gtol * (1 + 1e-7)) or (rr == 1) for g, rr in zip(gj, reasons))
        ic_ok = (x['ic_reason'] in (1, 2, 4)) and ((x['ic_joint_stationarity'] <= gtol * (1 + 1e-7))
                                                   or (x['ic_relative_residual'] <= IC_RESIDUAL_FLOOR))
        return bool(step_ok and step_grad_ok and ic_ok)

    dev = []
    for x in roms:
        gtol = x['gtol']
        step_ok = all(rr in (1, 2, 4) for rr in x['stop_reasons'])
        conv5 = design5(x, DESIGN5_GTOL)            # the pre-registered rule, fixed threshold
        conv_own = design5(x, gtol)                 # the rule against the query's own gtol (pre-A13 flag)
        strict = bool(x['worst_joint_stationarity'] <= gtol * (1 + 1e-7) and step_ok and x['ic_reason'] in (1, 2, 4))
        # the driver's own `converged` is the own-gtol flag; it and `converged_strict` must reproduce
        if conv_own != x['converged'] or strict != x['converged_strict']:
            dev.append(dict(name=x['name'], case=x['case'], rep=x['rep']))
        x['converged_design5_audit'], x['converged_own_gtol_audit'], x['converged_strict_audit'] = conv5, conv_own, strict
    gate('convergence_flags_reproduced', not dev, dev[:5],
         'the driver\'s converged / converged_strict flags (own-gtol rule) reproduce from the raw per-step quantities')
    info('design5_flag_differs_from_own_gtol',
         None, sorted({x['name'] for x in roms if x['converged_own_gtol_audit'] and not x['converged_design5_audit']}),
         'reduced subjects converged at their own gtol but not under DESIGN §5\'s fixed 1e-6 (DESIGN §A13)')

    # ------------------------------------------------ per-subject rows --------
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    table = {}
    for x in inv + fno_rows:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], family=x['family'], q=x.get('q'), k=x.get('k'), M=x.get('M'), m=x.get('m'),
            quadrature=x.get('quadrature'), gtol=x.get('gtol'), dt=x.get('dt'), solved_dimension=x.get('solved_dimension'),
            rule_kind=x.get('rule_kind'), gpu_ms=[], host_ms=[], case_ref={}, case_all={}, case_ev={}, case_t0={},
            per_time={}, iters=[], maxit=[], stat=[], stepstat=[], icstat=[], icrel=[], conv5=[], convo=[], strict=[], comp=[], be=[], exits=[],
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
            t['stepstat'].append(float(np.max(x['step_joint_stationarity'])))
            t['icstat'].append(x['ic_joint_stationarity'])
            t['icrel'].append(x['ic_relative_residual'])
            t['conv5'].append(x['converged_design5_audit'])
            t['convo'].append(x['converged_own_gtol_audit'])
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
        conv5 = bool(all(t['conv5'])) if t['conv5'] else None
        convo = bool(all(t['convo'])) if t['convo'] else None
        basis = rule.get('basis')
        certified = (basis in ('primary', 'secondary')) if t['quadrature'] == 'eq' else None

        def adm(conv):
            if t['kind'] in ('fom', 'fno'):
                return True
            if t['family'] == 'fast':
                return bool(conv) and fast_ok and (certified is not False)
            return bool(conv) and (certified is not False)
        admissible, admissible_own = adm(conv5), adm(convo)
        rows.append(dict(
            arm=name, kind=t['kind'], family=t['family'], q=t['q'], k=t['k'], M=t['M'], m=t['m'],
            quadrature=t['quadrature'], gtol=t['gtol'], dt=t['dt'], solved_dimension=t['solved_dimension'],
            rule_kind=t['rule_kind'], rule_basis=basis, rule_m=rule.get('m'), rho_max=rule.get('rho_max', rule.get('source_rho_max')),
            rule_set=rule.get('rule_set'), rule_status=rule.get('construction_status'), rule_export_basis=rule.get('export_basis'),
            rule_source_lane=rule.get('source_lane'), rule_source_attempt=rule.get('source_attempt'),
            rule_fit_states=rule.get('source_fit_states'), rule_file=rule.get('file'), rule_file_sha256=rule.get('sha256'),
            rule_construction_note=rule.get('construction_note'),
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
            max_step_stationarity=(float(np.max(t['stepstat'])) if t['stepstat'] else None),
            max_ic_stationarity=(float(np.max(t['icstat'])) if t['icstat'] else None),
            max_ic_relative_residual=(float(np.max(t['icrel'])) if t['icrel'] else None),
            converged_design5=conv5, converged_own_gtol=convo,
            converged_strict=(bool(all(t['strict'])) if t['strict'] else None),
            completed=(bool(all(t['comp'])) if t['comp'] else None),
            total_budget_exits=(int(np.sum(t['be'])) if t['be'] else None),
            exit_reason_counts=({str(k): int(v) for k, v in zip(*np.unique(t['exits'], return_counts=True))} if t['exits'] else None),
            # `admissible` is defined by DESIGN §5 (converged_design5); the pre-A13 flag is kept beside it
            admissible=bool(admissible), admissible_design5=bool(admissible), admissible_own_gtol=bool(admissible_own)))
    for row in rows:
        row.pop('median_of_case_median_gpu_ms', None)
    floors = {}
    for e in r['reconstruction']:
        key = ('pod', e.get('k')) if e.get('family') == 'pod' else ('rom', e.get('q'))
        floors[key] = dict(best_found_percent=100 * e['worst_best_found'],
                           bank_projection_percent=(100 * e['worst_bank_projection']
                                                    if 'worst_bank_projection' in e else None))
    for x in rows:
        if x['family'] == 'pod':
            fl = floors.get(('pod', x['k'])) or {}
            best = fl.get('best_found_percent')
        elif x['family'] in ('rom', 'fast'):
            fl = floors.get(('rom', x['q'])) or {}
            best = fl.get('best_found_percent')
        elif x['family'] == 'free':
            # the unrestricted bank solves ALL R coefficients, so its representation floor is
            # the bank projection itself, not a head manifold
            fl = floors.get(('rom', 0)) or {}
            best = fl.get('bank_projection_percent')
        else:
            fl, best = {}, None
        x['best_found_percent'] = best
        x['best_found_kind'] = ('bank projection (all R coefficients)' if x['family'] == 'free'
                                else ('POD span projection' if x['family'] == 'pod'
                                      else ('head manifold best-found' if best is not None else None)))
        x['bank_projection_percent'] = fl.get('bank_projection_percent')
        x['solved_over_best_found'] = (x['worst_all_times_percent'] / best if best else None)
    by = {x['arm']: x for x in rows}
    rows.sort(key=lambda x: ({'rom': 0, 'fast': 1, 'pod': 2, 'free': 3, 'fno': 4, 'unet': 4.3,
                              'transolver': 4.6, 'fom': 5}.get(x['family'], 9),
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
                    converged_design5=[x['converged_design5'] for x in lad], converged_own_gtol=[x['converged_own_gtol'] for x in lad],
                    admissible=[x['admissible'] for x in lad], admissible_own_gtol=[x['admissible_own_gtol'] for x in lad],
                    monotone_evolved=mono(ev), monotone_all_times=mono(al),
                    all_converged=bool(all(x['converged_design5'] for x in lad)),
                    all_converged_own_gtol=bool(all(x['converged_own_gtol'] for x in lad)),
                    nondominated_converged_points=len(ndr),
                    error_span=(max(x['worst_evolved_percent'] for x in ndr) / max(min(x['worst_evolved_percent'] for x in ndr), 1e-300) if ndr else None),
                    cost_span=(max(x['median_gpu_ms'] for x in ndr) / max(min(x['median_gpu_ms'] for x in ndr), 1e-300) if ndr else None))

    g6 = f"g{cfg['strict']['gtol']:g}".replace('-', 'm').replace('.', 'p')
    ladders = dict(
        dense=ladder(lambda x: x['family'] == 'rom' and x['quadrature'] == 'dense' and x['M'] == 4 * (K + x['q']) and x['arm'].endswith(g6), lambda x: x['q']))
    # one EQ ladder per (rule set, tolerance): a job may carry several sets (DESIGN A5.1 / A8)
    for kind in sorted({x['rule_kind'] for x in rows if x['family'] == 'rom' and x['quadrature'] == 'eq' and x['rule_kind']}):
        for g in sorted({x['gtol'] for x in rows if x['family'] == 'rom' and x['quadrature'] == 'eq'}, reverse=True):
            gs = f'g{g:g}'.replace('-', 'm').replace('.', 'p')
            ladders[f'eq_{kind}_{gs}'] = ladder(lambda x, kind=kind, g=g: x['family'] == 'rom' and x['quadrature'] == 'eq'
                                                and x['rule_kind'] == kind and x['gtol'] == g, lambda x: x['q'])
    ladders['pod'] = ladder(lambda x: x['family'] == 'pod', lambda x: x['k'])

    # ------------------------------------------------ non-dominated sets ------
    nd = {}
    reduced_fams = {'rom', 'fast', 'pod', 'free'}
    for tag, (ck_, ek) in PAIRS.items():
        adm = [x for x in rows if x['admissible']]                 # DESIGN §5 (primary)
        adm_own = [x for x in rows if x['admissible_own_gtol']]    # pre-A13 flag (secondary, labelled)
        nd[tag] = dict(cost=ck_, error=ek, all=nondominated(rows, ck_, ek),
                       admissible=nondominated(adm, ck_, ek),
                       admissible_own_gtol=nondominated(adm_own, ck_, ek),
                       # DESIGN.md A4, added after seeing job 1: the frontier among REDUCED
                       # subjects only, which is what "does the nonlinear manifold beat POD"
                       # asks. It changes no pre-registered criterion and is labelled post-hoc.
                       reduced_only=nondominated([x for x in adm if x['family'] in reduced_fams], ck_, ek),
                       reduced_only_own_gtol=nondominated([x for x in adm_own if x['family'] in reduced_fams], ck_, ek))
    # the FOM controls' own reference error (the mesh's discretisation error) for context
    disc = {x['arm']: x['worst_reference_percent'] for x in rows if x['family'] == 'fom'}

    parts = Path(a.result).resolve().parts
    attempt = parts[parts.index('runs') + 1] if 'runs' in parts else cfg.get('attempt')
    out = dict(result=str(Path(a.result).resolve()), reference_mesh=cfg.get('reference_mesh'),
               result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'), attempt=attempt, intervals=L, dt=dt,
               admissibility_rule=ADMISSIBILITY_RULE, design5_gtol=DESIGN5_GTOL,
               mem_fraction=r.get('mem_fraction'), elapsed_seconds=r.get('elapsed_seconds'), K=K, R=r['R'],
               output_times=r['output_times'], checks=checks, failed=sorted(fail), arms=rows, ladders=ladders,
               nondominated=nd, fom_discretisation_error_percent=disc, dropped=r['dropped'], rules=r['rules'],
               transfer=r.get('transfer', []), directions=r['directions'], reconstruction=r['reconstruction'],
               operators=operator_meta or None,
               fno=fno_meta and dict(model=fno_meta['model'], checkpoint_sha256=fno_meta['checkpoint_sha256'],
                                     real_parameter_count=fno_meta['real_parameter_count'],
                                     device_query_pooled=fno_meta['device_query_pooled'],
                                     host_transfer_pooled=fno_meta['host_transfer_pooled'], deviation=fno_meta.get('deviation')),
               compile_warmup=r.get('compile_warmup'), timed_subjects=r.get('timed_subjects'))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(failed=sorted(fail), arms=len(rows), dropped=len(r['dropped']),
                          nondominated_gpu_evolved_admissible=nd['gpu_evolved']['admissible'],
                          nondominated_gpu_evolved_admissible_own_gtol=nd['gpu_evolved']['admissible_own_gtol']), indent=2))
    print('AUDIT WROTE', a.out)


if __name__ == '__main__':
    main()
