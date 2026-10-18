"""Offline picks and pair labels for one cell (DESIGN 6, 9, A1.6, A1.10-A1.12, A2.12, A2.17, A4.8, A5.3, A22.0,
A26.5), from the Stage-B job's data only. CPU, numpy. Writes checks/VERDICT-<cell>.json (+ .sha256).

Per admitted X (G-ours passed in the tuning and Stage-B jobs):
  - candidate set = Stage-B members Q1-qualified on the verdict cohort whose A-B-A timing against X is complete with
    every timed output equal to its scored output (<= 1e-12);
  - errors on the eligible case-kinds of the verdict cohort (ineligible ones shown, never ranked); W = worst, med =
    median; time = T (median over the 6 timing cases of the per-case median valid ratio t_B / mean(t_A1, t_A2));
  - roles B_acc / B_cost from the X-specific front in (T, W) (stats.picks); pair labels by stats.pair_label for the
    picks (registered) and for every other candidate whose worst envelope is not entirely above X's (exploratory);
  - cell flags carried into every line: adequacy / saturation, ECSW quality, budget relation (A2.5 / A25.2), 3D
    batch-1 diagnostic-only (A22.0).

Usage: python analyze.py <cell> <tune_output_dir> <stageb_output_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stats as ST  # noqa: E402

TIMED_TOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rho_of(cases_rec, reps=3, max_retry=3):
    """A32.9: the timing contract recomputed from raw evidence. Per timing case: exactly `reps` repeats, attempts
    numbered 0..k contiguously with k <= max_retry, a triple is valid iff all three times are finite and positive,
    |tA1/tA2 - 1| <= 0.25 (drift recomputed, the stored flag is ignored) and every output discrepancy is finite and
    <= 1e-12; a valid attempt must end its repeat. Returns (rho, problems)."""
    out, bad = {}, []
    for t, rc in cases_rec.items():
        vals, by = [], {}
        for tr in rc['triples']:
            by.setdefault(tr['rep'], []).append(tr)
        if sorted(by) != list(range(reps)):
            bad.append((t, 'repeat inventory'))
        for rep, ats in by.items():
            if [a['attempt'] for a in ats] != list(range(len(ats))) or len(ats) > max_retry + 1:
                bad.append((t, rep, 'attempt inventory'))
            for k, tr in enumerate(ats):
                ts = (tr['tA1'], tr['tB'], tr['tA2'])
                ok = all(np.isfinite(x) and x > 0 for x in ts) and abs(tr['tA1'] / tr['tA2'] - 1.) <= 0.25
                outs = all(x is not None and np.isfinite(x) and x <= TIMED_TOL for x in (tr['dA1'], tr['dB'], tr['dA2']))
                if not outs:
                    bad.append((t, rep, k, 'output check'))
                if ok and k < len(ats) - 1:
                    bad.append((t, rep, k, 'valid attempt not last'))
                if (not ok) and k == len(ats) - 1 and len(ats) < max_retry + 1:
                    bad.append((t, rep, k, 'invalid final attempt before retries exhausted'))
                if ok and outs:
                    vals.append(tr['tB'] / (0.5 * (tr['tA1'] + tr['tA2'])))
        out[int(t.split('|')[1])] = vals
    return out, bad


T_PHASES = ('basis', 'saturation', 'adequacy', 'adv_pod', 'generator_probe_basis', 'ecsw_states', 'ecsw_r')
V_TUNE = ('A|', 'EXT|', 'generator_probe_runs', 'ours_pass')
V_STB = ('B|', 'aba', 'ours_pass')


def phase_sum(phases, prefixes):
    return sum(v['seconds'] for k, v in phases.items() if k.startswith(prefixes)) / 3600


def accounting(T, B):
    """A3.1/A35.2: non-overlapping T (training-only construction) and V (validation-informed) GPU-hours from the
    checkpointed phase timers of both jobs; comparator qualification (G-ours replays, warm timings) is V; the
    full-rank limiting comparator is reported apart. A job that did not finish has an unfinished active phase: its
    expenditure is bounded below by the recorded phases and above by the job's measured allocation (ledger)."""
    out = dict(T_hours=phase_sum(T['phases'], T_PHASES),
               V_hours_tuning=phase_sum(T['phases'], V_TUNE), V_hours_stageB=phase_sum(B['phases'], V_STB),
               limiting_comparator_hours=phase_sum(T['phases'], ('fom_limiting',)),
               tuning_finished='total_seconds' in T, stageB_finished='total_seconds' in B,
               active_phase=dict(tuning=T.get('active_phase'), stageB=B.get('active_phase')))
    out['V_hours'] = out['V_hours_tuning'] + out['V_hours_stageB']
    out['V_is_lower_bound'] = not (out['tuning_finished'] and out['stageB_finished'])
    # A36.2: upper bound = the measured allocations of the two attempts in the ledger (unknown if not collected)
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    alloc = {}
    for role, att in (('tuning', ATTEMPTS.get('tune')), ('stageB', ATTEMPTS.get('stageb'))):
        rec = [j for j in led['jobs'] if j['attempt'] == att] if att else []
        alloc[role] = (float(rec[0]['measured_gpu_hours']) if rec and rec[0].get('measured_gpu_hours') is not None
                       else None)
    out['allocation_hours'] = alloc
    out['V_upper_bound_hours'] = (None if None in alloc.values() else alloc['tuning'] + alloc['stageB'])
    return out


ATTEMPTS = {}


def main():
    cell, tdir, bdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    for a in sys.argv:                                 # --tune-attempt=<name> --stageb-attempt=<name> (ledger binding)
        if a.startswith('--tune-attempt='):
            ATTEMPTS['tune'] = a.split('=', 1)[1]
        if a.startswith('--stageb-attempt='):
            ATTEMPTS['stageb'] = a.split('=', 1)[1]
    T = json.loads((tdir / 'tune.json').read_text())
    B = json.loads((bdir / 'stageb.json').read_text())
    fz = HERE / B['config']['frozen']
    assert sha(fz) == B['config']['frozen_sha256']
    FZ = json.loads(fz.read_text())
    assert sha(tdir / 'tune.json') == FZ['tune_json_sha256'], 'tuning JSON differs from the frozen one'   # A32.9
    d, L_ = T['config']['d'], T['config']['L']
    if d == 2:
        exp = json.loads((HERE / 'inputs' / 'p1_analysis.json').read_text())['configurations']
        eX_all = {k: exp[k]['e0_per_case'] for k in FZ['X']}
    else:
        exp = json.loads((HERE / 'inputs' / 'speed3d_batch1_e0.json').read_text())['configs']
        eX_all = {k: {f'val64|{j}': v for j, v in exp[k]['e0'].items()} for k in FZ['X']}
    members = B['members']
    elig = list(B['eligible_cases'])                                     # A32.1: the store's fixed mask
    eps = np.array([B['eps_ref'][c] for c in elig])                      # A35.1: from the store, not from any run
    # ---- coverage: registered tuning inventory, frozen members, comparators (A28.5, A35.1) ----------------------------
    inv = T.get('inventory') or {}
    tune_unfinished = sorted(c for c, v in inv.items() if v['status'] in ('not_run', 'run_failed'))
    tune_constr = sorted(c for c, v in inv.items() if v['status'] == 'construction_failed')
    tune_deferred = sorted(c for c, v in inv.items() if v['status'] == 'deferred')
    missing_members = {m: (B['inventory'].get(m) or {}).get('status', 'absent') for m in FZ['members']
                       if (B['inventory'].get(m) or {}).get('status') != 'complete'}       # replay failures included
    expected_X = [k for k, x in FZ['X'].items() if 'front' in x]
    # ---- saturation / adequacy (A1.3, A2.2): unavailable = unassessed = withheld ---------------------------------------
    sat256 = None
    if d == 2 and L_ == 1024:
        sp = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--sat256=')]
        assert sp, '--sat256=<tune.json of the 256^2 cell>:<sha256> is required for the 1024^2 analysis'
        path, h = sp[0].rsplit(':', 1)
        assert sha(path) == h, 'saturation source differs from its pinned hash'
        T256 = json.loads(Path(path).read_text())
        assert T256['config']['d'] == 2 and T256['config']['L'] == 256 and T256['config'].get('saturation'), 'not 256^2'
        sat256 = (T256.get('saturation') or {}).get('adequate')
    elif d == 2:
        sat256 = (T.get('saturation') or {}).get('adequate')
    adequate = (T.get('adequacy') or {}).get('adequate')
    adequate, sat256 = (None if adequate is None else bool(adequate)), (None if sat256 is None else bool(sat256))
    cf = T.get('construction_failures') or {}
    ecsw_bad = [c for c in inv if c.startswith('ecsw|') and inv[c]['status'] != 'complete']
    limited = sorted({str(v.get('quality')) for v in T['stageA'].values() if v.get('kind') == 'ecsw'} - {'ok', 'None'})
    acct = accounting(T, B)
    restrictions = []
    if adequate is not True:
        restrictions.append('basis adequacy ' + ('failed' if adequate is False else 'unassessed') + ' (A2.2)')
    if d == 2 and sat256 is not True:
        restrictions.append(('snapshot-limited' if L_ == 1024 else 'saturation') + ' check ' +
                            ('failed' if sat256 is False else 'unassessed') + ' (A1.3)')
    if tune_unfinished:
        restrictions.append(f'tuning coverage incomplete ({len(tune_unfinished)} registered configurations unfinished)')
    if missing_members:
        restrictions.append(f'Stage B incomplete ({len(missing_members)} frozen members without valid results: '
                            f'{sorted(set(missing_members.values()))})')
    if ecsw_bad or any(k.startswith('ecsw|') for k in cf):
        restrictions.append('ECSW coverage incomplete (A3.11)')
    if limited:
        restrictions.append(f'ECSW construction quality-limited: {limited} (A16.5/A24.3)')
    flags = dict(adequate=adequate, saturation_256=sat256, basis_used=T.get('basis_used'),
                 restrictions=restrictions, tune_unfinished=tune_unfinished, tune_construction_failed=tune_constr,
                 tune_deferred=tune_deferred, missing_members=missing_members, ecsw_not_complete=ecsw_bad,
                 ecsw_quality_limited=limited, accounting=acct, allocation_hours=T['config'].get('search_allocation_hours'),
                 budget_note='dimension-level V ratio and band wording are applied by make_report.py (A4.3)')
    blocks_favourable = bool(restrictions and any(not r.startswith('ECSW construction quality') for r in restrictions))
    out = dict(cell=cell, flags=flags, eligible_cases=elig, X={}, stageb_sha256=sha(bdir / 'stageb.json'),
               tune_sha256=sha(tdir / 'tune.json'), frozen_sha256=B['config']['frozen_sha256'])

    def tag_of(key):
        fx = FZ['X'].get(key, {})
        return dict(batch=fx.get('batch'), qualification=fx.get('qualification'),
                    diagnostic_only=bool(d == 3 and fx.get('batch') == 1))

    def qualify(tag, text):                            # one formatter for every emitted conclusion (A32.3, A35.1)
        pre = ('DIAGNOSTIC ONLY (3D batch 1, A22.0): ' if tag['diagnostic_only'] else
               'qualified subset, not a completeness claim (3D batch 2, A29): ' if tag['batch'] == 2 and d == 3 else '')
        if restrictions:
            fav = not text.startswith('baseline dominates')
            fav = 'baseline dominates' not in text
            if fav and blocks_favourable and any(w in text for w in ('ours dominates', 'matches', 'neither', 'no observed')):
                text += ' (withheld: ' + '; '.join(restrictions) + ')'
            else:
                text += ' [' + '; '.join(restrictions) + ']'
        return pre + text + ' [developmental, in-sample, unadjusted]'

    def pairs_inventory(key, outcome):                 # A35.1/A36.1: every pair has a qualified outcome
        tg = tag_of(key)
        return {m: qualify(tg, outcome.get(m, 'not assessed')) for m in FZ['members']}

    for key in FZ['X']:
        tag = tag_of(key)
        if key not in expected_X:
            out['X'][key] = dict(tag, label=qualify(tag, 'not assessed (X not admitted in the tuning job)'),
                                 pairs=pairs_inventory(key, {}), n_assessed_pairs=0)
            continue
        xr = B['ours'].get(key)
        if not (xr and xr.get('G_ours')):
            out['X'][key] = dict(tag, label=qualify(tag, 'not assessed (G-ours failed or comparator missing in Stage B)'),
                                 pairs=pairs_inventory(key, {}), n_assessed_pairs=0)
            continue
        if not elig or any(c not in eX_all[key] for c in elig):
            out['X'][key] = dict(tag, label=qualify(tag, 'not assessed (no eligible case-kind or missing exported errors)'),
                                 pairs=pairs_inventory(key, {}), n_assessed_pairs=0)
            continue
        eX = np.array([eX_all[key][c] for c in elig])
        envX = ST.envelope(eX, eps)
        cand, outcome, incomplete = [], {}, []
        for m in FZ['members']:
            mv = members.get(m)
            if mv is None:
                outcome[m] = f'missing ({missing_members.get(m, "absent")})'
                incomplete.append(m)
                continue
            if not mv['q1']['q1']:
                outcome[m] = 'excluded: not Q1-qualified on the verdict cohort'
                continue
            if not mv.get('replay_ok'):
                outcome[m] = 'missing (replay failed)'
                incomplete.append(m)
                continue
            evs = [mv['e0'].get(c, {}).get('e') for c in elig]                        # A36.4: full, finite scores
            if any(e is None or not np.isfinite(e) for e in evs):
                outcome[m] = 'missing (incomplete or non-finite eligible-case scores)'
                incomplete.append(m)
                continue
            tm = B['timing'].get(f'{key}||{m}') or {}
            if tm.get('status') != 'complete' or not tm.get('outputs_ok'):
                outcome[m] = f'timing {tm.get("status", "absent")}'
                incomplete.append(m)
                continue
            rho, bad = rho_of(tm['cases'])
            if bad or sorted(rho) != list(ST.TIMING_CASES) or not all(rho.values()):
                outcome[m] = f'timing contract failed: {bad[:3]}'
                incomplete.append(m)
                continue
            eB = np.array([mv['e0'][c]['e'] for c in elig])
            ts = ST.time_stats(rho)
            kind, r, hp, scheme, dt, solver, gen = m.split('|')
            cand.append(dict(id=m, ratio=ts['median'], W=float(np.max(eB)), med=float(np.median(eB)), r=int(r),
                             kind=kind, hr_points=int(FZ['member_info'][m]['hr_points']), scheme=scheme, dt=float(dt),
                             solver=solver, eB=eB, rho=rho, quality=FZ['member_info'][m].get('quality')))
        rec = dict(tag, W_X=float(np.max(eX)), med_X=float(np.median(eX)), roles={}, exploratory={})
        if incomplete:                                 # A30.1 (2), A35.1: never fill a role from an incomplete set
            rec.update(label=qualify(tag, 'role incomplete (' + ', '.join(sorted({outcome[m].split(':')[0].split(' (')[0]
                                                                              for m in incomplete})) + ')'),
                       pairs=pairs_inventory(key, outcome), n_assessed_pairs=0, incomplete=incomplete)
            out['X'][key] = rec
            continue
        if not cand:
            rec.update(label=qualify(tag, 'baseline incomplete at this mesh'), pairs=pairs_inventory(key, outcome),
                       n_assessed_pairs=0)
            out['X'][key] = rec
            continue
        b_acc, b_cost, F = ST.picks([{k: v for k, v in c.items() if k not in ('eB', 'rho')} for c in cand],
                                    float(np.max(eX)))
        byid = {c['id']: c for c in cand}
        rec['front'] = [f['id'] for f in F]
        substantive = set()
        for role, p_ in (('B_acc', b_acc), ('B_cost', b_cost)):
            if p_ is None:
                rec['roles'][role] = dict(label=qualify(tag, 'no baseline in role'))
                continue
            c = byid[p_['id']]
            lab = ST.pair_label(eX, c['eB'], eps, c['rho'])
            rr = int(c['r'])
            rec['roles'][role] = dict(id=c['id'], quality=c['quality'], T=c['ratio'], W=c['W'], med=c['med'], **{
                k: v for k, v in lab.items() if k in ('time', 'envX', 'envB', 'qualifier', 'reason')})
            rec['roles'][role]['label'] = qualify(tag, lab['label'])
            rec['roles'][role]['construction_hours'] = dict(           # A3.1: T beside the pick
                cell_T=acct['T_hours'], ecsw_path=(T['phases'].get(f'ecsw_r{rr}', {}).get('seconds', 0.) / 3600
                                                   if c['kind'] == 'ecsw' else 0.))
            if lab['label'] != 'not assessed':
                substantive.add(c['id'])
                outcome[c['id']] = f'{role}: {lab["label"]}'
        for c in cand:
            if c['id'] in substantive or c['id'] in [v.get('id') for v in rec['roles'].values()]:
                continue
            envB = ST.envelope(c['eB'], eps)
            if envB['worst'][0] > envX['worst'][1]:
                outcome[c['id']] = 'not labelled: worst envelope entirely above X'
                continue
            lab = ST.pair_label(eX, c['eB'], eps, c['rho'])
            rec['exploratory'][c['id']] = dict(label=qualify(tag, lab['label'] + ' (exploratory)'), T=c['ratio'],
                                               W=c['W'], med=c['med'], quality=c['quality'])
            if lab['label'] != 'not assessed':
                substantive.add(c['id'])
                outcome[c['id']] = f'exploratory: {lab["label"]}'
        rec['pairs'] = pairs_inventory(key, outcome)
        rec['n_assessed_pairs'] = len(substantive)                          # unique (X, B) with a substantive label
        rec['n_not_assessed'] = len(FZ['members']) - len(substantive)
        dom_exp = [i for i, v in rec['exploratory'].items() if 'baseline dominates' in v['label']]
        dom_reg = [r_ for r_, v in rec['roles'].items() if 'baseline dominates' in v.get('label', '')]
        line = ('baseline dominates X: observed domination by a registered pick' if dom_reg else
                'baseline dominates X: observed domination by a non-registered Stage-B configuration' if dom_exp else
                f'no observed domination among the {len(substantive)} pairs assessed')
        rec['cell_line'] = qualify(tag, line)
        out['X'][key] = rec
    p = HERE / 'checks' / f'VERDICT-{cell}.json'
    p.write_text(json.dumps(out, indent=1, default=float) + '\n')
    (HERE / 'checks' / f'VERDICT-{cell}.sha256').write_text(f'{sha(p)}  {p.name}\n')
    for k, v in out['X'].items():
        print(k, v.get('cell_line', v.get('label')), {r_: x.get('label') for r_, x in v.get('roles', {}).items()})


if __name__ == '__main__':
    main()
