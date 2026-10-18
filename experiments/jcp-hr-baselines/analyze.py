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


def main():
    cell, tdir, bdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    T = json.loads((tdir / 'tune.json').read_text())
    B = json.loads((bdir / 'stageb.json').read_text())
    fz = HERE / B['config']['frozen']
    assert sha(fz) == B['config']['frozen_sha256']
    FZ = json.loads(fz.read_text())
    d = T['config']['d']
    if d == 2:
        exp = json.loads((HERE / 'inputs' / 'p1_analysis.json').read_text())['configurations']
        eX_all = {k: exp[k]['e0_per_case'] for k in B['ours']}
    else:
        exp = json.loads((HERE / 'inputs' / 'speed3d_batch1_e0.json').read_text())['configs']
        eX_all = {k: {f'val64|{j}': v for j, v in exp[k]['e0'].items()} for k in B['ours']}
    assert sha(tdir / 'tune.json') == FZ['tune_json_sha256'], 'tuning JSON differs from the frozen one'   # A32.9
    members = B['members']
    elig = list(B['eligible_cases'])                                           # A32.1: the store's fixed mask
    any_m = next(iter(members.values()), None)
    eps = np.array([any_m['e0'][c]['eps_ref'] for c in elig]) if any_m else np.zeros(0)
    # A33.3 reconciliation: frozen members and comparators vs what the Stage-B job recorded
    missing_members = {m: (B['inventory'].get(m) or {}).get('status', 'absent') for m in FZ['members']
                       if m not in members}
    expected_X = [k for k, x in FZ['X'].items() if 'front' in x]
    missing_X = [k for k in expected_X if k not in B['ours']]
    # A33.3: saturation of the 256^2 cell is required (hashed, identity-checked) for the 1024^2 analysis
    L_ = T['config']['L']
    sat256 = None
    if d == 2 and L_ == 1024:
        sp = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--sat256=')]
        assert sp, '--sat256=<tune.json of the 256^2 cell>:<sha256> is required for the 1024^2 analysis'
        path, h = sp[0].rsplit(':', 1)
        assert sha(path) == h, 'saturation source differs from its pinned hash'
        T256 = json.loads(Path(path).read_text())
        assert T256['config']['d'] == 2 and T256['config']['L'] == 256 and T256['config'].get('saturation'), 'not the 256^2 cell'
        sat256 = (T256.get('saturation') or {}).get('adequate')
    elif d == 2 and L_ == 256:
        sat256 = (T.get('saturation') or {}).get('adequate')
    def tri(x):                                        # True / False / None (unassessed) -- never a silent pass
        return None if x is None else bool(x)
    adequate, saturation = tri((T.get('adequacy') or {}).get('adequate')), tri(sat256)
    cf = T.get('construction_failures') or {}
    inv = T.get('inventory') or {}
    ecsw_exp = [c for c in inv if c.startswith('ecsw|')]
    ecsw_bad = [c for c in ecsw_exp if inv[c]['status'] != 'complete']
    limited = sorted({str(v.get('quality')) for v in T['stageA'].values() if v.get('kind') == 'ecsw'} - {'ok', 'None'})
    # search (V) accounting from checkpointed classified phases (A3.1, A25.2, A28.3): tuning A|/EXT| phases, the probe
    # E0 runs, and every Stage-B selection-relevant phase (B| member runs and the A-B-A timing that decides the picks)
    tune_v = sum(v['seconds'] for k, v in T['phases'].items() if k.startswith(('A|', 'EXT|', 'generator_probe_runs')))
    stb_v = sum(v['seconds'] for k, v in B['phases'].items() if k.startswith(('B|', 'aba')))
    complete_run = 'total_seconds' in T and 'total_seconds' in B
    flags = dict(adequate=adequate, saturation_256=saturation, basis_used=T.get('basis_used'),
                 snapshot_limited=(None if (d == 2 and L_ == 1024 and saturation is None) else
                                   bool(d == 2 and L_ == 1024 and saturation is False)),
                 ecsw_coverage=('ECSW coverage incomplete' if ecsw_bad or any(k.startswith('ecsw|') for k in cf)
                                else 'complete'),
                 ecsw_not_complete=ecsw_bad, ecsw_quality_limited=limited,
                 V_hours=(tune_v + stb_v) / 3600, V_hours_tuning=tune_v / 3600, V_hours_stageB=stb_v / 3600,
                 V_complete=complete_run, allocation_hours=T['config'].get('search_allocation_hours'),
                 inventory=T.get('inventory_counts'), deferred=T.get('deferred'),
                 stageB_inventory={s_: sum(1 for v in B['inventory'].values() if v['status'] == s_)
                                   for s_ in ('complete', 'replay_failed', 'run_failed', 'not_run')},
                 missing_members=missing_members, missing_comparators=missing_X,
                 budget_note='dimension-level V ratio and band wording are applied by make_report.py (A4.3)')
    withhold = (adequate is not True) or (d == 2 and saturation is not True)    # unavailable = unassessed = withheld
    out = dict(cell=cell, flags=flags, eligible_cases=elig, withhold=withhold, X={}, stageb_sha256=sha(bdir / 'stageb.json'),
               tune_sha256=sha(tdir / 'tune.json'), frozen_sha256=B['config']['frozen_sha256'])

    def qualify(tag, text):                            # one formatter for every emitted conclusion (A32.3, A33.3)
        pre = ('DIAGNOSTIC ONLY (3D batch 1, A22.0): ' if tag['diagnostic_only'] else
               'qualified subset, not a completeness claim (3D batch 2, A29): ' if tag['batch'] == 2 and d == 3 else '')
        return pre + text + ' [developmental, in-sample, unadjusted]'
    for key in missing_X:
        fx = FZ['X'][key]
        tag = dict(batch=fx.get('batch'), qualification=fx.get('qualification'),
                   diagnostic_only=bool(d == 3 and fx.get('batch') == 1))
        out['X'][key] = dict(tag, label=qualify(tag, 'not assessed (comparator missing from the Stage-B job)'))
    for key, xr in B['ours'].items():
        fx = FZ['X'].get(key, {})
        tag = dict(batch=fx.get('batch'), qualification=fx.get('qualification'),
                   diagnostic_only=bool(d == 3 and fx.get('batch') == 1))
        if not (xr.get('G_ours') and fx.get('front') is not None):
            out['X'][key] = dict(tag, label=qualify(tag, 'not assessed'), reason='G-ours failed or X not admitted')
            continue
        if not elig or any(c not in eX_all[key] for c in elig):
            out['X'][key] = dict(tag, label=qualify(tag, 'not assessed'), reason='no eligible case-kind or missing exported errors')
            continue
        eX = np.array([eX_all[key][c] for c in elig])
        envX = ST.envelope(eX, eps)
        cand, excluded, incomplete = [], {}, []
        for m, mv in members.items():
            if not (mv['q1']['q1'] and mv.get('replay_ok')):
                excluded[m] = 'not Q1-qualified on the verdict cohort' if not mv['q1']['q1'] else 'replay failed'
                continue
            tm = B['timing'].get(f'{key}||{m}') or {}
            if tm.get('status') != 'complete' or not tm.get('outputs_ok'):
                incomplete.append(m)
                continue
            rho, bad = rho_of(tm['cases'])
            if bad or sorted(rho) != list(ST.TIMING_CASES) or not all(rho.values()):
                incomplete.append(m)
                excluded[m] = f'timing contract: {bad[:3]}'
                continue
            eB = np.array([mv['e0'][c]['e'] for c in elig])
            ts = ST.time_stats(rho)
            kind, r, hp, scheme, dt, solver, gen = m.split('|')
            cand.append(dict(id=m, ratio=ts['median'], W=float(np.max(eB)), med=float(np.median(eB)), r=int(r),
                             kind=kind, hr_points=int(FZ['member_info'][m]['hr_points']), scheme=scheme, dt=float(dt),
                             solver=solver, eB=eB, rho=rho, quality=FZ['member_info'][m].get('quality')))
        if incomplete or missing_members:              # A30.1 (2), A33.3: never fill a role from an incomplete set
            out['X'][key] = dict(tag, label=qualify(tag, 'role incomplete (' + ('timing' if incomplete else 'members')
                                                    + ')'), incomplete=incomplete, missing_members=missing_members,
                                 excluded=excluded,
                                 W_X=float(np.max(eX)))
            continue
        if not cand:
            out['X'][key] = dict(tag, label=qualify(tag, 'baseline incomplete at this mesh'), W_X=float(np.max(eX)),
                                 excluded=excluded)
            continue
        b_acc, b_cost, F = ST.picks([{k: v for k, v in c.items() if k not in ('eB', 'rho')} for c in cand],
                                    float(np.max(eX)))
        byid = {c['id']: c for c in cand}
        rec = dict(tag, W_X=float(np.max(eX)), med_X=float(np.median(eX)), front=[f['id'] for f in F], roles={},
                   exploratory={}, excluded=excluded, n_candidates=len(cand))
        for role, p_ in (('B_acc', b_acc), ('B_cost', b_cost)):
            if p_ is None:
                rec['roles'][role] = dict(label=qualify(tag, 'no baseline in role'))
                continue
            c = byid[p_['id']]
            lab = ST.pair_label(eX, c['eB'], eps, c['rho'])
            rec['roles'][role] = dict(id=c['id'], quality=c['quality'], T=c['ratio'], W=c['W'], med=c['med'], **{
                k: v for k, v in lab.items() if k in ('time', 'envX', 'envB', 'qualifier', 'reason')})
            rec['roles'][role]['label'] = qualify(tag, lab['label'])
        role_ids = {v['id'] for v in rec['roles'].values() if 'id' in v}
        for c in cand:
            if c['id'] in role_ids:
                continue                                  # a registered pick is not re-labelled as exploratory
            envB = ST.envelope(c['eB'], eps)
            if envB['worst'][0] > envX['worst'][1]:
                continue                                  # worst envelope entirely above X's: not labelled
            lab = ST.pair_label(eX, c['eB'], eps, c['rho'])
            rec['exploratory'][c['id']] = dict(label=qualify(tag, lab['label'] + ' (exploratory)'), T=c['ratio'],
                                               W=c['W'], med=c['med'], quality=c['quality'])
        assessed = len(role_ids) + len(rec['exploratory'])                      # unique (X, B) pairs
        rec['n_assessed_pairs'] = assessed
        rec['n_not_assessed'] = len(FZ['members']) - assessed
        rec['not_assessed_reasons'] = dict(excluded, **{c['id']: 'worst envelope entirely above X' for c in cand
                                                        if c['id'] not in role_ids and c['id'] not in rec['exploratory']})
        dom_reg = [r_ for r_, v in rec['roles'].items() if 'baseline dominates' in v.get('label', '')]
        dom_exp = [i for i, v in rec['exploratory'].items() if 'baseline dominates' in v['label']]
        line = ('observed domination of X by a registered pick' if dom_reg else
                'observed domination of X by a non-registered Stage-B configuration' if dom_exp else
                f'no observed domination among the {assessed} pairs assessed')
        if withhold and not dom_reg and not dom_exp:
            line += ' (withheld: basis possibly under-sampled / snapshot-limited / check unassessed, A1.3/A2.2)'
        if flags['ecsw_coverage'] != 'complete' or limited:
            line += f' [ECSW: {flags["ecsw_coverage"]}; quality-limited: {limited}; no "adequately tuned ECSW" wording]'
        rec['cell_line'] = qualify(tag, line)
        out['X'][key] = rec
    p = HERE / 'checks' / f'VERDICT-{cell}.json'
    p.write_text(json.dumps(out, indent=1, default=float) + '\n')
    (HERE / 'checks' / f'VERDICT-{cell}.sha256').write_text(f'{sha(p)}  {p.name}\n')
    for k, v in out['X'].items():
        print(k, v.get('cell_line', v.get('label')), {r_: x.get('label') for r_, x in v.get('roles', {}).items()})


if __name__ == '__main__':
    main()
