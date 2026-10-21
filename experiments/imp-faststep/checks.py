"""imp-faststep offline acceptance checker (DESIGN A5.10; CPU, NumPy only, no JAX). Recomputes every registered predicate
from the raw job outputs and the in-job rows, and never trusts an in-job flag it can recompute:

  * E0 errors of every run from the saved coefficients with an independent decode (bank: npcheck.decode257; POD:
    Phi at the scoring-lattice nodes) against the local E0 store; in-job agreement to 1e-9 relative (A5.6);
  * Q1 / Q2 labels per (arm, r) (A1.3, A1.4, A2.2, A3.1, A5.1-A5.3), pooled over steps where registered;
  * the E5 stop rule per new arm with the two worst-error predicates reported separately (A2.5, A3.3), F5;
  * timing statistics (A3.6, A5.7, A5.8), in-job selection recomputed (A3.4, A5.9), match statuses per population;
  * the control gate (A5.10): in-job controls, C9, C11 (ecsw job), offline C10, C17, C19-C21, C23, C24;
  * the alternation diagnostic (A2.3) from the saved states.

    python checks.py <fsrun output dir> --ecsw <ecsw.json> [--phi-lat <Phi_lat.npz>] --out checks/<job>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'jcp-p1-scaling'))
sys.path.insert(0, str(HERE.parent / 'jcp-references'))
sys.path.insert(0, str(HERE))
import npcheck as NC  # noqa: E402
import refstore as RS  # noqa: E402
import fsselect as FS  # noqa: E402

NEW_ARMS = ('G-CN', 'G-E4a', 'G-E4b', 'G-R1')
BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3


def agree(a, b, rel=1e-9):
    return bool(abs(a - b) <= rel * abs(b) + 1e-15)


def dmax(xs):
    xs = [float('inf') if x is None else float(x) for x in xs]
    return float('inf') if not xs or not all(math.isfinite(x) for x in xs) else max(xs)


def e0_offline(f, case):
    coh, c = case.split('|')
    return float(RS.score(f, 2, coh, int(c), 'ST')['e'])


# ------------------------------------------------------------------ per-arm labels ----

def bank_labels(rows, A, kind, rp, cases, dev):
    """Q1/Q2 of one (arm, r) from rows + saved arrays. Returns (labels, disagreements)."""
    dis = []
    by = {r_['case']: r_ for r_ in rows}
    if [str(x) for x in A['case_ids']] != cases or set(by) != set(cases):
        dis.append(('population', sorted(set(cases) ^ set(by))))
    out = dict(per_case={})
    for j, c in enumerate(cases):
        r_ = by.get(c)
        if r_ is None:
            continue
        f = NC.decode257(A['W'][j], rp)
        e_off = e0_offline(f, c)
        if not agree(e_off, r_['e0']['e']):
            dis.append((c, 'e0', e_off, r_['e0']['e']))
        n0 = float(A['n0'][j])
        if 'dist_tight' in r_:
            d_t = NC.fdist(f, NC.decode257(A['W_tight'][j], rp), n0)
            if not agree(d_t, r_['dist_tight']):
                dis.append((c, 'dist_tight', d_t, r_['dist_tight']))
        d_q = NC.fdist(f, NC.decode257(A['W_qp'][j], rp), n0)
        if not agree(d_q, r_['dist_qp']):
            dis.append((c, 'dist_qp', d_q, r_['dist_qp']))
        out['per_case'][c] = dict(e=e_off, dist_qp=d_q)
    S = None
    if kind == 'lspg':
        g = np.asarray(A['gamma'], float)
        nonst = int(np.sum(~(g <= GAMMA)))
        pooled_ok = bool(g.size and nonst <= .01 * g.size)
        S = dict(pooled_nonstationary=nonst, pooled_steps=int(g.size))
    elif kind == 'cn':
        rn, nA = np.asarray(A['cert_rn'], float), np.asarray(A['cert_nA'], float)
        bad = ~(rn <= (1 + 1e-6) * 1e-10 * nA)
        pooled_ok = bool(rn.size and np.sum(bad) <= .01 * rn.size)
        S = dict(pooled_unconverged=int(np.sum(bad)), pooled_steps=int(rn.size))
    elif kind == 'etd':
        mis = np.asarray(A['cert_mismatch'], float)
        pooled_ok = bool(mis.size and np.all(mis <= 1e-10))
        S = dict(reconstruction_failures=int(np.sum(~(mis <= 1e-10))), worst=float(np.nanmax(mis)))
    else:
        jr, lr = np.asarray(A['cert_jac'], float), np.asarray(A['cert_lin'], float)
        pooled_ok = bool(jr.size and np.all(jr <= 1e-9) and np.all(lr <= 1e-9))
        S = dict(failures=int(np.sum(~((jr <= 1e-9) & (lr <= 1e-9)))))
    valid = all(by[c]['valid'] for c in cases if c in by)
    stable = all(by[c]['stable'] for c in cases if c in by)
    q1 = dict(all_valid=valid, certificate=pooled_ok, **S)
    if kind in ('lspg', 'cn'):
        q1['anchors'] = all(by[c].get('tight_anchor_ok') for c in cases if c in by) and \
            all(by[c].get('tighter_anchor_ok') for c in dev if c in by)
        q1['distance'] = dmax(by[c]['dist_tight'] for c in cases if c in by) <= BAR_Q1
        q1['tight_tighter'] = dmax(by[c].get('dist_tight_tighter') for c in dev if c in by) <= BAR_TT
    else:
        q1['twin'] = dmax(by[c].get('twin_dist') for c in dev if c in by) <= BAR_TT
    if kind == 'lspg':
        q1['gcert'] = all(by[c].get('gcert_ok') for c in cases if c in by)
    q1['offline_agrees'] = not dis
    q1['pass'] = all(v for k, v in q1.items() if isinstance(v, bool))
    rho = np.asarray(A['rho'], float).ravel()
    rho_ok = rho[~np.isnan(rho)]
    q2 = dict(rollout=dmax(r_['dist_qp'] for r_ in rows) <= BAR_Q2, worst_dist_qp=dmax(r_['dist_qp'] for r_ in rows),
              successor_valid=all(r_.get('qp_valid') for r_ in rows),
              rho_max=float(rho_ok.max()) if rho_ok.size else None, rho_trivial=int(np.sum(np.isnan(rho))))
    q2['rho'] = bool(rho_ok.size) and bool(np.isfinite(rho_ok).all()) and float(rho_ok.max()) <= BAR_RHO
    if not q2['successor_valid']:
        q2['label'] = 'unverified (successor invalid)'
    elif not rho_ok.size:
        q2['label'] = 'unverified (no rho indicators)'
    else:
        q2['label'] = 'pass' if (q2['rollout'] and q2['rho']) else 'fail'
    return dict(Q1=q1, Q2=q2, valid=valid, stable=stable, qualified=bool(q1['pass'] and q2['label'] == 'pass'),
                per_case=out['per_case']), dis


def pod_labels(mem, P_, Phi_lat, nodes_ok):
    """Offline e0 of every POD run via Phi at the lattice nodes; Q1 (validity, reconstruction) and Q2-POD labels."""
    dis = []
    r = mem['r']
    Pl = Phi_lat[:, :r]
    for j, row in enumerate(mem['rows']):
        v = np.asarray(P_['W'][j]) @ Pl.T
        f = np.pad(v.reshape(6, 255, 255), ((0, 0), (1, 1), (1, 1)))
        e_off = e0_offline(f, row['case'])
        if not agree(e_off, row['e0']['e']):
            dis.append((row['case'], 'e0', e_off, row['e0']['e']))
    rows = mem['rows']
    q1 = dict(all_valid=all(x['valid'] for x in rows), reconstruction=all(x['cert_ok'] for x in rows),
              twin=dmax(x.get('twin_dist') for x in rows if 'twin_dist' in x) <= BAR_TT, lattice_nodes_ok=nodes_ok,
              offline_agrees=not dis)
    q1['pass'] = all(q1.values())
    if mem['kind'] == 'none':
        q2 = dict(label='HR-free')
    else:
        rho = np.asarray([x.get('succ_rho') for x in rows], float).ravel()
        ok = rho[~np.isnan(rho)]
        q2 = dict(rollout=dmax(x.get('succ_dist') for x in rows) <= BAR_Q2, successor_certified=all(x.get('succ_certified') for x in rows),
                  rho_max=float(ok.max()) if ok.size else None)
        if not q2['successor_certified']:
            q2['label'] = 'HR-unqualified (successor not certified)'
        elif not ok.size:
            q2['label'] = 'unverified'
        else:
            q2['label'] = 'pass' if (q2['rollout'] and np.isfinite(ok).all() and ok.max() <= BAR_RHO) else 'HR-unqualified'
    return dict(Q1=q1, Q2=q2, qualified=bool(q1['pass'] and q2['label'] in ('pass', 'HR-free'))), dis


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--ecsw', required=True)
    ap.add_argument('--phi-lat', default=str(HERE / 'runs' / '_cache' / 'Phi_lat.npz'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    od = Path(a.outdir)
    R = json.loads((od / 'result.json').read_text())
    cfg = R['config']
    cases = R['setup']['cases']
    dev = [c for c in cases if c.startswith('dev6|')]
    val = [c for c in cases if c.startswith('val32|')]
    res = dict(job=R.get('job_id'), mesh=R['mesh'], gpu=R['gpu'], executed=R.get('executed'), labels={}, disagreements=[],
               stop={}, f5={}, timing={}, controls={}, selection={}, pod={}, alternation={})
    if RS.store_version() != cfg['e0_store_version']:
        res['disagreements'].append(('e0 store version', RS.store_version()))
    rows_by = {}
    for r_ in R['runs']:
        rows_by.setdefault((r_['arm'], r_['Rp']), []).append(r_)
    kinds = {'L-CNR': 'lspg', 'L-CNR-M1152': 'lspg', 'G-CN': 'cn', 'G-E4a': 'etd', 'G-E4b': 'etd', 'G-R1': 'ros'}
    for (arm, rp), rows in sorted(rows_by.items()):
        A = np.load(od / f'A_{arm}_{rp}.npz')
        lab, dis = bank_labels(rows, A, kinds[arm], rp, cases, dev)
        res['labels'][f'{arm}|{rp}'] = {k: v for k, v in lab.items() if k != 'per_case'}
        res['disagreements'] += [(f'{arm}|{rp}',) + tuple(d) for d in dis]
        if 'Z' in A.files:                                                # A2.3 alternation diagnostic (reported)
            D = np.asarray(A['D'])
            Z = np.asarray(A['Z'])
            mask = D > np.median(D)
            ai = []
            for j in range(len(Z)):
                dz = np.diff(Z[j], axis=0)[:, mask]
                thr = 1e-5 * np.max(np.abs(Z[j, 0]))
                pr = dz[1:] * dz[:-1]
                qual = np.minimum(np.abs(dz[1:]), np.abs(dz[:-1])) > thr
                ev = qual & (pr < 0)
                runs = np.zeros(mask.sum())
                num = 0.
                for t in range(len(pr)):
                    runs = np.where(ev[t], runs + 1, 0)
                    num += float(np.sum(np.where(ev[t] & (runs >= 3), np.abs(pr[t]), 0.)))
                den = float(np.sum(np.where(qual, np.abs(pr), 0.)))
                ai.append(num / den if den > 0 else 0.)
            res['alternation'][f'{arm}|{rp}'] = dict(max=float(max(ai)), median=float(np.median(ai)))
    # ---------------------------------------------------- stop rule per new arm (A1.7, A2.5, A3.3) ----
    ranks = sorted({rp for (_, rp) in rows_by})
    elig_cases = sorted({x['case'] for x in R['runs'] if x['arm'] == 'L-CNR' and x['e0']['eligible']})

    def wv(rows, cs, elig=False):
        """Worst E0 error over cs (eligible: over the store-eligible cases only); missing / non-finite -> inf."""
        use = [c for c in cs if c in elig_cases] if elig else list(cs)
        return FS.worst([dict(case=x['case'], e=x['e0']['e']) for x in rows], use) if use else float('nan')
    tim = R.get('timing') or {}
    ident_ok = tim.get('status') == 'ok'
    reps = tim.get('reps', 3)
    espeed = {}
    for sid in {f'{a_}|{rp}' for (a_, rp) in rows_by}:
        espeed[sid] = FS.case_ratio_stats([t for t in tim.get('records', {}).get('E', []) if t['B'] == sid], dev, reps) \
            if tim else dict(status='not run', value=None)
    res['timing']['evolve_speedup'] = espeed
    res['timing']['status'] = tim.get('status')
    for arm in NEW_ARMS:
        if not any(a_ == arm for (a_, _) in rows_by):
            continue
        cond = dict(unstable_or_invalid=False, worst_all38={}, worst_eligible={}, speed={})
        for rp in ranks:
            X, A1 = rows_by.get((arm, rp), []), rows_by.get(('L-CNR', rp), [])
            if len(X) != len(cases) or not all(x['valid'] and x['stable'] for x in X):
                cond['unstable_or_invalid'] = True
            wx, w1 = wv(X, cases), wv(A1, cases)
            wxe, w1e = wv(X, cases, True), wv(A1, cases, True)
            cond['worst_all38'][rp] = dict(X=wx, arm1=w1, fired=bool(not wx <= w1))
            cond['worst_eligible'][rp] = dict(X=wxe, arm1=w1e, fired=bool(wxe == wxe and not wxe <= w1e),
                                              n_eligible=len(elig_cases))
            res_cmp = dict(better=0, worse=0, tie=0, not_rankable=0)
            byx = {x['case']: x for x in X}
            for y in A1:
                x = byx.get(y['case'])
                if x is None or not (x['e0']['eligible'] and y['e0']['eligible']) or x['e0']['eps_ref'] is None:
                    res_cmp['not_rankable'] += 1
                elif abs(x['e0']['e'] - y['e0']['e']) <= 2 * x['e0']['eps_ref']:
                    res_cmp['tie'] += 1
                elif x['e0']['e'] < y['e0']['e']:
                    res_cmp['better'] += 1
                else:
                    res_cmp['worse'] += 1
            cond['worst_all38'][rp]['ranking_vs_arm1'] = res_cmp
        sp = espeed.get(f'{arm}|384', dict(status='not run'))
        cond['speed'] = dict(statistic=sp, identity_ok=ident_ok, headline=bool(tim.get('headline')),
                             fired=bool(not (ident_ok and sp.get('status') == 'ok' and sp['value'] >= 3)))
        c2 = any(v['fired'] for v in cond['worst_all38'].values()) or any(v['fired'] for v in cond['worst_eligible'].values())
        cond['fired'] = dict(stability_validity=cond['unstable_or_invalid'], worst_error=c2, speed=cond['speed']['fired'])
        cond['passes'] = not any(cond['fired'].values())
        cond['qualified'] = all(res['labels'].get(f'{arm}|{rp}', {}).get('qualified') for rp in ranks)
        res['stop'][arm] = cond
    res['E5_fixed_on_completed_scope'] = any(v['passes'] and v['qualified'] for v in res['stop'].values())
    res['E5_full_grid'] = 'see the per-mesh checks; incomplete if any registered cell was deferred'
    # ---------------------------------------------------- F5 (arm 4) ----
    for rp in ranks:
        blk = (R['blocks'].get(str(rp)) or {}).get('order_F5')
        if not blk:
            continue
        conv = all(o['ref_converged'] for o in blk)
        sl = [o['slope'] for o in blk]
        d = [t for t in tim.get('records', {}).get('D', []) if t.get('pair') == f'G-CN|{rp}~G-R1|{rp}']
        gain = FS.case_ratio_stats(d, dev, reps) if tim else dict(status='not run')
        g = 1. / gain['value'] if gain.get('status') == 'ok' and gain['value'] else None
        w4, w2 = wv(rows_by.get(('G-R1', rp), []), cases), wv(rows_by.get(('G-CN', rp), []), cases)
        res['f5'][rp] = dict(slopes=sl, median_slope=float(np.median(sl)), reference_converged=conv,
                             order='unresolved' if not conv else ('pass' if np.median(sl) >= 1.8 else 'fail'),
                             worst_vs_arm2=dict(arm4=w4, arm2=w2, fired=bool(not w4 <= w2)), end_to_end_gain=g,
                             gain_fired=bool(g is None or g < 1.3))
    # ---------------------------------------------------- POD ----
    pod = R.get('pod')
    if pod:
        Z = np.load(a.phi_lat)
        sys.path.append(str(HERE.parent / 'jcp-hr-baselines'))
        lat_nodes = Z['nodes']
        L = R['mesh']
        m = 256
        s = L // m
        ax = np.arange(1, m) * s - 1
        g1, g2 = np.meshgrid(ax, ax, indexing='ij')
        nodes_ok = bool(np.array_equal(lat_nodes, np.ravel_multi_index((g1.ravel(), g2.ravel()), (L - 1, L - 1))))
        Phi_lat = np.asarray(Z['Phi_lat'])
        for mid, mem in pod['members'].items():
            P_ = np.load(od / f'P_{mid.replace("|", "_")}.npz')
            lab, dis = pod_labels(mem, P_, Phi_lat, nodes_ok)
            res['pod'][mid] = dict(lab, worst=FS.worst([dict(case=x['case'], e=x['e0']['e']) for x in mem['rows']], cases),
                                   kappas=mem.get('kappas'))
            res['disagreements'] += [(mid,) + tuple(d_) for d_ in dis]
        for mid, mem in pod.get('native', {}).items():
            res['pod'][mid] = dict(native=True, worst=FS.worst([dict(case=x['case'], e=x['e0']['e']) for x in mem['rows']], cases),
                                   valid=all(x['valid'] for x in mem['rows']), stable=all(x['stable'] for x in mem['rows']),
                                   anchors_dev6=all(x.get('tight_anchor_ok') and x.get('tighter_anchor_ok') for x in mem['rows'] if 'tight_anchor_ok' in x))
    # ---------------------------------------------------- selection recomputed (A3.4, A5.9) ----
    if tim:
        screen = {k: (v['value'] if v['status'] == 'ok' else None) for k, v in tim['screen'].items()}
        sys.path.insert(0, str(HERE))
        import fsrun as FR                                      # pure-Python helpers only (no JAX work at import)
        fake = dict(runs=R['runs'], pod=pod)
        pairs = FR.select_pairs(fake, screen, tim['records'], [i for i in range(len(dev))], [tuple(c.split('|')) for c in dev],
                                dict(cfg, ranks=cfg['ranks']))
        res['selection'] = dict(recomputed=fake['selection'], in_job=R.get('selection'),
                                agrees=bool(fake['selection']['matched'] == R['selection']['matched'] and
                                            fake['selection']['best_stepper'] == R['selection']['best_stepper']))
        bank_rows = {k: [dict(case=x['case'], valid=x['valid'], stable=x['stable'], e=x['e0']['e']) for x in v]
                     for k, v in {f'{a_}|{rp}': rr for (a_, rp), rr in rows_by.items()}.items()}
        pool = {}
        for grp in ('members', 'native'):
            for mid, mem in (pod or {}).get(grp, {}).items():
                pool[mid] = [dict(case=x['case'], valid=x['valid'], stable=x['stable'], e=x['e0']['e']) for x in mem['rows']]
        matches = {}
        for X, B in (R['selection'].get('matched') or {}).items():
            if B is None:
                continue
            st = FS.match_status(bank_rows[X], pool[B], dict(dev6=dev, val32=val, all38=cases))
            d = [t for t in tim['records']['D'] if t.get('pair') == f'{X}~{B}']
            matches[X] = dict(B=B, status=st, direct=FS.case_ratio_stats(d, dev, reps))
        res['selection']['matches'] = matches
    # ---------------------------------------------------- control gate (A5.10) ----
    C = {}
    for rp in ranks:
        cc = (R['blocks'].get(str(rp)) or {}).get('controls')
        C[f'in_job_{rp}'] = bool(cc and cc['summary']['all_ok'])
    for rp in ranks:                                                       # C9 self timing
        st_ = espeed.get(f'L-CNR|{rp}', {})
        C[f'C9_{rp}'] = bool(st_.get('status') == 'ok' and .9 <= st_['value'] <= 1.1)
    E = json.loads(Path(a.ecsw).read_text())
    C['C11'] = bool(E.get('accepted') and E.get('c11_ok'))
    C10 = []
    for rp in ranks:                                                       # C10 perturbation witness on real data
        A1 = rows_by.get(('L-CNR', rp), [])
        w1 = wv(A1, cases)
        worst_row = max(A1, key=lambda x: x['e0']['e'])
        pert = [dict(x) for x in A1]
        for x in pert:
            if x['case'] == worst_row['case']:
                x['e0'] = dict(x['e0'], e=2 * x['e0']['e'])           # f + (f - R): the error doubles exactly
        C10.append(bool(not wv(pert, cases) <= w1) and bool(wv(A1, cases) <= w1))
    C['C10'] = all(C10)
    try:                                                                   # C17: a missing Q2 case must raise
        rows0 = rows_by[('G-E4a', ranks[0])][1:]
        A0 = np.load(od / f'A_G-E4a_{ranks[0]}.npz')
        _, d17 = bank_labels(rows0, A0, 'etd', ranks[0], cases, dev)
        C['C17'] = any(d_[0] == 'population' for d_ in d17)
    except Exception:  # noqa: BLE001
        C['C17'] = False
    mem_ = [('a', 'h1'), ('b', 'h2'), ('c', 'h3'), ('n', 'none')]
    C['C19'] = FS.pod_successors([x for x in mem_ if x[0] != 'b']) == {'a': 'c', 'c': 'n'} and FS.pod_successors(mem_)['a'] == 'b'
    C['C21'] = FS.pod_successors([('a', 'h1'), ('b', 'h1'), ('c', 'h2'), ('n', 'none')]) == {'a': 'c', 'b': 'c', 'c': 'n'}
    Xr = [dict(case=c, valid=True, stable=True, e=.01) for c in cases]
    Br = [dict(case=c, valid=True, stable=True, e=(.009 if c in dev else .011)) for c in cases]
    C['C20'] = FS.match_status(Xr, Br, dict(dev6=dev, val32=val, all38=cases)) == dict(dev6=True, val32=False, all38=False)
    Xr2 = [dict(case=c, valid=True, stable=True, e=(.03 if c in dev else .009)) for c in cases]
    Br2 = [dict(case=c, valid=True, stable=True, e=(.01 if c in dev else .011)) for c in cases]
    C['C24'] = (FS.match_status(Xr2, Br2, dict(dev6=dev, val32=val, all38=cases)) == dict(dev6=True, val32=False, all38=True)) \
        if val else None
    trip = [dict(case=c, ratio=1., valid=True, identity_ok=True) for c in dev for _ in range(reps)]
    C['C23'] = (FS.case_ratio_stats(trip[reps:], dev, reps)['status'] == 'incomplete' and
                FS.case_ratio_stats([dict(trip[0], identity_ok=False)] + trip[1:], dev, reps)['status'] == 'failed (identity)')
    res['controls'] = C
    res['controls_all_ok'] = all(v for v in C.values() if v is not None)
    res['offline_agrees'] = not res['disagreements']
    res['accepted'] = bool(R.get('executed') and res['controls_all_ok'] and res['offline_agrees'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    print(json.dumps(dict(accepted=res['accepted'], controls=C, disagreements=len(res['disagreements']),
                          stop={k: v['fired'] for k, v in res['stop'].items()}, E5=res['E5_fixed_on_completed_scope']), indent=1))


if __name__ == '__main__':
    main()
