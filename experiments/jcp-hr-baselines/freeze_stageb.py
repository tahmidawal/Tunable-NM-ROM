"""Offline Stage-B freeze for one cell (DESIGN A2.6, A5.3, A24.4, A26.5), from a collected tuning job.

Reads runs/<attempt>/archive/output/tune.json, recomputes from its per-case evidence (never from summary fields alone):
  - the eligible set: Q1-qualified (q1 recomputed from per-run certificates is in tune.json; the distances are re-derived
    in check mode from arrays/ and Phi_lat when --verify is given), timing_ok, finite W and t;
  - the absolute-time screening front F and up to 24 near-front additions (A2.6);
  - for every admitted X (G-ours true and timing_ok in the tuning job), the X-specific front in (median over the timing
    cases of the per-case ratio t_B / t_X, worst error) (A5.3);
and writes checks/FROZEN-STAGEB-<cell>.json (+ .sha256) with the membership, the admitted X list, the source tune.json
sha256 and the artifact inventory the Stage-B job must verify.

Usage: python freeze_stageb.py <cell> <tune_output_dir> [--verify]
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

TIMING = ST.TIMING_CASES


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recheck(tdir, T, cid, P, n0):
    """A32.8: Q1 and timing acceptance of one screened configuration recomputed from its evidence (certificates, the
    saved output coefficients through Phi_lat, warm-run agreement), never from the stored flags."""
    sys.path.insert(0, str(HERE / 'e0'))
    import hrcore as H
    import score as SC
    import pipeline as PL
    d = T['config']['d']
    v = T['stageA'][cid]
    a = np.load(tdir / 'arrays' / f"{v['stage']}~{cid.replace('|', '~')}.npz")
    r = v['r']
    lat = lambda w: H.to_lattice(w @ P[:, :r].T, d)
    certs = [v['runs'][str(j)]['cert'] for j in TIMING]
    tc = [v['anchors']['tight'][str(j)] for j in TIMING]
    tr = [v['anchors']['tighter'][str(j)] for j in TIMING[:2]]
    dist = [SC.lattice_distance(lat(a['W'][i]), lat(a['Wtight'][i]), n0[j]) for i, j in enumerate(TIMING)]
    dtt = [SC.lattice_distance(lat(a['Wtight'][i]), lat(a['Wtighter'][i]), n0[j]) for i, j in enumerate(TIMING[:2])]
    st = sum(c['steps'] for c in certs)
    ns = sum(c['nonstat'] for c in certs)
    q1 = bool(all(c['valid'] for c in certs) and st > 0 and ns <= PL.Q1_NONSTAT * st
              and all(c['anchor_ok'] for c in tc + tr) and all(np.isfinite(x) and x <= PL.Q1_DIST for x in dist)
              and all(np.isfinite(x) and x <= PL.Q1_ANCHOR for x in dtt))
    tv = [v['runs'][str(j)]['timed_vs_scored'] for j in TIMING]
    timing_ok = bool(all(x is not None and np.isfinite(x) and x <= 1e-12 for x in tv))
    ref = [v['q1']['dist'][str(j)] for j in TIMING]
    mism = max(abs(x - y) for x, y in zip(dist, ref))
    return q1, timing_ok, mism


def ecsw_validate(tdir, T, members):
    """A33.3 offline ECSW artifact validator (before freezing): per ECSW member, the online sample (unique in-mesh
    integer nodes, finite strictly positive weights, count = the reported size) and the selected path's terminal
    evidence; the rank's quality label recomputed from the selected attempt's drift values and ban count."""
    d, L = T['config']['d'], T['config']['L']
    n = (L - 1) ** d
    out = {}
    for m in members:
        kind, r, hp = m.split('|')[:3]
        if kind != 'ecsw':
            continue
        z = np.load(tdir / 'artifacts' / f'HR~ecsw~{r}~{hp}.npz')
        nd, w = z['nodes'], z['weights']
        ok = bool(nd.ndim == 1 and np.issubdtype(nd.dtype, np.integer) and np.all((nd >= 0) & (nd < n))
                  and len(set(nd.tolist())) == len(nd) and np.all(np.isfinite(w)) and np.all(w > 0)
                  and len(w) == len(nd) == int(T['stageA'][m]['hr_points']))
        P_ = T[f'ecsw_r{r}']
        sel = P_['attempts'][P_['selected']]
        dr = sel.get('qr_drift') or []
        lim = []
        if any(not (np.isfinite(v) and v <= 1e-8) for v in dr):
            lim.append('limited (orthogonality)')
        if sel.get('banned', 0) > 0.01 * P_['n_cand']:
            lim.append('limited (banning)')
        q = '; '.join(lim) if lim else 'ok'
        zt = np.load(tdir / 'artifacts' / f'ECSW_r{r}.npz')
        a = P_['selected']
        tw, tn, tr = zt[f'{a}~terminal~weights'], zt[f'{a}~terminal~nodes'], zt[f'{a}~terminal~resvec']
        bn = float(np.asarray(zt[f'{a}~bnorm']))
        term_ok = bool(len(tw) == len(tn) == sel['final_support'] and np.all(np.isfinite(tw)) and np.all(tw > 0)
                       and np.issubdtype(tn.dtype, np.integer) and np.all((tn >= 0) & (tn < n))
                       and len(set(tn.tolist())) == len(tn) and np.all(np.isfinite(tr)) and bn > 0
                       and abs(float(np.linalg.norm(tr)) / bn - sel['final_relres'])
                       <= 1e-10 * max(sel['final_relres'], 1e-300) + 1e-14)
        if sel['reason'] == 'kkt':                     # A23/A24.1: a KKT stop must carry the solver's certificate
            k_ = sel.get('kkt') or {}
            vals = [k_.get('support_max_abs_gradient'), k_.get('off_support_max_gradient'), k_.get('tol')]
            term_ok &= bool(k_.get('ok') is True and all(v is not None and np.isfinite(v) for v in vals)
                            and k_['tol'] >= 0 and vals[0] <= k_['tol'] and vals[1] <= k_['tol'])   # smoke_accept's
        # the online sample is exactly the selected path's checkpoint (or, for an optimum alias, its terminal solution)
        cn, cw = zt[f'{a}~{hp}~nodes'], zt[f'{a}~{hp}~weights']
        o1, o2 = np.argsort(nd), np.argsort(cn)
        term_ok &= bool(len(cn) == len(nd) and np.array_equal(nd[o1], cn[o2]) and np.array_equal(w[o1], cw[o2]))
        cp = sel['checkpoints'].get(hp) or {}
        if cp.get('optimum'):
            ot1, ot2 = np.argsort(tn), np.argsort(nd)
            term_ok &= bool(len(tn) == len(nd) and np.array_equal(tn[ot1], nd[ot2]) and np.array_equal(tw[ot1], w[ot2]))
        ok &= term_ok and q == P_['quality'] == T['stageA'][m]['quality']
        out[m] = dict(ok=ok, quality_recomputed=q, terminal_ok=term_ok)
    assert all(v['ok'] for v in out.values()), {k: v for k, v in out.items() if not v['ok']}
    return out


def main():
    cell, tdir = sys.argv[1], Path(sys.argv[2])
    T = json.loads((tdir / 'tune.json').read_text())
    global TIMING
    if T['config'].get('dry'):                          # local CPU dry run only: 3 screening cases
        TIMING = (0, 1, 2)
    A = T['stageA']
    sys.path.insert(0, str(HERE / 'e0'))
    import hrcore as H
    d = T['config']['d']
    z = np.load(tdir / 'artifacts' / ('Phi_lat.npz' if d == 2 else 'Phi.npz'))
    P = z['Phi_lat'] if d == 2 else z['Phi']
    n0 = {int(k): v for k, v in T['n0'].items()}
    EL = [int(j) for j in T['eligible_screen_cases']]                       # A32.1
    elig, checks, mism_all = {}, {}, 0.
    for cid, v in A.items():
        if T['inventory'].get(cid, {}).get('status') != 'complete':
            continue
        q1, tok, mism = recheck(tdir, T, cid, P, n0)
        mism_all = max(mism_all, mism)
        assert q1 == bool(v['q1']['q1']) and tok == bool(v['timing_ok']), ('stored flags disagree', cid, q1, tok)
        es = [v['e0'][str(j)]['e'] for j in EL]
        W = float(np.max(es)) if es and all(np.isfinite(es)) else float('nan')
        assert (np.isnan(W) and np.isnan(v['W'])) or W == v['W'], cid
        tt = {int(j): float(r['seconds_warm']) for j, r in v['runs'].items()}
        assert sorted(tt) == list(TIMING) and np.median([tt[j] for j in TIMING]) == v['t'], cid
        if not (q1 and tok and np.isfinite(W) and np.isfinite(v['t'])):
            continue
        elig[cid] = dict(id=cid, t=v['t'], W=W, med=float(np.median(es)), tcase=tt, r=v['r'], kind=v['kind'],
                         hr_points=v['hr_points'] or 0, scheme=v['scheme'], dt=v['dt'], solver=v['solver'],
                         stage=v['stage'], quality=v.get('quality'))
    assert mism_all <= 1e-12, ('Q1 distance recomputation mismatch', mism_all)          # A32.8: blocking
    items = [dict(id=c['id'], t=c['t'], W=c['W']) for c in elig.values()]
    F = ST.front(items)
    near = ST.near_front(items, F)
    X = {}
    for key, rec in (T.get('ours_pass') or {}).items():
        if rec.get('status') == 'complete' and rec.get('G_ours') and rec.get('timing_ok'):
            tx = {int(j): float(c['seconds_warm']) for j, c in rec['cases'].items()}
            ratios = [dict(id=c['id'], t=float(np.median([c['tcase'][j] / tx[j] for j in TIMING])), W=c['W'])
                      for c in elig.values()]
            X[key] = dict(front=[f['id'] for f in ST.front(ratios)], batch=rec.get('batch'),
                          qualification=rec.get('qualification'),
                          W_X_screen=max(rec['cases'][str(j)]['e0']['e'] for j in EL) if EL else None)
        else:
            X[key] = dict(admitted=False, status=rec.get('status'), G_ours=rec.get('G_ours'), timing_ok=rec.get('timing_ok'),
                          batch=rec.get('batch'), qualification=rec.get('qualification'))
    members = ST.stage_b(items, [[dict(id=i) for i in x['front']] for x in X.values() if 'front' in x])
    arts = sorted(p.relative_to(tdir).as_posix() for p in list((tdir / 'artifacts').glob('*.npz')) +
                  list((tdir / 'arrays').glob('*.npz')))
    needed = {'artifacts/Phi.npz'} | ({'artifacts/Phi_lat.npz'} if (tdir / 'artifacts' / 'Phi_lat.npz').exists() else set())
    for m in members:
        kind, r, hp = m.split('|')[:3]
        needed.add(f"arrays/{A[m]['stage']}~{m.replace('|', '~')}.npz")            # replay dependency (A32.8)
        if kind == 'ecsw':
            needed.add(f'artifacts/ECSW_r{r}.npz')                                 # validator input (A35.1)
        if kind != 'none':
            needed.add(f'artifacts/HR~{kind}~{r}~{hp}.npz')
    assert needed <= set(arts), sorted(needed - set(arts))
    ecsw_ok = ecsw_validate(tdir, T, members)
    # A33.3 membership-derived resource projection (model, from the tuning job's own warm times; certified factor 2)
    d_ = T['config']['d']
    ncase = 38 if d_ == 2 else 64
    grp = {}
    for m in members:
        grp.setdefault(tuple(m.split('|')[:5]), []).append(m)
    sec = 0.
    for g, ms in grp.items():
        tg = max(elig[m]['t'] for m in ms)
        sec += 1.6 * tg * (ncase + 2)                                        # tight on all cases + tighter on two
        sec += sum(1.6 * elig[m]['t'] * ncase for m in ms)                   # prod runs
    adm = [k for k, x in X.items() if 'front' in x]
    tx = {k: float(np.median([c['seconds_warm'] for c in T['ours_pass'][k]['cases'].values()])) for k in adm}
    sec += sum(len(TIMING) * 3 * 1.5 * (2 * tx[k] + elig[m]['t'] + 0.3) for k in adm for m in members)
    sec += len(adm) * 2 * 300.                                              # comparator setup twice (G-ours, timing)
    sec += len(members) * 60. + len(grp) * 60.                              # compilation, checkpoint writes
    # A35.3 memory: one basis (Phi, all ranks), one live HR data set, one comparator, timing-case fields on the host
    # A36.3 memory model of simultaneous allocations: the cell keeps Phi, LPhi, DPhi at full basis width; the live
    # online data set holds at most three r_max-wide slices (none: Phi_r, LPhi_r, DPhi_r); one comparator at a time,
    # bounded by the tuning job's ours_pass device peak (that phase runs first, after refs, so its process-cumulative
    # peak is the comparators' own); workspace margin 25 %.
    nn = (T['config']['L'] - 1) ** d_
    rmax = max(int(m.split('|')[1]) for m in members) if members else 0
    rphi = int(np.load(tdir / 'artifacts' / 'Phi.npz', mmap_mode='r')['Phi'].shape[1])
    comp = int((T['phases'].get('ours_pass') or {}).get('device_peak') or 0)
    dev = 1.25 * (8 * nn * (3 * rphi + 3 * rmax) + comp)
    lat = 257 ** 2 if d_ == 2 else 63 ** 3
    host = 1.25 * (8 * nn * rphi                                         # Phi loaded on the host
                   + 8 * ncase * 6 * lat                                  # reference fields of the cohort
                   + 8 * len(members) * len(TIMING) * 6 * lat             # certified timing-case fields
                   + 8 * len(adm) * len(TIMING) * 6 * lat                 # comparator scored fields
                   + 8 * 2 * nn * rmax)                                   # transient slices
    projection = dict(seconds_estimate=sec, hours_estimate=2 * sec / 3600,
                      note='an estimate (fixed multipliers); the hard GPU-hour controls are the wall limit and the ledger',
                      device_bytes_model=dev, host_bytes_model=host, comparator_peak_bytes=comp)
    ecsw_ok = dict(ecsw_ok)
    ver = dict(projection=projection, ecsw_members_validated=len(ecsw_ok), configs_rechecked=sum(1 for c in A if T['inventory'].get(c, {}).get('status') == 'complete'),
               worst_abs_distance_mismatch=mism_all, ok=True)
    out = dict(cell=cell, verify=ver, inventory=T['inventory'], eligible_screen_cases=EL, tune_json_sha256=sha(tdir / 'tune.json'), job=T['env'].get('job'), commit=T['env'].get('commit'),
               n_eligible=len(elig), front=[f['id'] for f in F], near_front=[c['id'] for c in near],
               X=X, members=members, member_info={m: elig[m] for m in members},
               artifacts={a: sha(tdir / a) for a in sorted(needed)},
               rule='A2.6 + A5.3 + A26.5: membership frozen before the Stage-B job; picks after it')
    p = HERE / 'checks' / f'FROZEN-STAGEB-{cell}.json'
    p.write_text(json.dumps(out, indent=1, default=float) + '\n')
    (HERE / 'checks' / f'FROZEN-STAGEB-{cell}.sha256').write_text(f'{sha(p)}  {p.name}\n')
    rem = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--remote=')]
    wall = [float(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--wall=')]
    if rem:                                             # the Stage-B job configuration (stage.py stages stage_files)
        sc = dict(d=T['config']['d'], L=T['config']['L'], cell=cell, frozen=f'checks/{p.name}', frozen_sha256=sha(p),
                  tune_dir=rem[0], stage_files=[f'checks/{p.name}', f'checks/FROZEN-STAGEB-{cell}.sha256'],
                  wall_hours=wall[0], budget_hours=wall[0] - 0.25)
        (HERE / 'configs' / f'stageb_{cell}.json').write_text(json.dumps(sc, indent=1) + '\n')
    print(cell, 'eligible', len(elig), 'front', len(F), 'near', len(near), 'members', len(members),
          'X admitted', sum('front' in x for x in X.values()))


if __name__ == '__main__':
    main()
