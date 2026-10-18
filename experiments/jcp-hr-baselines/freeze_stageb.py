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


def verify(tdir, T, members):
    """Q1 distances of the members re-derived from the saved output coefficients: fields = W Phi_lat^T (A26.2)."""
    sys.path.insert(0, str(HERE / 'e0'))
    import hrcore as H
    import score as SC
    d = T['config']['d']
    z = np.load(tdir / 'artifacts' / ('Phi_lat.npz' if d == 2 else 'Phi.npz'))
    P = z['Phi_lat'] if d == 2 else z['Phi']
    n0 = {int(k): v for k, v in T['n0'].items()}
    worst = 0.
    for m in members:
        v = T['stageA'][m]
        a = np.load(tdir / 'arrays' / f"{v['stage']}~{m.replace('|', '~')}.npz")
        r = v['r']
        for i, j in enumerate(TIMING):
            f = H.to_lattice(a['W'][i] @ P[:, :r].T, d)
            ft = H.to_lattice(a['Wtight'][i] @ P[:, :r].T, d)
            dist = SC.lattice_distance(f, ft, n0[j])
            ref = v['q1']['dist'][str(j)]
            worst = max(worst, abs(dist - ref))
    return dict(members=len(members), worst_abs_distance_mismatch=worst, ok=bool(worst <= 1e-12))


def main():
    cell, tdir = sys.argv[1], Path(sys.argv[2])
    T = json.loads((tdir / 'tune.json').read_text())
    global TIMING
    if T['config'].get('dry'):                          # local CPU dry run only: 3 screening cases
        TIMING = (0, 1, 2)
    A = T['stageA']
    elig = {}
    for cid, v in A.items():
        runs = v['runs']
        if not (v['q1']['q1'] and v['timing_ok'] and np.isfinite(v['W']) and np.isfinite(v['t'])):
            continue
        assert sorted(int(j) for j in runs) == list(TIMING), cid           # the screening cases are the timing cases
        es = [v['e0'][str(j)]['e'] if str(j) in v['e0'] else v['e0'][j]['e'] for j in TIMING]
        assert np.isclose(max(es), v['W'], rtol=0, atol=0), cid           # W recomputed from per-case records
        tt = {int(j): float(r['seconds_warm']) for j, r in runs.items()}
        assert np.median([tt[j] for j in TIMING]) == v['t'], cid
        elig[cid] = dict(id=cid, t=v['t'], W=v['W'], med=float(np.median(es)), tcase=tt, r=v['r'], kind=v['kind'],
                         hr_points=v['hr_points'] or 0, scheme=v['scheme'], dt=v['dt'], solver=v['solver'],
                         stage=v['stage'], quality=v.get('quality'))
    items = [dict(id=c['id'], t=c['t'], W=c['W']) for c in elig.values()]
    F = ST.front(items)
    near = ST.near_front(items, F)
    X = {}
    for key, rec in (T.get('ours_pass') or {}).items():
        if rec.get('status') == 'complete' and rec.get('G_ours') and rec.get('timing_ok'):
            tx = {int(j): float(c['seconds_warm']) for j, c in rec['cases'].items()}
            ratios = [dict(id=c['id'], t=float(np.median([c['tcase'][j] / tx[j] for j in TIMING])), W=c['W'])
                      for c in elig.values()]
            X[key] = dict(front=[f['id'] for f in ST.front(ratios)], W_X_screen=max(c['e0']['e'] for c in rec['cases'].values()))
        else:
            X[key] = dict(admitted=False, status=rec.get('status'), G_ours=rec.get('G_ours'), timing_ok=rec.get('timing_ok'))
    members = ST.stage_b(items, [[dict(id=i) for i in x['front']] for x in X.values() if 'front' in x])
    arts = sorted(p.relative_to(tdir).as_posix() for p in (tdir / 'artifacts').glob('*.npz'))
    needed = {'artifacts/Phi.npz'} | ({'artifacts/Phi_lat.npz'} if (tdir / 'artifacts' / 'Phi_lat.npz').exists() else set())
    for m in members:
        kind, r, hp = m.split('|')[:3]
        if kind != 'none':
            needed.add(f'artifacts/HR~{kind}~{r}~{hp}.npz')
    assert needed <= set(arts), sorted(needed - set(arts))
    ver = verify(tdir, T, members) if '--verify' in sys.argv else None
    out = dict(cell=cell, verify=ver, tune_json_sha256=sha(tdir / 'tune.json'), job=T['env'].get('job'), commit=T['env'].get('commit'),
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
