"""jcp-combined offline scorer (DESIGN 4 "one-switch rescoring", A1.8, A2.9/A3.5 G-score, A2.10).

Decodes saved output coefficients W (6, R') through the bank evaluated AT THE REFERENCE'S OWN LATTICE COORDINATES
(independent of the job's mesh-row decode) and scores them against a named reference from refs.json. The normaliser is
the physical initial field evaluated at the reference nodes from the case formula (never the decoded W0).

    python score.py --refs <name> --run <runs/<attempt>/archive/output> [--check] [--out checks/scores-<attempt>-<name>.json]

--check: compare with the in-job errors (|x - y| <= 1e-13 + 1e-10 |y|, G-score) and run the negative controls (case swap,
output-time shift, ST/S swap, one-node coordinate shift on dev6 case 0): each must break agreement by > 1e-6.
Registry kinds:
  2d_f257   2D references on the 257^2 nodes x = i/256 (files ref_<tag>_<cohort>_<case:03d>.npz, key f257); derived
            tags (TX) are linear combinations declared in the registry.
  3d_l65    3D references on the 63^3 lattice x = k/64 (one npz with arrays c<case> (6, 63^3)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REG = json.loads((HERE / 'refs.json').read_text())


def tol_ok(x, y):
    return abs(x - y) <= 1e-13 + 1e-10 * abs(y)


# ------------------------------------------------------------------------------------------------ 2D
def bank2d(Rp_max, shift=0):
    import t2core as T2
    Q = T2.load_qcore()
    import jax.numpy as jnp
    mdl = Q.Model(ckpt=T2.CKPT)
    x = (np.arange(1, 256) + shift) / 256
    xy = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    G = np.asarray(mdl.base.at(xy, chunk=8192) @ jnp.asarray(mdl.T[:, :Rp_max]))       # (255^2, R')
    return Q, G


def decode2d(G, W):
    U = np.zeros((W.shape[0], 257, 257))
    U[:, 1:-1, 1:-1] = (W @ G[:, :W.shape[1]].T).reshape(-1, 255, 255)
    return U


def load_ref2d(ent, coh, c):
    out = {}
    for tag, fn in ent['tags'].items():
        out[tag] = np.load(Path(ent['dir']) / fn.format(cohort=coh, case=c))['f257']
    for tag, comb in ent.get('derived', {}).items():
        out[tag] = sum(w * out[t] for t, w in comb.items())
    return out


def score2d(ent, run, check):
    res = json.loads((run / 'result.json').read_text())
    cohorts = res['cohorts']
    Rps = [res['setup'][s['name']]['Rp'] for s in res['config']['settings']]
    Q, G = bank2d(max(Rps))
    import engines as E   # noqa: F401  (qcore put it on the path)
    out, gate = {}, dict(compared=0, worst=0., passed=True)
    inrow = {(r['setting'], r['run'], r['cohort'], r['case']): r for r in res['rows']}
    for s in res['config']['settings']:
        z = np.load(run / f"W_{s['name']}.npz")
        W, keys, cases = z['W'], [str(k) for k in z['runs']], [str(c) for c in z['cases']]
        for ci, ck in enumerate(cases):
            coh, c = ck.split('|')
            c = int(c)
            ph = np.asarray(cohorts[coh]['physical'][c])
            n0 = float(np.linalg.norm(Q.e.initial(256, ph)))
            refs = load_ref2d(ent, coh, c)
            for ri, k in enumerate(keys):
                U = decode2d(G, W[ci, ri])
                pe = {t: [float(np.linalg.norm(U[j] - R[j])) / n0 for j in range(6)] for t, R in refs.items()}
                out[f"{s['name']}|{k}|{coh}|{c}"] = {t: dict(per_time=v, worst=max(v[1:])) for t, v in pe.items()}
                if check and (s['name'], k, coh, c) in inrow:
                    r = inrow[(s['name'], k, coh, c)]
                    for t in ('ST', 'S', 'TX'):
                        if t in pe:
                            gate['compared'] += 1
                            d = abs(max(pe[t][1:]) - r[f'e_{t}'])
                            gate['worst'] = max(gate['worst'], d)
                            gate['passed'] &= tol_ok(max(pe[t][1:]), r[f'e_{t}'])
    ctrl = {}
    if check:
        s0 = res['config']['settings'][0]['name']
        z = np.load(run / f'W_{s0}.npz')
        W, keys, cases = z['W'], [str(k) for k in z['runs']], [str(c) for c in z['cases']]
        k0 = keys.index('LSPG|BE|0.005|prod')
        coh, c = cases[0].split('|')
        c = int(c)
        r = inrow[(s0, keys[k0], coh, int(c))]
        ph = np.asarray(cohorts[coh]['physical'][c])
        n0 = float(np.linalg.norm(Q.e.initial(256, ph)))
        refs = load_ref2d(ent, coh, c)
        U = decode2d(G, W[0, k0])
        e = lambda U_, R_: max(float(np.linalg.norm(U_[j] - R_[j])) / n0 for j in range(1, 6))
        ctrl['case_swap'] = abs(e(decode2d(G, W[1, k0]), refs['ST']) - r['e_ST']) if len(cases) > 1 else None
        ctrl['time_shift'] = abs(e(np.roll(U, 1, 0), refs['ST']) - r['e_ST'])
        ctrl['ST_S_swap'] = abs(e(U, refs['S']) - r['e_ST'])
        _, Gs = bank2d(W.shape[-1], shift=1)
        ctrl['coordinate_shift'] = abs(e(decode2d(Gs, W[0, k0]), refs['ST']) - r['e_ST'])
        ctrl['all_break'] = all(v is None or v > 1e-6 for v in ctrl.values())
    return out, gate, ctrl


# ------------------------------------------------------------------------------------------------ 3D
def score3d(ent, run, check):
    import c3common as C3
    res = json.loads((run / 'result.json').read_text())
    z = np.load(ent['file'])
    cfg = res['config']
    tab = C3.C.table(cfg['cohort_seed'], cfg['cohort_count'])
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    i65 = C3.lattice65_index(65)
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j))[i65])) for j in cases}
    Wz = np.load(run / f"W_{res['mode']}.npz")
    out, gate = {}, dict(compared=0, worst=0., passed=True)
    inrow = {(r['setting'], r['rule'], r['spec'], r['case']): r for r in res['rows']}
    G = {}
    for s in cfg['settings']:
        bk = C3.load_bank(s['bank'])
        G[s['name']] = np.asarray(C3.TB.feature_rows(bk['params'], bk['T'][:, :int(s['Rp'])], X65))
    for key in Wz.files:
        if key.startswith('w0|'):
            continue
        sname, rule, spec = key.split('|', 2)
        A = Wz[key]
        for ci, j in enumerate(cases):
            F = A[ci] @ G[sname]
            R = np.asarray(z[f'c{j}'])
            pe = [float(np.linalg.norm(F[k] - R[k])) / n0[j] for k in range(6)]
            out[f'{key}|{j}'] = dict(ref=dict(per_time=pe, worst=max(pe[1:])))
            r = inrow.get((sname, rule, spec, j))
            if check and r is not None and np.isfinite(pe).all():
                gate['compared'] += 1
                gate['worst'] = max(gate['worst'], abs(max(pe[1:]) - r['e_ref']))
                gate['passed'] &= tol_ok(max(pe[1:]), r['e_ref'])
    return out, gate, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refs', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    ent = REG[a.refs]
    run = Path(a.run)
    fn = score2d if ent['kind'] == '2d_f257' else score3d
    out, gate, ctrl = fn(ent, run, a.check)
    rep = dict(refs=a.refs, registry_entry=ent, run=str(run), gate=gate, controls=ctrl, scores=out)
    dest = Path(a.out or HERE / 'checks' / f"scores-{run.parents[1].name}-{a.refs}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rep) + '\n')
    print(dest, 'gate', gate, 'controls', ctrl)


if __name__ == '__main__':
    main()
