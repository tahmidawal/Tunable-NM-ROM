"""Phase-sensitive comparison of two transient runs (R1b time / R1c space gates).

Usage: compare.py --a <run dir> --b <finer run dir> --out <json>
Fields are compared at common save times with exact assembled FE mass matrices on
the finer space; a lower-order state is embedded exactly (same curved mesh).
Forces are compared at common step times. Metrics follow DESIGN-G5.md §4.4.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))


def load_run(d):
    d = Path(d)
    cfg = json.loads((d / 'config.json').read_text())
    st = json.loads((d / 'stages.json').read_text())
    ts, xs = [], []
    for rec in st['snapshots']:
        z = np.load(d / rec['file'])
        ts.append(z['t']); xs.append(z['x'])
    h = np.load(d / 'history.npz')
    return cfg, np.concatenate(ts), np.concatenate(xs), h['rows']


def common(ta, tb, tol=1e-9):
    ia, ib = [], []
    j = 0
    for i, t in enumerate(ta):
        while j < len(tb) and tb[j] < t - tol:
            j += 1
        if j < len(tb) and abs(tb[j] - t) <= tol:
            ia.append(i); ib.append(j)
    return np.array(ia, int), np.array(ib, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import ngsolve as ng
    ca, ta, xa, ha = load_run(args.a)
    cb, tb, xb, hb = load_run(args.b)
    if ca['mesh_sha256'] != cb['mesh_sha256'] or ca['nu'] != cb['nu']:
        raise ValueError('runs differ in mesh or viscosity')
    mesh = ng.Mesh(ca['mesh_path'])
    def space(k):
        V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        return V, ng.H1(mesh, order=k - 1)
    ka, kb = ca['velocity_order'], cb['velocity_order']
    kf = max(ka, kb)
    Vf, Qf = space(kf)
    Xf = Vf * Qf
    (u, p), (v, q) = Xf.TnT()
    mv = ng.BilinearForm(Xf); mv += u * v * ng.dx; mv.Assemble()
    mp = ng.BilinearForm(Xf); mp += p * q * ng.dx; mp.Assemble()
    vol = ng.Integrate(1, mesh, order=8)
    gf = ng.GridFunction(Xf)
    def embed(x, k):
        if k == kf:
            return x
        Vs, Qs = space(k)
        gs = ng.GridFunction(Vs * Qs); gs.vec.FV().NumPy()[:] = x
        gf.components[0].Set(gs.components[0]); gf.components[1].Set(gs.components[1])
        return gf.vec.FV().NumPy().copy()
    tmp, out = gf.vec.CreateVector(), gf.vec.CreateVector()
    def quad(mat, x):
        tmp.FV().NumPy()[:] = x
        out.data = mat.mat * tmp
        return float(x @ out.FV().NumPy())
    ia, ib = common(ta, tb)
    if len(ia) < 2:
        raise ValueError('fewer than two common save times')
    eu, ep = [], []
    for i, j in zip(ia, ib):
        a, b = embed(xa[i], ka), embed(xb[j], kb)
        d = a - b
        eu.append(np.sqrt(quad(mv, d) / quad(mv, b)))
        ep.append(np.sqrt(quad(mp, d)) / (1.0 ** 2 * np.sqrt(vol)))
    fa, fb = common(ha[:, 0], hb[:, 0])
    fa, fb = fa[1:], fb[1:]  # stage-start row has no reaction force
    cda, cdb, cla, clb = ha[fa, 1], hb[fb, 1], ha[fa, 2], hb[fb, 2]
    res = {'a': str(args.a), 'b': str(args.b), 'common_save_times': len(ia), 'common_force_times': len(fa),
           'window': [float(ta[ia[0]]), float(ta[ia[-1]])],
           'e_u_max': float(max(eu)), 'e_p_max': float(max(ep)),
           'e_D': float(np.max(np.abs(cda - cdb)) / np.max(np.abs(cdb))),
           'e_L': float(np.max(np.abs(cla - clb))),
           'e_u_series': [float(z) for z in eu], 'e_p_series': [float(z) for z in ep],
           'definitions': 'DESIGN-G5 4.4: max over common times; velocity relative L2, pressure L2/(U^2 sqrt|Omega|)'}
    Path(args.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('e_u_max', 'e_p_max', 'e_D', 'e_L', 'common_save_times')}))


if __name__ == '__main__':
    main()
