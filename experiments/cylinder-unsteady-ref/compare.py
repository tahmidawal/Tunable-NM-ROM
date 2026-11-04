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
    if st.get('status') != 'completed' or any(s.get('status') != 'completed' for s in st['stages']):
        raise ValueError(f'run not completed: {d} ({st.get("status")})')
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
    ap.add_argument('--t0', type=float, required=True, help='window start (DESIGN-G5: S* time)')
    ap.add_argument('--t1', type=float, required=True, help='window end (S* + W)')
    args = ap.parse_args()
    import ngsolve as ng
    ca, ta, xa, ha = load_run(args.a)
    cb, tb, xb, hb = load_run(args.b)
    if ca['mesh_sha256'] != cb['mesh_sha256'] or ca['nu'] != cb['nu']:
        raise ValueError('runs differ in mesh or viscosity')
    if ca['initial'] != cb['initial'] or ca['initial'].get('kind') != 'state':
        raise ValueError('runs do not start from the identical hash-bound state')
    mesh = ng.Mesh(ca['mesh_path'])
    def space(k):
        V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        return V, ng.H1(mesh, order=k - 1)
    ka, kb = ca['velocity_order'], cb['velocity_order']
    kf = max(ka, kb)
    Vf, Qf = space(kf)
    Xf = Vf * Qf
    (u, p), (v, q) = Xf.TnT()
    dxb = ng.dx(bonus_intorder=6)
    mv = ng.BilinearForm(Xf); mv += u * v * dxb; mv.Assemble()
    mp = ng.BilinearForm(Xf); mp += p * q * dxb; mp.Assemble()
    vol = ng.Integrate(1, mesh, order=2 * kf + 6)
    gf = ng.GridFunction(Xf)
    def embed(x, k):
        if k == kf:
            return x
        Vs, Qs = space(k)
        gs = ng.GridFunction(Vs * Qs); gs.vec.FV().NumPy()[:] = x
        gf.components[0].Set(gs.components[0], bonus_intorder=6); gf.components[1].Set(gs.components[1], bonus_intorder=6)
        return gf.vec.FV().NumPy().copy()
    tmp, out = gf.vec.CreateVector(), gf.vec.CreateVector()
    def quad(mat, x):
        tmp.FV().NumPy()[:] = x
        out.data = mat.mat * tmp
        return float(x @ out.FV().NumPy())
    win = lambda tt: (tt >= args.t0 - 1e-9) & (tt <= args.t1 + 1e-9)
    ia, ib = common(ta, tb)
    keep = win(ta[ia]); ia, ib = ia[keep], ib[keep]
    expected = int(round((args.t1 - args.t0) * 400)) + 1
    tc = ta[ia]
    reasons = []
    if len(ia) != expected:
        reasons.append(f'common saves {len(ia)} != expected {expected}')
    elif abs(tc[0] - args.t0) > 1e-9 or abs(tc[-1] - args.t1) > 1e-9 or np.max(np.abs(np.diff(tc) - 1 / 400)) > 1e-9:
        reasons.append('save window/cadence differs from [t0, t1] at 1/400')
    if reasons:
        Path(args.out).write_text(json.dumps({'valid': False, 'invalid_reason': '; '.join(reasons), 'a': args.a, 'b': args.b}) + '\n')
        raise SystemExit('invalid comparison: ' + '; '.join(reasons))
    eu, ep = [], []
    for i, j in zip(ia, ib):
        a, b = embed(xa[i], ka), embed(xb[j], kb)
        d = a - b
        eu.append(np.sqrt(quad(mv, d) / quad(mv, b)))
        ep.append(np.sqrt(quad(mp, d)) / (1.0 ** 2 * np.sqrt(vol)))
    fa, fb = common(ha[:, 0], hb[:, 0])
    ok = win(ha[fa, 0]) & np.isfinite(ha[fa, 1:3]).all(1) & np.isfinite(hb[fb, 1:3]).all(1)
    fa, fb = fa[ok], fb[ok]  # stage-start/startup rows carry no reaction force
    dt_coarse = max(ca['stages'][0]['dt'], cb['stages'][0]['dt'])
    expected_f = int(round((args.t1 - args.t0) / dt_coarse)) - 1   # all coarse steps but the two startup rows
    if len(fa) < expected_f:
        Path(args.out).write_text(json.dumps({'valid': False, 'invalid_reason': f'force times {len(fa)} < {expected_f}'}) + '\n')
        raise SystemExit('invalid comparison: incomplete force coverage')
    cda, cdb, cla, clb = ha[fa, 1], hb[fb, 1], ha[fa, 2], hb[fb, 2]
    res = {'valid': True, 'a': str(args.a), 'b': str(args.b), 'common_save_times': len(ia), 'common_force_times': len(fa),
           'initial_state_sha256': ca['initial']['sha256'],
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
