"""Second rebuttal push, PROTOCOL.md A9b: operator-lane data preparation for the extra cases (NGSolve, CPU).

Adapted copy of the operator lane's prep.py (commit in ops/VENDORED.txt).  Differences, none of which touches how a
field is sampled: any number of saves per case (the long windows), the scoring-quadrature reference is stored for every
case, no training-case floor rule, held-out cases need --allow-heldout and go to the same output directory (each role
has its own job), and the geometry file is compared array by array with the one the frozen checkpoints were trained
with (--geom-ref; agreement to 1e-7 required, differences recorded); the evaluation reads that original file.
  <case>_n2.npy / _n3.npy / _grid.npy / _t.npy / _hist.npy / _Q.npy, floors_<case>.json, geom.npz, cases.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

NX, NY = 1024, 192
LX, LY = 2.2, 0.41
CENTER, RADIUS = (0.2, 0.2), 0.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recorded(run_dir, stage_name):
    run_dir = Path(run_dir)
    st = json.loads((run_dir / 'stages.json').read_text())
    (rec,) = [s for s in st['stages'] if s['name'] == stage_name]
    if rec['status'] != 'completed':
        raise ValueError('stage incomplete')
    t0, t1 = rec['t_start'], rec['t_end']
    for r in st['snapshots']:
        p = run_dir / r['file']
        if sha(p) != r['sha256']:
            raise ValueError(f'snapshot hash mismatch {p}')
        z = np.load(p)
        for t, x in zip(z['t'], z['x']):
            if t0 - 1e-9 <= t <= t1 + 1e-9:
                yield float(t), x


def count_saves(run_dir, stage_name):
    run_dir = Path(run_dir)
    st = json.loads((run_dir / 'stages.json').read_text())
    (rec,) = [s for s in st['stages'] if s['name'] == stage_name]
    n = 0
    for r in st['snapshots']:
        t = np.load(run_dir / r['file'])['t']
        n += int(((t >= rec['t_start'] - 1e-9) & (t <= rec['t_end'] + 1e-9)).sum())
    return n


def history(run_dir, stage_name):
    st = json.loads((Path(run_dir) / 'stages.json').read_text())
    (s,) = [x for x in st['stages'] if x['name'] == stage_name]
    h = np.load(Path(run_dir) / 'history.npz')['rows']
    keep = (h[:, 0] >= s['t_start'] - 1e-9) & (h[:, 0] <= s['t_end'] + 1e-9)
    return h[keep]


def lagrange_nodes(k):
    return np.array([(i / k, j / k) for j in range(k + 1) for i in range(k + 1 - j)])


def monomials(pts, k):
    return np.stack([pts[:, 0] ** a * pts[:, 1] ** b for b in range(k + 1) for a in range(k + 1 - b)], axis=1)


def lagrange_at(pts, k):
    """Values of the degree-k Lagrange basis (equispaced nodes) at reference points: (npts, nnodes)."""
    return monomials(pts, k) @ np.linalg.inv(monomials(lagrange_nodes(k), k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='smoke: only this many saves per case')
    ap.add_argument('--geom-ref', required=True, help='geom.npz of the operator lane (must be reproduced exactly)')
    ap.add_argument('--allow-heldout', action='store_true')
    a = ap.parse_args()
    import ngsolve as ng
    ng.SetNumThreads(a.threads)
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    if any(c['split'] == 'heldout' for c in man['cases']) and not a.allow_heldout:
        raise PermissionError('confirmation cases need --allow-heldout')
    if sha(man['mesh_path']) != man['mesh_sha256']:
        raise ValueError('mesh hash')
    t_start = time.time()
    mesh = ng.Mesh(man['mesh_path'])
    k = man['velocity_order']
    V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle'); Qs = ng.H1(mesh, order=k - 1)
    nv = V.ndof
    gv = ng.GridFunction(V); gp = ng.GridFunction(Qs)
    with ng.TaskManager():
        # scoring quadrature, identical construction to rom_export.FE.rule
        ir = ng.IntegrationRule(ng.TRIG, man['quadrature_order'])
        ref_q = np.array([(ip.point[0], ip.point[1]) for ip in ir])
        nq = len(ref_q)
        wq = []
        for el in mesh.Elements(ng.VOL):
            tr = mesh.GetTrafo(el)
            for ip in ir:
                wq.append(float(ip.weight * tr(ip).measure))
        wq = np.asarray(wq)
        mapped_q = mesh.MapToAllElements(ir, ng.VOL)
        xq = np.asarray(ng.CF((ng.x, ng.y))(mapped_q))
        ne = mesh.ne
        assert xq.shape == (ne * nq, 2) and len(wq) == ne * nq
        geom = {'Q_points': xq, 'Q_weights': wq, 'ne': ne, 'nq': nq}
        print('quadrature', xq.shape, 'area', wq.sum(), flush=True)
        # node sets
        nodesets = {}
        for kk in (2, 3):
            rn = lagrange_nodes(kk); nl = len(rn)
            irk = ng.IntegrationRule(points=[(float(x), float(y)) for x, y in rn], weights=[0.0] * nl)
            mk = mesh.MapToAllElements(irk, ng.VOL)
            xy = np.asarray(ng.CF((ng.x, ng.y))(mk))
            # check ordering on the first elements against an element loop
            chk = []
            for i, el in enumerate(mesh.Elements(ng.VOL)):
                if i >= 5:
                    break
                tr = mesh.GetTrafo(el)
                chk += [tuple(tr(ip).point) for ip in irk]
            assert np.max(np.abs(np.asarray(chk) - xy[:len(chk)])) < 1e-12
            key = np.round(xy / 1e-7).astype(np.int64)
            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            expected = mesh.nv + (kk - 1) * mesh.nedge + ((kk - 1) * (kk - 2) // 2) * ne
            print(f'N{kk}: unique nodes {len(first)} expected {expected}', flush=True)
            if len(first) != expected:
                raise AssertionError('node merge count mismatch')
            B = lagrange_at(ref_q, kk)                       # (nq, nl)
            rows = np.repeat(np.arange(ne * nq), nl)
            cols = inv.reshape(ne, nl)[:, None, :].repeat(nq, axis=1).reshape(-1)
            vals = np.tile(B.reshape(-1), ne)
            A = sp.csr_matrix((vals, (rows, cols)), shape=(ne * nq, len(first)))
            A.sum_duplicates()
            nodesets[kk] = (mk, first, A)
            geom.update({f'n{kk}_xy': xy[first], f'n{kk}_A_data': A.data, f'n{kk}_A_indices': A.indices,
                         f'n{kk}_A_indptr': A.indptr, f'n{kk}_weights': A.T @ wq})
        # regular grid
        xc = (np.arange(NX) + 0.5) * LX / NX; yc = (np.arange(NY) + 0.5) * LY / NY
        X, Y = np.meshgrid(xc, yc)                            # (NY, NX)
        fluid = np.hypot(X - CENTER[0], Y - CENTER[1]) > RADIUS
        gx, gy = X[fluid], Y[fluid]
        mips = mesh(gx, gy)
        found = np.asarray(mips['nr']) >= 0
        print('grid fluid cells', int(fluid.sum()), 'not located', int((~found).sum()), flush=True)
        if (~found).sum() > 50:
            raise AssertionError('too many fluid grid points not located in the mesh')
        fl = fluid.copy(); fl[fluid] = found
        fluid = fl
        mips = mesh(X[fluid], Y[fluid])
        _, near = ndimage.distance_transform_edt(~fluid, return_indices=True)
        # inlet profile at the cell-centre heights (prescribed boundary datum; steady)
        xa = np.concatenate([[0.0], xc, [LX]]); ya = np.concatenate([[0.0], yc, [LY]])
        ix = np.clip(np.searchsorted(xa, xq[:, 0], side='right') - 1, 0, NX)
        iy = np.clip(np.searchsorted(ya, xq[:, 1], side='right') - 1, 0, NY)
        tx = (xq[:, 0] - xa[ix]) / (xa[ix + 1] - xa[ix]); ty = (xq[:, 1] - ya[iy]) / (ya[iy + 1] - ya[iy])
        r = np.arange(len(xq))
        rows = np.concatenate([r, r, r, r])
        cols = np.concatenate([iy * (NX + 2) + ix, iy * (NX + 2) + ix + 1, (iy + 1) * (NX + 2) + ix, (iy + 1) * (NX + 2) + ix + 1])
        vals = np.concatenate([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
        G = sp.csr_matrix((vals, (rows, cols)), shape=(len(xq), (NX + 2) * (NY + 2)))
        geom.update({'grid_xc': xc, 'grid_yc': yc, 'grid_fluid': fluid, 'G_data': G.data, 'G_indices': G.indices, 'G_indptr': G.indptr})
        inlet_pts = mesh(np.full(NY, 1e-9), yc)

        def aug(g, inlet_u):
            """(3, NY, NX) grid field -> (3, NY+2, NX+2) with boundary data (PROTOCOL: walls, inlet, outlet copy)."""
            o = np.zeros((3, NY + 2, NX + 2), g.dtype)
            o[:, 1:-1, 1:-1] = g
            o[:, 1:-1, -1] = g[:, :, -1]
            o[2, 1:-1, 0] = g[2, :, 0]
            o[0, 1:-1, 0] = inlet_u
            o[2, 0, :] = o[2, 1, :]; o[2, -1, :] = o[2, -2, :]
            return o

        floors = {}
        inlet_u = None
        vol = wq.sum()
        for c in man['cases']:
            name = c['name']; held = False
            d = out
            store_q = True
            n_total = count_saves(c['run_dir'], c['stage'])
            ts, n2, n3, gr = [], [], [], []
            qmm = None
            fl_rows = []
            w0 = time.time()
            for i, (t, x) in enumerate(recorded(c['run_dir'], c['stage'])):
                if a.limit and i >= a.limit:
                    break
                gv.vec.FV().NumPy()[:] = x[:nv]; gp.vec.FV().NumPy()[:] = x[nv:]
                ts.append(t)
                per = {}
                for kk, (mk, first, A) in nodesets.items():
                    u = np.asarray(gv(mk)).reshape(-1, 2)[first]; p = np.asarray(gp(mk)).reshape(-1)[first]
                    per[kk] = np.concatenate([u, p[:, None]], axis=1)
                n2.append(per[2].astype(np.float32)); n3.append(per[3].astype(np.float32))
                g = np.zeros((3, NY, NX))
                ug = np.asarray(gv(mips)).reshape(-1, 2)
                g[0][fluid] = ug[:, 0]; g[1][fluid] = ug[:, 1]; g[2][fluid] = np.asarray(gp(mips)).reshape(-1)
                g[2] = g[2][near[0], near[1]]
                gr.append(g.astype(np.float32))
                if inlet_u is None:
                    inlet_u = np.asarray(gv(inlet_pts)).reshape(-1, 2)[:, 0].copy()
                    geom['inlet_u'] = inlet_u
                want_floor = i % 10 == 0
                if store_q or want_floor:
                    uq = np.asarray(gv(mapped_q)).reshape(-1, 2); pq = np.asarray(gp(mapped_q)).reshape(-1)
                    if store_q:
                        if qmm is None:
                            n_s = a.limit or n_total
                            qmm = np.lib.format.open_memmap(d / f'{name}_Q.npy', mode='w+', dtype=np.float32, shape=(n_s, len(xq), 3))
                        qmm[i, :, :2] = uq; qmm[i, :, 2] = pq
                    if want_floor:
                        un = np.sqrt(wq @ (uq ** 2).sum(1))
                        row = {'t': t}
                        for kk, (mk, first, A) in nodesets.items():
                            f = A @ per[kk].astype(np.float32).astype(np.float64)
                            row[f'n{kk}_e_u'] = float(np.sqrt(wq @ ((f[:, :2] - uq) ** 2).sum(1)) / un)
                            row[f'n{kk}_e_p'] = float(np.sqrt(wq @ (f[:, 2] - pq) ** 2) / np.sqrt(vol))
                        f = (G @ aug(g.astype(np.float32).astype(np.float64), inlet_u).reshape(3, -1).T)
                        row['grid_e_u'] = float(np.sqrt(wq @ ((f[:, :2] - uq) ** 2).sum(1)) / un)
                        row['grid_e_p'] = float(np.sqrt(wq @ (f[:, 2] - pq) ** 2) / np.sqrt(vol))
                        fl_rows.append(row)
                if i % 100 == 0:
                    print(name, i, f'{time.time() - w0:.0f}s', fl_rows[-1] if fl_rows else '', flush=True)
            if qmm is not None:
                qmm.flush(); del qmm
            np.save(d / f'{name}_t.npy', np.asarray(ts))
            np.save(d / f'{name}_n2.npy', np.stack(n2)); np.save(d / f'{name}_n3.npy', np.stack(n3))
            np.save(d / f'{name}_grid.npy', np.stack(gr))
            np.save(d / f'{name}_hist.npy', history(c['run_dir'], c['stage']))
            keys = [k_ for k_ in fl_rows[0] if k_ != 't']
            fsum = {k_: {'rms': float(np.sqrt(np.mean([r_[k_] ** 2 for r_ in fl_rows]))), 'max': float(max(r_[k_] for r_ in fl_rows))} for k_ in keys}
            fsum.update(Re=c['Re'], split=c['split'], saves=len(ts), sampled=len(fl_rows))
            (d / f'floors_{name}.json').write_text(json.dumps(fsum, indent=2) + '\n')
            if not held:
                floors[name] = fsum
            print('case done', name, len(ts), f'{time.time() - w0:.0f}s', json.dumps(fsum), flush=True)
    np.savez(out / 'geom.npz', **geom)
    zr = np.load(a.geom_ref)
    diff = {k_: (float(np.max(np.abs(np.asarray(geom[k_], dtype=np.float64) - zr[k_].astype(np.float64)))) if np.shape(geom[k_]) == zr[k_].shape else 'shape')
            for k_ in zr.files}
    floors['geometry_check'] = {'reference': a.geom_ref, 'reference_sha256': sha(a.geom_ref), 'max_abs_difference': diff,
                                'identical': all(v != 'shape' and v <= 1e-7 for v in diff.values()) and set(zr.files) == set(geom)}
    print('geometry_check', json.dumps(floors['geometry_check']), flush=True)
    if not floors['geometry_check']['identical']:
        raise AssertionError('geometry differs from the operator lane geometry')
    floors['elapsed_seconds'] = time.time() - t_start
    (out / 'floors.json').write_text(json.dumps(floors, indent=2) + '\n')
    (out / 'cases.json').write_text(json.dumps({c['name']: {'Re': c['Re'], 'split': c['split']} for c in man['cases']}, indent=2) + '\n')
    print('prep_done', flush=True)


if __name__ == '__main__':
    main()
