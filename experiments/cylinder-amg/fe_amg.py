"""AMG lane (PROTOCOL.md of experiments/cylinder-amg): fe_ladder2.py with the linear solver of the frozen scheme as an
extra entry element.  Copied from experiments/cylinder-unsteady-rom/fe_ladder2.py; the only changes are the solver spec
(6th entry element, a dict with 'name' and the amg_stepper.AmgStepper spec), the stepper class, the solver statistics in
the record and --entries-file.  Original docstring follows.

Second rebuttal push, PROTOCOL.md A7: finite-element comparator ladder over meshes, orders, steps and schemes.

One entry = [mesh name, Taylor-Hood order k, step 1/N, scheme, refresh K] with the vendored stepper
(vendor/ref/transient.py, unchanged): scheme frozen_oseen (frozen convection, LU every K steps), sbdf2
(explicit convection, one LU) or oseen_bdf2 (extrapolated convection, GMRES, LU every K steps).
The entry starts from the reference state at the first save of the recorded window, projected onto the
entry's space by weighted least squares on the reference quadrature Q (Dirichlet values re-imposed), runs
the whole window with the ROM's output contract (`timing_mode`: reaction forces and probe pressures every
step, state at every save) and is scored with the six targets of rom_r4.TARGETS:
  e_u(t) = ||u - u_ref||_Q / ||u_ref||_Q,  e_p(t) = ||p - p_ref||_Q / sqrt(|Omega|)   (Q = mapped order-q rule
  of the admitted mesh; points of Q that are outside a coarse mesh get the value 0 and are counted),
  St, CD_max, CL_max from cycles.py on the entry's own per-step reaction-force history.
Saves scored: all reference saves that lie on the entry's step grid.  Warm cost = stage wall seconds.
Usage: fe_ladder2.py --manifest M --case re95 --meshes MESHES.json --entries '[["c2",2,400,"frozen_oseen",40]]'
       --out DIR [--reps 1] [--allow-heldout]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'ref'))
import rom_r4  # noqa: E402
import amg_stepper  # noqa: E402

SAVE_INV = 400


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def eval_matrix(ng, mesh, S, loc, inside):
    """Sparse (points x dofs) matrix of the scalar H1 space S at located points (zero rows outside)."""
    nr, xr, yr = loc['nr'], loc['x'], loc['y']
    order = np.argsort(nr, kind='stable')
    order = order[inside[order]]
    rows, cols, vals = [], [], []
    snr = nr[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(snr)) + 1, len(snr)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        ei = ng.ElementId(ng.VOL, int(snr[start]))
        fe = S.GetFE(ei)
        dofs = np.array(S.GetDofNrs(ei), dtype=np.int64)
        idx = order[start:stop]
        sh = np.array([np.asarray(fe.CalcShape(float(xr[i]), float(yr[i]))) for i in idx]).reshape(len(idx), -1)
        rows.append(np.repeat(idx, len(dofs))); cols.append(np.tile(dofs, len(idx))); vals.append(sh.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(nr), S.ndof))


class Space:
    """One (mesh, order) Taylor-Hood space with evaluation matrices on Q."""
    def __init__(self, ng, mesh, k, loc, inside, w, cfg0, rng):
        self.k = k
        self.V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        self.Q = ng.H1(mesh, order=k - 1)
        self.X = self.V * self.Q
        Sv = ng.H1(mesh, order=k)
        self.n = Sv.ndof
        if self.V.ndof != 2 * self.n:
            raise AssertionError('VectorH1 layout')
        self.Ev = eval_matrix(ng, mesh, Sv, loc, inside)
        self.Ep = eval_matrix(ng, mesh, self.Q, loc, inside)
        self.free = np.array([bool(b) for b in self.X.FreeDofs()], dtype=bool)
        lay = hashlib.sha256()
        lay.update(np.array([self.V.ndof, self.Q.ndof, self.X.ndof], dtype=np.int64).tobytes())
        lay.update(np.packbits(self.free).tobytes())
        self.layout = lay.hexdigest()
        # self-check of the evaluation matrices against NGSolve point evaluation (random vector, random points)
        g = ng.GridFunction(self.X)
        r = rng.standard_normal(self.X.ndof)
        g.vec.FV().NumPy()[:] = r
        pick = rng.choice(np.flatnonzero(inside), size=min(3000, int(inside.sum())), replace=False)
        mips = mesh(loc['px'][pick], loc['py'][pick])
        uv = np.asarray(g.components[0](mips)).reshape(-1, 2)
        pp = np.asarray(g.components[1](mips)).reshape(-1)
        mine = self.values(r)
        self.check = float(max(np.abs(mine[0][pick] - uv[:, 0]).max(), np.abs(mine[1][pick] - uv[:, 1]).max(),
                               np.abs(mine[2][pick] - pp).max()))
        if self.check > 1e-9:
            raise AssertionError(f'evaluation matrix differs from NGSolve point evaluation: {self.check}')
        gb = ng.GridFunction(self.X)
        gb.components[0].Set(ng.CF((6 * ng.y * (cfg0['H'] - ng.y) / cfg0['H'] ** 2, 0)), definedon=mesh.Boundaries('inlet'))
        self.bc = cfg0['U_mean'] * gb.vec.FV().NumPy().copy()
        W = sp.diags(w)
        self.Av = spl.factorized((self.Ev.T @ W @ self.Ev).tocsc())
        self.Ap = spl.factorized((self.Ep.T @ W @ self.Ep).tocsc())
        self.w = w

    def values(self, x):
        n = self.n
        return self.Ev @ x[:n], self.Ev @ x[n:2 * n], self.Ep @ x[2 * n:]

    def project(self, ux, uy, p):
        w = self.w
        x = np.concatenate([self.Av(self.Ev.T @ (w * ux)), self.Av(self.Ev.T @ (w * uy)), self.Ap(self.Ep.T @ (w * p))])
        x[~self.free] = self.bc[~self.free]
        return x


def main():
    ap = argparse.ArgumentParser()
    for k in ('--manifest', '--case', '--meshes', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--entries-file', required=True)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--allow-heldout', action='store_true')
    ap.add_argument('--duration', type=float, help='smoke tests only: shorter window')
    a = ap.parse_args()
    import ngsolve as ng
    import transient as tr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    if case['split'] == 'heldout' and not a.allow_heldout:
        raise PermissionError('held-out case needs --allow-heldout')
    meshes = json.loads(Path(a.meshes).read_text())
    entries = json.loads(Path(a.entries_file).read_text())
    run = Path(case['run_dir'])
    cfg0 = json.loads((run / 'config.json').read_text())
    st0 = json.loads((run / 'stages.json').read_text())
    (rec0,) = [s for s in st0['stages'] if s['name'] == case['stage']]
    t0, duration = rec0['t_start'], cfg0['stages'][-1]['duration']
    if a.duration:
        duration = a.duration
    ng.SetNumThreads(int(cfg0['threads']))
    rng = np.random.default_rng(0)

    # ---- reference: quadrature Q of the admitted mesh and the reference fields on it
    if meshes['m0']['sha256'] != cfg0['mesh_sha256'] or sha(cfg0['mesh_path']) != cfg0['mesh_sha256']:
        raise ValueError('admitted mesh hash')
    mesh0 = ng.Mesh(cfg0['mesh_path'])
    kf = cfg0['velocity_order']
    X0 = ng.VectorH1(mesh0, order=kf, dirichlet='inlet|wall.*|obstacle') * ng.H1(mesh0, order=kf - 1)
    g0 = ng.GridFunction(X0)
    ir = ng.IntegrationRule(ng.TRIG, int(man['quadrature_order']))
    wts = []
    for el in mesh0.Elements(ng.VOL):
        trf = mesh0.GetTrafo(el)
        wts.extend(float(ip.weight * trf(ip).measure) for ip in ir)
    w = np.asarray(wts)
    mapped = mesh0.MapToAllElements(ir, ng.VOL)
    xy = np.asarray(ng.CF((ng.x, ng.y))(mapped)).reshape(-1, 2)
    if len(xy) != len(w) or np.any(w <= 0):
        raise AssertionError('quadrature')
    vol = float(w.sum())

    def ref_fields(x):
        g0.vec.FV().NumPy()[:] = x
        uv = np.asarray(g0.components[0](mapped)).reshape(-1, 2)
        return uv[:, 0].copy(), uv[:, 1].copy(), np.asarray(g0.components[1](mapped)).reshape(-1).copy()

    files = st0['snapshots']
    need = set()
    for _, _, n_inv, _, _, _ in entries:
        need.update(range(0, int(round(duration * SAVE_INV)) + 1, SAVE_INV // math.gcd(SAVE_INV, n_inv)))
    tb = time.perf_counter()
    ref = {}
    x_first = None
    with ng.TaskManager():
        for sn in files:
            if sha(run / sn['file']) != sn['sha256']:
                raise ValueError(f'snapshot hash mismatch {sn["file"]}')
            z = np.load(run / sn['file'])
            for row, t in enumerate(z['t']):
                if t < t0 - 1e-9:
                    continue
                j = int(round((t - t0) * SAVE_INV))
                if abs(t0 + j / SAVE_INV - t) > 1e-8:
                    raise ValueError('reference saves are not on the 1/400 grid')
                if j in need:
                    f = ref_fields(z['x'][row])
                    if j == 0:
                        x_first = f
                    ref[j] = (np.float32(f[0]), np.float32(f[1]), np.float32(f[2]),
                              float(np.sum(w * (f[0] ** 2 + f[1] ** 2))))
    if sorted(ref) != sorted(need):
        raise ValueError('missing reference saves')
    print('reference on Q', len(ref), 'saves', len(w), 'points', round(time.perf_counter() - tb, 1), 's', flush=True)

    href = rom_r4.reference_history(run, case['stage'])
    if a.duration:
        href = href[href[:, 0] <= t0 + duration + 1e-9]
    ref_stats = rom_r4.stats(href[1:, 0], href[1:, 1], href[1:, 2], href[1:, 5] - href[1:, 6])
    summary = {'case': a.case, 'Re': case['Re'], 'window': {'t_start': t0, 'duration': duration},
               'reference': {'run': str(run), 'velocity_order': kf, 'dt': cfg0['stages'][-1]['dt'], 'ndof': int(X0.ndof),
                             'record_stage_wall_seconds': rec0['wall_seconds'], 'cycles': ref_stats},
               'quadrature': {'points': int(len(w)), 'order': int(man['quadrature_order']), 'area': vol},
               'targets': rom_r4.TARGETS, 'threads': cfg0['threads'], 'meshes': {}, 'entries': {}}
    located, spaces = {}, {}
    for mname, k, n_inv, scheme, lag, solver in entries:
        name = f'{mname}_p{k}_dt{n_inv}_{scheme}_K{lag}_{solver["name"]}'
        rec = {'mesh': mname, 'velocity_order': k, 'pressure_order': k - 1, 'dt_inverse': n_inv, 'scheme': scheme,
               'refactor_every': lag, 'solver': solver, 'entry': [mname, k, n_inv, scheme, lag, solver]}
        try:
            d = out / name; d.mkdir()
            mi = dict(meshes[mname])
            if mname == 'm0':
                mi['path'] = cfg0['mesh_path']
            if mname not in located:
                if sha(mi['path']) != mi['sha256']:
                    raise ValueError('mesh hash')
                tl = time.perf_counter()
                mesh = ng.Mesh(mi['path'])
                if int(mesh.GetCurveOrder()) != cfg0['geometry_order']:
                    raise ValueError('curve order')
                mp = mesh(xy[:, 0], xy[:, 1])
                loc = {'nr': np.asarray(mp['nr']).astype(np.int64), 'x': np.asarray(mp['x']).astype(float),
                       'y': np.asarray(mp['y']).astype(float), 'px': xy[:, 0].copy(), 'py': xy[:, 1].copy()}
                inside = loc['nr'] >= 0
                located[mname] = (mesh, loc, inside)
                summary['meshes'][mname] = {'path': mi['path'], 'sha256': mi['sha256'], 'elements': int(mesh.ne),
                                            'points_outside': int((~inside).sum()),
                                            'weight_outside': float(w[~inside].sum()),
                                            'locate_seconds': time.perf_counter() - tl}
            mesh, loc, inside = located[mname]
            if (mname, k) not in spaces:
                ts_ = time.perf_counter()
                spaces = {kk: v for kk, v in spaces.items() if kk[0] == mname}   # keep memory bounded
                spaces[(mname, k)] = Space(ng, mesh, k, loc, inside, w, cfg0, rng)
                print('space', mname, k, spaces[(mname, k)].X.ndof, 'check', spaces[(mname, k)].check,
                      round(time.perf_counter() - ts_, 1), 's', flush=True)
            S = spaces[(mname, k)]
            x0 = S.project(*x_first)
            v0 = S.values(x0)
            rec['start_projection'] = {
                'e_u': float(np.sqrt(np.sum(w * ((v0[0] - x_first[0]) ** 2 + (v0[1] - x_first[1]) ** 2)) / ref[0][3])),
                'e_p': float(np.sqrt(np.sum(w * (v0[2] - x_first[2]) ** 2) / vol)), 'evaluation_check': S.check}
            state = d / 'start_state.npz'
            np.savez(state, x=x0, t=t0, nu=cfg0['nu'], velocity_order=k, mesh_sha256=mi['sha256'], layout_sha256=S.layout)
            g = math.gcd(SAVE_INV, n_inv)
            cfg = dict(cfg0, mesh_path=mi['path'], mesh_sha256=mi['sha256'], velocity_order=k, pressure_order=k - 1,
                       scheme=scheme, refactor_every=lag, kick=None, timing_mode=True,
                       initial={'kind': 'state', 'path': str(state), 'sha256': sha(state)},
                       snapshot_chunk=10 ** 6, wall_limit_seconds=10 ** 6,
                       stages=[{'name': 'ladder', 'dt': 1.0 / n_inv, 'duration': duration, 'save_every_steps': n_inv // g,
                                'diagnostics_every_saves': 0}])
            reps = []
            for rep in range(a.reps):
                dr = d / f'run{rep}'; dr.mkdir()
                tc = time.perf_counter()
                with ng.TaskManager():
                    s = amg_stepper.AmgStepper(cfg, dr, ng, solver)
                    construct = time.perf_counter() - tc
                    s.run()
                sj = json.loads((dr / 'stages.json').read_text())
                reps.append({'warm_seconds': sj['stages'][0]['wall_seconds'], 'construct_seconds': construct,
                             'setup_seconds': sj['setup_seconds'], 'steps': sj['stages'][0]['steps'],
                             'lu_refreshes': sj['factorizations'], 'timers': sj['timers_seconds'],
                             'linear_checks_max': sj['linear_checks_max'], 'solver_stats': s.solver_stats()})
                print('  rep', rep, name, round(sj['stages'][0]['wall_seconds'], 2), sj['timers_seconds'],
                      {kk: vv for kk, vv in s.solver_stats().items() if kk != 'solver'}, flush=True)
                if rep:
                    for f in sj['snapshots']:
                        (dr / f['file']).unlink()
            ws = [r['warm_seconds'] for r in reps]
            rec.update(ndof=int(S.X.ndof), timing=reps, warm_seconds_median=float(np.median(ws)),
                       warm_seconds_min=float(min(ws)), warm_seconds_max=float(max(ws)))
            dr = d / 'run0'
            sj = json.loads((dr / 'stages.json').read_text())
            ts = np.concatenate([np.load(dr / f['file'])['t'] for f in sj['snapshots']])
            xs = np.concatenate([np.load(dr / f['file'])['x'] for f in sj['snapshots']])
            e_u, e_p = [], []
            for i, t in enumerate(ts):
                j = int(round((t - t0) * SAVE_INV))
                if abs(t0 + j / SAVE_INV - t) > 1e-8 or j not in ref:
                    raise ValueError('entry save is not a reference save')
                ux, uy, pp = S.values(xs[i])
                rx, ry, rp, nrm = ref[j]
                e_u.append(np.sqrt(np.sum(w * ((ux - rx) ** 2 + (uy - ry) ** 2)) / nrm))
                e_p.append(np.sqrt(np.sum(w * (pp - rp) ** 2) / vol))
            e_u, e_p = np.array(e_u), np.array(e_p)
            h = np.load(dr / 'history.npz')['rows']
            ok = np.isfinite(h[:, 1])
            stt = rom_r4.stats(h[ok, 0], h[ok, 1], h[ok, 2], (h[:, 5] - h[:, 6])[ok])
            rcd = np.interp(h[ok, 0], href[:, 0], href[:, 1]); rcl = np.interp(h[ok, 0], href[:, 0], href[:, 2])
            rec['force'] = {'CD_maxabs_rel': float(np.max(np.abs(h[ok, 1] - rcd)) / np.max(np.abs(rcd))),
                            'CL_maxabs': float(np.max(np.abs(h[ok, 2] - rcl)))}
            rec['cycles'] = stt
            ce = {}
            if stt and ref_stats:
                ce = {'St_rel': abs(stt['St'] - ref_stats['St']) / ref_stats['St'],
                      'CD_max_rel': abs(stt['CD_max'] - ref_stats['CD_max']) / ref_stats['CD_max'],
                      'CL_max_abs': abs(stt['CL_max'] - ref_stats['CL_max'])}
            rec['cycle_errors'] = ce
            rec['e_u'] = {'rms': rom_r4.rms(e_u), 'max': float(e_u.max()), 'at_start': float(e_u[0]), 'saves': len(e_u)}
            rec['e_p'] = {'rms': rom_r4.rms(e_p), 'max': float(e_p.max()), 'at_start': float(e_p[0])}
            got = {'e_u_rms': rec['e_u']['rms'], 'e_u_max': rec['e_u']['max'], 'e_p_rms': rec['e_p']['rms'], **ce}
            rec['targets'] = {kk: {'value': got.get(kk), 'target': vv, 'met': got.get(kk) is not None and got[kk] <= vv}
                              for kk, vv in rom_r4.TARGETS.items()}
            rec['targets_met'] = int(sum(z['met'] for z in rec['targets'].values()))
            np.savez(d / 'series.npz', t_save=ts, e_u=e_u, e_p=e_p, t=h[:, 0], CD=h[:, 1], CL=h[:, 2],
                     p_front=h[:, 5], p_back=h[:, 6])
            for f in sj['snapshots']:
                (dr / f['file']).unlink()
            state.unlink()
            del xs
            rec['status'] = 'completed'
        except Exception as exc:  # retained failure
            import traceback
            rec['status'] = 'failed'; rec['error'] = repr(exc); rec['traceback'] = traceback.format_exc()
        summary['entries'][name] = rec
        print(name, rec['status'], rec.get('ndof'), rec.get('warm_seconds_median'), rec.get('e_u'), rec.get('e_p'),
              rec.get('cycle_errors'), rec.get('targets_met'), rec.get('error', ''), flush=True)
        (out / f'AMG-{a.case}.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
