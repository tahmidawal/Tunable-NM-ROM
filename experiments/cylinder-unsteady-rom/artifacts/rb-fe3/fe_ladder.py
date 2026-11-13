"""Rebuttal push, PROTOCOL.md step 4: accuracy-matched finite-element ladder on one recorded window.

One entry = (velocity order k, step 1/N, frozen-convection refresh every K steps) on the SAME mesh
and the SAME integrator (frozen-Oseen BDF2) as the reference.  The entry starts from the reference
state at the first save of the recorded window (order 4: the state itself; lower order: nodal/L2
`Set` of the reference fields into the lower-order space, Dirichlet values re-imposed), runs the
whole recorded window with the ROM's output contract (reaction forces and probe pressures every
step, state at every save; `timing_mode`, snapshots kept in memory), and is scored against the
recorded reference with exactly the six targets used for the reduced models (rom_r4.TARGETS):
  e_u(t) = ||u - u_ref||_L2 / ||u_ref||_L2 and e_p(t) = ||p - p_ref||_L2 / sqrt(|Omega|) at the common
  saves (assembled order-4 mass matrices, lower order embedded), RMS and max over saves;
  St, CD_max, CL_max from cycles.py on the entry's own per-step force history.
The stage wall time (stepping only, including the start-up substeps and LU refreshes) is the
entry's warm cost for the whole window; `--reps` repeats the timed run.

Usage: fe_ladder.py --manifest M --case re95 --entries '[[2,400,5],[3,800,10]]' --out DIR [--reps 1]
       [--allow-heldout]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'ref'))
import rom_r4  # noqa: E402

SAVE_INV = 400


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ('--manifest', '--case', '--entries', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--allow-heldout', action='store_true')
    a = ap.parse_args()
    import ngsolve as ng
    import transient as tr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    if case['split'] == 'heldout' and not a.allow_heldout:
        raise PermissionError('held-out case needs --allow-heldout')
    run = Path(case['run_dir'])
    cfg0 = json.loads((run / 'config.json').read_text())
    st0 = json.loads((run / 'stages.json').read_text())
    (rec0,) = [s for s in st0['stages'] if s['name'] == case['stage']]
    t0, duration = rec0['t_start'], cfg0['stages'][-1]['duration']
    ng.SetNumThreads(int(cfg0['threads']))
    mesh = ng.Mesh(cfg0['mesh_path'])

    def spaces(k):
        V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        Q = ng.H1(mesh, order=k - 1)
        return V, Q, V * Q

    # reference saves (lazy, by file)
    files = st0['snapshots']
    ref_index = []          # (file number, row, save index j)
    for fi, sn in enumerate(files):
        if sha(run / sn['file']) != sn['sha256']:
            raise ValueError(f'snapshot hash mismatch {sn["file"]}')
        tt = np.load(run / sn['file'])['t']
        for row, t in enumerate(tt):
            if t >= t0 - 1e-9:
                ref_index.append((fi, row, int(round((t - t0) * SAVE_INV))))
    if ref_index[0][2] != 0 or [r[2] for r in ref_index] != list(range(len(ref_index))):
        raise ValueError('reference saves are not the consecutive 1/400 grid from the window start')
    cache = {}
    def ref_state(j):
        fi, row, _ = ref_index[j]
        if cache.get('fi') != fi:
            cache.update(fi=fi, x=np.load(run / files[fi]['file'])['x'])
        return cache['x'][row]

    kf = cfg0['velocity_order']
    Vf, Qf, Xf = spaces(kf)
    (u, p), (v, q) = Xf.TnT()
    dxb = ng.dx(bonus_intorder=6)
    with ng.TaskManager():
        mv = ng.BilinearForm(Xf); mv += u * v * dxb; mv.Assemble()
        mp = ng.BilinearForm(Xf); mp += p * q * dxb; mp.Assemble()
    vol = float(ng.Integrate(1, mesh, order=2 * kf + 6))
    gf = ng.GridFunction(Xf)
    tmp, res = gf.vec.CreateVector(), gf.vec.CreateVector()
    def quad(mat, x):
        tmp.FV().NumPy()[:] = x
        res.data = mat * tmp
        return float(x @ res.FV().NumPy())

    href = rom_r4.reference_history(run, case['stage'])
    ref_stats = rom_r4.stats(href[1:, 0], href[1:, 1], href[1:, 2], href[1:, 5] - href[1:, 6])
    x_ref0 = np.array(ref_state(0), dtype=np.float64)
    summary = {'case': a.case, 'Re': case['Re'], 'window': {'t_start': t0, 'duration': duration},
               'reference': {'run': str(run), 'velocity_order': kf, 'dt': cfg0['stages'][-1]['dt'], 'ndof': int(Xf.ndof),
                             'refactor_every': cfg0['refactor_every'], 'record_stage_wall_seconds': rec0['wall_seconds'],
                             'cycles': ref_stats},
               'targets': rom_r4.TARGETS, 'threads': cfg0['threads'], 'entries': {}}
    for k, n_inv, lag in json.loads(a.entries):
        name = f'p{k}_dt{n_inv}_K{lag}'
        rec = {'velocity_order': k, 'pressure_order': k - 1, 'dt_inverse': n_inv, 'refactor_every': lag,
               'frozen_lag_seconds': lag / n_inv}
        try:
            d = out / name; d.mkdir()
            Vs, Qs, Xs = spaces(k)
            free = np.array([bool(b) for b in Xs.FreeDofs()], dtype=bool)
            lay = hashlib.sha256()
            lay.update(np.array([Vs.ndof, Qs.ndof, Xs.ndof], dtype=np.int64).tobytes())
            lay.update(np.packbits(free).tobytes())
            gs = ng.GridFunction(Xs)
            if k == kf:
                x0 = x_ref0.copy()
            else:
                gf.vec.FV().NumPy()[:] = x_ref0
                with ng.TaskManager():
                    gs.components[0].Set(gf.components[0], bonus_intorder=6)
                    gs.components[1].Set(gf.components[1], bonus_intorder=6)
                x0 = gs.vec.FV().NumPy().copy()
                gb = ng.GridFunction(Xs)
                gb.components[0].Set(ng.CF((6 * ng.y * (cfg0['H'] - ng.y) / cfg0['H'] ** 2, 0)), definedon=mesh.Boundaries('inlet'))
                x0[~free] = cfg0['U_mean'] * gb.vec.FV().NumPy()[~free]
            state = d / 'start_state.npz'
            np.savez(state, x=x0, t=t0, nu=cfg0['nu'], velocity_order=k, mesh_sha256=cfg0['mesh_sha256'],
                     layout_sha256=lay.hexdigest())
            dt = 1.0 / n_inv
            if SAVE_INV % n_inv == 0:
                save_every, save_stride = 1, SAVE_INV // n_inv
            else:
                save_every, save_stride = n_inv // SAVE_INV, 1
                if n_inv % SAVE_INV:
                    raise ValueError('step is not commensurate with the 1/400 saves')
            cfg = dict(cfg0, velocity_order=k, pressure_order=k - 1, refactor_every=lag, kick=None, timing_mode=True,
                       initial={'kind': 'state', 'path': str(state), 'sha256': sha(state)},
                       snapshot_chunk=10 ** 6, wall_limit_seconds=10 ** 6,
                       stages=[{'name': 'ladder', 'dt': dt, 'duration': duration, 'save_every_steps': save_every,
                                'diagnostics_every_saves': 0}])
            reps = []
            for rep in range(a.reps):
                dr = d / f'run{rep}'; dr.mkdir()
                tc = time.perf_counter()
                with ng.TaskManager():
                    s = tr.Stepper(cfg, dr, ng)
                    construct = time.perf_counter() - tc
                    s.run()
                sj = json.loads((dr / 'stages.json').read_text())
                reps.append({'warm_seconds': sj['stages'][0]['wall_seconds'], 'construct_seconds': construct,
                             'setup_seconds': sj['setup_seconds'], 'steps': sj['stages'][0]['steps'],
                             'lu_refreshes': sj['factorizations'], 'timers': sj['timers_seconds']})
                if rep:
                    for f in sj['snapshots']:
                        (dr / f['file']).unlink()
            rec.update(ndof=int(Xs.ndof), timing=reps, warm_seconds_median=float(np.median([r['warm_seconds'] for r in reps])),
                       warm_seconds_min=float(min(r['warm_seconds'] for r in reps)),
                       warm_seconds_max=float(max(r['warm_seconds'] for r in reps)))
            # ---- score run0 against the recorded reference
            dr = d / 'run0'
            sj = json.loads((dr / 'stages.json').read_text())
            ts = np.concatenate([np.load(dr / f['file'])['t'] for f in sj['snapshots']])
            xs = np.concatenate([np.load(dr / f['file'])['x'] for f in sj['snapshots']])
            e_u, e_p, tsave = [], [], []
            tm = ng.TaskManager(); tm.__enter__()
            for i, t in enumerate(ts):
                j = int(round((t - t0) * SAVE_INV))
                if abs(t0 + j / SAVE_INV - t) > 1e-8 or j >= len(ref_index):
                    raise ValueError('entry save is not a reference save')
                if k == kf:
                    xe = xs[i]
                else:
                    gs.vec.FV().NumPy()[:] = xs[i]
                    gf.components[0].Set(gs.components[0], bonus_intorder=6)
                    gf.components[1].Set(gs.components[1], bonus_intorder=6)
                    xe = gf.vec.FV().NumPy().copy()
                xr = np.asarray(ref_state(j), dtype=np.float64)
                dd = xe - xr
                e_u.append(np.sqrt(max(quad(mv.mat, dd), 0) / quad(mv.mat, xr)))
                e_p.append(np.sqrt(max(quad(mp.mat, dd), 0)) / np.sqrt(vol))
                tsave.append(t)
            tm.__exit__(None, None, None)
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
            np.savez(d / 'series.npz', t_save=np.array(tsave), e_u=e_u, e_p=e_p, t=h[:, 0], CD=h[:, 1], CL=h[:, 2],
                     p_front=h[:, 5], p_back=h[:, 6])
            for f in sj['snapshots']:
                (dr / f['file']).unlink()
            del xs
            rec['status'] = 'completed'
        except Exception as exc:  # retained failure
            import traceback
            rec['status'] = 'failed'; rec['error'] = repr(exc); rec['traceback'] = traceback.format_exc()
        summary['entries'][name] = rec
        print(name, rec['status'], rec.get('ndof'), rec.get('warm_seconds_median'), rec.get('e_u'), rec.get('e_p'),
              rec.get('cycle_errors'), rec.get('targets_met'), rec.get('error', ''), flush=True)
        (out / f'LADDER-{a.case}.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
