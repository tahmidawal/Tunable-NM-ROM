"""jcp-hr-baselines phase-I real-size smoke (DESIGN.md 10; A1.9, A2.9-A2.11, A2.19, A3.2, A3.3, A3.6, A3.9, A4.1,
A4.4, A5.1, A5.4, A6).

One cluster job, one GPU, <= 1 h wall (sbatch --time=01:00:00). For each cell (2D L=1024, 3D L=64; fixtures in A5.4):
reduced snapshot sets with the maximal registered shapes (memory/time pilots), a small configuration grid with the
complete Q1 machinery, every control at real size, the A-B-A harness and ours' adapters. Writes <out>/smoke.json after
EVERY phase and <out>/arrays_*.npz (coefficients, lattice bases, fields) for the offline checks. Nothing here declares
acceptance: smoke_accept.py recomputes every predicate from these raw outputs.

Usage: python smoke.py <config.json> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'e0'))
import hrcore as H  # noqa: E402
import rom as RM  # noqa: E402
import build as BD  # noqa: E402
import pipeline as PL  # noqa: E402
import score as SC  # noqa: E402

T_START = time.perf_counter()
TIMING_CASES = list(range(6))          # 2D dev6 0-5, 3D 923801 0-5 (A2.19)


def log(*a):
    print(f'[{time.perf_counter() - T_START:7.1f}s]', *a, flush=True)


def left(cfg):
    return cfg['wall_seconds'] - (time.perf_counter() - T_START)


class Saver:
    def __init__(self, path, res):
        self.path, self.res = path, res

    def __call__(self):
        tmp = self.path.with_suffix('.json.tmp')          # atomic: a deadline kill never leaves a truncated JSON
        PL.dump(tmp, self.res)
        os.replace(tmp, self.path)


def phase(res, name, fn, need=0., cfg=None, save=None, store=None):
    """Run one phase; record ok / error / skipped(time); never abort the smoke; install the result under `store`
    (res[store]) and THEN checkpoint the JSON (code audit 2, item 15)."""
    if cfg is not None and left(cfg) < need:
        res['phases'][name] = dict(status='skipped', reason=f'time left {left(cfg):.0f}s < {need:.0f}s')
        log(f'SKIP {name}')
        if save:
            save()
        return None
    t0 = time.perf_counter()
    try:
        v = fn()
        res['phases'][name] = dict(status='ok', seconds=time.perf_counter() - t0, device_peak=PL.device_peak(),
                                   host_maxrss=PL.host_rss())
        log(f'OK   {name} ({time.perf_counter() - t0:.1f}s)')
    except Exception as e:  # noqa: BLE001
        res['phases'][name] = dict(status='error', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-3000:],
                                   seconds=time.perf_counter() - t0)
        log(f'ERR  {name}: {type(e).__name__}: {e}')
        v = None
    if store is not None:
        res[store] = v
    if save:
        save()
    return v


def disc(a, b):
    """Discrepancy of two outputs: max |a - b| over finite entries; matching non-finite entries (same positions, same
    nan/inf pattern) count as agreement; any mismatch of the non-finite pattern -> nan (code audit 6, item 2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float('nan')
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb) or not np.array_equal(np.isnan(a), np.isnan(b)) or \
            not np.array_equal(np.where(np.isinf(a), np.sign(a), 0.), np.where(np.isinf(b), np.sign(b), 0.)):
        return float('nan')
    return float(np.max(np.abs(a[fa] - b[fa]))) if fa.any() else 0.


def disc_finite(a, b):
    """Timing-call discrepancy (code audit 7): both outputs must be entirely finite, else nan (fails acceptance)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float('nan')
    return float(np.max(np.abs(a - b)))


def fmax(vals):
    """max of a list of discrepancies; NaN if any is non-finite (never hidden by Python's max, code audit 4 item 25)."""
    v = np.asarray([float(x) for x in vals], float)
    return float(np.max(v)) if v.size and np.all(np.isfinite(v)) else float('nan')


_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    """Keep the GPU busy for `seconds` before a timed call (combined lane's burn, c3common.burn)."""
    a = jax.block_until_ready(_BURN(jnp.ones((512, 512), jnp.float64) * .01))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def timed_triples(fA, fB, argsA, argsB, checkA, checkB, reps=3, burn_s=0.1, max_retry=3):
    """A-B-A triples (A3.9, A4.10, A6.6): `reps` original triples; each drifting triple (|tA1/tA2 - 1| > 0.25) is
    re-measured up to `max_retry` times. Every attempt is kept: dict(rep, attempt, tA1, tB, tA2, valid, dA1, dB, dA2),
    d* = discrepancy of THAT call's output against its separately scored reference (checkA / checkB, evaluated after the
    timer stops)."""
    rec = []
    for rep in range(reps):
        for attempt in range(max_retry + 1):
            ts, ds = [], []
            for f, a, chk in ((fA, argsA, checkA), (fB, argsB, checkB), (fA, argsA, checkA)):
                burn(burn_s)
                t0 = time.perf_counter()
                o = jax.block_until_ready(f(*a))
                ts.append(time.perf_counter() - t0)
                ds.append([float(x) for x in np.atleast_1d(chk(o))])      # one discrepancy per output of the call
            ok = abs(ts[0] / ts[2] - 1.) <= 0.25
            rec.append(dict(rep=rep, attempt=attempt, tA1=ts[0], tB=ts[1], tA2=ts[2], valid=bool(ok),
                            dA1=ds[0], dB=ds[1], dA2=ds[2]))
            if ok:
                break
    return rec


def run_cell(cfg, c, res, outdir, save):
    d, L = c['d'], c['L']
    key = f'{d}d_L{L}'
    R = res.setdefault(key, dict(phases={}))
    cell = PL.Cell(d, L, log=log)
    fam_tr = H.Family(d, 'train')
    coh = 'dev6' if d == 2 else 'val64'
    fam = H.Family(d, coh, count=None if d == 2 else max(TIMING_CASES + c['cases'] + c['under_fixtures']) + 1)
    cases = c['cases']
    fx = c['under_fixtures']
    opt_main = H.OPTIONS[d][0]                  # BE at dt0
    rs = c['rs']
    tau_run = f"k{c['kappa_run']}"                # the ECSW fixture's knob value (A15.1); name kept for brevity
    arrays = {}
    P = lambda name, fn, need, store=None: phase(R, name, fn, need, cfg, save, store)

    arrdir = outdir / f'arrays_{key}'
    arrdir.mkdir(parents=True, exist_ok=True)

    def chunk(name, **kw):
        """Append-only array evidence: one .npz per completed block (code audit 3, item 15); written atomically."""
        fn = arrdir / f"{name.replace('|', '~')}.npz"
        tmp = arrdir / (fn.stem + '.tmp.npz')
        np.savez(tmp, **kw)
        os.replace(tmp, fn)

    # ---- references and IC checks
    refs = P('refs', lambda: SC.Refs(d, coh, fam, sorted(set(cases + fx + TIMING_CASES[:1]))), 60)
    if refs is not None:
        R['n0'] = {j: refs.n0(j) for j in refs.f}

    # ---- G-fom at the real mesh (BE and CN at dt0)
    def g_fom():
        import refcore as RC
        u0, nu = fam.ic(L, 0), fam.nu(0)
        fom = H.make_fom(d, L)
        ref = RC.make_solver(d, L, adv='central')
        out = {}
        for sc_h, sc_r in (('BE', 'be'), ('CN', 'cn')):
            dt = H.DT0[d]
            t0 = time.perf_counter()
            a = np.asarray(fom(u0, nu, H.sched(sc_h, dt), 1e-10, 1e-12)['out'])
            ta = time.perf_counter() - t0
            b = np.asarray(ref(u0, nu, dt, sc_r, 1e-10, 1e-12)['out'])
            out[sc_h] = dict(max_diff=float(np.max(np.linalg.norm(a - b, axis=1))), norm0=float(np.linalg.norm(b[0])),
                             fom_seconds=ta)
        return out
    P('G-fom', g_fom, 120, store='G-fom')

    # ---- snapshots + POD (maximal rank) + trust; adequacy; generator probe (unit cost)
    rows = list(range(c['n_traj']))
    bb = P('basis', lambda: cell.build_basis(fam_tr, rows, c['rmax'], c['ranks']), 300)
    if bb is None:
        return arrays
    S, Phi, sig, trust = bb
    bb = None
    cell.set_basis(Phi, trust)
    R['basis'] = cell.info['basis[central]']
    chunk('basis', Phi_lat=np.asarray(cell.Phi[jnp.asarray(H.lattice_nodes(d, L)), :rs]), cases=np.asarray(cases))

    def adequacy():
        held = (3, 5, 6, 8, 12, 17, 22, 35, 45)
        rr = list(range(c['n_adequacy']))
        Sh, _ = BD.snapshots(d, L, fam_tr, rr, save_steps=held, log=None)
        Si = S[:len(rr) * len(H.SNAP_STEPS)]
        out = {}
        for r in c['ranks']:
            Pr = np.asarray(Phi[:, :r])
            e = lambda X: float(np.sqrt(np.mean(np.sum((X - (X @ Pr) @ Pr.T) ** 2, 1) / np.sum(X * X, 1))))
            out[r] = dict(held=e(Sh), inset=e(Si))
        return out
    P('adequacy', adequacy, 120, store='adequacy')

    def probe_cost():
        t0 = time.perf_counter()
        _, info = BD.snapshots(d, L, fam_tr, list(range(c['n_probe'])), scheme='BE', adv='upwind', log=None)
        return dict(seconds_per_traj=(time.perf_counter() - t0) / c['n_probe'], worst=info['worst_rel_residual'])
    P('generator_probe_cost', probe_cost, 120, store='probe_cost')

    # ---- advection POD (maximal p) with certificates at every p the smoke uses; Q-DEIM maximal-shape pilot
    ladder_p = []
    p = rs // 8
    while p >= 1:
        ladder_p.append(p)
        p //= 2
    p_list = sorted(set([c['ps'], c['pmax']] + ladder_p))         # p_max: reported only (A14.1)
    R['p_list'] = p_list
    P('adv_pod', lambda: cell.build_adv(S, c['pmax'], p_list), 240)
    del S
    R['adv'] = {k: v for k, v in cell.info.get('adv', {}).items() if k != 'sigma'}

    def qdeim_pilot():
        t0 = time.perf_counter()
        # maximal B-QR shape: a MEMORY/SHAPE pilot (A3.6, A14.1); the reduced smoke snapshot set need not certify p_max
        dat, info = cell.qdeim(c['rmax'], c['pmax'], allow_uncertified_p=True)
        info = dict(info, p_max_certified=c['pmax'] in getattr(cell, 'constructible_p', set()))
        jax.block_until_ready(dat['T1'])
        out = dict(info, seconds=time.perf_counter() - t0, device_peak=PL.device_peak(), r=c['rmax'],
                   B_shape=[cell.n, 2 * c['rmax'] + c['pmax']], T1_shape=list(dat['T1'].shape))
        del dat
        return out
    P('qdeim_max_shape', qdeim_pilot, 240, store='qdeim_pilot')

    # ---- ECSW: states, smoke-rank path, maximal-shape pilot
    P('ecsw_states', lambda: cell.ecsw_states(fam_tr), 120, store='ecsw_states')
    P('ecsw_smoke_rank', lambda: {str(t): (None if v is None else dict(size=v['size'], relres=v['relres']))
                                  for t, v in cell.ecsw(rs).items()}, 240, store='ecsw_rs')
    if rs in getattr(cell, 'ecsw_paths_raw', {}):           # A16: both attempts' checkpoint supports and weights
        kw = {}
        for att, cps in cell.ecsw_paths_raw[rs].items():
            kw[f'{att}~drops'] = np.asarray(cps['_drops'])
            kw[f'{att}~bnorm'] = np.asarray(cps['_bnorm'])
            kw[f'{att}~terminal~nodes'] = np.asarray(cps['_terminal']['nodes'])
            kw[f'{att}~terminal~weights'] = np.asarray(cps['_terminal']['weights'])
            kw[f'{att}~terminal~resvec'] = np.asarray(cps['_terminal']['resvec'])
            for k, v in cps.items():
                if k.startswith('_'):
                    continue
                if v is not None:
                    kw[f'{att}~{k}~nodes'] = np.asarray(v['nodes'])
                    kw[f'{att}~{k}~weights'] = np.asarray(v['weights'])
                    kw[f'{att}~{k}~resvec'] = np.asarray(v['resvec'])
        chunk('ECSWpath', **kw)
        R['ecsw_path_summary'] = cell.info['ecsw'][rs]['paths']
        save()

    def ecsw_pilot():
        t0 = time.perf_counter()
        G, b, cand, cinfo = BD.ecsw_contrib(cell.states, cell.Phi[:, :c['rmax']], d, L)
        jax.block_until_ready(G)
        tg = time.perf_counter() - t0
        shape = tuple(G.shape)
        t1 = time.perf_counter()
        o = BD.NN.greedy_nnls(G, b, taus=(), cap=c['pilot_support'])
        tn = time.perf_counter() - t1
        rv = np.asarray(b - G[:, jnp.asarray(o['support'])] @ jnp.asarray(o['weights']))
        chunk('ECSWpilot', support=np.asarray(o['support']), weights=np.asarray(o['weights']), resvec=rv,
              bnorm=np.asarray(o['bnorm']), drops=np.asarray(o['drops']))
        del G, b
        return dict(shape=shape, contrib_seconds=tg, nnls_seconds=tn, support=int(len(o['support'])),
                    relres=o['relres'], reason=o['reason'], iterations=len(o['path']), device_peak=PL.device_peak(),
                    host_maxrss=PL.host_rss(), kkt=o['kkt'], banned=o['banned'])
    P('ecsw_max_shape', ecsw_pilot, 300, store='ecsw_pilot')

    def none_pilot():
        dat, _ = cell.arm_data('none', c['rmax'], '-')
        rec = cell.run('none', c['rmax'], dat, 'native', opt_main, 'prod', fam.ic(L, 0), fam.nu(0))
        return dict(seconds=rec['seconds'], cert=rec['cert'], device_peak=PL.device_peak(), r=c['rmax'],
                    J_shape=[cell.n, c['rmax']], W_finite=bool(np.all(np.isfinite(rec['W']))),
                    C_finite=bool(np.all(np.isfinite(rec['C']))), fields_finite=bool(np.all(np.isfinite(rec['fields_lat']))))
    P('none_max_shape', none_pilot, 240, store='none_pilot')

    # ---- small grid with the complete Q1 machinery (A5.4)
    datas = {}
    for kind, hrp in (('none', '-'), ('ecsw', tau_run), ('qdeim', c['ps'])):
        try:
            if kind == 'ecsw' and cell.ecsw_rec.get(rs, {}).get(hrp) is None:
                raise PL.ConstructionError(f'ECSW support target {hrp} not reached at r={rs}')
            datas[(kind, hrp)] = cell.arm_data(kind, rs, hrp)
        except Exception as e:  # noqa: BLE001
            R.setdefault('construction_failures', {})[f'{kind}|{hrp}'] = f'{type(e).__name__}: {e}'
    R['arm_info'] = {f'{k[0]}|{k[1]}': v[1] for k, v in datas.items()}
    R['ecsw_info'] = {str(k): v for k, v in cell.info.get('ecsw', {}).items()}
    grid = {}
    subset = cases[:2]

    def summarize(g):
        return dict(q1={k: v for k, v in g.get('q1', {}).items()}, e0=g.get('e0'),
                    anchors={k: v for k, v in g.get('anchors', {}).items() if not k.endswith('_F')},
                    runs={j: dict(seconds=g['recs'][j]['seconds'], seconds_warm=g['recs'][j].get('seconds_warm'),
                                  timed_vs_scored=g['recs'][j].get('timed_vs_scored'), cert=g['recs'][j]['cert'])
                          for j in g['recs']})

    def grid_run():
        for (kind, hrp), (dat, _) in datas.items():
            for option in c['options']:
                option = tuple(option)
                for solver in ('native', 'common'):
                    cid = PL.cfg_id(kind, rs, hrp, option[0], option[1], solver)
                    recs = {}
                    for j in cases:
                        recs[j] = cell.run(kind, rs, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j))
                        warm = cell.run(kind, rs, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j), certify=False,
                                        burn_s=0.1)
                        recs[j]['seconds_warm'] = warm['seconds']
                        recs[j]['timed_vs_scored'] = disc(warm['fields_lat'], recs[j]['fields_lat'])
                    grid[cid] = dict(recs=recs)
                tight = {j: cell.run(kind, rs, dat, 'native', option, 'tight', fam.ic(L, j), fam.nu(j)) for j in cases}
                tighter = {j: cell.run(kind, rs, dat, 'native', option, 'tighter', fam.ic(L, j), fam.nu(j)) for j in subset}
                for solver in ('native', 'common'):
                    cid = PL.cfg_id(kind, rs, hrp, option[0], option[1], solver)
                    g = grid[cid]
                    g['q1'] = PL.q1_config(g['recs'], tight, tighter, R['n0'], subset, cases)
                    g['anchors'] = dict(tight={j: tight[j]['cert'] for j in tight},
                                        tighter={j: tighter[j]['cert'] for j in tighter},
                                        tight_F={j: tight[j]['fields_lat'] for j in tight},
                                        tighter_F={j: tighter[j]['fields_lat'] for j in tighter})
                    if refs is not None:
                        g['e0'] = {j: refs.score_safe(g['recs'][j]['fields_lat'], j) for j in cases}
                    chunk(f'grid|{cid}', W=np.stack([g['recs'][j]['W'] for j in cases]),
                          F=np.stack([g['recs'][j]['fields_lat'] for j in cases]),
                          Ftight=np.stack([tight[j]['fields_lat'] for j in cases]),
                          Ftighter=np.stack([tighter[j]['fields_lat'] for j in subset]),
                          C=np.stack([g['recs'][j]['C'] for j in cases]), Ctight=np.stack([tight[j]['C'] for j in cases]),
                          Ctighter=np.stack([tighter[j]['C'] for j in subset]))
                    R.setdefault('grid', {})[cid] = summarize(g)
                    save()
                    log(f'   {cid}: q1={g["q1"]["q1"]} ' + ' '.join(
                        f"{j}:{g['recs'][j]['seconds_warm']:.3f}s ns{g['recs'][j]['cert']['nonstat']}" for j in cases))
        return True
    P('grid', grid_run, 600)
    R['grid'] = {cid: summarize(g) for cid, g in grid.items() if 'q1' in g}
    save()

    # ---- G-fd at real size (A1.9 G-fd-a/b/c, N-fd; A2.1 Sigma bounds)
    u0, nu = jnp.asarray(fam.ic(L, cases[0])), fam.nu(cases[0])

    def g_fd():
        import refcore as RC
        out = {}
        rng = np.random.default_rng(0)
        sh = H.shape_of(d, L)
        Fr = lambda u: (RC.adv_central(u.reshape(sh), L) - nu * RC.lap(u.reshape(sh), L)).reshape(-1)
        Nr = lambda u: RC.adv_central(u.reshape(sh), L).reshape(-1)
        # G-fd-a (A1.9): a random state; stencil values at 4096 sampled nodes and the stencil-Jacobian ENTRIES of 64
        # sampled rows against refcore (rows of its Jacobian by vjp; independent arithmetic)
        nodes = np.sort(rng.choice(cell.n, min(4096, cell.n), replace=False))
        idx = H.stencil_index(nodes, d, L)
        u = u0 + 0.1 * float(jnp.max(jnp.abs(u0))) * jnp.asarray(rng.standard_normal(cell.n))
        us = jnp.concatenate([u, jnp.zeros(1)])[jnp.asarray(idx)]
        Fs, _ = H.stencil_F(us, nu, L, d)
        Fref = Fr(u)[jnp.asarray(nodes)]
        dF, _ = H.stencil_dF(us, nu, L, d)
        _, vjp = jax.vjp(Fr, u)
        errs_j, scales_j = [], []
        for ii in range(min(64, len(nodes))):
            row = np.asarray(vjp(jnp.zeros(cell.n).at[int(nodes[ii])].set(1.))[0])
            want = np.zeros(cell.n)
            for sl in range(2 * d + 1):
                col = int(idx[ii, sl])
                if col < cell.n:
                    want[col] += float(dF[ii, sl])
            errs_j.append(float(np.max(np.abs(row - want))))
            scales_j.append(float(np.max(np.abs(row))))
        out['G-fd-a'] = dict(rows=float(jnp.max(jnp.abs(Fs - Fref)) / jnp.max(jnp.abs(Fref))), jac=fmax(errs_j) / fmax(scales_j),
                             n_rows=len(errs_j))
        # G-fd-b / N-fd: directional FD and jacfwd of each arm's online residual; G-fd-c (ECSW rows, Q-DEIM Sigma)
        c0 = cell.Phi[:, :rs].T @ u0
        cn = c0 + 1e-3 * jnp.asarray(rng.standard_normal(rs)) * jnp.linalg.norm(c0) / np.sqrt(rs)
        co = jnp.asarray([1., -1., 0., .5, .5])
        dt = H.DT0[d]
        for (kind, hrp), (dat, _) in datas.items():
            scf, rJt = RM.make_arm(kind, d, L)
            resf = RM.make_res(kind, d, L)              # residual only: no Jacobian tangent under autodiff (audit 5)
            const = scf(cn, c0, nu, co, dt, dat)
            c1 = cn + 1e-3 * jnp.asarray(rng.standard_normal(rs)) * jnp.linalg.norm(c0) / np.sqrt(rs)
            for mut in (None, 'drop_adv_jac'):
                _, rJ = RM.make_arm(kind, d, L, mutation=mut)
                r, J = rJ(c1, const, nu, co, dt, dat)
                efds = []
                for _ in range(5):
                    vv = jnp.asarray(rng.standard_normal(rs))
                    vv = vv / jnp.linalg.norm(vv)
                    h = 1e-6 * max(1., float(jnp.linalg.norm(c1)))
                    fd = (resf(c1 + h * vv, const, nu, co, dt, dat) - resf(c1 - h * vv, const, nu, co, dt, dat)) / (2 * h)
                    efds.append(float(jnp.linalg.norm(fd - J @ vv) / jnp.linalg.norm(fd)))
                rec = dict(fd=fmax(efds), fd_list=efds)
                if mut is None:
                    Ja = jax.jit(jax.jacfwd(resf))(c1, const, nu, co, dt, dat)
                    rec['jacfwd'] = float(jnp.max(jnp.abs(J - Ja)) / jnp.max(jnp.abs(Ja)))
                    rec['res_vs_rJ'] = float(jnp.max(jnp.abs(resf(c1, const, nu, co, dt, dat) - r)) / jnp.max(jnp.abs(r)))
                    P_ = cell.Phi[:, :rs]
                    u1, un = P_ @ c1, P_ @ cn
                    if kind == 'ecsw':
                        Rf = u1 - un + dt * (.5 * Fr(u1) + .5 * Fr(un))
                        nd = cell.ecsw_rec[rs][hrp]['nodes']
                        rec['rows'] = float(jnp.max(jnp.abs(r - dat['sw'] * Rf[jnp.asarray(nd)])) / jnp.max(jnp.abs(Rf)))
                    if kind == 'qdeim':
                        U = cell.U[:, :hrp]
                        Pp = jnp.asarray(cell.piv[hrp])
                        M = U[Pp]
                        LP = cell.LPhi[:, :rs]

                        def parts(cc):
                            t1 = P_ @ (cc - cn)
                            t2 = -dt * nu * (LP @ (.5 * cc + .5 * cn))
                            t3 = U @ jnp.linalg.solve(M, dt * (.5 * Nr(P_ @ cc)[Pp] + .5 * Nr(un)[Pp]))
                            return t1, t2, t3
                        Rd = lambda cc: sum(parts(cc))
                        cs = c1                      # cancellation candidate (A2.1): 8 GN iterations on the assembly
                        for _ in range(8):
                            Jd = jax.jacfwd(Rd)(cs)
                            cs = cs - jnp.linalg.lstsq(Jd, Rd(cs))[0]
                        for tag, cc in (('generic', c1), ('cancel', cs)):
                            t1, t2, t3 = parts(cc)
                            Bz = t1 + t2 + t3
                            Sig = float(jnp.linalg.norm(t1) + jnp.linalg.norm(t2) + jnp.linalg.norm(t3))
                            rq, Jq = rJt(cc, const, nu, co, dt, dat)       # analytic J is the object under test
                            g_ass = jax.grad(lambda x: 0.5 * jnp.sum(Rd(x) ** 2))(cc)
                            rec[f'norm_{tag}'] = abs(float(jnp.linalg.norm(rq)) - float(jnp.linalg.norm(Bz))) / Sig
                            rec[f'grad_{tag}'] = float(jnp.linalg.norm(Jq.T @ rq - g_ass)) / (Sig * float(jnp.linalg.norm(Jq)))
                            rec[f'cancel_ratio_{tag}'] = float(jnp.linalg.norm(Bz)) / Sig
                out[f'{kind}|{mut}'] = rec
        return out
    P('G-fd', g_fd, 240, store='G-fd')

    def g_all():
        nodes = np.arange(cell.n)
        de, _ = BD.ecsw_data(cell.Phi[:, :rs], nodes, np.ones(cell.n), d, L)
        de = dict(de, Phi=cell.Phi[:, :rs])
        a = cell.run('ecsw', rs, de, 'native', opt_main, 'tight', u0, nu)
        b = cell.run('none', rs, datas[('none', '-')][0], 'native', opt_main, 'tight', u0, nu)
        chunk('G-all', Fa=a['fields_lat'], Fb=b['fields_lat'], Ca=a['C'], Cb=b['C'])
        return dict(dist=SC.lattice_distance(a['fields_lat'], b['fields_lat'], R['n0'][cases[0]]),
                    valid=[a['cert']['valid'], b['cert']['valid']], cert_a=a['cert'], cert_b=b['cert'])
    if refs is not None and ('none', '-') in datas:
        P('G-all', g_all, 300, store='G-all')

    # ---- N-under (A1.9, A2.10, A6.3): ladder levels with complete Q1 on the two fixtures
    def q1_level(kind, dat, tag, support=None):
        fxs = sorted(fx)
        prod = {j: cell.run(kind, rs, dat, 'native', opt_main, 'prod', fam.ic(L, j), fam.nu(j)) for j in fxs}
        tight = {j: cell.run(kind, rs, dat, 'native', opt_main, 'tight', fam.ic(L, j), fam.nu(j)) for j in fxs}
        tighter = {j: cell.run(kind, rs, dat, 'native', opt_main, 'tighter', fam.ic(L, j), fam.nu(j)) for j in fxs}
        q = PL.q1_config(prod, tight, tighter, R['n0'], fxs, fxs)
        if support is not None and support < rs:        # A6.3: underdetermined LSPG is never Q1-qualified
            q = dict(q, q1=False, reason=f'underdetermined: support {support} < r {rs}')
        kw = {}
        for j in fxs:
            kw.update({f'{j}~F': prod[j]['fields_lat'], f'{j}~Ftight': tight[j]['fields_lat'],
                       f'{j}~Ftighter': tighter[j]['fields_lat'], f'{j}~C': prod[j]['C'], f'{j}~Ctight': tight[j]['C'],
                       f'{j}~Ctighter': tighter[j]['C']})
        chunk(f'NU|{tag}', **kw)
        return dict(e={j: refs.score_safe(prod[j]['fields_lat'], j)['e'] for j in fxs}, q1=q, support=support,
                    cert={j: prod[j]['cert'] for j in fxs}, tight={j: tight[j]['cert'] for j in fxs},
                    tighter={j: tighter[j]['cert'] for j in fxs})

    def n_under():
        out = {'none': q1_level('none', datas[('none', '-')][0], 'none')}
        R['N-under-partial'] = out
        save()
        G, b, cand, _ = BD.ecsw_contrib(cell.states, cell.Phi[:, :rs], d, L)
        for mult in (3., 2.5, 2., 1.5, 1.25):          # A15.2: under-sampled relative to the 4r fixture, all > r
            cap = int(np.ceil(mult * rs))
            o = BD.NN.greedy_nnls(G, b, taus=(), cap=cap)
            de, _ = BD.ecsw_data(cell.Phi[:, :rs], cand[o['support']], o['weights'], d, L)
            out[f'ecsw_cap{cap}'] = q1_level('ecsw', dict(de, Phi=cell.Phi[:, :rs]), f'ecsw_cap{cap}',
                                             support=int(len(o['support'])))
            R['N-under-partial'] = out
            save()
        del G, b
        for p in ladder_p:
            try:
                dq, _ = cell.arm_data('qdeim', rs, p)
                out[f'qdeim_p{p}'] = q1_level('qdeim', dq, f'qdeim_p{p}')
            except PL.ConstructionError as e:
                out[f'qdeim_p{p}'] = dict(construction_failure=str(e))
            R['N-under-partial'] = out
            save()
        R.pop('N-under-partial', None)
        return out
    if refs is not None and ('none', '-') in datas:
        P('N-under', n_under, 420, store='N-under')

    # ---- N-grad (A4.1): frozen fixture ecsw|r_s|tau|BE|dt0|native on fixture[0]; shared isolated-tuple certifier
    def n_grad():
        if ('ecsw', tau_run) not in datas:
            return dict(status='fixture unavailable', reason='ECSW fixture not constructed')
        dat = datas[('ecsw', tau_run)][0]
        j = fx[0]
        u0j, nuj = jnp.asarray(fam.ic(L, j)), fam.nu(j)
        base = cell.run('ecsw', rs, dat, 'native', opt_main, 'prod', u0j, nuj)
        if not base['cert']['valid']:
            return dict(status='fixture unavailable', reason='base run not certified-valid', base_cert=base['cert'])
        tg = jax.jit(RM.make_tuple_gamma('ecsw', d, L))
        scf, rJ = RM.make_arm('ecsw', d, L)
        resf = RM.make_res('ecsw', d, L)
        jac_or = jax.jit(jax.jacfwd(resf))
        sch = RM.schedule(d, opt_main[0], opt_main[1], 'prod')
        Cs = jnp.asarray(base['C'])
        tuples, skipped, skipped_steps = [], 0, []
        for n_ in range(0, int(sch['steps']), 5):
            co = RM.step_coefficients(sch, n_)
            cn_, cm_ = Cs[n_], Cs[max(n_ - 1, 0)]
            const = scf(cn_, cm_, nuj, co, sch['dt'], dat)
            c1 = Cs[n_ + 1]
            g_unpert = float(tg(cn_, cm_, c1, jnp.float64(nuj), co, sch['dt'], dat))
            tuples.append(dict(step=n_, k=None, sign=0., cert=g_unpert, cert_traj=float(base['gamma'][n_])))
            r1, J1 = rJ(c1, const, nuj, co, sch['dt'], dat)
            _, sv, vt = jnp.linalg.svd(J1, full_matrices=False)
            if float(sv[0]) == 0.:
                skipped += 1
                skipped_steps.append(dict(step=n_, sigma1=float(sv[0])))
                continue
            t0_ = max(100. * float(jnp.linalg.norm(r1)) / float(sv[0]), 1e-6 * float(jnp.linalg.norm(c1)), 1e-12)
            for k in range(7):
                for sgn in (1., -1.):
                    cp = c1 + sgn * t0_ * 2 ** k * vt[0]
                    g_cert = float(tg(cn_, cm_, cp, jnp.float64(nuj), co, sch['dt'], dat))
                    r = resf(cp, const, nuj, co, sch['dt'], dat)
                    Ja = jac_or(cp, const, jnp.float64(nuj), co, sch['dt'], dat)
                    g_or = float(jnp.linalg.norm(Ja.T @ r) / (jnp.linalg.norm(Ja) * jnp.linalg.norm(r) + 1e-300))
                    tuples.append(dict(step=n_, k=k, sign=sgn, t0=t0_, cert=g_cert, oracle=g_or))
        # production classification (RM.certify) on per-variant gamma arrays: every tested step replaced by its perturbed
        # tuple's gamma; the count of non-stationary steps must equal the oracle's count (code audit 2, item 10)
        base_g = np.asarray(base['gamma'], float)
        classify = []
        for k in range(7):
            for sgn in (1., -1.):
                tt = [t for t in tuples if t['k'] == k and t['sign'] == sgn]
                g_var = base_g.copy()
                for t in tt:
                    g_var[t['step']] = t['cert']
                cert_v = RM.certify(dict(stats=dict(reason=base['reason'], lam=base['lam'], it=base['it']), C=base['C'],
                                         scale=base['scale']), g_var, sch, 'prod')
                o_var = base_g.copy()
                for t in tt:
                    o_var[t['step']] = t['oracle']
                classify.append(dict(k=k, sign=sgn, nonstat_production=cert_v['nonstat'],
                                     nonstat_oracle=int(np.sum(~(np.isfinite(o_var) & (o_var <= 1e-3))))))
        chunk('NG', base_C=np.asarray(base['C']), base_gamma=np.asarray(base['gamma'], float))
        # part (b): the wrong-gradient solver on both fixtures, with complete Q1 (reported, A2.9 (b))
        fxs = sorted(fx)
        mut = {jj: cell.run('ecsw', rs, dat, 'native', opt_main, 'prod', fam.ic(L, jj), fam.nu(jj), mutation='drop_adv_jac')
               for jj in fxs}
        tight = {jj: cell.run('ecsw', rs, dat, 'native', opt_main, 'tight', fam.ic(L, jj), fam.nu(jj)) for jj in fxs}
        tighter = {jj: cell.run('ecsw', rs, dat, 'native', opt_main, 'tighter', fam.ic(L, jj), fam.nu(jj)) for jj in fxs}
        kwb = {}
        for jj in fxs:
            kwb.update({f'{jj}~F': mut[jj]['fields_lat'], f'{jj}~Ftight': tight[jj]['fields_lat'],
                        f'{jj}~Ftighter': tighter[jj]['fields_lat'], f'{jj}~C': mut[jj]['C'], f'{jj}~Ctight': tight[jj]['C'],
                        f'{jj}~Ctighter': tighter[jj]['C']})
        chunk('NGb', **kwb)
        return dict(status='run', skipped_steps=skipped_steps,
                    b_cert=dict(prod={jj: mut[jj]['cert'] for jj in fxs}, tight={jj: tight[jj]['cert'] for jj in fxs},
                                tighter={jj: tighter[jj]['cert'] for jj in fxs}), fixture=PL.cfg_id('ecsw', rs, tau_run, opt_main[0], opt_main[1], 'native'), case=j,
                    steps=int(sch['steps']), tested_steps=list(range(0, int(sch['steps']), 5)),
                    tuples=tuples, skipped=skipped, base_cert=base['cert'], classify=classify,
                    b=dict(q1=PL.q1_config(mut, tight, tighter, R['n0'], fxs, fxs), cert={jj: mut[jj]['cert'] for jj in fxs}))
    if refs is not None:
        P('N-grad', n_grad, 240, store='N-grad')

    # ---- N-swap (A3.3 (i), A4.4, A5.1)
    def n_swap():
        out = {}
        j0, j1 = cases[0], cases[1]
        Rf = refs.f[j0]
        out['eps_max'] = max(refs.score(Rf, j)['eps_ref'] for j in (j0, j1))
        out['ref_fixture'] = dict(true=refs.score(Rf, j0)['e'], case_swap=refs.score(Rf, j0, ref_case=j1)['e'],
                                  time_shift=refs.score(Rf, j0, shift=1)['e'])
        rom = None
        cands = {}
        for cid in (PL.cfg_id('ecsw', rs, tau_run, opt_main[0], opt_main[1], 'native'),
                    PL.cfg_id('none', rs, '-', opt_main[0], opt_main[1], 'native')):
            g = grid.get(cid)
            cands[cid] = 'absent' if g is None else ('Q1-qualified' if g.get('q1', {}).get('q1') else 'not Q1-qualified')
            if g is not None and g.get('q1', {}).get('q1'):                # A15.5: the ROM fixture must be Q1-qualified
                F = g['recs'][j0]['fields_lat']
                rom = dict(cid=cid, true=refs.score(F, j0)['e'], case_swap=refs.score(F, j0, ref_case=j1)['e'],
                           time_shift=refs.score(F, j0, shift=1)['e'], refstore=SC.RS.score(F, d, coh, j0, 'ST')['e'])
                break
        out['rom_fixture'] = rom if rom is not None else 'fixture unavailable'
        out['rom_candidates'] = cands
        kinds = {}
        for dd, kd, op in ((3, 'S', ('BE', 0.005)), (3, 'S', ('BE', 0.01)), (2, 'S', ('CNR', 0.005)),
                           (2, 'S', ('BE', 0.01)), (2, 'S', ('BE', 0.005)), (2, 'ST', ('CNR', 0.005))):
            try:
                SC.kind_allowed(dd, kd, op)
                kinds[f'{dd}|{kd}|{op[0]}|{op[1]}'] = 'allowed'
            except PermissionError:
                kinds[f'{dd}|{kd}|{op[0]}|{op[1]}'] = 'raised'
        out['kinds'] = kinds
        return out
    if refs is not None:
        P('N-swap', n_swap, 60, store='N-swap')

    # ---- N-time (A3.9) on the 6 timing cases, and ours' adapter (A-side) against the ECSW fixture
    def n_time_and_ours():
        out = {}
        # N-time uses the heavier `none` arm (stable call times); ours' A-B-A pairs with the ECSW fixture (A10.1)
        kind, hrp = ('none', '-')
        dat = datas[(kind, hrp)][0]
        q = cell.query(kind, rs, 'native')
        sch = RM.schedule(d, opt_main[0], opt_main[1], 'prod_native')     # the native solver's production level (A8.1)
        fA = lambda u, v: q(u, v, dat, sch)['fields']

        def fB(u, v):                                   # the same case twice, two synchronised host calls
            o1 = jax.block_until_ready(q(u, v, dat, sch)['fields'])
            return o1, q(u, v, dat, sch)['fields']
        nt = {}
        for j in TIMING_CASES:
            uj, vj = jnp.asarray(fam.ic(L, j)), jnp.float64(fam.nu(j))
            jax.block_until_ready(fA(uj, vj))           # warm
            scored = cell.run(kind, rs, dat, 'native', opt_main, 'prod', uj, fam.nu(j))['fields_lat']
            chkA = lambda o, sc=scored: disc_finite(cell.lattice(o), sc)
            chkB = lambda o, sc=scored: [disc_finite(cell.lattice(o[0]), sc), disc_finite(cell.lattice(o[1]), sc)]
            nt[j] = dict(triples=timed_triples(fA, fB, (uj, vj), (uj, vj), chkA, chkB))
        out['N-time'] = dict(config=PL.cfg_id(kind, rs, hrp, opt_main[0], opt_main[1], 'native'), cases=nt)
        R['timing-partial'] = out
        save()
        if cfg.get('dry_skip_ours'):                    # local CPU dry run only (never set in configs/smoke.json)
            out['ours'] = dict(skipped='dry run')
            R.pop('timing-partial', None)
            return out
        import ours as OU
        if d == 2:
            ox = OU.Ours2D(L, [dict(name='acc', Rp=384, rule='gauss128')], log=log)
            fX0 = ox.fn('acc', 'LSPG', 'CNR', 0.005)
            xid = 'combined acc R384 gauss128 LSPG-CNR 0.005'
        else:
            ox = OU.Ours3D('W1024', 256, 'gl24', a=6, log=log)
            fX0 = ox.fn(('fs', 'BE', 0.01, 1))
            xid = 'combined W256 gl24 FS-K1 BE 0.01'
        fX = lambda u, v: fX0(u, v)['fields']
        kind, hrp = ('ecsw', tau_run) if ('ecsw', tau_run) in datas else ('none', '-')
        dat = datas[(kind, hrp)][0]
        q = cell.query(kind, rs, 'native')
        fA = lambda u, v: q(u, v, dat, sch)['fields']
        out['ours_B'] = PL.cfg_id(kind, rs, hrp, opt_main[0], opt_main[1], 'native')
        ab = {}
        for j in TIMING_CASES:
            ux, vj = ox.u0(fam, j), jnp.float64(fam.nu(j))
            ub = jnp.asarray(fam.ic(L, j))
            jax.block_until_ready(fX(ux, vj))
            scoredX = ox.lattice(fX0(ux, vj))
            scoredB = cell.run(kind, rs, dat, 'native', opt_main, 'prod', ub, fam.nu(j))['fields_lat']
            chkX = lambda o, sc=scoredX: disc_finite(ox.lattice(dict(fields=o)), sc)
            chkB = lambda o, sc=scoredB: disc_finite(cell.lattice(o), sc)
            ab[j] = dict(triples=timed_triples(fX, fA, (ux, vj), (ub, vj), chkX, chkB))
            if j == 0:
                chunk('ours', F=scoredX)
                F_ours0 = scoredX
        out['ours'] = dict(id=xid, e0=refs.score_safe(F_ours0, 0) if refs is not None and 0 in refs.f else None,
                           note='phase-I adapter execution check; NOT G-ours (exported-error replay is phase II)',
                           cases=ab, setup_seconds=ox.setup_seconds)
        R.pop('timing-partial', None)
        return out
    P('N-time+ours', n_time_and_ours, 300, store='timing')

    R['timers'] = cell.T.t
    return arrays


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    log(f'jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64} devices={jax.devices()}')
    res = dict(config=cfg, phases={}, env=dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
                                               jax=jax.__version__, job=os.environ.get('SLURM_JOB_ID'),
                                               commit=cfg.get('commit')))
    save = Saver(outdir / 'smoke.json', res)
    res['disjoint'] = phase(res, 'disjoint', lambda: [H.assert_disjoint(2), H.assert_disjoint(3)], save=save)
    for c in cfg['cells']:
        arrays = {}
        try:
            arrays = run_cell(cfg, c, res, outdir, save)
        except Exception as e:  # noqa: BLE001
            res[f"{c['d']}d_L{c['L']}_fatal"] = f'{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}'
            log('FATAL in cell', c['d'], c['L'], e)
        save()
    res['total_seconds'] = time.perf_counter() - T_START
    save()
    log('done')


if __name__ == '__main__':
    main()
