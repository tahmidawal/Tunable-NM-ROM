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
        PL.dump(self.path, self.res)


def phase(res, name, fn, need=0., cfg=None, save=None):
    """Run one phase; record ok / error / skipped(time); never abort the smoke; checkpoint the JSON afterwards."""
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
    if save:
        save()
    return v


_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    """Keep the GPU busy for `seconds` before a timed call (combined lane's burn, c3common.burn)."""
    a = jax.block_until_ready(_BURN(jnp.ones((512, 512), jnp.float64) * .01))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def timed_triples(fA, fB, argsA, argsB, reps=3, burn_s=0.1, max_retry=3):
    """A-B-A triples (A3.9, A4.10): `reps` original triples; each drifting triple (|tA1/tA2 - 1| > 0.25) is re-measured
    up to `max_retry` times. Every attempt is kept: list of dict(rep, attempt, tA1, tB, tA2, valid). Returns also the
    outputs of the last A and B calls (timed outputs, compared with the scored ones by the caller)."""
    rec, last = [], {}
    for rep in range(reps):
        for attempt in range(max_retry + 1):
            ts = []
            for tag, f, a in (('A', fA, argsA), ('B', fB, argsB), ('A', fA, argsA)):
                burn(burn_s)
                t0 = time.perf_counter()
                o = jax.block_until_ready(f(*a))
                ts.append(time.perf_counter() - t0)
                last[tag] = o
            ok = abs(ts[0] / ts[2] - 1.) <= 0.25
            rec.append(dict(rep=rep, attempt=attempt, tA1=ts[0], tB=ts[1], tA2=ts[2], valid=bool(ok)))
            if ok:
                break
    return rec, last


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
    tau_run = c['taus'][-1]
    arrays = {}
    P = lambda name, fn, need: phase(R, name, fn, need, cfg, save)

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
    R['G-fom'] = P('G-fom', g_fom, 120)

    # ---- snapshots + POD (maximal rank) + trust; adequacy; generator probe (unit cost)
    rows = list(range(c['n_traj']))
    bb = P('basis', lambda: cell.build_basis(fam_tr, rows, c['rmax'], c['ranks']), 300)
    if bb is None:
        return arrays
    S, Phi, sig, trust = bb
    bb = None
    cell.set_basis(Phi, trust)
    R['basis'] = cell.info['basis[central]']

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
    R['adequacy'] = P('adequacy', adequacy, 120)

    def probe_cost():
        t0 = time.perf_counter()
        _, info = BD.snapshots(d, L, fam_tr, list(range(c['n_probe'])), scheme='BE', adv='upwind', log=None)
        return dict(seconds_per_traj=(time.perf_counter() - t0) / c['n_probe'], worst=info['worst_rel_residual'])
    R['probe_cost'] = P('generator_probe_cost', probe_cost, 120)

    # ---- advection POD (maximal p) with certificates at every p the smoke uses; Q-DEIM maximal-shape pilot
    ladder_p = []
    p = rs // 8
    while p >= 1:
        ladder_p.append(p)
        p //= 2
    p_list = sorted(set([c['ps'], c['pmax']] + ladder_p))
    R['p_list'] = p_list
    P('adv_pod', lambda: cell.build_adv(S, c['pmax'], p_list), 240)
    del S
    R['adv'] = {k: v for k, v in cell.info.get('adv', {}).items() if k != 'sigma'}

    def qdeim_pilot():
        t0 = time.perf_counter()
        dat, info = cell.qdeim(c['rmax'], c['pmax'])               # maximal B-QR shape (memory pilot)
        jax.block_until_ready(dat['T1'])
        out = dict(info, seconds=time.perf_counter() - t0, device_peak=PL.device_peak())
        del dat
        return out
    R['qdeim_pilot'] = P('qdeim_max_shape', qdeim_pilot, 240)

    # ---- ECSW: states, smoke-rank path, maximal-shape pilot
    R['ecsw_states'] = P('ecsw_states', lambda: cell.ecsw_states(fam_tr), 120)
    R['ecsw_rs'] = P('ecsw_smoke_rank', lambda: {str(t): (None if v is None else dict(size=v['size'], relres=v['relres']))
                                                for t, v in cell.ecsw(rs, taus=c['taus']).items()}, 240)

    def ecsw_pilot():
        t0 = time.perf_counter()
        G, b, cand, cinfo = BD.ecsw_contrib(cell.states, cell.Phi[:, :c['rmax']], d, L)
        jax.block_until_ready(G)
        tg = time.perf_counter() - t0
        shape = tuple(G.shape)
        t1 = time.perf_counter()
        o = BD.NN.greedy_nnls(G, b, taus=(), cap=c['pilot_support'])
        tn = time.perf_counter() - t1
        del G, b
        return dict(shape=shape, contrib_seconds=tg, nnls_seconds=tn, support=int(len(o['support'])),
                    relres=o['relres'], reason=o['reason'], iterations=len(o['path']), device_peak=PL.device_peak(),
                    host_maxrss=PL.host_rss())
    R['ecsw_pilot'] = P('ecsw_max_shape', ecsw_pilot, 300)

    def none_pilot():
        dat, _ = cell.arm_data('none', c['rmax'], '-')
        rec = cell.run('none', c['rmax'], dat, 'native', opt_main, 'prod', fam.ic(L, 0), fam.nu(0))
        return dict(seconds=rec['seconds'], cert=rec['cert'], device_peak=PL.device_peak())
    R['none_pilot'] = P('none_max_shape', none_pilot, 240)

    # ---- small grid with the complete Q1 machinery (A5.4)
    datas = {}
    for kind, hrp in (('none', '-'), ('ecsw', tau_run), ('qdeim', c['ps'])):
        try:
            if kind == 'ecsw' and cell.ecsw_rec.get(rs, {}).get(hrp) is None:
                raise PL.ConstructionError(f'ECSW tau={hrp} not reached at r={rs}')
            datas[(kind, hrp)] = cell.arm_data(kind, rs, hrp)
        except Exception as e:  # noqa: BLE001
            R.setdefault('construction_failures', {})[f'{kind}|{hrp}'] = f'{type(e).__name__}: {e}'
    R['arm_info'] = {f'{k[0]}|{k[1]}': v[1] for k, v in datas.items()}
    R['ecsw_info'] = {str(k): v for k, v in cell.info.get('ecsw', {}).items()}
    grid = {}
    subset = cases[:2]

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
                        recs[j]['timed_vs_scored'] = float(np.max(np.abs(warm['fields_lat'] - recs[j]['fields_lat'])))
                    grid[cid] = dict(recs=recs)
                tight = {j: cell.run(kind, rs, dat, 'common', option, 'tight', fam.ic(L, j), fam.nu(j)) for j in cases}
                tighter = {j: cell.run(kind, rs, dat, 'common', option, 'tighter', fam.ic(L, j), fam.nu(j)) for j in subset}
                for solver in ('native', 'common'):
                    cid = PL.cfg_id(kind, rs, hrp, option[0], option[1], solver)
                    g = grid[cid]
                    g['q1'] = PL.q1_config(g['recs'], tight, tighter, R['n0'], subset)
                    g['anchors'] = dict(tight={j: tight[j]['cert'] for j in tight},
                                        tighter={j: tighter[j]['cert'] for j in tighter},
                                        tight_F={j: tight[j]['fields_lat'] for j in tight},
                                        tighter_F={j: tighter[j]['fields_lat'] for j in tighter})
                    if refs is not None:
                        g['e0'] = {j: refs.score(g['recs'][j]['fields_lat'], j) for j in cases}
                    arrays[f'{cid}|W'] = np.stack([g['recs'][j]['W'] for j in cases])
                    arrays[f'{cid}|F'] = np.stack([g['recs'][j]['fields_lat'] for j in cases])
                    arrays[f'{cid}|Ftight'] = np.stack([tight[j]['fields_lat'] for j in cases])
                    arrays[f'{cid}|Ftighter'] = np.stack([tighter[j]['fields_lat'] for j in subset])
                    log(f'   {cid}: q1={g["q1"]["q1"]} ' + ' '.join(
                        f"{j}:{g['recs'][j]['seconds_warm']:.3f}s ns{g['recs'][j]['cert']['nonstat']}" for j in cases))
        return True
    P('grid', grid_run, 600)
    R['grid'] = {cid: dict(q1={k: v for k, v in g.get('q1', {}).items()}, e0=g.get('e0'),
                           anchors={k: v for k, v in g.get('anchors', {}).items() if not k.endswith('_F')},
                           runs={j: dict(seconds=g['recs'][j]['seconds'], seconds_warm=g['recs'][j].get('seconds_warm'),
                                         timed_vs_scored=g['recs'][j].get('timed_vs_scored'),
                                         cert=g['recs'][j]['cert'], gamma=g['recs'][j]['gamma'])
                                 for j in g['recs']}) for cid, g in grid.items()}
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
        # G-fd-a: raw stencil values and Jacobian action at sampled nodes vs refcore (independent arithmetic)
        nodes = np.sort(rng.choice(cell.n, min(4096, cell.n), replace=False))
        idx = H.stencil_index(nodes, d, L)
        u = cell.Phi[:, :rs] @ (cell.Phi[:, :rs].T @ u0)
        us = jnp.concatenate([u, jnp.zeros(1)])[jnp.asarray(idx)]
        Fs, _ = H.stencil_F(us, nu, L, d)
        Fref = Fr(u)[jnp.asarray(nodes)]
        dF, _ = H.stencil_dF(us, nu, L, d)
        v = jnp.asarray(rng.standard_normal(cell.n))
        vs = jnp.concatenate([v, jnp.zeros(1)])[jnp.asarray(idx)]
        Jv_st = jnp.sum(dF * vs, 1)
        Jv_ref = jax.jvp(Fr, (u,), (v,))[1][jnp.asarray(nodes)]
        out['G-fd-a'] = dict(rows=float(jnp.max(jnp.abs(Fs - Fref)) / jnp.max(jnp.abs(Fref))),
                             jac=float(jnp.max(jnp.abs(Jv_st - Jv_ref)) / jnp.max(jnp.abs(Jv_ref))))
        # G-fd-b / N-fd: directional FD and jacfwd of each arm's online residual; G-fd-c (ECSW rows, Q-DEIM Sigma)
        c0 = cell.Phi[:, :rs].T @ u0
        cn = c0 + 1e-3 * jnp.asarray(rng.standard_normal(rs)) * jnp.linalg.norm(c0) / np.sqrt(rs)
        co = jnp.asarray([1., -1., 0., .5, .5])
        dt = H.DT0[d]
        for (kind, hrp), (dat, _) in datas.items():
            scf, rJt = RM.make_arm(kind, d, L)
            const = scf(cn, c0, nu, co, dt, dat)
            c1 = cn + 1e-3 * jnp.asarray(rng.standard_normal(rs)) * jnp.linalg.norm(c0) / np.sqrt(rs)
            for mut in (None, 'drop_adv_jac'):
                _, rJ = RM.make_arm(kind, d, L, mutation=mut)
                r, J = rJ(c1, const, nu, co, dt, dat)
                efd = 0.
                for _ in range(5):
                    vv = jnp.asarray(rng.standard_normal(rs))
                    vv = vv / jnp.linalg.norm(vv)
                    h = 1e-6 * max(1., float(jnp.linalg.norm(c1)))
                    fd = (rJt(c1 + h * vv, const, nu, co, dt, dat)[0] - rJt(c1 - h * vv, const, nu, co, dt, dat)[0]) / (2 * h)
                    efd = max(efd, float(jnp.linalg.norm(fd - J @ vv) / jnp.linalg.norm(fd)))
                rec = dict(fd=efd)
                if mut is None:
                    Ja = jax.jacfwd(lambda x: rJt(x, const, nu, co, dt, dat)[0])(c1)
                    rec['jacfwd'] = float(jnp.max(jnp.abs(J - Ja)) / jnp.max(jnp.abs(Ja)))
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
                            rq, Jq = rJt(cc, const, nu, co, dt, dat)
                            g_ass = jax.grad(lambda x: 0.5 * jnp.sum(Rd(x) ** 2))(cc)
                            rec[f'norm_{tag}'] = abs(float(jnp.linalg.norm(rq)) - float(jnp.linalg.norm(Bz))) / Sig
                            rec[f'grad_{tag}'] = float(jnp.linalg.norm(Jq.T @ rq - g_ass)) / (Sig * float(jnp.linalg.norm(Jq)))
                            rec[f'cancel_ratio_{tag}'] = float(jnp.linalg.norm(Bz)) / Sig
                out[f'{kind}|{mut}'] = rec
        return out
    R['G-fd'] = P('G-fd', g_fd, 240)

    def g_all():
        nodes = np.arange(cell.n)
        de, _ = BD.ecsw_data(cell.Phi[:, :rs], nodes, np.ones(cell.n), d, L)
        de = dict(de, Phi=cell.Phi[:, :rs])
        a = cell.run('ecsw', rs, de, 'native', opt_main, 'tight', u0, nu)
        b = cell.run('none', rs, datas[('none', '-')][0], 'native', opt_main, 'tight', u0, nu)
        return dict(dist=SC.lattice_distance(a['fields_lat'], b['fields_lat'], R['n0'][cases[0]]),
                    valid=[a['cert']['valid'], b['cert']['valid']])
    if refs is not None and ('none', '-') in datas:
        R['G-all'] = P('G-all', g_all, 300)

    # ---- N-under (A1.9, A2.10, A6.3): ladder levels with complete Q1 on the two fixtures
    def q1_level(kind, dat):
        prod = {j: cell.run(kind, rs, dat, 'native', opt_main, 'prod', fam.ic(L, j), fam.nu(j)) for j in fx}
        tight = {j: cell.run(kind, rs, dat, 'common', opt_main, 'tight', fam.ic(L, j), fam.nu(j)) for j in fx}
        tighter = {j: cell.run(kind, rs, dat, 'common', opt_main, 'tighter', fam.ic(L, j), fam.nu(j)) for j in fx}
        q = PL.q1_config(prod, tight, tighter, R['n0'], list(fx))
        return dict(e={j: refs.score(prod[j]['fields_lat'], j)['e'] for j in fx}, q1=q,
                    cert={j: prod[j]['cert'] for j in fx})

    def n_under():
        out = {'none': q1_level('none', datas[('none', '-')][0])}
        G, b, cand, _ = BD.ecsw_contrib(cell.states, cell.Phi[:, :rs], d, L)
        for div in (1, 2, 4, 8):
            cap = int(np.ceil(rs / div))
            o = BD.NN.greedy_nnls(G, b, taus=(), cap=cap)
            de, _ = BD.ecsw_data(cell.Phi[:, :rs], cand[o['support']], o['weights'], d, L)
            out[f'ecsw_cap{cap}'] = dict(q1_level('ecsw', dict(de, Phi=cell.Phi[:, :rs])), support=int(len(o['support'])))
            R['N-under-partial'] = out
            save()
        del G, b
        for p in ladder_p:
            try:
                dq, _ = cell.arm_data('qdeim', rs, p)
                out[f'qdeim_p{p}'] = q1_level('qdeim', dq)
            except PL.ConstructionError as e:
                out[f'qdeim_p{p}'] = dict(construction_failure=str(e))
            R['N-under-partial'] = out
            save()
        R.pop('N-under-partial', None)
        return out
    if refs is not None and ('none', '-') in datas:
        R['N-under'] = P('N-under', n_under, 420)

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
        sch = RM.schedule(d, opt_main[0], opt_main[1], 'prod')
        Cs = jnp.asarray(base['C'])
        tuples, skipped = [], 0
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
                continue
            t0_ = max(100. * float(jnp.linalg.norm(r1)) / float(sv[0]), 1e-6 * float(jnp.linalg.norm(c1)), 1e-12)
            for k in range(7):
                for sgn in (1., -1.):
                    cp = c1 + sgn * t0_ * 2 ** k * vt[0]
                    g_cert = float(tg(cn_, cm_, cp, jnp.float64(nuj), co, sch['dt'], dat))
                    r, _ = rJ(cp, const, nuj, co, sch['dt'], dat)
                    Ja = jax.jacfwd(lambda x: rJ(x, const, nuj, co, sch['dt'], dat)[0])(cp)
                    g_or = float(jnp.linalg.norm(Ja.T @ r) / (jnp.linalg.norm(Ja) * jnp.linalg.norm(r) + 1e-300))
                    tuples.append(dict(step=n_, k=k, sign=sgn, cert=g_cert, oracle=g_or))
        mut = cell.run('ecsw', rs, dat, 'native', opt_main, 'prod', u0j, nuj, mutation='drop_adv_jac')
        tight = cell.run('ecsw', rs, dat, 'common', opt_main, 'tight', u0j, nuj)
        return dict(status='run', tuples=tuples, skipped=skipped, base_cert=base['cert'],
                    b=dict(cert=mut['cert'], dist=SC.lattice_distance(mut['fields_lat'], tight['fields_lat'], R['n0'][j])))
    if refs is not None:
        R['N-grad'] = P('N-grad', n_grad, 240)

    # ---- N-swap (A3.3 (i), A4.4, A5.1)
    def n_swap():
        out = {}
        j0, j1 = cases[0], cases[1]
        Rf = refs.f[j0]
        out['eps_max'] = max(refs.score(Rf, j)['eps_ref'] for j in (j0, j1))
        out['ref_fixture'] = dict(true=refs.score(Rf, j0)['e'], case_swap=refs.score(Rf, j0, ref_case=j1)['e'],
                                  time_shift=refs.score(Rf, j0, shift=1)['e'])
        rom = None
        for cid in (PL.cfg_id('ecsw', rs, tau_run, opt_main[0], opt_main[1], 'native'),
                    PL.cfg_id('none', rs, '-', opt_main[0], opt_main[1], 'native')):
            g = grid.get(cid)
            if g is not None and g['recs'][j0]['cert']['valid']:
                F = g['recs'][j0]['fields_lat']
                rom = dict(cid=cid, true=refs.score(F, j0)['e'], case_swap=refs.score(F, j0, ref_case=j1)['e'],
                           time_shift=refs.score(F, j0, shift=1)['e'], refstore=SC.RS.score(F, d, coh, j0, 'ST')['e'])
                break
        out['rom_fixture'] = rom if rom is not None else 'fixture unavailable'
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
        R['N-swap'] = P('N-swap', n_swap, 60)

    # ---- N-time (A3.9) on the 6 timing cases, and ours' adapter (A-side) against the ECSW fixture
    def n_time_and_ours():
        out = {}
        kind, hrp = ('ecsw', tau_run) if ('ecsw', tau_run) in datas else ('none', '-')
        dat = datas[(kind, hrp)][0]
        q = cell.query(kind, rs, 'native')
        sch = RM.schedule(d, opt_main[0], opt_main[1], 'prod')
        fA = lambda u, v: q(u, v, dat, sch)['fields']

        def fB(u, v):                                   # the same case twice, two synchronised host calls
            jax.block_until_ready(q(u, v, dat, sch)['fields'])
            return q(u, v, dat, sch)['fields']
        nt = {}
        for j in TIMING_CASES:
            uj, vj = jnp.asarray(fam.ic(L, j)), jnp.float64(fam.nu(j))
            jax.block_until_ready(fA(uj, vj))           # warm
            rec, last = timed_triples(fA, fB, (uj, vj), (uj, vj))
            scored = cell.run(kind, rs, dat, 'native', opt_main, 'prod', uj, fam.nu(j))['fields_lat']
            nt[j] = dict(triples=rec, timed_vs_scored=float(np.max(np.abs(cell.lattice(last['A']) - scored))))
        out['N-time'] = dict(config=PL.cfg_id(kind, rs, hrp, opt_main[0], opt_main[1], 'native'), cases=nt)
        R['timing-partial'] = out
        save()
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
        ab = {}
        for j in TIMING_CASES:
            ux, vj = ox.u0(fam, j), jnp.float64(fam.nu(j))
            ub = jnp.asarray(fam.ic(L, j))
            jax.block_until_ready(fX(ux, vj))
            rec, last = timed_triples(fX, fA, (ux, vj), (ub, vj))
            scoredX = ox.lattice(fX0(ux, vj))
            ab[j] = dict(triples=rec, timed_vs_scored_X=float(np.max(np.abs(ox.lattice(dict(fields=last['A'])) - scoredX))))
            if j == 0:
                arrays['ours|F'] = scoredX
        out['ours'] = dict(id=xid, e0=refs.score(arrays['ours|F'], 0) if refs is not None and 0 in refs.f else None,
                           cases=ab, setup_seconds=ox.setup_seconds)
        R.pop('timing-partial', None)
        return out
    R['timing'] = P('N-time+ours', n_time_and_ours, 300)

    idx = H.lattice_nodes(d, L)
    arrays['Phi_lat'] = np.asarray(cell.Phi[jnp.asarray(idx), :rs])
    arrays['cases'] = np.asarray(cases)
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
        if arrays:
            np.savez(outdir / f"arrays_{c['d']}d_L{c['L']}.npz", **{k.replace('|', '~'): v for k, v in arrays.items()})
        save()
    res['total_seconds'] = time.perf_counter() - T_START
    save()
    log('done')


if __name__ == '__main__':
    main()
