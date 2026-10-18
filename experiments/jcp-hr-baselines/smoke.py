"""jcp-hr-baselines phase-I real-size smoke (DESIGN.md 10; A1.9, A2.9-A2.11, A2.19, A3.2, A3.3, A3.6, A3.9).

One cluster job, one GPU, <= 1 h wall (sbatch --time=01:00:00). For each cell (2D L=1024, 3D L=64): reduced snapshot
sets with the maximal registered shapes (memory/time pilots), a small configuration grid with Q1 machinery, every
control at real size, the A-B-A harness and ours' adapters. Writes <out>/smoke.json progressively and <out>/arrays_*.npz
(coefficients, lattice bases, fields) for the offline G-score check. Predicates are recomputed offline by
smoke_accept.py from these raw outputs; nothing here declares acceptance.

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
import hrcore as H  # noqa: E402
import rom as RM  # noqa: E402
import build as BD  # noqa: E402
import pipeline as PL  # noqa: E402
import score as SC  # noqa: E402

T_START = time.perf_counter()


def log(*a):
    print(f'[{time.perf_counter() - T_START:7.1f}s]', *a, flush=True)


def left(cfg):
    return cfg['wall_seconds'] - (time.perf_counter() - T_START)


def phase(res, name, fn, need=0., cfg=None):
    """Run one phase; record ok / error / skipped(time); never abort the smoke."""
    if cfg is not None and left(cfg) < need:
        res['phases'][name] = dict(status='skipped', reason=f'time left {left(cfg):.0f}s < {need:.0f}s')
        log(f'SKIP {name}')
        return None
    t0 = time.perf_counter()
    try:
        v = fn()
        res['phases'][name] = dict(status='ok', seconds=time.perf_counter() - t0, device_peak=PL.device_peak(),
                                   host_maxrss=PL.host_rss())
        log(f'OK   {name} ({time.perf_counter() - t0:.1f}s)')
        return v
    except Exception as e:  # noqa: BLE001
        res['phases'][name] = dict(status='error', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-3000:],
                                   seconds=time.perf_counter() - t0)
        log(f'ERR  {name}: {type(e).__name__}: {e}')
        return None


_BURN = jax.jit(lambda a: a @ a / 512 + .01)


def burn(seconds):
    """Keep the GPU busy for `seconds` before a timed call (combined lane's burn, c3common.burn)."""
    a = jax.block_until_ready(_BURN(jnp.ones((512, 512), jnp.float64) * .01))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def timed_pair(fA, fB, args_A, args_B, reps=3, burn_s=0.1, max_retry=3):
    """A-B-A triples (A3.9): returns list of (tA1, tB, tA2, valid)."""
    out = []
    for _ in range(reps):
        for attempt in range(max_retry + 1):
            ts = []
            for f, a in ((fA, args_A), (fB, args_B), (fA, args_A)):
                burn(burn_s)
                t0 = time.perf_counter()
                o = f(*a)
                jax.block_until_ready(o)
                ts.append(time.perf_counter() - t0)
            ok = abs(ts[0] / ts[2] - 1.) <= 0.25
            if ok or attempt == max_retry:
                out.append((ts[0], ts[1], ts[2], ok))
                break
    return out


def run_cell(cfg, c, res, outdir):
    d, L = c['d'], c['L']
    key = f'{d}d_L{L}'
    R = res.setdefault(key, dict(phases={}))
    cell = PL.Cell(d, L, log=log)
    fam_tr = H.Family(d, 'train')
    coh = 'dev6' if d == 2 else 'val64'
    fam = H.Family(d, coh, count=None if d == 2 else max(c['cases'] + c['under_fixtures']) + 1)
    cases = c['cases']
    opt_main = H.OPTIONS[d][0]
    arrays = {}

    # ---- P0 references and IC checks
    refs = phase(R, 'refs', lambda: SC.Refs(d, coh, fam, sorted(set(cases + c['under_fixtures']))), 60, cfg)
    if refs is not None:
        R['n0'] = {j: refs.n0(j) for j in refs.f}

    # ---- P1 G-fom at the real mesh (BE and CN at dt0)
    def g_fom():
        sys.path.insert(0, str(HERE / 'e0'))
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
            out[sc_h] = dict(rel=float(np.max(np.linalg.norm(a - b, axis=1)) / np.linalg.norm(b[0])), fom_seconds=ta)
        return out
    R['G-fom'] = phase(R, 'G-fom', g_fom, 120, cfg)

    # ---- P2 snapshots + POD (maximal rank) + trust; P3 adequacy; P4 generator probe (unit cost)
    rows = list(range(c['n_traj']))
    bb = phase(R, 'basis', lambda: cell.build_basis(fam_tr, rows, c['rmax'], c['ranks']), 300, cfg)
    if bb is None:
        return
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
            out[r]['ratio'] = out[r]['held'] / max(out[r]['inset'], 1e-300)
        return out
    R['adequacy'] = phase(R, 'adequacy', adequacy, 120, cfg)

    def probe_cost():
        t0 = time.perf_counter()
        Su, info = BD.snapshots(d, L, fam_tr, list(range(c['n_probe'])), scheme='BE', adv='upwind', log=None)
        return dict(seconds_per_traj=(time.perf_counter() - t0) / c['n_probe'], worst=info['worst_rel_residual'])
    R['probe_cost'] = phase(R, 'generator_probe_cost', probe_cost, 120, cfg)

    # ---- P5 advection POD (maximal p), Q-DEIM pilots
    rhoN = phase(R, 'adv_pod', lambda: cell.build_adv(S, c['pmax'], [c['ps'], c['pmax']]), 240, cfg)
    del S
    R['adv'] = {k: v for k, v in cell.info.get('adv', {}).items() if k != 'sigma'}

    def qdeim_pilot():
        out = {}
        t0 = time.perf_counter()
        dat, info = cell.qdeim(c['rmax'], c['pmax'])             # maximal B-QR shape (memory pilot)
        sync = jax.block_until_ready(dat['T1'])
        out['max'] = dict(info, seconds=time.perf_counter() - t0, device_peak=PL.device_peak())
        del dat, sync
        return out
    R['qdeim_pilot'] = phase(R, 'qdeim_max_shape', qdeim_pilot, 240, cfg)

    # ---- P6 ECSW: states, smoke-rank path, maximal-shape pilot
    R['ecsw_states'] = phase(R, 'ecsw_states', lambda: cell.ecsw_states(fam_tr), 120, cfg)
    R['ecsw_rs'] = phase(R, 'ecsw_smoke_rank', lambda: {str(t): (None if v is None else dict(size=v['size'], relres=v['relres']))
                                                         for t, v in cell.ecsw(c['rs'], taus=c['taus']).items()}, 240, cfg)

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
    R['ecsw_pilot'] = phase(R, 'ecsw_max_shape', ecsw_pilot, 300, cfg)

    # ---- P8 small grid with Q1 machinery
    rs = c['rs']
    datas = {}
    tau_run = c['taus'][-1]
    if cell.ecsw_rec.get(rs, {}).get(tau_run) is not None:
        datas[('ecsw', tau_run)] = cell.arm_data('ecsw', rs, tau_run)
    datas[('none', '-')] = cell.arm_data('none', rs, '-')
    if rhoN is not None:
        datas[('qdeim', c['ps'])] = cell.arm_data('qdeim', rs, c['ps'])
    R['arm_info'] = {f'{k[0]}|{k[1]}': v[1] for k, v in datas.items()}
    grid = {}

    def grid_run():
        for (kind, hrp), (dat, _) in datas.items():
            for option in c['options']:
                option = tuple(option)
                for solver in ('native', 'common'):
                    cid = PL.cfg_id(kind, rs, hrp, option[0], option[1], solver)
                    recs = {j: cell.run(kind, rs, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j)) for j in cases}
                    # second timed pass (warm) -> screening time
                    for j in cases:
                        recs[j]['seconds_warm'] = cell.run(kind, rs, dat, solver, option, 'prod', fam.ic(L, j), fam.nu(j),
                                                           certify=False)['seconds']
                    grid[cid] = dict(recs=recs)
                    log(f'   {cid}: ' + ' '.join(f"{j}:{recs[j]['seconds_warm']:.3f}s ns{recs[j]['cert']['nonstat']}"
                                                 for j in cases))
                # anchors (common solver, shared by both solvers)
                tight = {j: cell.run(kind, rs, dat, 'common', option, 'tight', fam.ic(L, j), fam.nu(j)) for j in cases}
                tighter = {cases[0]: cell.run(kind, rs, dat, 'common', option, 'tighter', fam.ic(L, cases[0]),
                                              fam.nu(cases[0]))}
                for solver in ('native', 'common'):
                    cid = PL.cfg_id(kind, rs, hrp, option[0], option[1], solver)
                    grid[cid]['q1'] = PL.q1_config(grid[cid]['recs'], tight, tighter, R['n0'])
                    if refs is not None:
                        grid[cid]['e0'] = {j: refs.score(grid[cid]['recs'][j]['fields_lat'], j) for j in cases}
                    arrays[f'{cid}|W'] = np.stack([grid[cid]['recs'][j]['W'] for j in cases])
                    arrays[f'{cid}|F'] = np.stack([grid[cid]['recs'][j]['fields_lat'] for j in cases])
        return True
    phase(R, 'grid', grid_run, 600, cfg)
    R['grid'] = {cid: dict(q1={k: v for k, v in g.get('q1', {}).items()}, e0=g.get('e0'),
                           runs={j: dict(seconds=g['recs'][j]['seconds'], seconds_warm=g['recs'][j]['seconds_warm'],
                                         cert=g['recs'][j]['cert'], worst_gamma=g['recs'][j]['cert']['worst_gamma'])
                                 for j in g['recs']}) for cid, g in grid.items()}

    # ---- P9 controls at real size
    u0, nu = jnp.asarray(fam.ic(L, cases[0])), fam.nu(cases[0])

    def g_fd():
        out = {}
        rng = np.random.default_rng(0)
        c0 = cell.Phi[:, :rs].T @ u0
        cn = c0 + 1e-3 * jnp.asarray(rng.standard_normal(rs)) * jnp.linalg.norm(c0) / np.sqrt(rs)
        co = jnp.asarray([1., -1., 0., .5, .5])
        dt = H.DT0[d]
        sh = H.shape_of(d, L)
        sys.path.insert(0, str(HERE / 'e0'))
        import refcore as RC
        Fr = lambda u: (RC.adv_central(u.reshape(sh), L) - nu * RC.lap(u.reshape(sh), L)).reshape(-1)
        for (kind, hrp), (dat, _) in datas.items():
            for mut in (None, 'drop_adv_jac'):
                scf, rJ = RM.make_arm(kind, d, L, mutation=mut)
                _, rJt = RM.make_arm(kind, d, L)
                const = scf(cn, c0, nu, co, dt, dat)
                c1 = cn + 1e-3 * jnp.asarray(rng.standard_normal(rs)) * jnp.linalg.norm(c0) / np.sqrt(rs)
                r, J = rJ(c1, const, nu, co, dt, dat)
                efd = 0.
                for _ in range(5):
                    v = jnp.asarray(rng.standard_normal(rs))
                    v = v / jnp.linalg.norm(v)
                    h = 1e-6 * max(1., float(jnp.linalg.norm(c1)))
                    fd = (rJt(c1 + h * v, const, nu, co, dt, dat)[0] - rJt(c1 - h * v, const, nu, co, dt, dat)[0]) / (2 * h)
                    efd = max(efd, float(jnp.linalg.norm(fd - J @ v) / jnp.linalg.norm(fd)))
                rec = dict(fd=efd)
                if mut is None:
                    Ja = jax.jacfwd(lambda x: rJt(x, const, nu, co, dt, dat)[0])(c1)
                    rec['jacfwd'] = float(jnp.max(jnp.abs(J - Ja)) / jnp.max(jnp.abs(Ja)))
                    if kind == 'ecsw':
                        u1, un = cell.Phi[:, :rs] @ c1, cell.Phi[:, :rs] @ cn
                        Rf = u1 - un + dt * (.5 * Fr(u1) + .5 * Fr(un))
                        nodes = cell.ecsw_rec[rs][hrp]['nodes']
                        rec['rows'] = float(jnp.max(jnp.abs(r - dat['sw'] * Rf[jnp.asarray(nodes)])) / jnp.max(jnp.abs(Rf)))
                out[f'{kind}|{mut}'] = rec
        return out
    R['G-fd'] = phase(R, 'G-fd', g_fd, 240, cfg)

    def g_all():
        nodes = np.arange(cell.n)
        de, _ = BD.ecsw_data(cell.Phi[:, :rs], nodes, np.ones(cell.n), d, L)
        de = dict(de, Phi=cell.Phi[:, :rs])
        a = cell.run('ecsw', rs, de, 'native', opt_main, 'tight', u0, nu)
        b = cell.run('none', rs, datas[('none', '-')][0], 'native', opt_main, 'tight', u0, nu)
        return dict(dist=SC.lattice_distance(a['fields_lat'], b['fields_lat'], R['n0'][cases[0]]))
    R['G-all'] = phase(R, 'G-all', g_all, 300, cfg)

    def n_under():
        out = {}
        fx = c['under_fixtures']
        base = {j: cell.run('none', rs, datas[('none', '-')][0], 'native', opt_main, 'prod', fam.ic(L, j), fam.nu(j))
                for j in fx}
        out['none'] = {j: refs.score(base[j]['fields_lat'], j)['e'] for j in fx}
        # ECSW support caps r/4, r/8 (NNLS to the cap on the smoke-rank contributions)
        G, b, cand, _ = BD.ecsw_contrib(cell.states, cell.Phi[:, :rs], d, L)
        for div in (4, 8):
            cap = int(np.ceil(rs / div))
            o = BD.NN.greedy_nnls(G, b, taus=(), cap=cap)
            de, _ = BD.ecsw_data(cell.Phi[:, :rs], cand[o['support']], o['weights'], d, L)
            de = dict(de, Phi=cell.Phi[:, :rs])
            rr = {j: cell.run('ecsw', rs, de, 'native', opt_main, 'prod', fam.ic(L, j), fam.nu(j)) for j in fx}
            out[f'ecsw_cap{cap}'] = {j: dict(e=refs.score(rr[j]['fields_lat'], j)['e'], valid=rr[j]['cert']['valid'],
                                             nonstat=rr[j]['cert']['nonstat']) for j in fx}
        del G, b
        p = int(np.ceil(rs / 8))
        while p >= 1:
            dq, _ = cell.arm_data('qdeim', rs, p)
            rr = {j: cell.run('qdeim', rs, dq, 'native', opt_main, 'prod', fam.ic(L, j), fam.nu(j)) for j in fx}
            out[f'qdeim_p{p}'] = {j: dict(e=refs.score(rr[j]['fields_lat'], j)['e'], valid=rr[j]['cert']['valid'],
                                          nonstat=rr[j]['cert']['nonstat']) for j in fx}
            p //= 2
        return out
    if refs is not None:
        R['N-under'] = phase(R, 'N-under', n_under, 420, cfg)

    def n_grad():
        kind, hrp = ('ecsw', tau_run) if ('ecsw', tau_run) in datas else ('none', '-')
        dat = datas[(kind, hrp)][0]
        j = c['under_fixtures'][0]
        u0j, nuj = jnp.asarray(fam.ic(L, j)), fam.nu(j)
        base = cell.run(kind, rs, dat, 'native', opt_main, 'prod', u0j, nuj)
        # (a) isolated tuples, perturbation ladder, independent jacfwd oracle (A4.1)
        scf, rJ = RM.make_arm(kind, d, L)
        sch = RM.schedule(d, opt_main[0], opt_main[1], 'prod')
        Cs = jnp.asarray(base['C'])
        be = jnp.asarray(H.BE_CO)
        tuples, skipped = [], 0
        for n_ in range(0, int(sch['steps']), 5):
            co = be if n_ < int(sch['nstart']) else sch['co']
            const = scf(Cs[n_], Cs[max(n_ - 1, 0)], nuj, co, sch['dt'], dat)
            c1 = Cs[n_ + 1]
            r1, J1 = rJ(c1, const, nuj, co, sch['dt'], dat)
            _, sv, vt = jnp.linalg.svd(J1, full_matrices=False)
            if float(sv[0]) == 0.:
                skipped += 1
                continue
            t0_ = max(100. * float(jnp.linalg.norm(r1)) / float(sv[0]), 1e-6 * float(jnp.linalg.norm(c1)), 1e-12)
            for k in range(7):
                for sgn in (1., -1.):
                    cp = c1 + sgn * t0_ * 2 ** k * vt[0]
                    r, J = rJ(cp, const, nuj, co, sch['dt'], dat)
                    g_cert = float(RM.ratio(J, r))
                    Ja = jax.jacfwd(lambda x: rJ(x, const, nuj, co, sch['dt'], dat)[0])(cp)
                    g_or = float(jnp.linalg.norm(Ja.T @ r) / (jnp.linalg.norm(Ja) * jnp.linalg.norm(r) + 1e-300))
                    tuples.append(dict(step=n_, k=k, sign=sgn, cert=g_cert, oracle=g_or))
        a = dict(tuples=tuples, skipped=skipped)
        # (b) wrong-gradient solver (reported only)
        mut = cell.run(kind, rs, dat, 'native', opt_main, 'prod', u0j, nuj, mutation='drop_adv_jac')
        tight = cell.run(kind, rs, dat, 'common', opt_main, 'tight', u0j, nuj)
        b_ = dict(cert=mut['cert'], dist=SC.lattice_distance(mut['fields_lat'], tight['fields_lat'], R['n0'][j]))
        return dict(a=a, b=b_)
    if refs is not None:
        R['N-grad'] = phase(R, 'N-grad', n_grad, 240, cfg)

    def n_swap():
        out = {}
        j0, j1 = cases[0], cases[1]
        Rf = refs.f[j0]
        out['ref_fixture'] = dict(true=refs.score(Rf, j0)['e'], case_swap=refs.score(Rf, j0, ref_case=j1)['e'],
                                  time_shift=refs.score(Rf, j0, shift=1)['e'])
        cid = next((k for k in grid if k.startswith('ecsw|') and '|native|' in k), None)
        if cid is not None:
            F = grid[cid]['recs'][j0]['fields_lat']
            out['rom_fixture'] = dict(cid=cid, true=refs.score(F, j0)['e'], case_swap=refs.score(F, j0, ref_case=j1)['e'],
                                      time_shift=refs.score(F, j0, shift=1)['e'],
                                      refstore_score=SC.RS.score(F, d, coh, j0, 'ST')['e'])
        out['eps_max'] = max(refs.score(Rf, j)['eps_ref'] for j in (j0, j1))
        for dd, kd, op in ((3, 'S', ('BE', 0.005)), (3, 'S', ('BE', 0.01)), (2, 'S', ('CNR', 0.005)),
                           (2, 'S', ('BE', 0.01)), (2, 'S', ('BE', 0.005))):
            try:
                SC.kind_allowed(dd, kd, op)
                out[f'kind|{dd}|{op}'] = 'allowed'
            except PermissionError:
                out[f'kind|{dd}|{op}'] = 'raised'
        return out
    if refs is not None:
        R['N-swap'] = phase(R, 'N-swap', n_swap, 60, cfg)

    def n_time_and_ours():
        out = {}
        kind, hrp = ('ecsw', tau_run) if ('ecsw', tau_run) in datas else ('none', '-')
        dat = datas[(kind, hrp)][0]
        q = cell.query(kind, rs, 'native')
        sch = RM.schedule(d, opt_main[0], opt_main[1], 'prod')
        fA = lambda u, v: q(u, v, dat, sch)['fields']
        fB = lambda u, v: (jax.block_until_ready(q(u, v, dat, sch)['fields']), q(u, v, dat, sch)['fields'])[1]
        tri = timed_pair(fA, fB, (u0, jnp.float64(nu)), (u0, jnp.float64(nu)))
        out['N-time'] = dict(triples=tri, ratio=float(np.median([b / (.5 * (a + c_)) for a, b, c_, _ in tri])))
        # ours' adapter at real size (A-side), timed against the baseline config, and its lattice fields
        import ours as OU
        if d == 2:
            o2 = OU.Ours2D(L, [dict(name='acc', Rp=384, rule='gauss128')], log=log)
            fX = o2.fn('acc', 'LSPG', 'CNR', 0.005)
            uX = o2.u0(fam, cases[0])
            outX = fX(uX, jnp.float64(nu))
            FX = o2.lattice(outX)
        else:
            o3 = OU.Ours3D('W1024', 256, 'gl24', a=6, log=log)
            fX = o3.fn(('fs', 'BE', 0.01, 1))
            uX = o3.u0(fam, cases[0])
            outX = fX(uX, jnp.float64(nu))
            FX = o3.lattice(outX)
        fXf = lambda u, v: fX(u, v)['fields']
        trip = timed_pair(fXf, fA, (uX, jnp.float64(nu)), (u0, jnp.float64(nu)))
        out['ours'] = dict(e0=refs.score(FX, cases[0]) if refs is not None else None,
                           triples=trip, ratio_B_over_ours=float(np.median([b / (.5 * (a + c_)) for a, b, c_, _ in trip])))
        arrays['ours|F'] = FX
        return out
    R['timing'] = phase(R, 'N-time+ours', n_time_and_ours, 300, cfg)

    # lattice basis rows for offline G-score (decode W on the CPU)
    idx = H.lattice_nodes(d, L)
    arrays['Phi_lat'] = np.asarray(cell.Phi[jnp.asarray(idx), :rs])
    arrays['cases'] = np.asarray(cases)
    np.savez(outdir / f'arrays_{key}.npz', **{k.replace('|', '~'): v for k, v in arrays.items()})
    R['timers'] = cell.T.t
    R['info_keys'] = list(cell.info)


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
    res['disjoint'] = phase(res, 'disjoint', lambda: [H.assert_disjoint(2), H.assert_disjoint(3)])
    for c in cfg['cells']:
        try:
            run_cell(cfg, c, res, outdir)
        except Exception as e:  # noqa: BLE001
            res[f"{c['d']}d_L{c['L']}_fatal"] = f'{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}'
            log('FATAL in cell', c['d'], c['L'], e)
        PL.dump(outdir / 'smoke.json', res)
    res['total_seconds'] = time.perf_counter() - T_START
    PL.dump(outdir / 'smoke.json', res)
    log('done')


if __name__ == '__main__':
    main()
