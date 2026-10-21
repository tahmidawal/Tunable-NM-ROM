"""imp-geometry cell driver (DESIGN A1). One cluster job = one (domain, family, level) cell, stages in order, each
checkpointed in the job directory (re-running skips finished stages):

  ref    FD references (3 levels) for dev / val / test (test written to refs_test.npz, never read before `test`),
         time check, Richardson labels, allowance U, interaction inputs (lattice fields), zero-field baseline
  data   192 training trajectories on the data grid; POD (R_max = 512) in-job; bank samples; ECSW states
  bank   Stage-A bank (envelope of the domain), rotation on the reference rule
  ladder competitor + reduced ladders on val32 (+ dev8 for Q1/Q2), rows -> ladder.jsonl
  time   finalists per (tau, statistic) and family (pre-registered selection), Q1/Q2, paired A-B-A timing -> final.json
  test   (separate job) frozen finalists on test32: accuracy, Q1, paired re-timing (MF-LEAK manifest check)

  python run_cell.py --dom L --fam wide --lev S1 --out <jobdir> --stages ref,data,bank,ladder,time [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geom as G        # noqa: E402
import fd               # noqa: E402
import sem              # noqa: E402
import spec             # noqa: E402
import rom as RM        # noqa: E402
import pod as PD        # noqa: E402
import score as SC      # noqa: E402
import ladder as LD     # noqa: E402
import finalize as FZ   # noqa: E402

REF_LEVELS = {'S1': ((128, 1000), (256, 2000), (512, 4000)), 'S2': ((256, 2000), (512, 4000), (1024, 8000))}
DATA_GRID = {'S1': (512, 4000), 'S2': (768, 8000)}
BANK_SAMPLE = {'S1': 128, 'S2': 256}
FD_SIZES = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768)
SEM_P = (4, 6, 8, 10, 12)
SEM_K = (1, 2, 3, 4, 6, 8)
SPEC_N = (16, 24, 32, 48, 64, 96, 128, 192)
R_LADDER = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
Q_LADDER = (4, 6, 8, 10, 12)
SEM_FINE_STEPS = 2560


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def env_record():
    return dict(jax=jax.__version__, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), x64=bool(jax.config.read('jax_enable_x64')),
                host=os.uname().nodename, commit=os.environ.get('LANE_COMMIT'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohorts(fam, lev, smoke):
    D = {s: G.draw(fam, lev, s) for s in G.SPLITS}
    if smoke:
        D = {'train': D['train'][:12], 'dev': D['dev'][:2], 'val': D['val'][:4], 'test': D['test'][:2]}
    return D


# ------------------------------------------------------------------------------------------------ stage ref ----

def stage_ref(a, out, D):
    lev = a.lev
    levels = REF_LEVELS[lev]
    if a.smoke:
        levels = ((64, 500), (128, 1000), (256, 2000))
    res = {}
    for split in ('dev', 'val', 'test'):
        Us, stats = [], []
        for (n, nt) in levels:
            q = fd.make_sbdf2(a.dom, n, nt, 1e-10, maxit=60)
            O, st, sec = LD.run_cohort(q, D[split])
            Us.append(O)
            stats.append(st)
            log(f'ref {split} n{n} nt{nt}: {sec:.0f}s, mg max {st[:, 1].max()}, unconverged {st[:, 2].sum()}')
        Us = np.stack(Us)                                     # (3, cases, 6, P)
        conv = np.stack(stats)[:, :, 2] == 0                  # every solve converged
        rich = [SC.richardson(Us[0, j], Us[1, j], Us[2, j]) for j in range(Us.shape[1])]
        for j, r in enumerate(rich):
            if not conv[:, j].all():
                r['label'], r['reason'] = 'ineligible', 'MG solve not converged'
        f = out / ('refs_test.npz' if split == 'test' else f'refs_{split}.npz')
        np.savez(f, U=Us[2], U1=Us[0], U2=Us[1])
        res[split] = dict(rich=rich, file=f.name, sha=sha(f), mg_unconverged=int((~conv).sum()))
    # time check at the finest h: n_t vs 2 n_t on dev[0:4] + val[0:4] (selection cohorts only)
    n, nt = levels[-1]
    Dt = np.concatenate([D['dev'][:4], D['val'][:4]])
    O1 = LD.run_cohort(fd.make_sbdf2(a.dom, n, nt, 1e-10, maxit=60), Dt)[0]
    O2 = LD.run_cohort(fd.make_sbdf2(a.dom, n, 2 * nt, 1e-10, maxit=60), Dt)[0]
    dt_ = float(np.max(np.linalg.norm(O1[:, 1:] - O2[:, 1:], axis=-1) / np.linalg.norm(O2[:, :1], axis=-1)))
    allow = SC.allowance(res['val']['rich'], dt_)
    P128, I128, J128 = G.lattice(a.dom, 128)
    s128 = n // 128
    O128 = LD.run_cohort(fd.make_sbdf2(a.dom, n, nt, 1e-10, maxit=60, out_idx=(I128 * s128 - 1, J128 * s128 - 1)),
                         D['val'][:8])[0]
    np.savez(out / 'refs_val128.npz', U=O128)
    Uv = np.load(out / 'refs_val.npz')['U']
    zero = np.max(np.linalg.norm(Uv[:, 1:], axis=-1), axis=-1) / np.linalg.norm(Uv[:, 0], axis=-1)
    decay = np.linalg.norm(Uv[:, 5], axis=-1) / np.linalg.norm(Uv[:, 0], axis=-1)
    # MF-REF on real data: finest replaced by the middle level must be ineligible
    U = np.load(out / 'refs_val.npz')
    mf = SC.richardson(U['U1'][0], U['U2'][0], U['U2'][0].copy())['label'] == 'ineligible'
    rec = dict(levels=levels, splits={k: dict(v, rich=v['rich']) for k, v in res.items()}, delta_t=dt_,
               allowance=allow, zero_baseline=dict(worst=float(zero.max()), median=float(np.median(zero))),
               decay_median=float(np.median(decay)), MF_REF_real=bool(mf), MF_REF_fixtures=SC.selftest(), env=env_record())
    (out / 'ref.json').write_text(json.dumps(rec, indent=1))
    log('ref allowance', allow)


# ----------------------------------------------------------------------------------------------- stage data ---

def stage_data(a, out, D):
    n, nt = DATA_GRID[a.lev]
    nb = BANK_SAMPLE[a.lev]
    if a.smoke:
        n, nt, nb = 128, 1000, 64
    gr = fd.Grid(a.dom, n)
    m = gr.mask_np
    flat = jnp.asarray(np.where(m.ravel())[0])
    # bank sample sub-lattice (per-unit 1/nb) restricted to the domain
    s = n // nb
    sub = np.zeros_like(m)
    sub[s - 1::s, s - 1::s] = True
    sub &= m
    bI, bJ = np.nonzero(sub)
    q = fd.make_sbdf2(a.dom, n, nt, 1e-10, n_out=20, out_idx=(np.nonzero(m)[0], np.nonzero(m)[1]), maxit=60)
    Dtr = D['train']
    t0 = time.perf_counter()
    snaps = []
    for j in range(0, len(Dtr), 8):
        O, st = jax.vmap(q)(jnp.asarray(Dtr[j:j + 8]))
        assert int(np.asarray(st)[:, 2].sum()) == 0, 'training MG solve not converged'
        snaps.append(np.asarray(O))                           # (8, 21, n_in)
    U = np.concatenate(snaps)                                  # (n_traj, 21, n_in)
    sec = time.perf_counter() - t0
    Xin = np.stack([gr.X[m], gr.Y[m]], -1)
    S_all = U.reshape(-1, U.shape[-1])
    Phi, sv = PD.pod(S_all, gr.h, 512 if not a.smoke else 64)
    Z = PD.project(S_all, Phi, gr.h)                            # projected coefficients (states, R)
    # bank samples (values at the sub-lattice nodes), coordinates
    pos = -np.ones(m.shape, int)
    pos[m] = np.arange(int(m.sum()))
    bidx = pos[bI, bJ]
    np.save(out / 'bank_U.npy', S_all[:, bidx])
    np.save(out / 'bank_X.npy', Xin[bidx])
    np.save(out / 'pod_Phi.npy', Phi)
    np.save(out / 'pod_Z.npy', Z)
    rec = dict(n=n, nt=nt, n_traj=len(Dtr), states=int(S_all.shape[0]), n_in=int(S_all.shape[1]), seconds=sec,
               sv=sv[:1024].tolist(), bank_points=int(len(bidx)), bank_sample=nb,
               files={f: sha(out / f) for f in ('bank_U.npy', 'bank_X.npy', 'pod_Phi.npy', 'pod_Z.npy')})
    (out / 'data.json').write_text(json.dumps(rec, indent=1))
    log('data', {k: v for k, v in rec.items() if k not in ('sv', 'files')})


# ----------------------------------------------------------------------------------------------- stage bank ---

def bank_rule_k(dom):
    return {'L': 4, 'H': 4, 'B': 4}[dom]


def stage_bank(a, out, D):
    import bank as BK
    U = np.load(out / 'bank_U.npy')
    X = np.load(out / 'bank_X.npy')
    steps = 2000 if a.smoke else None
    seed = 7_300_000 + 100 * G.FAM_ID[a.fam] + 10 * G.LEV_ID[a.lev] + {'L': 1, 'H': 2, 'B': 3}[a.dom]
    params, Z, info = BK.train(a.dom, X, U, seed, steps=steps, log=log)
    kq = bank_rule_k(a.dom)
    Xq, wq = G.composite_rule(a.dom, kq, 16)
    T, sv, condR, dropped = BK.rotation(a.dom, params, Z, Xq, wq)
    h1 = BK.save(out / 'bank.pkl', dict(params=params, Z=Z, T=T, k_q=kq, q_ref=16, dom=a.dom, info=info))
    rec = dict(info=info, sv=sv[:600].tolist(), condR=condR, dropped_codes=dropped, R_eff=int(T.shape[1]), sha=h1,
               seed=seed, env=env_record())
    (out / 'bank.json').write_text(json.dumps(rec, indent=1))
    log('bank', info['seconds'], 'recon', info['recon_mean'], info['recon_max'], 'R_eff', T.shape[1])


# --------------------------------------------------------------------------------------------- stage ladder ---

def load_refs(out, split):
    z = np.load(out / f'refs_{split}.npz')
    return z['U']


def pod_space(a, out):
    n, _ = DATA_GRID[a.lev]
    if a.smoke:
        n = 128
    Phi = np.load(out / 'pod_Phi.npy')
    return PD.pod_space(a.dom, n, Phi, n_init=min(128, n))


def bank_spaces(a, out):
    import bank as BK
    b = BK.load(out / 'bank.pkl')
    return {q: BK.bank_space(a.dom, b['params'], b['T'], b['k_q'], q) for q in Q_LADDER + (14,)}, b


def stage_ladder(a, out, D):
    rj = json.loads((out / 'ref.json').read_text())
    U_allow = rj['allowance']['U']
    Uv = load_refs(out, 'val')
    Dv = D['val']
    L = LD.Ladder(out / 'ladder.jsonl', Uv, Dv, U_allow, tag='val')
    dom = a.dom
    sizes = FD_SIZES[:2] if a.smoke else FD_SIZES
    # ---- C-FDMG (SBDF2 + MG): step ladders per size at tol 1e-8, then the tolerance ladder at the Pareto sizes
    if dom != 'B' or a.box_fd:
        for n in sizes:
            n0 = 5 * 2 ** max(0, math.ceil(math.log2(max(0.25 * 2.0 * n / 5.0, 1))))
            L.step_ladder(f'fdmg|n{n}|tol1e-08', dict(fam='fdmg', n=n, tol=1e-8), build_fdmg(dom), n0)
            n0r = fd.rk4_stable_n(dom, n, G.LEVELS[a.lev][1], theta_min=fd.Grid(dom, n).theta_min ** 2)
            if n0r <= LD.N_CAP:
                L.step_ladder(f'rk4|n{n}', dict(fam='rk4', n=n), build_rk4(dom), n0r)
        for row in pareto(L, 'fdmg'):
            for tol in (1e-3, 1e-4, 1e-6):
                c = dict(row['cfg'], tol=tol)
                L.run(f"fdmg|n{c['n']}|tol{tol:.0e}|nt{c['nsteps']}", c, build_fdmg(dom))
    # ---- C-SEM
    if dom != 'B':
        grades = ((None, 0.5) if a.smoke else (None, 0.5, 0.7)) if dom == 'L' else (None,)
        P_ = SEM_P[:2] if a.smoke else SEM_P
        K_ = SEM_K[:2] if a.smoke else SEM_K
        fine = SEM_FINE_STEPS // (8 if a.smoke else 1)
        spatial = []
        for p in P_:
            for k in K_:
                for g in grades:
                    c = dict(fam='sem', p=p, k=k, g=g, dea=False, ord=3, tol=1e-10, solver='pcg', nsteps=fine)
                    row, _ = L.run(sem_key(c) + f'|nt{fine}', c, build_sem(dom))
                    spatial.append(row)
        for row in space_pareto(spatial):                 # over-integration only on the spatial Pareto set (A3)
            c = dict(row['cfg'], dea=True)
            r2, _ = L.run(sem_key(c) + f'|nt{fine}', c, build_sem(dom))
            spatial.append(r2)
        keep = space_pareto(spatial)
        for row in keep:
            for order in (2, 3):
                c = dict(row['cfg'], ord=order)
                c.pop('nsteps')
                L.step_ladder(sem_key(c), c, build_sem(dom), 5)
        for row in pareto(L, 'sem'):
            for tol in (1e-4, 1e-6, 1e-8):
                c = dict(row['cfg'], tol=tol)
                L.run(sem_key(c) + f"|nt{c['nsteps']}", c, build_sem(dom))
            c = dict(row['cfg'], solver='eig')
            L.run(sem_key(c) + f"|nt{c['nsteps']}", c, build_sem(dom))
    else:
        # ---- C-SPEC (box only)
        for N in (SPEC_N[:3] if a.smoke else SPEC_N):
            for ni in (64, 128, 256):
                if ni < N:
                    continue
                L.step_ladder(f'spec|N{N}|ni{ni}', dict(fam='spec', N=N, ni=ni), build_spec(), 5)
    # ---- reduced arms
    sp = pod_space(a, out)
    Rl = [r for r in R_LADDER if r <= sp.R]
    steppers = RM.STEPPERS
    if a.smoke:
        Rl, steppers = Rl[:2], ('lspg_cnr', 'gal_etdrk4')
    for r in Rl:
        for st in steppers:
            form = 'tensor'
            L.step_ladder(f'podg|r{r}|{st}|{form}', dict(fam='podg', r=r, st=st, form=form, tol='prod'),
                          build_rom(sp, 'podg'), 10)
    Z = np.load(out / 'pod_Z.npy')
    ecsw_states = ecsw_population(Z, len(D['train']))
    ecsw = {}
    for r in Rl:
        for st in (('gal_etdrk4',) if a.smoke else ('lspg_cnr', 'gal_etdrk4')):
            Mt = min(4 * r, sp.R) if st == 'lspg_cnr' else r
            fit, info = PD.ecsw_fit(sp, r, Mt, ecsw_states)
            for kap, x in fit.items():
                if x is None:
                    continue
                ecsw[(r, Mt, kap)] = x
                sub = sp.with_weights(x['nodes'], x['weights'])
                L.step_ladder(f'ecsw|r{r}|k{kap}|{st}', dict(fam='ecsw', r=r, kappa=kap, st=st, form='point',
                                                             tol='prod', relres=x['relres'], size=x['size']),
                              build_rom(sub, 'ecsw'), 10)
    np.save(out / 'ecsw.npy', np.array([dict(key=k, **v) for k, v in ecsw.items()], dtype=object), allow_pickle=True)
    if (out / 'bank.pkl').exists():
        sps, _ = bank_spaces(a, out)
        for r in Rl:
            for st in a.bank_steppers.split(','):
                rows = L.step_ladder(f'bank|r{r}|q8|{st}', dict(fam='bank', r=r, q=8, st=st, form='point', tol='prod'),
                                     build_rom(sps[8], 'bank'), 10)
                okn = [x['cfg']['nsteps'] for x in rows if x.get('valid')][-2:]   # step-converged rung and predecessor
                for nt in okn:
                    for q in Q_LADDER:
                        if q == 8:
                            continue
                        c = dict(fam='bank', r=r, q=q, st=st, form='point', tol='prod', nsteps=nt)
                        L.run(f'bank|r{r}|q{q}|{st}|nt{nt}', c, build_rom(sps[q], 'bank'))
                        L.run(f'bank|r{r}|q{q}|{st}|tensor|nt{nt}', dict(c, form='tensor'), build_rom(sps[q], 'bank'))
    log('ladder rows', len(L.done))


def ecsw_population(Z, n_traj, n_use=6):
    """6 training trajectories x 11 snapshots (t = 0.025 k; A4 memory bound): states are rows j*21 + 2*i."""
    idx = [j * 21 + 2 * i for j in range(min(n_use, n_traj)) for i in range(11)]
    return Z[idx]


def sem_key(c):
    return f"sem|p{c['p']}|k{c['k']}|g{c['g']}|d{int(c['dea'])}|o{c['ord']}|tol{c['tol']:.0e}|{c['solver']}"


def space_pareto(rows):
    """Spatial configurations on the (DOF, worst) or (DOF, median) Pareto front with error <= 0.1."""
    ok = [r for r in rows if r.get('valid')]
    dof = lambda r: r['cfg']['k'] ** 2 * r['cfg']['p'] ** 2 * (1 + 0.1 * r['cfg']['dea'])
    keep = {}
    for stat in ('worst', 'median'):
        best = np.inf
        for r in sorted(ok, key=dof):
            if r[stat] <= 0.1 and r[stat] < best:
                best = r[stat]
                keep[r['key']] = r
    return list(keep.values())


def pareto(L, fam):
    """Visited valid rows of a family that are not dominated in (cohort seconds, worst) or (seconds, median)."""
    rows = [r for r in L.done.values() if r['cfg'].get('fam') == fam and r.get('valid')]
    keep = {}
    for stat in ('worst', 'median'):
        best = np.inf
        for r in sorted(rows, key=lambda r: r['cohort_seconds']):
            if r[stat] < best and r[stat] <= 0.1:
                best = r[stat]
                keep[r['key']] = r
    return list(keep.values())


# ------------------------------------------------------------------------------------------------ builders -----

def build_fdmg(dom):
    return lambda c: fd.make_sbdf2(dom, c['n'], c['nsteps'], c['tol'])


def build_rk4(dom):
    return lambda c: fd.make_rk4(dom, c['n'], c['nsteps'])


def build_sem(dom):
    return lambda c: sem.make_sem(dom, c['p'], c['k'], c['nsteps'], c['tol'], order=c['ord'], grade=c['g'],
                                  dealias=c['dea'], solver=c['solver'])[0]


def build_spec():
    return lambda c: spec.make_spec(c['N'], c['nsteps'], n_init=c['ni'])


def build_rom(sp, fam):
    return lambda c: RM.make_query(sp, c['r'], c['st'], c['nsteps'], form=c['form'], tol=c['tol'],
                                   cert=c.get('cert', False), perturb_cert=c.get('perturb', 0.0))


# ----------------------------------------------------------------------------------------------- stage time ---

def builders(a, out):
    """fam -> build(cfg) for every family of this cell (spaces loaded once); '_space'(fam, cfg, lat) for diagnostics."""
    import bank as BK
    dom = a.dom
    B = dict(fdmg=build_fdmg(dom), rk4=build_rk4(dom), sem=build_sem(dom), spec=build_spec())
    sp = pod_space(a, out)
    B['podg'] = build_rom(sp, 'podg')
    ecsw = {}
    if (out / 'ecsw.npy').exists():
        for x in np.load(out / 'ecsw.npy', allow_pickle=True):
            ecsw[tuple(x['key'])] = x

    def ecsw_space(c, base):
        Mt = min(4 * c['r'], base.R) if c['st'] == 'lspg_cnr' else c['r']
        x = ecsw[(c['r'], Mt, c['kappa'])]
        return base.with_weights(x['nodes'], x['weights'])
    B['ecsw'] = lambda c: build_rom(ecsw_space(c, sp), 'ecsw')(c)
    B['_ecsw'] = ecsw
    spaces = None
    bk = None
    if (out / 'bank.pkl').exists():
        spaces, bk = bank_spaces(a, out)
        B['bank'] = lambda c: build_rom(spaces[c['q']], 'bank')(c)

    def space_at(fam, c, lat):
        if fam in ('podg', 'ecsw'):
            n, _ = DATA_GRID[a.lev]
            n = 128 if a.smoke else n
            base = PD.pod_space(dom, n, np.load(out / 'pod_Phi.npy'), n_init=min(128, n), lat=lat)
            return base if fam == 'podg' else ecsw_space(c, base)
        pts = G.lattice(dom, lat)[0]
        return BK.bank_space(dom, bk['params'], bk['T'], bk['k_q'], c['q'], lattice_pts=pts)
    B['_space'] = space_at
    return B, spaces


def ecsw_q2(B, c, D, U):
    """C-ECSW support successor (kappa -> 2 kappa) rollout distance, reported only (A3.4)."""
    k2 = 2 * c['kappa']
    Mt = min(4 * c['r'], 512) if c['st'] == 'lspg_cnr' else c['r']
    if (c['r'], Mt, k2) not in B['_ecsw']:
        return dict(available=False)
    O1, _ = FZ.run_unbatched(B['ecsw'](c), D)
    O2, _ = FZ.run_unbatched(B['ecsw'](dict(c, kappa=k2)), D)
    return dict(available=True, rollout_max=float(np.max(FZ.dist(O1, O2, U))))


def select(L, B, U, D_val, D_dev, Uv, Ud, spaces, fam, tau, stat, log_rows, n_final=2):
    """A3.4 selection: screen the K_SCREEN cheapest-by-cohort-seconds candidates meeting tau (warm one-case median of
    3); in screen order, unbatched official re-run (must still meet tau); the bank must pass Q1 and Q2; competitors
    must be valid with converged solves (their Q1 is measured and reported). Returns up to n_final finalists."""
    rows = [r for r in L.done.values() if r['cfg'].get('fam') == fam and r.get('valid')
            and SC.three_way(r[stat], U, tau) == 'meets']
    rows = sorted(rows, key=lambda r: r['cohort_seconds'])
    screened = []
    for r in rows[:FZ.K_SCREEN]:
        try:
            screened.append((FZ.screen_time(B[fam](r['cfg']), D_val[0]), r))
        except Exception as ex:                                     # noqa: BLE001  (recorded)
            log_rows.append(dict(fam=fam, key=r['key'], screen_error=repr(ex)[:200]))
    screened.sort(key=lambda t: t[0])
    finals = []
    for ts, r in screened:
        if len(finals) >= n_final:
            break
        q = B[fam](r['cfg'])
        O, st = FZ.run_unbatched(q, D_val)
        e = SC.err(O, Uv)
        sv = float(np.max(e) if stat == 'worst' else np.median(e))
        item = dict(fam=fam, tau=tau, stat=stat, key=r['key'], screen=ts, worst=float(np.max(e)),
                    median=float(np.median(e)), meets_unbatched=SC.three_way(sv, U, tau) == 'meets')
        if not item['meets_unbatched']:
            log_rows.append(item)
            continue
        item['Q1'] = FZ.q1(fam, r['cfg'], B[fam], D_val, D_dev, Uv, Ud, O, st)
        if fam == 'bank':
            ok = item['Q1']['passed']
            if ok:
                item['Q2'] = FZ.q2_bank(r['cfg'], spaces, np.concatenate([D_dev, D_val]), np.concatenate([Ud, Uv]))
                ok = item['Q2']['passed']
        else:
            ok = bool(np.all(np.isfinite(O)) and (fam not in ('fdmg', 'sem') or float(st[:, 2].sum()) == 0))
            if fam == 'ecsw':
                item['Q2_reported'] = ecsw_q2(B, r['cfg'], D_val[:8], Uv[:8])
        item['eligible'] = bool(ok)
        log_rows.append(item)
        if ok:
            finals.append(dict(item, cfg=r['cfg'], e=e.tolist(), outputs=O))
    return finals


def stage_time(a, out, D):
    rj = json.loads((out / 'ref.json').read_text())
    U = rj['allowance']['U']
    Uv, Ud = load_refs(out, 'val'), load_refs(out, 'dev')
    D_val, D_dev = D['val'], D['dev']
    L = LD.Ladder(out / 'ladder.jsonl', Uv, D_val, U, tag='val')
    B, spaces = builders(a, out)
    fams = [f for f in FZ.COMPETITORS if any(r['cfg'].get('fam') == f for r in L.done.values())]
    log_rows, fin = [], {}
    for tau in SC.TAUS:
        for stat in ('worst', 'median'):
            for fam in fams + (['bank'] if 'bank' in B else []):
                fin[(fam, tau, stat)] = select(L, B, U, D_val, D_dev, Uv, Ud, spaces, fam, tau, stat, log_rows)
                log(f'finalists {fam} tau={tau} {stat}:', [s['key'] for s in fin[(fam, tau, stat)]])
    Dt = D_val[:FZ.N_TIME]
    cc = FZ.CompileCounter()
    logging.getLogger('jax').addHandler(cc)
    qcache = {}

    def getq(fam, s):
        k = (fam, s['key'])
        if k not in qcache:
            qcache[k] = B[fam](s['cfg'])
            # T-setup: warm-up with a nu different from every timed case
            jax.block_until_ready(qcache[k](jnp.asarray(Dt[0] * np.array([1, 1, 1, 1, 1.0001]))))
        return qcache[k]
    timing, summary = {}, {}
    jax.config.update('jax_log_compiles', True)
    for tau in SC.TAUS:
        for stat in ('worst', 'median'):
            bks = fin.get(('bank', tau, stat)) or []
            per_fam = {}
            for fam in fams:
                for sa in fin.get((fam, tau, stat)) or []:
                    for sb in bks:
                        qA, qB = getq(fam, sa), getq('bank', sb)
                        n_before = cc.n
                        p = FZ.paired(qA, qB, Dt)
                        p['compiles_during_timing'] = cc.n - n_before
                        oa = np.asarray(qA(jnp.asarray(Dt[0]))[0])
                        ob = np.asarray(qB(jnp.asarray(Dt[0]))[0])
                        p['T_id'] = float(max(np.max(np.abs(oa - sa['outputs'][0])) / np.max(np.abs(sa['outputs'][0])),
                                              np.max(np.abs(ob - sb['outputs'][0])) / np.max(np.abs(sb['outputs'][0]))))
                        timing[f"{fam}|{tau}|{stat}|{sa['key']}|{sb['key']}"] = p
                # family S: min over its finalists of (max over bank finalists = the bank's faster finalist)
                vals = [max(timing[f"{fam}|{tau}|{stat}|{sa['key']}|{sb['key']}"]['S'] for sb in bks)
                        for sa in fin.get((fam, tau, stat)) or []] if bks else []
                if vals:
                    per_fam[fam] = min(vals)
            if per_fam:
                best = min(per_fam, key=per_fam.get)
                summary[f'{tau}|{stat}'] = dict(S=per_fam[best], best_competitor=best, per_family=per_fam,
                                                within5pct=[f for f, v in per_fam.items() if v <= 1.05 * per_fam[best]])
            else:
                summary[f'{tau}|{stat}'] = dict(S=None, bank_matched=bool(bks),
                                                competitor_matched=any(fin.get((f, tau, stat)) for f in fams))
            log(f'S tau={tau} {stat}:', summary[f'{tau}|{stat}'])
    jax.config.update('jax_log_compiles', False)
    tdbl = {}
    for (fam, tau, stat), ss in fin.items():
        if ss and fam not in tdbl:
            tdbl[fam] = FZ.t_dbl(getq(fam, ss[0]), Dt)
    # A3.7 finer-lattice diagnostic for every finalist (val[0:8])
    finer = {}
    if (out / 'refs_val128.npz').exists():
        U128 = np.load(out / 'refs_val128.npz')['U']
        pts128 = G.lattice(a.dom, 128)[0]
        for (fam, tau, stat), ss in fin.items():
            for s in ss:
                if s['key'] in finer:
                    continue
                try:
                    finer[s['key']] = FZ.finer_diag(a, out, fam, s['cfg'], B, D_val[:8], U128, pts128)
                except Exception as ex:                                 # noqa: BLE001  (recorded)
                    finer[s['key']] = dict(error=repr(ex)[:200])
    manifest = {f'{k[0]}|{k[1]}|{k[2]}': [dict(key=s['key'], cfg=s['cfg']) for s in v] for k, v in fin.items()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))
    fins = {f'{k[0]}|{k[1]}|{k[2]}': [{kk: vv for kk, vv in s.items() if kk != 'outputs'} for s in v]
            for k, v in fin.items()}
    res = dict(U=U, finalists=fins, selection_log=log_rows, timing=timing, summary=summary, T_dbl=tdbl, finer=finer,
               manifest_sha=FZ.manifest_sha(out / 'manifest.json'), env=env_record())
    (out / 'final.json').write_text(json.dumps(res, indent=1, default=float))


def interaction_split(a, out, D, split):
    """A3.2: D_bnd (box evolution of the non-box IC), D_geo (box IC) and the IC mismatch, FD n = 256 (S1) / 512 (S2)."""
    if a.dom == 'B':
        return None
    n, nt = (128, 1000) if a.smoke else {'S1': (256, 2000), 'S2': (512, 4000)}[a.lev]
    P, I, J = G.lattice(a.dom)
    s = n // G.LAT
    qG = fd.make_sbdf2(a.dom, n, nt, 1e-10, out_idx=(I * s - 1, J * s - 1))
    qBg = fd.make_sbdf2('B', n, nt, 1e-10, out_idx=(I * s - 1, J * s - 1), ic_dom=a.dom)
    qB = fd.make_sbdf2('B', n, nt, 1e-10, out_idx=(I * s - 1, J * s - 1))
    Dd = D[split]
    OG, OBg, OB = (LD.run_cohort(q, Dd)[0] for q in (qG, qBg, qB))
    n0 = np.linalg.norm(OG[:, 0], axis=-1)
    dist_ = lambda X: np.max(np.linalg.norm(OG[:, 1:] - X[:, 1:], axis=-1), axis=-1) / n0
    d_bnd, d_geo = dist_(OBg), dist_(OB)
    ic_mis = np.linalg.norm(OG[:, 0] - OB[:, 0], axis=-1) / n0
    return dict(split=split, n=n, nt=nt, D_bnd=d_bnd.tolist(), D_bnd_median=float(np.median(d_bnd)),
                D_geo=d_geo.tolist(), D_geo_median=float(np.median(d_geo)), ic_mismatch_median=float(np.median(ic_mis)),
                gate=bool(np.median(d_bnd) >= 0.05))


def stage_inter(a, out, D):
    res = interaction_split(a, out, D, 'val')
    (out / 'inter.json').write_text(json.dumps(dict(val=res, env=env_record()), indent=1))


def stage_test(a, out, D):
    """Test-once on the frozen finalists (MF-LEAK: the manifest sha must equal --manifest-sha, the committed copy)."""
    man = out / 'manifest.json'
    if not man.exists() or not a.manifest_sha or FZ.manifest_sha(man) != a.manifest_sha:
        raise SystemExit('MF-LEAK: manifest missing or sha mismatch; test refused')
    M = json.loads(man.read_text())
    rj = json.loads((out / 'ref.json').read_text())
    rich_t = rj['splits']['test']['rich']
    Ut = np.load(out / 'refs_test.npz')['U']
    U_t = SC.allowance(rich_t, rj['delta_t'])
    Ud = load_refs(out, 'dev')
    B, spaces = builders(a, out)
    res, qs = {}, {}
    for k, lst in M.items():
        fam, tau, stat = k.split('|')
        for v in lst:
            q = B[fam](v['cfg'])
            O, st = FZ.run_unbatched(q, D['test'])
            e = SC.err(O, Ut)
            sv = float(np.max(e) if stat == 'worst' else np.median(e))
            item = dict(key=v['key'], e=e.tolist(), stat=sv, verdict=SC.three_way(sv, U_t['U'], float(tau)),
                        Q1=FZ.q1(fam, v['cfg'], B[fam], D['test'], D['dev'], Ut, Ud, O, st))
            if fam == 'bank':
                item['Q2'] = FZ.q2_bank(v['cfg'], spaces, D['test'], Ut)
            res[f"{k}|{v['key']}"] = item
            qs[(fam, v['key'])] = q
    timing = {}
    Dt = D['test'][:FZ.N_TIME]
    for k, lst in M.items():
        fam, tau, stat = k.split('|')
        if fam == 'bank':
            continue
        for sa in lst:
            for sb in M.get(f'bank|{tau}|{stat}', []):
                timing[f"{k}|{sa['key']}|{sb['key']}"] = FZ.paired(qs[(fam, sa['key'])], qs[('bank', sb['key'])], Dt)
    inter = interaction_split(a, out, D, 'test')
    (out / 'test.json').write_text(json.dumps(dict(results=res, timing=timing, allowance_test=U_t, inter_test=inter,
                                                   env=env_record()), indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dom', required=True, choices=G.DOMS)
    ap.add_argument('--fam', required=True, choices=list(G.FAMS))
    ap.add_argument('--lev', required=True, choices=list(G.LEVELS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--stages', default='ref,data,bank,ladder')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--box-fd', action='store_true', help='also run FD competitors on the box (diagnostic)')
    ap.add_argument('--bank-steppers', default='lspg_cnr')
    ap.add_argument('--allow-cpu', action='store_true')
    ap.add_argument('--manifest-sha', default=None)
    a = ap.parse_args()
    if jax.default_backend() != 'gpu' and not a.allow_cpu:
        raise SystemExit(42)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = cohorts(a.fam, a.lev, a.smoke)
    G.check_cohorts()
    if a.smoke:                                   # smoke only: a loose extra target so every finalist path runs
        SC.TAUS = SC.TAUS + (0.2,)
    marks = {'ref': 'ref.json', 'data': 'data.json', 'bank': 'bank.json', 'ladder': 'ladder.done', 'time': 'final.json',
             'test': 'test.json', 'inter': 'inter.json'}
    for st in a.stages.split(','):
        mk = out / marks.get(st, f'{st}.done')
        if mk.exists():
            log(f'stage {st}: done ({mk.name})')
            continue
        t0 = time.perf_counter()
        log(f'stage {st}: start')
        dict(ref=stage_ref, data=stage_data, bank=stage_bank, ladder=stage_ladder, time=stage_time,
             test=stage_test, inter=stage_inter)[st](a, out, D)
        if st == 'ladder':
            mk.write_text('ok')
        with (out / 'stage_seconds.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=st, seconds=time.perf_counter() - t0, env=env_record())) + '\n')
        log(f'stage {st}: {time.perf_counter() - t0:.0f}s')


if __name__ == '__main__':
    main()
