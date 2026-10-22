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
R_LADDER = (16, 32, 64, 128, 256, 384)          # A9 = A5.2 (a)
Q_LADDER = (4, 6, 8, 10, 12, 'f8', 'f12')
SEM_FINE_STEPS = 2560
SMOKE_RELAX_Q2 = False
SCREEN_CACHE = {}                 # per-job fresh screen times (A9 rev. 2)


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
        f = out / ('refs_test.npz' if split == 'test' else f'refs_{split}.npz')
        np.savez(f, U=Us[2], U1=Us[0], U2=Us[1], converged=conv)
        if split == 'test':                    # labels of the test reference are computed only in the test stage
            res[split] = dict(file=f.name, sha=sha(f))
            continue
        rich = [SC.richardson(Us[0, j], Us[1, j], Us[2, j]) for j in range(Us.shape[1])]
        for j, r in enumerate(rich):
            if not conv[:, j].all():
                r['label'], r['reason'] = 'ineligible', 'MG solve not converged'
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
    # MF-MG at the finest reference level on the time-step operator (contraction <= 0.2; identity smoother fails)
    mg = fd.MG(a.dom, n)
    g0 = mg.levels[0]
    _, fm = fd.mms_fields(a.dom, g0.X, g0.Y, 1.5 * nt / G.T_END, G.LEVELS[a.lev][1])
    fj = jnp.asarray(fm * g0.mask_np)
    al = 1.5 * nt / G.T_END
    _, it_, ok_, rr_ = mg.solve(al, G.LEVELS[a.lev][1], fj, jnp.zeros_like(fj), 1e-10, 50)
    _, it0, ok0, rr0 = fd.MG(a.dom, n, smoother=False).solve(al, G.LEVELS[a.lev][1], fj, jnp.zeros_like(fj), 1e-10, 50)
    contr = float(10 ** (np.log10(max(float(rr_), 1e-300)) / max(int(it_), 1)))
    mfmg = dict(n=n, cycles=int(it_), converged=bool(ok_), contraction=contr, noop_converged=bool(ok0),
                fires=bool(ok_ and contr <= 0.2 and not ok0))
    rec = dict(levels=levels, splits=res, delta_t=dt_, MF_MG_finest=mfmg,
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
    return PD.pod_space(a.dom, n, Phi, n_init=n)


def rule_of(q, k_q):
    """(k_q, degree) of a rule label: int q -> (k_q, q); 'f<q>' -> (2 k_q, q) (A7)."""
    return (2 * k_q, int(q[1:])) if isinstance(q, str) else (k_q, q)


def bank_spaces(a, out):
    """Rules q in Q_LADDER (+14) on k_q = 4 elements and 'f8', 'f12', 'f16' = q 8, 12, 16 on 2 k_q elements (A7)."""
    import bank as BK
    b = BK.load(out / 'bank.pkl')
    sp = {q: BK.bank_space(a.dom, b['params'], b['T'], b['k_q'], q) for q in [x for x in Q_LADDER if isinstance(x, int)] + [14]}
    for q in (8, 12, 16):
        sp[f'f{q}'] = BK.bank_space(a.dom, b['params'], b['T'], 2 * b['k_q'], q)
    return sp, b


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
            n0r = fd.rk4_stable_n(dom, n, G.LEVELS[a.lev][1], theta_min=fd.Grid(dom, n).theta_min)
            if n0r <= LD.N_CAP:
                L.step_ladder(f'rk4|n{n}', dict(fam='rk4', n=n), build_rk4(dom), n0r)
        for row in pareto(L, 'fdmg'):
            for tol in (1e-3, 1e-4, 1e-6):
                c = dict(row['cfg'], tol=tol)
                L.run(f"fdmg|n{c['n']}|tol{tol:.0e}|nt{c['nsteps']}", c, build_fdmg(dom))
    # ---- C-SEM
    if dom != 'B':
        grades = (None, 0.5) if dom == 'L' else (None,)                       # A9 = A5.2 (c)
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
        feas = [x for x in spatial if x.get('valid') and min(x['worst'], x['median']) <= 0.1]
        for row in list(feas):                            # over-integration on every feasible spatial config (A6)
            c = dict(row['cfg'], dea=True)
            r2, _ = L.run(sem_key(c) + f'|nt{fine}', c, build_sem(dom))
            spatial.append(r2)
        keep = [x for x in spatial if x.get('valid') and min(x['worst'], x['median']) <= 0.1]
        for row in keep:
            for order in (2, 3):
                c = dict(row['cfg'], ord=order)
                c.pop('nsteps')
                L.step_ladder(sem_key(c), c, build_sem(dom), 5)
        for row in pareto(L, 'sem'):
            if row['cfg'].get('solver') != 'pcg':          # tolerance / eig variants only from pcg rows
                continue
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
    if (out / 'ecsw.npy').exists():                    # A9 resume: reuse fits already made in this cell
        for x in np.load(out / 'ecsw.npy', allow_pickle=True):
            ecsw[tuple(x['key'])] = {k_: v_ for k_, v_ in x.items() if k_ != 'key'}
    for r in Rl:
        fits = {}
        for st in (('gal_etdrk4',) if a.smoke else RM.STEPPERS):
            Mt = min(4 * r, sp.R) if st == 'lspg_cnr' else r
            if Mt not in fits and all((r, Mt, k_) in ecsw for k_ in PD.ECSW_KAPPA):
                fits[Mt] = {k_: ecsw[(r, Mt, k_)] for k_ in PD.ECSW_KAPPA}
            if Mt not in fits:
                t_fit = time.perf_counter()
                fits[Mt], finfo = PD.ecsw_fit(sp, r, Mt, ecsw_states)
                with (out / 'ecsw_fits.jsonl').open('a') as f_:   # A8: per-fit cost and termination record
                    f_.write(json.dumps(dict(r=r, Mt=Mt, seconds=time.perf_counter() - t_fit, info=finfo,
                                             sizes={str(k): (None if v is None else v['size'])
                                                    for k, v in fits[Mt].items()})) + '\n')
            for kap, x in fits[Mt].items():
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
                for q in Q_LADDER:
                    form_ = 'point' if r <= 128 and any(k_.startswith(f'bank|r{r}|q{q}|{st}|nt') for k_ in L.done) \
                        else 'tensor'                  # A9: tensor form in the ladder (identical outputs, MF-TENS)
                    L.step_ladder(f'bank|r{r}|q{q}|{st}', dict(fam='bank', r=r, q=q, st=st, form=form_, tol='prod'),
                                  build_rom(sps[q], 'bank'), 10)
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


BASE_TOL = {'fdmg': 1e-8, 'sem': 1e-10}


def pareto(L, fam):
    """Expansion frontier (A9): only BASE step-ladder rows (base tolerance, pcg, registered) that are not dominated in
    (cohort seconds, worst) or (seconds, median); tolerance / solver variants never feed the frontier, so a resumed run
    expands exactly the same frontier as an uninterrupted one."""
    rows = [r for r in L.done.values() if r['cfg'].get('fam') == fam and r.get('valid') and registered(r['cfg'])
            and r['cfg'].get('tol') == BASE_TOL[fam] and r['cfg'].get('solver', 'pcg') == 'pcg']
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
            base = PD.pod_space(dom, n, np.load(out / 'pod_Phi.npy'), n_init=n, lat=lat)
            return base if fam == 'podg' else ecsw_space(c, base)
        pts = G.lattice(dom, lat)[0]
        kq_, deg_ = rule_of(c['q'], bk['k_q'])
        return BK.bank_space(dom, bk['params'], bk['T'], kq_, deg_, lattice_pts=pts)
    B['_space'] = space_at
    B['_dom'] = dom
    return B, spaces


def dom_of(B):
    return B['_dom']


def ecsw_q2(B, c, D, U):
    """C-ECSW support successor (kappa -> 2 kappa) rollout distance, reported only (A3.4)."""
    k2 = 2 * c['kappa']
    Mt = min(4 * c['r'], 512) if c['st'] == 'lspg_cnr' else c['r']
    if (c['r'], Mt, k2) not in B['_ecsw']:
        return dict(available=False)
    O1, _ = FZ.run_unbatched(B['ecsw'](c), D)
    O2, _ = FZ.run_unbatched(B['ecsw'](dict(c, kappa=k2)), D)
    return dict(available=True, rollout_max=float(np.max(FZ.dist(O1, O2, U))))


def registered(cfg):
    """A9: rows outside the reduced registered ladders (recorded before the reduction) never enter selection."""
    if 'r' in cfg and cfg.get('fam') in ('podg', 'ecsw', 'bank') and cfg['r'] not in R_LADDER:
        return False
    if cfg.get('fam') == 'sem' and cfg.get('g') not in (None, 0.5):
        return False
    return True


def select(L, B, U, D_val, D_dev, Uv, Ud, spaces, fam, tau, stat, log_rows, n_final=2):
    """A3.4 selection: screen the K_SCREEN cheapest-by-cohort-seconds candidates meeting tau (warm one-case median of
    3); in screen order, unbatched official re-run (must still meet tau); the bank must pass Q1 and Q2; competitors
    must be valid with converged solves (their Q1 is measured and reported). Returns up to n_final finalists."""
    rows = [r for r in L.done.values() if r['cfg'].get('fam') == fam and r.get('valid') and registered(r['cfg'])
            and SC.three_way(r[stat], U, tau) == 'meets']
    if not rows:
        return []
    # A9 (rev. 2): EVERY eligible candidate is re-screened in THIS job (exhaustive; stored screens are not used for
    # ordering or truncation); a per-job cache shares the screens across targets and statistics.
    fresh = {}
    for x in sorted(rows, key=lambda x: x['key']):
        if x['key'] not in SCREEN_CACHE:
            try:
                SCREEN_CACHE[x['key']] = FZ.screen_time(B[fam](x['cfg']), D_val[0])
            except Exception as ex:                     # noqa: BLE001
                SCREEN_CACHE[x['key']] = None
                log_rows.append(dict(fam=fam, key=x['key'], screen_error=repr(ex)[:200]))
        if SCREEN_CACHE[x['key']] is not None:
            fresh[x['key']] = (SCREEN_CACHE[x['key']], x)
    log_rows.append(dict(fam=fam, tau=tau, stat=stat, rescreened=len(fresh)))
    screened = [(t_, x['cfg'], k_) for k_, (t_, x) in fresh.items()]
    for t_, cfg0, k0 in sorted(screened, key=lambda t: (t[0], t[2]))[:3]:
        variants = []
        if fam == 'sem' and cfg0.get('solver') == 'pcg':
            variants.append(dict(cfg0, solver='eig'))
        if fam == 'bank':
            variants.append(dict(cfg0, form='tensor' if cfg0.get('form') == 'point' else 'point'))
        for v in variants:
            vk = k0 + f"|{v.get('solver', v.get('form'))}"
            try:
                if vk not in SCREEN_CACHE:
                    SCREEN_CACHE[vk] = FZ.screen_time(B[fam](v), D_val[0])
                screened.append((SCREEN_CACHE[vk], v, vk))
            except Exception as ex:                     # noqa: BLE001  (recorded: e.g. eig not admissible)
                log_rows.append(dict(fam=fam, key=k0, variant_error=repr(ex)[:200]))
    screened.sort(key=lambda t: (t[0], t[2]))
    finals = []
    for ts, cfg_, key_ in screened:
        if len(finals) >= n_final:
            break
        r = dict(cfg=cfg_, key=key_)
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
            ok = item['Q1']['passed'] and LD.stats_valid(r['cfg'], st)
            if ok:
                item['Q2'] = FZ.q2_bank(r['cfg'], spaces, np.concatenate([D_dev, D_val]), np.concatenate([Ud, Uv]))
                ok = item['Q2']['passed'] or SMOKE_RELAX_Q2      # smoke only (labelled): exercise the timing path
                item['Q2_gate_relaxed_smoke'] = SMOKE_RELAX_Q2
        else:
            ok = bool(np.all(np.isfinite(O)) and LD.stats_valid(r['cfg'], st))
            if fam == 'ecsw':
                item['Q2_reported'] = ecsw_q2(B, r['cfg'], D_val[:8], Uv[:8])
        item['eligible'] = bool(ok)
        log_rows.append(item)
        if ok:
            P_, _, _ = G.lattice(dom_of(B), G.LAT)
            cf = (G.LC, G.LC) if dom_of(B) == 'L' else G.HC
            win = np.hypot(P_[:, 0] - cf[0], P_[:, 1] - cf[1]) < 0.2
            if win.any():
                item['feature_window_err'] = SC.err(O[..., win], Uv[..., win]).tolist()
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
    timing, summary, pair_cache = {}, {}, {}
    sent = next(((f2, ss[0]) for (f2, _t, _s), ss in fin.items() if ss and f2 != 'bank'), None)
    sentinel = {}
    if sent:
        sentinel['start'] = FZ.screen_time(getq(*sent), Dt[0], reps=5)
    jax.config.update('jax_log_compiles', True)
    for tau in SC.TAUS:
        for stat in ('worst', 'median'):
            bks = fin.get(('bank', tau, stat)) or []
            per_fam = {}
            for fam in fams:
                for sa in fin.get((fam, tau, stat)) or []:
                    for sb in bks:
                        pk = (fam, sa['key'], sb['key'])
                        if pk in pair_cache:               # identical finalist pair already timed (other tau/stat)
                            timing[f"{fam}|{tau}|{stat}|{sa['key']}|{sb['key']}"] = pair_cache[pk]
                            continue
                        qA, qB = getq(fam, sa), getq('bank', sb)
                        n_before = cc.n
                        p, oA, oB = FZ.paired(qA, qB, Dt)
                        p['compiles_during_timing'] = cc.n - n_before
                        p['T_id'] = max(FZ.t_id(oA, sa['outputs'][:len(Dt)]), FZ.t_id(oB, sb['outputs'][:len(Dt)]))
                        p['valid'] = bool(p['T_id'] <= 1e-12 and p['compiles_during_timing'] == 0)
                        timing[f"{fam}|{tau}|{stat}|{sa['key']}|{sb['key']}"] = p
                        pair_cache[pk] = p
            comp_keys = [(f2, sa['key']) for f2 in fams for sa in fin.get((f2, tau, stat)) or []]
            pairs = {}
            for f2, ka in comp_keys:
                for sb in bks:
                    t_ = timing.get(f"{f2}|{tau}|{stat}|{ka}|{sb['key']}")
                    pairs[(f2, ka, sb['key'])] = t_['S'] if t_ and t_['valid'] else None
            mm = FZ.maxmin(pairs, [sb['key'] for sb in bks], comp_keys)
            summary[f'{tau}|{stat}'] = mm if mm else dict(
                S=None, bank_matched=bool(bks), competitor_matched=bool(comp_keys))
            log(f'S tau={tau} {stat}:', summary[f'{tau}|{stat}'])
    jax.config.update('jax_log_compiles', False)
    if sent:
        sentinel['end'] = FZ.screen_time(getq(*sent), Dt[0], reps=5)
        sentinel['drift'] = abs(sentinel['end'] / sentinel['start'] - 1)
        sentinel['valid'] = bool(sentinel['drift'] <= 0.10)
    tdbl = {}
    td_k = {}
    for (fam, tau, stat), ss in fin.items():
        for s_ in ss:
            if (fam, s_['key']) not in td_k:
                td_k[(fam, s_['key'])] = FZ.t_dbl(getq(fam, s_), Dt)
    for (fam, key_), v_ in td_k.items():
        tdbl.setdefault(fam, dict(passed=True, ratios={}))
        tdbl[fam]['ratios'][key_] = v_['ratio']
        tdbl[fam]['passed'] = tdbl[fam]['passed'] and v_['passed']
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
    side = {}                                         # A4.2 per-side status
    for (fam, tau, stat), ss in fin.items():
        rows_ = [x for x in L.done.values() if x['cfg'].get('fam') == fam and registered(x['cfg'])]
        val_ = [x for x in rows_ if x.get('valid')]
        meets_ = [x for x in val_ if SC.three_way(x[stat], U, tau) == 'meets']
        if ss:
            stt = 'matched'
        elif not val_:
            stt = 'invalid'
        elif meets_:
            stt = 'unqualified'                      # candidates meet tau but none passed selection/Q1/Q2 -> no evidence
        elif any(SC.three_way(x[stat], U, tau) == 'unresolved' for x in val_):
            stt = 'unresolved-match'
        else:
            stt = 'censored'
        side[f'{fam}|{tau}|{stat}'] = stt
    for k_, sm_ in summary.items():
        bad = []
        if sentinel and not sentinel.get('valid', True):
            bad.append('sentinel drift > 10 %')
        if sm_.get('S') is not None:
            for fam_ in set(sm_['per_family']) | {'bank'}:
                if fam_ in tdbl and not tdbl[fam_]['passed']:
                    bad.append(f'T-dbl failed for {fam_}')
        if bad and sm_.get('S') is not None:
            sm_['S_unchecked'], sm_['S'], sm_['reason'] = sm_['S'], None, '; '.join(bad)
    manifest = {f'{k[0]}|{k[1]}|{k[2]}': [dict(key=s['key'], cfg=s['cfg']) for s in v] for k, v in fin.items()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))
    fins = {f'{k[0]}|{k[1]}|{k[2]}': [{kk: vv for kk, vv in s.items() if kk != 'outputs'} for s in v]
            for k, v in fin.items()}
    res = dict(U=U, finalists=fins, selection_log=log_rows, timing=timing, summary=summary, T_dbl=tdbl, finer=finer,
               sentinel=sentinel, side_status=side,
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
    runs_ = [LD.run_cohort(q, Dd) for q in (qG, qBg, qB)]
    OG, OBg, OB = (x[0] for x in runs_)
    solves_ok = all(bool(np.all(x[1][:, 2] == 0)) for x in runs_)
    n0 = np.linalg.norm(OG[:, 0], axis=-1)
    dist_ = lambda X: np.max(np.linalg.norm(OG[:, 1:] - X[:, 1:], axis=-1), axis=-1) / n0
    d_bnd, d_geo = dist_(OBg), dist_(OB)
    ic_mis = np.linalg.norm(OG[:, 0] - OB[:, 0], axis=-1) / n0
    return dict(split=split, n=n, nt=nt, D_bnd=d_bnd.tolist(), D_bnd_median=float(np.median(d_bnd)),
                D_geo=d_geo.tolist(), D_geo_median=float(np.median(d_geo)), ic_mismatch_median=float(np.median(ic_mis)),
                solves_converged=solves_ok, gate=bool(solves_ok and np.median(d_bnd) >= 0.05))


def stage_diag(a, out, D):
    """A6 diagnostics (reported): training-data accuracy on dev[0:4], SEM reference cross-check on dev8, G-FLOOR for every
    reduced finalist on val[0:8] (a run below its span floor is a bug)."""
    res = {}
    Ud = load_refs(out, 'dev')
    n, nt = DATA_GRID[a.lev] if not a.smoke else (128, 1000)
    Od = LD.run_cohort(fd.make_sbdf2(a.dom, n, nt, 1e-10, maxit=60), D['dev'][:4])[0]
    res['training_data_err'] = SC.err(Od, Ud[:4]).tolist()
    if a.dom != 'B':
        p, k, fine = (6, 2, 320) if a.smoke else (12, 8, SEM_FINE_STEPS)
        q, _ = sem.make_sem(a.dom, p, k, fine, 1e-12, order=3, grade=0.5 if a.dom == 'L' else None)
        Os = LD.run_cohort(q, D['dev'])[0]
        eps = max(r_['eps'] for r_ in json.loads((out / 'ref.json').read_text())['splits']['dev']['rich'])
        es = SC.err(Os, Ud)
        res['sem_crosscheck'] = dict(p=p, k=k, nsteps=fine, err=es.tolist(), bar=max(2 * eps, 1e-3),
                                     flagged=bool(np.max(es) > max(2 * eps, 1e-3)))
    fz = json.loads((out / 'final.json').read_text())
    Uv = load_refs(out, 'val')[:8]
    B, _ = builders(a, out)
    floors = {}
    for k_, lst in fz['finalists'].items():
        for s in lst:
            fam = s['cfg']['fam']
            if fam not in ('bank', 'podg', 'ecsw') or s['key'] in floors:
                continue
            sp_ = B['_space'](fam, s['cfg'], G.LAT)
            fl = [RM.floor_err(sp_, s['cfg']['r'], Uv[j]) for j in range(len(Uv))]
            e = s['e'][:len(Uv)]
            floors[s['key']] = dict(floor=fl, e=e, below_floor=bool(any(ei < fi - 1e-9 for ei, fi in zip(e, fl))))
    res['G_FLOOR'] = floors
    icd = {}
    for k_, lst in fz['finalists'].items():
        for s in lst:
            if s['cfg']['fam'] != 'sem' or s['key'] in icd:
                continue
            c = s['cfg']
            S_ = sem.SEMSpace(a.dom, c['p'], c['k'], c['g'], False, c['dea'])
            pts = G.lattice(a.dom)[0]
            el, Lx, Ly = S_.lattice_interp(pts)
            errs = []
            for d_ in D['val'][:8]:
                u0n = np.asarray(S_.ic(jnp.asarray(d_)))
                ui = np.einsum('pi,pij,pj->p', Lx, u0n[S_.conn[el]], Ly)
                ue = G.ic(a.dom, d_, pts[:, 0], pts[:, 1])
                errs.append(float(np.linalg.norm(ui - ue) / np.linalg.norm(ue)))
            icd[s['key']] = errs
    res['sem_ic_interp_err'] = icd
    res['G_FLOOR_ok'] = not any(v['below_floor'] for v in floors.values())
    (out / 'diag.json').write_text(json.dumps(res, indent=1, default=float))


def stage_freeze(a, out, D):
    """A8: after time, diag and inter: hash every decision-affecting artefact and the lane code files."""
    names = ('manifest.json', 'final.json', 'diag.json', 'inter.json', 'ref.json', 'bank.pkl', 'pod_Phi.npy', 'ecsw.npy',
             'refs_val.npz', 'refs_test.npz', 'refs_dev.npz', 'refs_val128.npz')
    need = ('manifest.json', 'final.json', 'diag.json', 'ref.json', 'refs_test.npz') + (('inter.json',) if a.dom != 'B' else ())
    missing = [n_ for n_ in need if not (out / n_).exists()]
    if missing:
        raise SystemExit(f'freeze refused: missing {missing}')
    arte = {n_: sha(out / n_) for n_ in names if (out / n_).exists()}
    code = {p_.name: sha(p_) for p_ in sorted(HERE.glob('*.py'))}
    code.update({f'vendor/{p_.name}': sha(p_) for p_ in sorted((HERE / 'vendor').glob('*.py'))})
    (out / 'frozen.json').write_text(json.dumps(dict(artefacts=arte, code=code, code_commit=os.environ.get('LANE_COMMIT'),
                                                     descriptors_sha=sha(HERE / 'runs' / 'descriptors.json')),
                                                indent=1, sort_keys=True))


def stage_inter(a, out, D):
    res = interaction_split(a, out, D, 'val')
    (out / 'inter.json').write_text(json.dumps(dict(val=res, env=env_record()), indent=1))


def stage_test(a, out, D):
    """Test-once on the frozen finalists (MF-LEAK: the manifest sha must equal --manifest-sha, the committed copy)."""
    # MF-LEAK (A6): the committed frozen.json (staged with the code, passed as --frozen) must equal the job's frozen.json
    # byte for byte, and every artefact hash in it must match the files now present
    fz = out / 'frozen.json'
    committed = HERE / 'runs' / 'frozen' / f'{a.dom}_{a.fam}_{a.lev}.json'      # staged from git archive of the commit
    if a.frozen and Path(a.frozen).resolve() != committed.resolve():
        raise SystemExit('MF-LEAK: --frozen must be the staged committed copy under code/runs/frozen/')
    if not committed.exists() or not fz.exists() or committed.read_bytes() != fz.read_bytes():
        raise SystemExit('MF-LEAK: frozen manifest missing or not the committed copy; test refused')
    froz = json.loads(fz.read_text())
    now = {p_.name: sha(p_) for p_ in sorted(HERE.glob('*.py'))}
    now.update({f'vendor/{p_.name}': sha(p_) for p_ in sorted((HERE / 'vendor').glob('*.py'))})
    if now != froz.get('code'):
        raise SystemExit('MF-LEAK: lane code differs from the frozen code; test refused')
    if froz.get('descriptors_sha') != (sha(HERE / 'runs' / 'descriptors.json') if (HERE / 'runs' / 'descriptors.json').exists() else None):
        raise SystemExit('MF-LEAK: descriptor manifest changed after the freeze')
    for f_, h_ in froz['artefacts'].items():
        if sha(out / f_) != h_:
            raise SystemExit(f'MF-LEAK: artefact {f_} changed after the freeze; test refused')
    M = json.loads((out / 'manifest.json').read_text())
    rj = json.loads((out / 'ref.json').read_text())
    zt = np.load(out / 'refs_test.npz')
    rich_t = [SC.richardson(zt['U1'][j], zt['U2'][j], zt['U'][j]) for j in range(len(zt['U']))]
    for j, rr_ in enumerate(rich_t):
        if not zt['converged'][:, j].all():
            rr_['label'], rr_['reason'] = 'ineligible', 'MG solve not converged'
    Ut = np.load(out / 'refs_test.npz')['U']
    U_t = SC.allowance(rich_t, rj['delta_t'])
    Ud = load_refs(out, 'dev')
    B, spaces = builders(a, out)
    res, qs, outs_t, res_by_key = {}, {}, {}, {}
    for k, lst in M.items():
        fam, tau, stat = k.split('|')
        for v in lst:
            if (fam, v['key']) in qs:
                base = res_by_key[(fam, v['key'])]
                e = np.asarray(base['e'])
                sv = float(np.max(e) if stat == 'worst' else np.median(e))
                res[f"{k}|{v['key']}"] = dict(base, stat=sv, verdict=SC.three_way(sv, U_t['U'], float(tau)), reused=True)
                continue
            q = B[fam](v['cfg'])
            O, st = FZ.run_unbatched(q, D['test'])
            outs_t[(fam, v['key'])] = O
            st_valid = LD.stats_valid(v['cfg'], st)
            e = SC.err(O, Ut)
            sv = float(np.max(e) if stat == 'worst' else np.median(e))
            item = dict(key=v['key'], e=e.tolist(), stat=sv, verdict=SC.three_way(sv, U_t['U'], float(tau)),
                        Q1=FZ.q1(fam, v['cfg'], B[fam], D['test'], D['dev'], Ut, Ud, O, st), stats_valid=st_valid)
            if fam == 'bank':
                item['Q2'] = FZ.q2_bank(v['cfg'], spaces, D['test'], Ut)
            res[f"{k}|{v['key']}"] = item
            res_by_key[(fam, v['key'])] = item
            qs[(fam, v['key'])] = q
    timing, cache, summary = {}, {}, {}
    Dt = D['test'][:FZ.N_TIME]
    cc = FZ.CompileCounter()
    logging.getLogger('jax').addHandler(cc)
    for q_ in qs.values():                      # warm every finalist (with a nu not among the timed cases)
        jax.block_until_ready(q_(jnp.asarray(Dt[0] * np.array([1, 1, 1, 1, 1.0001]))))
    sent_t = next(((k_, FZ.screen_time(q_, Dt[0], reps=5)) for k_, q_ in qs.items() if k_[0] != 'bank'), None)
    jax.config.update('jax_log_compiles', True)
    for k, lst in M.items():
        fam, tau, stat = k.split('|')
        if fam == 'bank':
            continue
        for sa in lst:
            for sb in M.get(f'bank|{tau}|{stat}', []):
                pk = (fam, sa['key'], sb['key'])
                if pk not in cache:
                    n0_ = cc.n
                    p, oA, oB = FZ.paired(qs[(fam, sa['key'])], qs[('bank', sb['key'])], Dt)
                    p['compiles_during_timing'] = cc.n - n0_
                    p['T_id'] = max(FZ.t_id(oA, outs_t[(fam, sa['key'])][:len(Dt)]),
                                    FZ.t_id(oB, outs_t[('bank', sb['key'])][:len(Dt)]))
                    p['valid'] = bool(p['T_id'] <= 1e-12 and p['compiles_during_timing'] == 0)
                    cache[pk] = p
                timing[f"{k}|{sa['key']}|{sb['key']}"] = cache[pk]
    for tau in SC.TAUS:
        for stat in ('worst', 'median'):
            def ok_(fam_, key_):
                it_ = res.get(f'{fam_}|{tau}|{stat}|{key_}')
                if not it_ or it_['verdict'] != 'meets' or not it_['stats_valid']:
                    return False
                if fam_ == 'bank':
                    return bool(it_['Q1']['passed'] and it_.get('Q2', {}).get('passed', False))
                return True
            bk = [sb['key'] for sb in M.get(f'bank|{tau}|{stat}', []) if ok_('bank', sb['key'])]
            comp_keys = [(k2.split('|')[0], sa['key']) for k2, lst in M.items() if not k2.startswith('bank')
                         and k2.split('|')[1:] == [str(tau), stat] for sa in lst if ok_(k2.split('|')[0], sa['key'])]
            pairs = {(fa, ka, kb): (cache[(fa, ka, kb)]['S'] if cache.get((fa, ka, kb), {}).get('valid') else None)
                     for fa, ka in comp_keys for kb in bk}
            summary[f'{tau}|{stat}'] = FZ.maxmin(pairs, bk, comp_keys)
    jax.config.update('jax_log_compiles', False)
    sentinel = {}
    if sent_t:
        sentinel = dict(start=sent_t[1], end=FZ.screen_time(qs[sent_t[0]], Dt[0], reps=5))
        sentinel['valid'] = bool(abs(sentinel['end'] / sentinel['start'] - 1) <= 0.10)
        if not sentinel['valid']:
            for sm_ in summary.values():
                if sm_ and sm_.get('S') is not None:
                    sm_['S_unchecked'], sm_['S'], sm_['reason'] = sm_['S'], None, 'sentinel drift > 10 %'
    td_k = {fk: FZ.t_dbl(q_, Dt) for fk, q_ in qs.items()}
    tdbl = {f_: dict(passed=all(v['passed'] for (g, _k), v in td_k.items() if g == f_)) for f_ in {g for g, _ in qs}}
    for k_, sm_ in summary.items():
        if sm_ and sm_.get('S') is not None:
            bad = [f_ for f_ in set(sm_['per_family']) | {'bank'} if not tdbl.get(f_, {'passed': True})['passed']]
            if bad:
                sm_['S_unchecked'], sm_['S'], sm_['reason'] = sm_['S'], None, f'T-dbl failed: {bad}'
    inter = interaction_split(a, out, D, 'test')
    (out / 'test.json').write_text(json.dumps(dict(results=res, timing=timing, summary=summary, rich_test=rich_t, T_dbl=tdbl, sentinel=sentinel,
                                                   allowance_test=U_t, inter_test=inter,
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
    ap.add_argument('--frozen', default=None, help='committed copy of frozen.json (staged with the code)')
    a = ap.parse_args()
    if jax.default_backend() != 'gpu' and not a.allow_cpu:
        raise SystemExit(42)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = cohorts(a.fam, a.lev, a.smoke)
    G.check_cohorts()
    if a.smoke:                                   # smoke only: a loose extra target so every finalist path runs
        SC.TAUS = SC.TAUS + (0.2,)
        globals()['SMOKE_RELAX_Q2'] = True
    marks = {'ref': 'ref.json', 'data': 'data.json', 'bank': 'bank.json', 'ladder': 'ladder.done', 'time': 'final.json',
             'test': 'test.json', 'inter': 'inter.json', 'diag': 'diag.json', 'freeze': 'frozen.json'}
    for st in a.stages.split(','):
        mk = out / marks.get(st, f'{st}.done')
        if mk.exists():
            log(f'stage {st}: done ({mk.name})')
            continue
        t0 = time.perf_counter()
        log(f'stage {st}: start')
        dict(ref=stage_ref, data=stage_data, bank=stage_bank, ladder=stage_ladder, time=stage_time,
             test=stage_test, inter=stage_inter, diag=stage_diag, freeze=stage_freeze)[st](a, out, D)
        if st == 'ladder':
            mk.write_text('ok')
        import resource
        dev_peak = max(((d.memory_stats() or {}).get('peak_bytes_in_use', 0) for d in jax.devices()), default=0)
        host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        with (out / 'stage_seconds.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=st, seconds=time.perf_counter() - t0, device_peak_bytes=int(dev_peak),
                                    host_peak_bytes=int(host_peak), env=env_record())) + '\n')
        log(f'stage {st}: {time.perf_counter() - t0:.0f}s')


if __name__ == '__main__':
    main()
