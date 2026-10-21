"""imp-geometry finalists, qualification and paired timing (DESIGN 4, 6, 7, A1.1, A1.7-A1.9).

stage `time` (val): per family and (tau, statistic): candidates = visited valid ladder rows that meet tau (three-way
rule with the cell allowance U), ordered by a warm one-case screen time (top K = 6 by cohort seconds are screened);
the screen-fastest candidate is re-run UNBATCHED on all val cases (official outputs), must still meet tau, and must pass
Q1 (and Q2 for the bank); otherwise the next candidate. Then paired A-B-A timing on val[0:8] for every
(competitor finalist A, bank finalist B) pair; T-id, T-dbl, T-setup controls. Writes final.json and manifest.json.

stage `test` (test-once): refuses unless the committed manifest's sha256 matches (MF-LEAK); runs the frozen finalists
on test32 (accuracy, Q1, paired re-timing on test[0:8]); never selects anything.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import score as SC
import rom as RM

COMPETITORS = ('sem', 'fdmg', 'rk4', 'ecsw', 'podg', 'spec')
K_SCREEN = 6
REPS = 5
N_TIME = 8
Q1_TIGHT = 2.5e-4
Q1_TIGHTER = 2.5e-5
RHO_BAR = 0.116


class CompileCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.n = 0

    def emit(self, record):
        if 'Compiling' in record.getMessage() or 'compil' in record.getMessage().lower():
            self.n += 1


def run_unbatched(q, D):
    outs, sts = [], []
    for d in D:
        o, st = q(jnp.asarray(d))
        outs.append(np.asarray(o))
        sts.append(np.asarray(st))
    return np.stack(outs), np.stack(sts)


def screen_time(q, d, reps=3):
    dj = jnp.asarray(d)
    jax.block_until_ready(q(dj))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(q(dj))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def dist(O1, O2, Uref):
    n0 = np.linalg.norm(Uref[:, 0], axis=-1)
    return np.max(np.linalg.norm(O1[:, 1:] - O2[:, 1:], axis=-1), axis=-1) / n0


# ------------------------------------------------------------------------------------------- qualification -----

def q1(fam, cfg, build, D_val, D_dev, Uv, Ud, O_prod, st_prod):
    """Q1 for a configuration (DESIGN 7, A1.8). Returns dict(passed, items)."""
    it = {}
    valid = bool(np.all(np.isfinite(O_prod)))
    it['valid'] = valid
    if fam in ('podg', 'ecsw', 'bank'):
        stp = cfg['st']
        if stp == 'gal_etdrk4':
            it['nonfinite_steps'] = float(st_prod[:, 3].sum())
            it['blowups'] = float(st_prod[:, 0].sum())
            it['passed'] = valid and it['nonfinite_steps'] == 0 and it['blowups'] == 0
            return it
        qc = build(dict(cfg, cert=True))
        Oc, stc = run_unbatched(qc, D_val)
        it['T_id_cert_replay'] = float(np.max(np.abs(Oc - O_prod)) / (np.max(np.abs(O_prod)) + 1e-300))
        n_ns, n_st = float(stc[:, 0].sum()), float(stc[:, 2].sum())
        it['nonstationary_frac'] = n_ns / max(n_st, 1)
        it['max_gamma'] = float(stc[:, 1].max())
        it['nonfinite'] = float(stc[:, 3].sum())
        it['damping_saturated'] = float(stc[:, 5].sum())
        tight = 'tight'
        if stp == 'gal_cnr_lin':
            Ot, _ = run_unbatched(build(dict(cfg, st='gal_cnr', tol='tight')), D_val)
            Od1, _ = run_unbatched(build(dict(cfg, st='gal_cnr', tol='tight')), D_dev)
            Od2, _ = run_unbatched(build(dict(cfg, st='gal_cnr', tol='tighter')), D_dev)
        else:
            Ot, _ = run_unbatched(build(dict(cfg, tol=tight)), D_val)
            Od1, _ = run_unbatched(build(dict(cfg, tol='tight')), D_dev)
            Od2, _ = run_unbatched(build(dict(cfg, tol='tighter')), D_dev)
        it['dist_tight_max'] = float(np.max(dist(O_prod, Ot, Uv)))
        it['tight_tighter_max'] = float(np.max(dist(Od1, Od2, Ud)))
        it['passed'] = bool(valid and it['nonstationary_frac'] <= 0.01 and it['nonfinite'] == 0
                            and it['damping_saturated'] == 0 and it['dist_tight_max'] <= Q1_TIGHT
                            and it['tight_tighter_max'] <= Q1_TIGHTER and it['T_id_cert_replay'] <= 1e-12)
        return it
    if fam in ('fdmg',) or (fam == 'sem' and cfg.get('solver') == 'pcg'):
        it['unconverged_solves'] = float(st_prod[:, 2].sum())
        Ot, stt = run_unbatched(build(dict(cfg, tol=1e-10)), D_val)
        Od1, _ = run_unbatched(build(dict(cfg, tol=1e-10)), D_dev)
        Od2, _ = run_unbatched(build(dict(cfg, tol=1e-12)), D_dev)
        it['dist_tight_max'] = float(np.max(dist(O_prod, Ot, Uv)))
        it['tight_tighter_max'] = float(np.max(dist(Od1, Od2, Ud)))
        it['passed'] = bool(valid and it['unconverged_solves'] == 0 and it['dist_tight_max'] <= Q1_TIGHT
                            and it['tight_tighter_max'] <= Q1_TIGHTER)
        return it
    it['passed'] = valid                      # explicit / direct solvers: validity only (rk4, spec, sem-eig)
    return it


def q2_bank(cfg, spaces, D_pop, Upop):
    """Bank Q2: rule q vs q+2 on the arm's own reached states (A1.8)."""
    q, r, stp = cfg['q'], cfg['r'], cfg['st']
    s1, s2 = spaces[q], spaces[q + 2]
    Mt = min(4 * r, s1.R) if stp == 'lspg_cnr' else r
    b1 = lambda c, sp: RM.make_query(sp, c['r'], c['st'], c['nsteps'], form=c['form'], tol=c['tol'], out='z')
    Z1, _ = run_unbatched(b1(cfg, s1), D_pop)
    O1, _ = run_unbatched(RM.make_query(s1, r, stp, cfg['nsteps'], form=cfg['form'], tol=cfg['tol']), D_pop)
    O2, _ = run_unbatched(RM.make_query(s2, r, stp, cfg['nsteps'], form=cfg['form'], tol=cfg['tol']), D_pop)

    def Nvec(sp, z):
        P, S = sp.P[:, :r], sp.S[:, :r]
        return sp.P[:, :Mt].T @ (sp.w * (P @ z) * (S @ z))
    rho, skipped = [], 0
    allN = [np.linalg.norm(Nvec(s2, z)) for z in Z1.reshape(-1, r)]
    mx = max(allN)
    for z, n2 in zip(Z1.reshape(-1, r), allN):
        if n2 <= 1e-12 * mx:
            skipped += 1
            continue
        rho.append(float(np.linalg.norm(Nvec(s1, z) - Nvec(s2, z)) / n2))
    dM = float(np.linalg.norm(s1.M[:r, :r] - s2.M[:r, :r]) / np.linalg.norm(s2.M[:r, :r]))
    dK = float(np.linalg.norm(s1.K[:r, :r] - s2.K[:r, :r]) / np.linalg.norm(s2.K[:r, :r]))
    roll = dist(O1, O2, Upop)
    it = dict(rho_max=float(max(rho)) if rho else float('nan'), rho_skipped=skipped, dM=dM, dK=dK,
              rollout_max=float(np.max(roll)))
    it['passed'] = bool(rho and it['rho_max'] <= RHO_BAR and dM <= 1e-6 and dK <= 1e-6 and it['rollout_max'] <= Q1_TIGHT)
    return it


# ----------------------------------------------------------------------------------------------- timing --------

def paired(qA, qB, D, reps=REPS):
    """A-B-A triplets per case and repetition; returns raw times and ratios t_A / t_B (t_A = mean of the two A)."""
    rows = []
    for d in D:
        dj = jnp.asarray(d)
        for f in (qA, qB):                     # warm (already compiled; burn-in 3)
            for _ in range(3):
                jax.block_until_ready(f(dj))
        for _ in range(reps):
            t0 = time.perf_counter(); oa = jax.block_until_ready(qA(dj)); t1 = time.perf_counter()
            ob = jax.block_until_ready(qB(dj)); t2 = time.perf_counter()
            jax.block_until_ready(qA(dj)); t3 = time.perf_counter()
            rows.append(dict(a1=t1 - t0, b=t2 - t1, a2=t3 - t2))
    ratios = [0.5 * (r['a1'] + r['a2']) / r['b'] for r in rows]
    return dict(raw=rows, S=float(np.median(ratios)), tA=float(np.median([0.5 * (r['a1'] + r['a2']) for r in rows])),
                tB=float(np.median([r['b'] for r in rows])))


def t_dbl(q, D):
    """T-dbl: two different cases back to back in one timed call vs single calls (ratio must be 1.8-2.2)."""
    d1, d2 = jnp.asarray(D[0]), jnp.asarray(D[1])
    both = jax.jit(lambda a, b: (q(a), q(b)))
    jax.block_until_ready(both(d1, d2)); jax.block_until_ready(q(d1))
    ts, td = [], []
    for _ in range(5):
        t = time.perf_counter(); jax.block_until_ready(q(d1)); ts.append(time.perf_counter() - t)
        t = time.perf_counter(); jax.block_until_ready(both(d1, d2)); td.append(time.perf_counter() - t)
    r = float(np.median(td) / np.median(ts))
    return dict(ratio=r, passed=bool(1.8 <= r <= 2.2))


def manifest_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
