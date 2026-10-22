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
# rule ladder (A7): q on k_q = 4 elements, then 'f8' / 'f12' = q 8 / 12 on k_q = 8; successor of each rule
SUCC = {4: 6, 6: 8, 8: 10, 10: 12, 12: 14, 14: 'f8', 'f8': 'f12', 'f12': 'f16'}


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
            Oc, st_prod = run_unbatched(build(dict(cfg, cert=True)), D_val)       # untimed certificate replay
            it['T_id_cert_replay'] = float(np.max(np.abs(Oc - O_prod)) / (np.max(np.abs(O_prod)) + 1e-300))
            it['nonfinite_steps'] = float(st_prod[:, 3].sum())
            it['blowup_frac'] = float(st_prod[:, 0].sum() / max(st_prod[:, 2].sum(), 1))
            it['coef_cert_max'] = float(st_prod[:, 1].max())
            it['update_defect_max'] = float(st_prod[:, 4].max())
            it['passed'] = bool(valid and it['nonfinite_steps'] == 0 and it['blowup_frac'] == 0
                                and it['coef_cert_max'] <= 1e-10 and it['update_defect_max'] <= 1e-9
                                and it['T_id_cert_replay'] <= 1e-12)
            return it
        qc = build(dict(cfg, cert=True))
        Oc, stc = run_unbatched(qc, D_val)
        it['T_id_cert_replay'] = float(np.max(np.abs(Oc - O_prod)) / (np.max(np.abs(O_prod)) + 1e-300))
        n_ns, n_st = float(stc[:, 0].sum()), float(stc[:, 2].sum())
        it['nonstationary_frac'] = n_ns / max(n_st, 1)
        it['max_gamma'] = float(stc[:, 1].max())
        it['nonfinite'] = float(stc[:, 3].sum())
        it['damping_saturated'] = float(stc[:, 5].sum())
        it['unconverged_steps'] = float(stc[:, 6].sum())
        tight = 'tight'
        if stp == 'gal_cnr_lin':
            cmp_cfg = [dict(cfg, st='gal_cnr', tol='tight'), dict(cfg, st='gal_cnr', tol='tight'), dict(cfg, st='gal_cnr', tol='tighter')]
        else:
            cmp_cfg = [dict(cfg, tol=tight), dict(cfg, tol='tight'), dict(cfg, tol='tighter')]
        import ladder as LD
        (Ot, s_t), (Od1, s_1), (Od2, s_2) = (run_unbatched(build(c_), D_) for c_, D_ in zip(cmp_cfg, (D_val, D_dev, D_dev)))
        it['comparison_solves_valid'] = bool(all(LD.stats_valid(c_, s_) for c_, s_ in zip(cmp_cfg, (s_t, s_1, s_2))))
        it['dist_tight_max'] = float(np.max(dist(O_prod, Ot, Uv)))
        it['tight_tighter_max'] = float(np.max(dist(Od1, Od2, Ud)))
        # LSPG: <= 1 % non-stationary steps (gamma); Galerkin / linearised CN certificates must hold at EVERY step
        ns_ok = it['nonstationary_frac'] <= 0.01 if stp == 'lspg_cnr' else it['nonstationary_frac'] == 0
        steps_ = max(float(stc[:, 2].sum()), 1.0)
        unc_ok = (it['unconverged_steps'] == 0 if stp == 'gal_cnr' else
                  (it['unconverged_steps'] / steps_ <= 0.01 if stp == 'lspg_cnr' else True))
        it['passed'] = bool(valid and ns_ok and unc_ok and it['comparison_solves_valid'] and it['nonfinite'] == 0
                            and it['damping_saturated'] == 0 and it['dist_tight_max'] <= Q1_TIGHT
                            and it['tight_tighter_max'] <= Q1_TIGHTER and it['T_id_cert_replay'] <= 1e-12)
        return it
    if fam in ('fdmg',) or (fam == 'sem' and cfg.get('solver') == 'pcg'):
        it['unconverged_solves'] = float(st_prod[:, 2].sum())
        import ladder as LD
        cc_ = [dict(cfg, tol=1e-10), dict(cfg, tol=1e-10), dict(cfg, tol=1e-12)]
        (Ot, stt), (Od1, s1_), (Od2, s2_) = (run_unbatched(build(c_), D_) for c_, D_ in zip(cc_, (D_val, D_dev, D_dev)))
        it['comparison_solves_valid'] = bool(all(LD.stats_valid(c_, s_) for c_, s_ in zip(cc_, (stt, s1_, s2_))))
        it['dist_tight_max'] = float(np.max(dist(O_prod, Ot, Uv)))
        it['tight_tighter_max'] = float(np.max(dist(Od1, Od2, Ud)))
        it['passed'] = bool(valid and it['unconverged_solves'] == 0 and it['comparison_solves_valid']
                            and it['dist_tight_max'] <= Q1_TIGHT
                            and it['tight_tighter_max'] <= Q1_TIGHTER)
        return it
    it['passed'] = valid                      # explicit / direct solvers: validity only (rk4, spec, sem-eig)
    return it


def rho_ok(rho):
    """The Q2 rho predicate (shared by q2_bank and the MF-Q2 control)."""
    return bool(np.isfinite(rho) and rho <= RHO_BAR)


def q2_bank(cfg, spaces, D_pop, Upop):
    """Bank Q2: rule q vs q+2 on the arm's own reached states (A1.8)."""
    q, r, stp = cfg['q'], cfg['r'], cfg['st']
    s1, s2 = spaces[q], spaces[SUCC[q]]
    Mt = min(4 * r, s1.R) if stp == 'lspg_cnr' else r
    b1 = lambda c, sp: RM.make_query(sp, c['r'], c['st'], c['nsteps'], form=c['form'], tol=c['tol'], out='z')
    import ladder as LD
    Z1, _ = run_unbatched(b1(cfg, s1), D_pop)
    O1, st1 = run_unbatched(RM.make_query(s1, r, stp, cfg['nsteps'], form=cfg['form'], tol=cfg['tol']), D_pop)
    O2, st2 = run_unbatched(RM.make_query(s2, r, stp, cfg['nsteps'], form=cfg['form'], tol=cfg['tol']), D_pop)
    solves_valid = bool(LD.stats_valid(cfg, st1) and LD.stats_valid(cfg, st2))

    def Nvec(sp, z):
        P, S = sp.P[:, :r], sp.S[:, :r]
        return sp.P[:, :Mt].T @ (sp.w * (P @ z) * (S @ z))
    rho, skipped = [], 0
    allN = [np.linalg.norm(Nvec(s2, z)) for z in Z1.reshape(-1, r)]
    mx = max(allN)                          # maximum over the whole population (all cases x 6 output states)
    for z, n2 in zip(Z1.reshape(-1, r), allN):
        if n2 <= 1e-12 * mx or n2 == 0.0:          # zero-norm states are always skipped (all-zero -> Q2 fails)
            skipped += 1
            continue
        rho.append(float(np.linalg.norm(Nvec(s1, z) - Nvec(s2, z)) / n2))
    dM = float(np.linalg.norm(s1.M[:Mt, :r] - s2.M[:Mt, :r]) / np.linalg.norm(s2.M[:Mt, :r]))   # tested operators
    dK = float(np.linalg.norm(s1.K[:Mt, :r] - s2.K[:Mt, :r]) / np.linalg.norm(s2.K[:Mt, :r]))
    Z2, _ = run_unbatched(b1(cfg, s2), D_pop)
    dIC = float(np.max(np.linalg.norm(Z1[:, 0] - Z2[:, 0], axis=-1) / np.linalg.norm(Z2[:, 0], axis=-1)))
    roll = dist(O1, O2, Upop)
    it = dict(rho_max=float(max(rho)) if rho else float('nan'), rho_skipped=skipped, dM=dM, dK=dK, dIC=dIC,
              solves_valid=solves_valid,
              rollout_max=float(np.max(roll)))
    it['skipped_frac'] = skipped / max(len(allN), 1)
    # A7: the gated Q2 bars are the brief's (rho, rollout on every case); M / K / IC differences are reported
    it['passed'] = bool(solves_valid and rho and rho_ok(it['rho_max']) and it['skipped_frac'] <= 0.05
                        and it['rollout_max'] <= Q1_TIGHT)
    return it


# ----------------------------------------------------------------------------------------------- timing --------

def paired(qA, qB, D, reps=REPS):
    """A-B-A triplets per case and repetition; raw times, ratios t_A / t_B (t_A = mean of the two A runs) and the
    outputs of the last timed A and B call per case (T-id against the official outputs)."""
    rows, outA, outB = [], [], []
    for d in D:
        dj = jnp.asarray(d)
        for f_ in (qA, qB):                    # burn-in 3 (already compiled)
            for _ in range(3):
                jax.block_until_ready(f_(dj))
        oas, obs = [], []
        for _ in range(reps):
            t0 = time.perf_counter(); oa = jax.block_until_ready(qA(dj)); t1 = time.perf_counter()
            ob = jax.block_until_ready(qB(dj)); t2 = time.perf_counter()
            oa2 = jax.block_until_ready(qA(dj)); t3 = time.perf_counter()
            rows.append(dict(a1=t1 - t0, b=t2 - t1, a2=t3 - t2))
            oas += [oa[0], oa2[0]]
            obs.append(ob[0])
        outA.append(np.stack([np.asarray(x) for x in oas]))      # every timed A output of this case
        outB.append(np.stack([np.asarray(x) for x in obs]))
    ratios = [0.5 * (r['a1'] + r['a2']) / r['b'] for r in rows]
    return dict(raw=rows, S=float(np.median(ratios)), tA=float(np.median([0.5 * (r['a1'] + r['a2']) for r in rows])),
                tB=float(np.median([r['b'] for r in rows]))), np.stack(outA), np.stack(outB)


def t_id(O_timed, O_official):
    """O_timed (cases, calls, 6, P) every timed output; O_official (cases, 6, P)."""
    return float(np.max(np.abs(O_timed - O_official[:, None])) / (np.max(np.abs(O_official)) + 1e-300))


def t_dbl(q, D):
    """T-dbl (A7): two different cases run back to back (serial, each blocked) in one timed region must take the sum of
    the two single-case times: ratio t(d1; d2) / (t(d1) + t(d2)) in [0.9, 1.1] (case-dependent iteration counts make a
    plain 2x bar wrong for iterative solvers)."""
    d1, d2 = jnp.asarray(D[0]), jnp.asarray(D[1])
    for _ in range(2):
        jax.block_until_ready(q(d1)); jax.block_until_ready(q(d2))
    t1, t2, tb = [], [], []
    for _ in range(5):
        t = time.perf_counter(); jax.block_until_ready(q(d1)); jax.block_until_ready(q(d2)); tb.append(time.perf_counter() - t)
        t = time.perf_counter(); jax.block_until_ready(q(d1)); t1.append(time.perf_counter() - t)
        t = time.perf_counter(); jax.block_until_ready(q(d2)); t2.append(time.perf_counter() - t)
    r = float(np.median(tb) / (np.median(t1) + np.median(t2)))
    return dict(ratio=r, passed=bool(0.9 <= r <= 1.1))


def maxmin(pairs, bank_keys, comp_keys):
    """S = max over bank finalists b of min over ALL competitor finalists a of S(a, b) (A4.2). pairs: {(fam, a_key,
    b_key): S or None (invalid timing)}; comp_keys: [(fam, a_key)] every competitor finalist. A bank finalist whose row
    of the matrix is incomplete or contains an invalid timing is excluded (no S without every competitor)."""
    if not comp_keys or not bank_keys:
        return None
    per_b, excluded = {}, []
    for kb in bank_keys:
        vals = [(pairs.get((fa, ka, kb)), fa, ka) for fa, ka in comp_keys]
        if any(v is None for v, _, _ in vals):
            excluded.append(kb)
            continue
        per_b[kb] = min(vals)
    if not per_b:
        return dict(S=None, reason='timing matrix incomplete or invalid for every bank finalist', excluded=excluded)
    kb = sorted(per_b, key=lambda k: (-per_b[k][0], k))[0]
    v, fam, ka = per_b[kb]
    per_fam = {}
    for fa, k2 in comp_keys:
        per_fam[fa] = min(per_fam.get(fa, np.inf), pairs[(fa, k2, kb)])
    return dict(S=v, best_competitor=fam, competitor_key=ka, bank_key=kb, per_family=per_fam, excluded=excluded,
                within5pct=sorted(fa for fa, x in per_fam.items() if x <= 1.05 * v))


def manifest_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finer_diag(a, out, fam, cfg, B, D8, U128, pts128):
    """A3.7: error of a matched configuration on the x = i/128 lattice (val[0:8]); FD by bilinear interpolation."""
    import geom as G
    import fd
    import sem
    import spec
    import run_cell as RC
    if fam in ('fdmg', 'rk4'):
        g = fd.Grid(a.dom, cfg['n'])
        I, J = np.nonzero(g.mask_np)
        q = (fd.make_sbdf2(a.dom, cfg['n'], cfg['nsteps'], cfg['tol'], out_idx=(I, J)) if fam == 'fdmg'
             else fd.make_rk4(a.dom, cfg['n'], cfg['nsteps'], out_idx=(I, J)))
        O = []
        for d in D8:
            v = np.asarray(q(jnp.asarray(d))[0])
            full = np.zeros((6, g.N1 + 2, g.N1 + 2))
            full[:, I + 1, J + 1] = v
            xs = pts128 * cfg['n']
            i0 = np.floor(xs[:, 0]).astype(int)
            j0 = np.floor(xs[:, 1]).astype(int)
            tx, ty = xs[:, 0] - i0, xs[:, 1] - j0
            O.append((1 - tx) * (1 - ty) * full[:, i0, j0] + tx * (1 - ty) * full[:, i0 + 1, j0]
                     + (1 - tx) * ty * full[:, i0, j0 + 1] + tx * ty * full[:, i0 + 1, j0 + 1])
        O = np.stack(O)
        label = 'bilinear interpolation of the FD grid'
    elif fam == 'sem':
        q, _ = sem.make_sem(a.dom, cfg['p'], cfg['k'], cfg['nsteps'], cfg['tol'], order=cfg['ord'], grade=cfg['g'],
                            dealias=cfg['dea'], solver=cfg['solver'], pts=pts128)
        O, _ = run_unbatched(q, D8)
        label = 'native (element interpolation)'
    elif fam == 'spec':
        q = spec.make_spec(cfg['N'], cfg['nsteps'], n_init=cfg['ni'], lat=128)
        O, _ = run_unbatched(q, D8)
        label = 'native (sine synthesis)'
    else:
        sp = B['_space'](fam, cfg, lat=128)
        q = RM.make_query(sp, cfg['r'], cfg['st'], cfg['nsteps'], form=cfg['form'], tol=cfg['tol'])
        O, _ = run_unbatched(q, D8)
        label = 'native (lattice rows)'
    e = SC.err(O, U128)
    return dict(worst=float(np.max(e)), median=float(np.median(e)), e=e.tolist(), how=label)
