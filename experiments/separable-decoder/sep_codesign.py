"""CO-DESIGN -- QUADRATURE-AWARE FINE-TUNE OF h AND THE NODES TOGETHER.

Design doc: understand/2026-08-26-codesign-design.md (main).  The decoder has
so far been trained knowing nothing about the fact that its residual will be
sampled at m points; stage 3 (sep_eq_nodefit.py) showed that with the decoder
frozen, moving the points buys ~3% and only at coarse m.  This driver flips
the dependency: starting from a committed checkpoint it fine-tunes the h-track
(params["h"], params["h_lin"]) JOINTLY with m continuous node positions so
that the m-point advection term matches the full-grid advection term -- value
(L_samp) and z-Jacobian (L_jac) -- while plain reconstruction (L_rec) and
optionally derivative reconstruction (L_sob) anchor the decoder to the data.

The BANK IS FROZEN (params["B"], params["g"], out_scale): every cached-bank
identity, A = Phi^T G exact-linear matrix, and the span floor are untouched.
Node weights w are never gradient-trained: they are re-solved by NNLS on the
exact loss-form rows every REFIT_EVERY steps (variable projection, the
sep_eq_nodefit.py recipe).

Loss (per arm, terms switched by env):

  L = REC_W * L_rec/L_rec0  + SOB_REL * L_sob/L_sob0
    + SAMP_REL * L_samp/L_samp0 + JAC_REL * L_jac/L_jac0
    + MINSEP_W * minsep(X)

Every term is normalized by its OWN value at the warm start, so all active
terms start at 1.0 and the *_REL knobs are direct relative weights.  The
mismatch terms compare the sampled advection projection against the
full-grid advection projection OF THE SAME CURRENT DECODER -- a mismatch
cannot be gamed by fooling the points, only by genuinely becoming easier to
integrate; the anti-cheat 'shrink the advection term' path is blocked by
stop-gradient denominators plus the L_rec anchor.

Fit states are TRAINING codes plus Gaussian latent perturbations (the
stage-2 lesson: light on solver-path iterates -- state-conditioned fits
overfit their fit distribution, +13% test error at N=256 dense_mid).  A
held-out slice of codes+perturbations is excluded from every gradient and
every NNLS row and drives the tripwire: if held-out reconstruction degrades
more than TRIP relative, the run flags itself (FREEZE_WDEC guard).

Gates (all before training):
  gate R : warm-start reconstruction vs regenerated snapshots (code->snapshot
           mapping is right; O(1) here means the r1 pick was not reproduced)
  gate H : h_of(hp0, z) == dec.head_fn()(z) bit-level (param split identity)
  gate 0 : grid-ops closure == blat_common.make_weak_ops at the NNLS nodes
  gate C : continuous machinery at the grid init == grid ops (<=1e-12)
  gate D : directional finite difference of the loss vs autodiff gradient
Certification (base AND final, same instrument): gate L (exlin linear vs
full-grid linear, per variant h), held-out ladder-lite rungs (b, c1, c3 vs
the variant's own full-grid ops), N_TEST test-trajectory rollouts with the
r5 LM rule, held-out reconstruction vs truth.  Certified checkpoint is saved
as a drop-in pkl (merged params, same Z_tr) + nodes npz.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc
import sep_solvers as ss

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402
import ctol_eq                                # noqa: E402
import exlin_common as xc                     # noqa: E402
from sep_burgers_r3 import build_test_full    # noqa: E402

F64 = jnp.float64

CKPT = os.environ["CKPT"]
N = int(os.environ.get("N", "64"))
N_TEST = int(os.environ.get("N_TEST", "4"))
EQ_M = int(os.environ.get("EQ_M", "64"))
EQ_M_FACTOR = int(os.environ.get("EQ_M_FACTOR", "4"))
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))

TRAIN_H = int(os.environ.get("TRAIN_H", "1"))
TRAIN_NODES = int(os.environ.get("TRAIN_NODES", "1"))
STEPS = int(os.environ.get("STEPS", "2000"))
LR = float(os.environ.get("LR", "3e-4"))
LR_NODES = float(os.environ.get("LR_NODES", "3e-3"))
REFIT_EVERY = int(os.environ.get("REFIT_EVERY", "200"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "250"))

REC_W = float(os.environ.get("REC_W", "1.0"))
SOB_REL = float(os.environ.get("SOB_REL", "0"))
SAMP_REL = float(os.environ.get("SAMP_REL", "1.0"))
JAC_REL = float(os.environ.get("JAC_REL", "0"))
MINSEP_W = float(os.environ.get("MINSEP_W", "0"))    # 0 -> 1/dx^2
MINSEP_D = float(os.environ.get("MINSEP_D", "0"))    # 0 -> dx/2
TRIP = float(os.environ.get("TRIP", "0.03"))

N_FIT_STATES = int(os.environ.get("N_FIT_STATES", "64"))
PERT_PER = int(os.environ.get("PERT_PER", "1"))
PERT_SIGMA = float(os.environ.get("PERT_SIGMA", "0.05"))
N_HELDOUT = int(os.environ.get("N_HELDOUT", "32"))
N_ANCHOR = int(os.environ.get("N_ANCHOR", "256"))
N_ANCHOR_HELD = int(os.environ.get("N_ANCHOR_HELD", "64"))

SEED0 = int(os.environ.get("SEED0", "0"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
STALL = float(os.environ.get("STALL", "1e-3"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
DATA_CACHE = os.environ.get("DATA_CACHE", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
OUT_TAG = os.environ.get("OUT_TAG", "")


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} CODESIGN N={N} "
           f"ckpt={os.path.basename(CKPT)} TRAIN_H={TRAIN_H} "
           f"TRAIN_NODES={TRAIN_NODES} m={EQ_M_FACTOR*EQ_M} steps={STEPS}")
    t_all = time.time()
    params, Z_tr, cfg = sc.load_pkl(CKPT)
    K, R = int(cfg["k"]), int(cfg["r"])
    assert int(cfg["N"]) == N, f"ckpt N={cfg['N']} != N={N}"
    dec = sc.SeparableDecoder(params, K, R)
    h_fn0 = dec.head_fn()
    tag = OUT_TAG or f"N{N}_K{K}_R{R}"
    OUT = f"{OUT_PREFIX}sep_codesign_{tag}.json"

    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    n_i2 = interior.size
    T = bc.NUM_STEPS + 1
    dx = 1.0 / (N - 1)
    global MINSEP_W, MINSEP_D
    if MINSEP_D == 0.0:
        MINSEP_D = 0.5 * dx
    if MINSEP_W == 0.0:
        MINSEP_W = 1.0 / (dx * dx)
    m_want = EQ_M_FACTOR * EQ_M

    report = dict(config=dict(
        pde="burgers2d", kind="codesign", N=N, k=K, r=R,
        ckpt=os.path.basename(CKPT), ckpt_cfg=cfg, n_test=N_TEST, eq_M=EQ_M,
        eq_m_factor=EQ_M_FACTOR, exlin=True, train_h=TRAIN_H,
        train_nodes=TRAIN_NODES, steps=STEPS, lr=LR, lr_nodes=LR_NODES,
        refit_every=REFIT_EVERY, eval_every=EVAL_EVERY, rec_w=REC_W,
        sob_rel=SOB_REL, samp_rel=SAMP_REL, jac_rel=JAC_REL,
        minsep_w=MINSEP_W, minsep_d=MINSEP_D, trip=TRIP,
        n_fit_states=N_FIT_STATES, pert_per=PERT_PER, pert_sigma=PERT_SIGMA,
        n_heldout=N_HELDOUT, n_anchor=N_ANCHOR, n_anchor_held=N_ANCHOR_HELD,
        step_tol=STEP_TOL, stall=STALL, extrap=EXTRAP,
        gn_budget=bc.GN_BUDGET, num_steps=bc.NUM_STEPS, dt=bc.DT,
        weak_alpha=bc.WEAK_ALPHA, seed=SEED0, data_seed=bc.SEED,
        test_seed=bc.TEST_SEED, x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        gates={}, init_terms={}, hist=[], variants={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- data: training snapshots + fresh test ------------------
    # Same regeneration as the r1 trainer that made the checkpoint; a local
    # cache (mini-pilot convenience) stores the arrays keyed by (N, seed).
    if DATA_CACHE and os.path.exists(DATA_CACHE):
        dz = np.load(DATA_CACHE)
        assert int(dz["N"]) == N and int(dz["seed"]) == bc.SEED
        U, nu_tr = dz["U"], dz["nu"]
        sc.log(f"  data cache hit: {DATA_CACHE}  U{U.shape}")
    else:
        d = bc.build_data(N)
        U, nu_tr = np.asarray(d["U"], dtype=np.float64), np.asarray(d["nu"])
        if DATA_CACHE:
            np.savez(DATA_CACHE, U=U, nu=nu_tr, N=N, seed=bc.SEED)
    n_traj = U.shape[0]
    assert U.shape[1] == T
    S_flat = U.reshape(n_traj * T, -1)
    U_test, nu_test, worst_res_test = build_test_full(N, N_TEST, sc.log)
    report["data"] = dict(n_traj=int(n_traj), T=int(T),
                          fingerprint=bc.data_fingerprint(U),
                          max_fom_rel_residual_test=worst_res_test)
    save()

    # r1 pick reproduction: row i of Z_tr <-> global state pick[i]
    rng = np.random.default_rng(SEED0)
    n_states = n_traj * T
    max_snaps = int(cfg.get("max_snaps", 8192))
    if n_states > max_snaps:
        pick = np.sort(rng.choice(n_states, max_snaps, replace=False))
    else:
        pick = np.arange(n_states)
    assert len(pick) == len(Z_tr), f"pick {len(pick)} != Z_tr {len(Z_tr)}"
    row_of = {int(s): i for i, s in enumerate(pick)}

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)
    zbar = Z_tr.mean(0)

    G_int = dec.feat_at(coords[interior])                    # (n_i2, R) frozen
    kx, ky, Phi, lam, _ = bc.test_modes(N, EQ_M)
    Phi_np = np.asarray(Phi)
    del Phi
    lam_np = np.asarray(lam, dtype=np.float64)
    lam_j = jnp.asarray(lam_np, dtype=F64)
    Phi_j = jnp.asarray(Phi_np)
    A_j = Phi_j.T @ G_int                                    # (M, R) exact-lin
    kx_j = jnp.asarray(np.asarray(kx, dtype=np.float64))
    ky_j = jnp.asarray(np.asarray(ky, dtype=np.float64))

    # ---------------- trainable split (bank FROZEN) --------------------------
    hp0 = {k_: params[k_] for k_ in ("h", "h_lin") if k_ in params}
    if "hB" in params:
        hp0["hB"] = params["hB"]         # fixed random freqs, carried not trained

    def h_of(hp, z):
        return sc.head({**params, **hp}, z)

    # gate H: split identity
    zt = jnp.asarray(Z_tr[:4])
    gH = float(jnp.max(jnp.abs(h_of(hp0, zt) - h_fn0(zt))))
    report["gates"]["gateH"] = gH
    sc.log(f"  GATE H (param split identity): {gH:.2e}")
    assert gH == 0.0

    u_full_j = jax.jit(lambda Gb, z: Gb @ h_fn0(z))
    adv_full = jax.jit(lambda uf: bc.upwind_adv_field(uf, N))

    # ---------------- gate R: warm recon vs regenerated snapshots ------------
    r_rng = np.random.default_rng(SEED0 + 300)
    chk_rows = r_rng.choice(len(Z_tr), 128, replace=False)
    uh = np.asarray(h_of(hp0, jnp.asarray(Z_tr[chk_rows])) @ G_int.T)
    ut = S_flat[pick[chk_rows]][:, interior]
    rec0_chk = float(np.mean(np.linalg.norm(uh - ut, axis=1)
                             / np.linalg.norm(ut, axis=1)))
    report["gates"]["gateR_recon"] = rec0_chk
    sc.log(f"  GATE R (warm recon on regenerated snapshots): {rec0_chk:.3e}")
    assert rec0_chk < 5e-2, "code->snapshot mapping broken"

    # ---------------- baseline NNLS nodes (stage-1 adv-only) -----------------
    cand_pos = ctol_eq.candidate_pool(n_i2, cap=EQ_CAND_CAP)
    eq_pick = np.sort(np.random.default_rng(SEED0).choice(
        len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    u_full_int = lambda z: u_full_j(G_int, z)
    keep, w0_np, eq_info = xc.eq_fit_burgers_adv(
        u_full_int, adv_full, Phi_np, cand_pos, Z_eq, K, m_want,
        f"codesign N={N} k={K} M={EQ_M} m={m_want} adv-only", bc.nnls_capped)
    idx0 = interior[cand_pos[keep]]
    X0 = coords[idx0].astype(np.float64)                      # (m, 2)
    m = len(X0)
    report["eq_fit"] = {k_: v for k_, v in eq_info.items()
                        if isinstance(v, (int, float, str, bool, type(None)))}

    # ---------------- state selection ----------------------------------------
    # base codes with an in-pick predecessor (prev = code of gid-1); fit and
    # held-out disjoint; perturbed copies share the base state's prev and nu.
    cands = [i for i, s in enumerate(pick)
             if (s % T) >= 1 and int(s - 1) in row_of]
    sel = np.random.default_rng(SEED0 + 130)
    base_pick = sel.choice(len(cands), N_FIT_STATES + N_HELDOUT, replace=False)
    prng = np.random.default_rng(SEED0 + 131)

    def expand(rows, n_pert):
        zs, zps, nus, gids = [], [], [], []
        for ci in rows:
            i = cands[ci]
            s = int(pick[i])
            zp = Z_tr[row_of[s - 1]]
            nu = float(nu_tr[s // T])
            for p_ in range(1 + n_pert):
                z = Z_tr[i] if p_ == 0 else \
                    Z_tr[i] + PERT_SIGMA * prng.standard_normal(K)
                zs.append(z); zps.append(zp); nus.append(nu); gids.append(s)
        return (np.asarray(zs), np.asarray(zps), np.asarray(nus), gids)

    Zb, Zbp, nu_b, fit_gids = expand(base_pick[:N_FIT_STATES], PERT_PER)
    Zh, Zhp, nu_h, held_gids = expand(base_pick[N_FIT_STATES:], PERT_PER)
    S, Sh = len(Zb), len(Zh)
    WT_b = (1.0 + bc.DT * nu_b[:, None] * lam_np[None, :]) ** (-bc.WEAK_ALPHA)
    WT_h = (1.0 + bc.DT * nu_h[:, None] * lam_np[None, :]) ** (-bc.WEAK_ALPHA)
    report["config"]["n_fit_total"] = S
    report["config"]["n_held_total"] = Sh

    # anchors: rows (with fields) for L_rec/L_sob; held anchors for tripwire
    a_rng = np.random.default_rng(SEED0 + 132)
    a_all = a_rng.choice(len(Z_tr), N_ANCHOR + N_ANCHOR_HELD, replace=False)
    a_fit, a_held = a_all[:N_ANCHOR], a_all[N_ANCHOR:]
    Za = jnp.asarray(Z_tr[a_fit])
    Ua = jnp.asarray(S_flat[pick[a_fit]][:, interior])
    Zah = jnp.asarray(Z_tr[a_held])
    Uah = jnp.asarray(S_flat[pick[a_held]][:, interior])

    # ---------------- differentiable machinery -------------------------------
    # box widened by dx/16 beyond the interior ring: grid-edge nodes (exactly
    # at dx, present in the NNLS support at N=64) must be strictly inside the
    # sigmoid box or the logit clip shifts them ~1e-6 and gate C fails at
    # 8e-7 (measured).  Worst stencil excursion outside the domain is dx/16.
    LO, HI = float(dx - dx / 16.0), float(1.0 - dx + dx / 16.0)
    OFF = jnp.asarray(np.array([[0.0, 0.0], [dx, 0.0], [-dx, 0.0],
                                [0.0, dx], [0.0, -dx]]))

    def x_of(theta):
        return LO + (HI - LO) * jax.nn.sigmoid(theta)

    def theta_of(X):
        p = np.clip((X - LO) / (HI - LO), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    def phi_at(X):
        sx = jnp.sin(jnp.pi * kx_j[None, :] * X[:, 0:1])
        sy = jnp.sin(jnp.pi * ky_j[None, :] * X[:, 1:2])
        return (sx * sy) / ((N - 1) / 2.0)                    # (m, M)

    def g5_of(X):
        Xs = (X[:, None, :] + OFF[None, :, :]).reshape(-1, 2)
        return sc.features(params, Xs).reshape(-1, 5, R)      # frozen bank

    def adv_nodes(G5, Hb):
        U5 = jnp.einsum("mfr,sr->smf", G5, Hb)                # (S, m, 5)
        c, xp, xm, yp, ym = (U5[..., 0], U5[..., 1], U5[..., 2],
                             U5[..., 3], U5[..., 4])
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
        return c * (ux + uy)                                  # (S, m)

    def adv_full_of(hp, Hb):
        Uf = Hb @ G_int.T                                     # (S, n_i2)
        return jax.vmap(lambda uf: bc.upwind_adv_field(uf, N))(Uf)

    Zb_j, Zbp_j = jnp.asarray(Zb), jnp.asarray(Zbp)
    Zh_j, Zhp_j = jnp.asarray(Zh), jnp.asarray(Zhp)
    WTb_j, WTh_j = jnp.asarray(WT_b), jnp.asarray(WT_h)

    # interior->full-grid embed for the Sobolev term (boundary is exactly 0)
    def sob_pair(pred, truth):
        def dgrad(v):
            f = jnp.zeros((v.shape[0], N, N), dtype=F64)
            f = f.reshape(v.shape[0], -1).at[:, interior].set(v).reshape(
                v.shape[0], N, N)
            gx = (f[:, 2:, 1:-1] - f[:, :-2, 1:-1]) / (2 * dx)
            gy = (f[:, 1:-1, 2:] - f[:, 1:-1, :-2]) / (2 * dx)
            return gx, gy
        px, py = dgrad(pred)
        tx, ty = dgrad(truth)
        num = jnp.sum((px - tx) ** 2, (1, 2)) + jnp.sum((py - ty) ** 2, (1, 2))
        den = jnp.sum(tx ** 2, (1, 2)) + jnp.sum(ty ** 2, (1, 2))
        return jnp.mean(num / (den + 1e-300))

    def term_rec(hp, Zc, Uc):
        pred = h_of(hp, Zc) @ G_int.T
        return jnp.mean(jnp.sum((pred - Uc) ** 2, 1)
                        / (jnp.sum(Uc ** 2, 1) + 1e-300))

    def term_sob(hp, Zc, Uc):
        pred = h_of(hp, Zc) @ G_int.T
        return sob_pair(pred, Uc)

    # per-state normalization denominators are FROZEN at the warm start
    # (constants): the loss is then genuinely smooth (gate D compares FD vs
    # autodiff exactly), and the NNLS refit rows share the identical
    # quadratic form.  The 'shrink the advection term' cheat this reopens is
    # blocked by the L_rec anchor, and the tripwire watches it.
    def t_jac_of(hp, Zc):
        def t_one(z):
            u = G_int @ h_of(hp, z)
            return Phi_j.T @ bc.upwind_adv_field(u, N)
        return jax.vmap(jax.jacfwd(t_one))(Zc)                # (S, M, K)

    def dens_of(hp, Zc, WTc):
        Hb = h_of(hp, Zc)
        t = adv_full_of(hp, Hb) @ Phi_j
        den_s = jnp.sum((WTc * t) ** 2, 1) + 1e-300
        Jt = t_jac_of(hp, Zc)
        den_j = jnp.sum((WTc[:, :, None] * Jt) ** 2, (1, 2)) + 1e-300
        return np.asarray(den_s), np.asarray(den_j)

    def term_samp(hp, X, w, Zc, WTc, den):
        Hb = h_of(hp, Zc)
        a = (w[None, :] * adv_nodes(g5_of(X), Hb)) @ phi_at(X)   # (S, M)
        t = adv_full_of(hp, Hb) @ Phi_j                          # (S, M)
        d = WTc * (a - t)
        return jnp.mean(jnp.sum(d * d, 1) / den)

    def term_jac(hp, X, w, Zc, WTc, den):
        G5, PhiX = g5_of(X), phi_at(X)

        def a_one(z):
            Hb = h_of(hp, z[None])
            return ((w[None, :] * adv_nodes(G5, Hb)) @ PhiX)[0]

        Ja = jax.vmap(jax.jacfwd(a_one))(Zc)                  # (S, M, K)
        Jt = t_jac_of(hp, Zc)
        d = WTc[:, :, None] * (Ja - Jt)
        return jnp.mean(jnp.sum(d * d, (1, 2)) / den)

    def minsep(X):
        d2 = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, -1)
        d2 = d2 + jnp.eye(X.shape[0]) * 1e9
        return jnp.sum(jax.nn.relu(MINSEP_D - jnp.sqrt(d2)) ** 2)

    # ---------------- init values (self-calibration) -------------------------
    theta0 = jnp.asarray(theta_of(X0))
    w0_j = jnp.asarray(np.asarray(w0_np, dtype=np.float64))
    den_s_b, den_j_b = dens_of(hp0, Zb_j, WTb_j)
    den_s_h, den_j_h = dens_of(hp0, Zh_j, WTh_j)
    DSb, DJb = jnp.asarray(den_s_b), jnp.asarray(den_j_b)
    DSh, DJh = jnp.asarray(den_s_h), jnp.asarray(den_j_h)
    t_terms = dict(
        rec=float(term_rec(hp0, Za, Ua)),
        sob=float(term_sob(hp0, Za, Ua)),
        samp=float(term_samp(hp0, x_of(theta0), w0_j, Zb_j, WTb_j, DSb)),
        jac=float(term_jac(hp0, x_of(theta0), w0_j, Zb_j, WTb_j, DJb)))
    held0 = dict(
        rec=float(term_rec(hp0, Zah, Uah)),
        samp=float(term_samp(hp0, x_of(theta0), w0_j, Zh_j, WTh_j, DSh)),
        jac=float(term_jac(hp0, x_of(theta0), w0_j, Zh_j, WTh_j, DJh)))
    report["init_terms"] = dict(fit=t_terms, held=held0)
    sc.log("  init terms (fit): " + "  ".join(
        f"{k_}={v:.3e}" for k_, v in t_terms.items()))
    sc.log("  init terms (held): " + "  ".join(
        f"{k_}={v:.3e}" for k_, v in held0.items()))
    NORM = {k_: max(v, 1e-300) for k_, v in t_terms.items()}

    def loss_fn(tv, w):
        hp = tv.get("hp", hp0)
        theta = tv.get("theta", theta0)
        X = x_of(theta)
        L = REC_W * term_rec(hp, Za, Ua) / NORM["rec"]
        aux = {}
        if SOB_REL > 0:
            aux["sob"] = term_sob(hp, Za, Ua)
            L = L + SOB_REL * aux["sob"] / NORM["sob"]
        aux["samp"] = term_samp(hp, X, w, Zb_j, WTb_j, DSb)
        L = L + SAMP_REL * aux["samp"] / NORM["samp"]
        if JAC_REL > 0:
            aux["jac"] = term_jac(hp, X, w, Zb_j, WTb_j, DJb)
            L = L + JAC_REL * aux["jac"] / NORM["jac"]
        if TRAIN_NODES:
            L = L + MINSEP_W * minsep(X)
        return L, aux

    trainables = {}
    if TRAIN_H:
        trainables["hp"] = hp0
    if TRAIN_NODES:
        trainables["theta"] = theta0
    do_train = STEPS > 0 and trainables

    # ---------------- gate 0 + gate C (grid reference at the NNLS init) ------
    # grid closure copied from sep_eq_nodefit.build_variant/mk (incumbent form)
    pos0 = np.searchsorted(interior, idx0)
    assert np.all(interior[pos0] == idx0)
    Phi_q0 = jnp.asarray(Phi_np[pos0]) * w0_j[:, None]
    st0 = bc.stencil_indices(idx0, N)
    G_st0 = dec.feat_at(coords[st0.reshape(-1)]).reshape(m, 5, R)

    def mk_grid():
        def u_and_N_fast(z):
            us = jnp.einsum("msr,r->ms", G_st0, h_fn0(z))
            c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2], us[:, 3],
                                 us[:, 4])
            ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
            uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
            return c, c * (ux + uy)

        def prev_of_fast(z):
            return jnp.einsum("mr,r->m", G_st0[:, 0, :], h_fn0(z))

        def parts_s(z, prev_c, nu):
            w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
            u, Nu = u_and_N_fast(z)
            pu = Phi_q0.T @ u
            lin = Phi_q0.T @ (u - prev_c) + bc.DT * nu * lam_j * pu
            adv = Phi_q0.T @ Nu
            return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

        def r_w_fast(z, prev_c, nu):
            return parts_s(z, prev_c, nu)[0]

        def rJ_fast(z, prev_c, nu):
            return (r_w_fast(z, prev_c, nu),
                    jax.jacfwd(r_w_fast)(z, prev_c, nu), None)

        def rJ_parts(z, prev_c, nu):
            R_, lin, adv = parts_s(z, prev_c, nu)
            return R_, jax.jacfwd(r_w_fast)(z, prev_c, nu), lin, adv
        return r_w_fast, rJ_fast, prev_of_fast, rJ_parts
    (r_w_g, rJ_g, prev_g, rJp_g) = mk_grid()
    cl0 = dict(kind="grid", idx=idx0, w=np.asarray(w0_np))
    ops_ref = bc.make_weak_ops(dec, N, cl0, kind="weak", M=EQ_M, solver="lspg")
    rJ_g_j = jax.jit(rJ_g)
    grng = np.random.default_rng(SEED0 + 50)
    g0 = []
    for _ in range(5):
        zt_ = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                          + 0.05 * grng.standard_normal(K))
        zp_ = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
        prev_c = ops_ref["prev_of"](zp_)
        nu = float(np.median(nu_test))
        ra, Ja, _ = ops_ref["rJ"](zt_, prev_c, nu)
        rb, Jb, _ = rJ_g_j(zt_, prev_c, nu)
        g0.append(max(
            float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
            float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
    gate0 = float(np.max(g0))
    sc.log(f"  GATE 0 (grid closure vs make_weak_ops): {gate0:.2e}")
    assert gate0 < 1e-12
    report["gates"]["gate0"] = gate0

    # gate C: continuous advection at the grid init == grid closure advection
    rJp_g_j = jax.jit(rJp_g)
    a_cont = np.asarray(
        (w0_j[None, :] * adv_nodes(g5_of(x_of(theta0)), h_of(hp0, Zb_j[:3])))
        @ phi_at(x_of(theta0)))
    gC = []
    for si in range(3):
        wt_r = WT_b[si]
        _, _, _, adv_grid = rJp_g_j(Zb_j[si],
                                    jax.jit(prev_g)(Zbp_j[si]), nu_b[si])
        a_grid = np.asarray(adv_grid) / (wt_r * bc.DT)
        gC.append(float(np.max(np.abs(a_cont[si] - a_grid))
                        / (np.max(np.abs(a_grid)) + 1e-300)))
    gateC = float(np.max(gC))
    sc.log(f"  GATE C (continuous machinery at grid init): {gateC:.2e}")
    assert gateC < 1e-12
    report["gates"]["gateC"] = gateC

    # gate D: directional FD of the loss vs autodiff, at init
    if do_train:
        vgrad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        (L0, _), gr0 = vgrad(trainables, w0_j)
        d_rng = np.random.default_rng(SEED0 + 7)
        vdir = jax.tree_util.tree_map(
            lambda x: jnp.asarray(d_rng.standard_normal(x.shape)), trainables)
        nrm = np.sqrt(sum(float(jnp.sum(v * v))
                          for v in jax.tree_util.tree_leaves(vdir)))
        vdir = jax.tree_util.tree_map(lambda x: x / nrm, vdir)
        loss_j = jax.jit(lambda tvv: loss_fn(tvv, w0_j)[0])

        def fd_probe(vd, eps):
            lp = float(loss_j(jax.tree_util.tree_map(
                lambda a, b: a + eps * b, trainables, vd)))
            lm_ = float(loss_j(jax.tree_util.tree_map(
                lambda a, b: a - eps * b, trainables, vd)))
            return (lp - lm_) / (2 * eps)

        def ad_dot(vd):
            return sum(float(jnp.sum(a * b)) for a, b in zip(
                jax.tree_util.tree_leaves(gr0),
                jax.tree_util.tree_leaves(vd)))

        if os.environ.get("GATED_SWEEP"):
            for name_ in list(trainables):
                vd = jax.tree_util.tree_map(jnp.zeros_like, vdir)
                vd[name_] = vdir[name_]
                nn = np.sqrt(sum(float(jnp.sum(v * v))
                                 for v in jax.tree_util.tree_leaves(vd)))
                vd = jax.tree_util.tree_map(lambda x: x / nn, vd)
                for eps in (1e-4, 1e-5, 1e-6, 1e-7):
                    sc.log(f"    D-sweep [{name_}] eps={eps:.0e}  "
                           f"fd={fd_probe(vd, eps):.6e}  ad={ad_dot(vd):.6e}")
            raise SystemExit("GATED_SWEEP done")
        # L_jac contains the upwind JACOBIAN, which jumps ~1e-7 whenever a
        # stencil sign flips -> the loss is piecewise-smooth with micro-jumps
        # spaced ~1e-6 along generic hp rays (measured in the D-sweep).  The
        # gradient itself is exact between kinks: probe INSIDE a kink-free
        # window (small eps) and accept the best agreement.
        ad = ad_dot(vdir)
        gateD = min(abs(fd_probe(vdir, e_) - ad) / (abs(ad) + 1e-300)
                    for e_ in (1e-7, 3e-7, 1e-8))
        sc.log(f"  GATE D (FD vs autodiff, kink-free window): ad={ad:.6e} "
               f"rel={gateD:.2e}")
        assert gateD < 1e-4
        report["gates"]["gateD"] = gateD

    # ---------------- NNLS weight refit on the exact loss form ---------------
    def refit_w(tv):
        hp = tv.get("hp", hp0)
        theta = tv.get("theta", theta0)
        X = jnp.asarray(x_of(theta))
        PhiX = np.asarray(phi_at(X))                          # (m, M)
        Hb = h_of(hp, Zb_j)
        Nn = np.asarray(adv_nodes(g5_of(X), Hb))              # (S, m)
        t_np = np.asarray(adv_full_of(hp, Hb) @ Phi_j)        # (S, M)
        Gr, br = [], []
        s_samp = np.sqrt(SAMP_REL / (S * NORM["samp"]))
        for si in range(S):
            base = PhiX.T * Nn[si][None, :]                   # (M, m)
            sw = (s_samp / np.sqrt(den_s_b[si])) * WT_b[si][:, None]
            Gr.append(sw * base)
            br.append(sw[:, 0] * t_np[si])
        if JAC_REL > 0:
            def a_rows_one(z):
                Hb1 = h_of(hp, z[None])
                return adv_nodes(g5_of(X), Hb1)[0]            # (m,)
            Jn = np.asarray(jax.vmap(jax.jacfwd(a_rows_one))(Zb_j))  # (S,m,K)
            Jt = np.asarray(t_jac_of(hp, Zb_j))                      # (S,M,K)
            s_jac = np.sqrt(JAC_REL / (S * NORM["jac"]))
            for si in range(S):
                for k_ in range(K):
                    rows = PhiX.T * Jn[si][:, k_][None, :]    # (M, m)
                    sw = (s_jac / np.sqrt(den_j_b[si])) * WT_b[si][:, None]
                    Gr.append(sw * rows)
                    br.append(sw[:, 0] * Jt[si][:, k_])
        Gr = np.concatenate(Gr, axis=0)
        br = np.concatenate(br)
        rng_w = np.random.default_rng(SEED0 + 777)
        pad = np.abs(Nn).mean(0)
        keep_w, ww, _ = ss._nnls_rows(Gr, br, Gr.shape[1], bc.nnls_capped,
                                      rng_w, ctol_eq.EQ_ROWS, pad)
        w_full = np.zeros(Gr.shape[1])
        w_full[keep_w] = ww
        return jnp.asarray(w_full)

    # ---------------- training loop ------------------------------------------
    tv = trainables
    w_cur = w0_j
    tripped = False
    if do_train:
        import optax
        labels = {k_: ("theta" if k_ == "theta" else "hp") for k_ in tv}
        opt = optax.multi_transform(
            dict(hp=optax.adam(LR), theta=optax.adam(LR_NODES)), labels)
        opt_state = opt.init(tv)
        t_opt = time.time()
        best = None
        for it in range(STEPS + 1):
            if it % REFIT_EVERY == 0 and it > 0:
                w_cur = refit_w(tv)
            (L, aux), gr = vgrad(tv, w_cur)
            Lf = float(L)
            if best is None or Lf < best["loss"]:
                best = dict(loss=Lf, tv=jax.tree_util.tree_map(np.asarray, tv),
                            step=it)
            if it % EVAL_EVERY == 0 or it == STEPS:
                hp_c = tv.get("hp", hp0)
                th_c = tv.get("theta", theta0)
                ev = dict(step=it, loss=Lf,
                          **{k_: float(v) for k_, v in aux.items()},
                          held_rec=float(term_rec(hp_c, Zah, Uah)),
                          held_samp=float(term_samp(hp_c, x_of(th_c), w_cur,
                                                    Zh_j, WTh_j, DSh)),
                          held_jac=float(term_jac(hp_c, x_of(th_c), w_cur,
                                                  Zh_j, WTh_j, DJh)))
                drift = np.sqrt(ev["held_rec"]) / np.sqrt(held0["rec"]) - 1.0
                ev["held_rec_drift"] = float(drift)
                if drift > TRIP:
                    tripped = True
                    ev["tripwire"] = True
                report["hist"].append(ev)
                sc.log(f"  step {it:5d}  loss {Lf:.4e}  "
                       f"samp {aux['samp']:.3e}  "
                       f"held samp {ev['held_samp']:.3e} "
                       f"jac {ev['held_jac']:.3e} "
                       f"rec-drift {drift:+.4f}"
                       f"{'  TRIPWIRE' if drift > TRIP else ''} "
                       f"[{time.time()-t_opt:.0f}s]")
            if it == STEPS:
                break
            upd, opt_state = opt.update(gr, opt_state)
            tv = optax.apply_updates(tv, upd)
        tv = jax.tree_util.tree_map(jnp.asarray, best["tv"])
        report["best_step"] = best["step"]
        sc.log(f"  train DONE: best loss {best['loss']:.4e} at step "
               f"{best['step']} [{time.time()-t_opt:.0f}s]")
    report["tripwire_fired"] = tripped
    hp_fin = tv.get("hp", hp0)
    th_fin = tv.get("theta", theta0)
    w_fin = refit_w(tv) if do_train else w0_j
    X_fin = np.asarray(x_of(th_fin))
    report["node_stats"] = dict(
        mean_move=float(np.mean(np.linalg.norm(X_fin - X0, axis=1))),
        max_move=float(np.max(np.linalg.norm(X_fin - X0, axis=1))),
        w_nonzero=int(np.sum(np.asarray(w_fin) > 0)))
    save()

    # save drop-in checkpoint + nodes
    CK2 = f"{OUT_PREFIX}sep_codesign_{tag}.pkl"
    sc.save_pkl(CK2, {**params, **hp_fin}, Z_tr,
                {**cfg, "codesign": report["config"]})
    np.savez(OUT.replace(".json", "_nodes.npz"), X=X_fin,
             w=np.asarray(w_fin), X0=X0, w0=np.asarray(w0_np))

    # ================= certification (same instrument for base & final) ======
    def make_full_v(h_fn):
        def parts(Gb, Ph, z, prev_full, nu):
            w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
            u = Gb @ h_fn(z)
            Nu = bc.upwind_adv_field(u, N)
            pu = Ph.T @ u
            lin = Ph.T @ (u - prev_full) + bc.DT * nu * lam_j * pu
            adv = Ph.T @ Nu
            return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

        def r_f(Gb, Ph, z, prev_full, nu):
            return parts(Gb, Ph, z, prev_full, nu)[0]

        def rJ_f(Gb, Ph, z, prev_full, nu):
            R_, lin, adv = parts(Gb, Ph, z, prev_full, nu)
            J = jax.jacfwd(r_f, argnums=2)(Gb, Ph, z, prev_full, nu)
            return R_, J, lin, adv
        return dict(rJ=jax.jit(rJ_f), r=jax.jit(r_f))

    def mk_var(h_fn, X_v, w_v):
        G_st = np.asarray(g5_of(jnp.asarray(X_v)))
        G_st = jnp.asarray(G_st)
        Phi_q = jnp.asarray(np.asarray(phi_at(jnp.asarray(X_v)))) \
            * jnp.asarray(w_v)[:, None]

        def u_and_N_fast(z):
            us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
            c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2], us[:, 3],
                                 us[:, 4])
            ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
            uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
            return c, c * (ux + uy)

        def prev_of_ex(z):
            return A_j @ h_fn(z)

        def parts_ex(z, prev_m, nu):
            w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
            Ah = A_j @ h_fn(z)
            _, Nu = u_and_N_fast(z)
            lin = (Ah - prev_m) + bc.DT * nu * lam_j * Ah
            adv = Phi_q.T @ Nu
            return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

        def r_w_ex(z, prev_m, nu):
            return parts_ex(z, prev_m, nu)[0]

        def d_c_ex(z):
            return u_and_N_fast(z)[0]

        def rJ_ex(z, prev_m, nu):
            return (r_w_ex(z, prev_m, nu), jax.jacfwd(r_w_ex)(z, prev_m, nu),
                    Phi_q.T @ jax.jacfwd(d_c_ex)(z))

        def rJ_parts_ex(z, prev_m, nu):
            R_, lin, adv = parts_ex(z, prev_m, nu)
            return R_, jax.jacfwd(r_w_ex)(z, prev_m, nu), lin, adv

        def full_ex(z):
            return G_st[:, 0, :] @ h_fn(z)
        ops = bc._finish_ops(rJ_ex, r_w_ex, prev_of_ex, full_ex,
                             len(X_v), "lspg")
        ops["M"] = EQ_M
        ops["tol_scale"] = float(np.sqrt(n_i2))
        return ops, jax.jit(rJ_parts_ex), jax.jit(prev_of_ex)

    def certify(vname, hp_v, X_v, w_v):
        h_fn = jax.tree_util.Partial(lambda hpp, z: h_of(hpp, z), hp_v)
        u_full_v = jax.jit(lambda z: G_int @ h_fn(z))
        fo = make_full_v(h_fn)
        ops, rJp_j, prev_j = mk_var(h_fn, X_v, w_v)
        out = dict(m=int(len(X_v)))

        # gate L: exlin linear vs full-grid linear, this variant's h
        gL = []
        vrng = np.random.default_rng(SEED0 + 50)
        for _ in range(5):
            zt_ = jnp.asarray(Z_tr[vrng.integers(len(Z_tr))]
                              + 0.05 * vrng.standard_normal(K))
            zp_ = jnp.asarray(Z_tr[vrng.integers(len(Z_tr))])
            nu = float(np.median(nu_test))
            pf = u_full_v(zp_)
            _, _, lin_f, _ = fo["rJ"](G_int, Phi_j, zt_, pf, nu)
            _, _, lin_x, _ = rJp_j(zt_, prev_j(zp_), nu)
            gL.append(float(jnp.max(jnp.abs(lin_x - lin_f))
                            / (jnp.max(jnp.abs(lin_f)) + 1e-300)))
        out["gateL"] = float(np.max(gL))
        sc.log(f"  GATE L [{vname}]: {out['gateL']:.2e}")
        assert out["gateL"] < 1e-12

        # held-out ladder-lite: rungs b / c1 / c3 vs this variant's full ops
        recs = []
        for si in range(Sh):
            zj = Zh_j[si]
            pf = u_full_v(Zhp_j[si])
            Rs, Js, _, _ = [np.asarray(v) for v in
                            rJp_j(zj, prev_j(Zhp_j[si]), float(nu_h[si]))]
            Rf, Jf, _, _ = [np.asarray(v) for v in
                            fo["rJ"](G_int, Phi_j, zj, pf, float(nu_h[si]))]
            gs, gf = Js.T @ Rs, Jf.T @ Rf
            Hs, Hf = Js.T @ Js, Jf.T @ Jf
            lam_lm = 1e-6
            Ds = np.diag(np.diag(Hs)) + 1e-30 * np.eye(K)
            Df = np.diag(np.diag(Hf)) + 1e-30 * np.eye(K)
            dzs = np.linalg.solve(Hs + lam_lm * Ds, -gs)
            dzf = np.linalg.solve(Hf + lam_lm * Df, -gf)
            recs.append(dict(b=rel(Rs, Rf), c1=rel(gs, gf),
                             c1_cos=cosine(gs, gf), c3=rel(dzs, dzf),
                             c3_cos=cosine(dzs, dzf)))
        out["heldout"] = {k_: float(np.mean([r_[k_] for r_ in recs]))
                          for k_ in recs[0]}
        # held-out reconstruction vs truth (decoder quality)
        pred = np.asarray(h_of(hp_v, Zah) @ G_int.T)
        ut = np.asarray(Uah)
        out["held_recon_rel"] = float(np.mean(
            np.linalg.norm(pred - ut, axis=1) / np.linalg.norm(ut, axis=1)))

        # rollouts (r5 LM rule, copied from sep_eq_nodefit collect loop)
        oracle_v = ss.make_oracle_lm_banked(h_fn, K, budget=200)
        orng = np.random.default_rng(SEED0 + 11)

        def ic_fit(u0_int):
            tj = jnp.asarray(u0_int[None], dtype=F64)
            z0s = [np.tile(zbar[None], (1, 1))]
            for _ in range(8):
                z0s.append(Z_tr[orng.integers(len(Z_tr), size=1)])
            z, v = ss.oracle_multi_init_banked(oracle_v, G_int, z0s, tj)
            return np.asarray(z)[0], float(np.asarray(v)[0])

        rolls = []
        for ti in range(N_TEST):
            nu = float(nu_test[ti])
            u0 = U_test[ti, 0]
            z0, v0 = ic_fit(u0[interior])
            u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
            tol_abs = STEP_TOL * u_scale * float(np.sqrt(n_i2))
            zs = [z0]
            errs = []
            uf = np.asarray(u_full_v(jnp.asarray(z0)))
            errs.append(float(np.linalg.norm(uf - u0[interior])
                              / np.linalg.norm(u0[interior])))
            t1 = time.time()
            for t in range(1, T):
                z_prev = zs[-1]
                z_init = z_prev if len(zs) < 2 or EXTRAP == 0 else \
                    z_prev + EXTRAP * (zs[-1] - zs[-2])
                prev_c = prev_j(jnp.asarray(z_prev, dtype=F64))
                z = jnp.asarray(z_init, dtype=F64)
                r, J, _ = ops["rJ"](z, prev_c, nu)
                rn = float(jnp.linalg.norm(r))
                lam_lm = 1e-6
                for attempt in range(1, bc.GN_BUDGET + 1):
                    H = J.T @ J
                    g = J.T @ r
                    D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
                    dz = jnp.linalg.solve(H + lam_lm * D, -g)
                    if not bool(jnp.all(jnp.isfinite(dz))):
                        lam_lm = min(lam_lm * 10.0, 1e12)
                        if lam_lm >= 1e12:
                            break
                        continue
                    ndz = float(jnp.linalg.norm(dz))
                    if ndz > bc.TR_DELTA:
                        lam_lm = min(lam_lm * 10.0, 1e12)
                        if lam_lm >= 1e12:
                            break
                        continue
                    if ndz <= 1e-12 * (1.0 + float(jnp.linalg.norm(z))):
                        break
                    z_new = z + dz
                    rn_new = float(ops["rn"](z_new, prev_c, nu))
                    if np.isfinite(rn_new) and rn_new < rn:
                        rel_dec = (rn - rn_new) / rn
                        z, rn = z_new, rn_new
                        if rn <= tol_abs:
                            break
                        r, J, _ = ops["rJ"](z, prev_c, nu)
                        lam_lm = max(lam_lm / 3.0, 1e-12)
                        if rel_dec < STALL:
                            break
                    else:
                        lam_lm = min(lam_lm * 10.0, 1e12)
                        if lam_lm >= 1e12:
                            break
                zs.append(np.asarray(z))
                uf = np.asarray(u_full_v(jnp.asarray(zs[-1])))
                ut_ = U_test[ti, t][interior]
                errs.append(float(np.linalg.norm(uf - ut_)
                                  / np.linalg.norm(ut_)))
            rolls.append(dict(traj=ti, nu=nu, ic_resid=v0,
                              err_mean=float(np.mean(errs)),
                              err_t0=errs[0], err_t1=errs[1],
                              err_last=errs[-1], secs=time.time() - t1))
            sc.log(f"  [{vname}] traj {ti}: rollout err {np.mean(errs):.3e} "
                   f"(t0 {errs[0]:.2e}, t50 {errs[-1]:.2e}) "
                   f"[{time.time()-t1:.0f}s]")
        out["rollout"] = rolls
        out["rollout_err_mean"] = float(np.mean([r_["err_mean"]
                                                 for r_ in rolls]))
        ho = out["heldout"]
        sc.log(f"  [{vname}] SUMMARY m={out['m']}: "
               f"held b {ho['b']:.3e} c1 {ho['c1']:.3e} "
               f"cos {ho['c1_cos']:.4f} c3 {ho['c3']:.3e} "
               f"cos {ho['c3_cos']:.4f}  recon {out['held_recon_rel']:.3e}  "
               f"rollout {out['rollout_err_mean']:.3e}")
        report["variants"][vname] = out
        save()

    certify("base", hp0, X0, np.asarray(w0_np, dtype=np.float64))
    if do_train:
        certify("cot", hp_fin, X_fin, np.asarray(w_fin, dtype=np.float64))

    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    sc.log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
