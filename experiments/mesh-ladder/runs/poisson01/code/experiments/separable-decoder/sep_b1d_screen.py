"""1D Burgers frozen-decoder node-learning screening (2026-08-27).

One process = one resolution N.  Everything regenerates from seeds in-job:

  1. data: 1D viscous Burgers family (b1d_common), N_TRAIN train + N_TEST
     fresh test trajectories, FOM residual gate <= 1e-8.
  2. decoder: separable u(x;z) = bc(x)<g(x),h(z)> trained normally
     (auto-decoder, TRAIN_STEPS), then FROZEN COMPLETELY -- no decoder
     parameter is touched by anything downstream.
  3. weak form: M lowest sine modes (exact 1D Laplacian eigenvectors), ALL
     linear terms exact via A = Phi^T G (exlin rule); only advection sampled.
  4. four arms, all certified with the same instrument:
       oracle      -- full-grid advection (no sampling error at all)
       base_tight  -- NNLS grid nodes at m = M (tight budget)
       nodes_tight -- learned continuous node positions at m = M (arm-n
                      recipe: sampled-advection + sampled-gradient agreement
                      losses, NNLS weight re-solve every REFIT_EVERY steps,
                      sigmoid box + min-separation)
       base_gen    -- NNLS grid nodes at m = min(4M, n_interior)
  5. certification: held-out ladder rungs (b, c1, c3) vs the full-grid ops,
     held-out reconstruction, full 50-step rollouts on the fresh test
     trajectories with the r5 LM rule; accuracy and cost from the SAME timed
     invocation, GPU burned in, every timing repetition persisted.

Gates: E (eigen identity), F (full-grid residual identity), C (continuous
node machinery == grid ops at grid init), D (FD vs autodiff of the node
loss in the kink-free window), plus the data residual gate in build_data_1d.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

import b1d_common as b1

import jax
import jax.numpy as jnp

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
K = int(os.environ.get("K", "8"))
R = int(os.environ.get("R", "32"))
M = int(os.environ.get("M", "32"))
M_GEN = int(os.environ.get("M_GEN", "0"))         # 0 -> min(4M, n_interior)
N_TRAIN = int(os.environ.get("N_TRAIN", "512"))
N_TEST = int(os.environ.get("N_TEST", "8"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "40000"))
SEED0 = int(os.environ.get("SEED0", "0"))
DATA_SEED = int(os.environ.get("DATA_SEED", "0"))
TEST_SEED = int(os.environ.get("TEST_SEED", "1"))

NODE_STEPS = int(os.environ.get("NODE_STEPS", "2000"))
LR_NODES = float(os.environ.get("LR_NODES", "3e-3"))
REFIT_EVERY = int(os.environ.get("REFIT_EVERY", "500"))
REFIT_JAC_STATES = int(os.environ.get("REFIT_JAC_STATES", "16"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "200"))
SAMP_REL = float(os.environ.get("SAMP_REL", "1.0"))
JAC_REL = float(os.environ.get("JAC_REL", "1.0"))

N_FIT_STATES = int(os.environ.get("N_FIT_STATES", "64"))
N_HELDOUT = int(os.environ.get("N_HELDOUT", "32"))
PERT_PER = int(os.environ.get("PERT_PER", "1"))
PERT_SIGMA = float(os.environ.get("PERT_SIGMA", "0.05"))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))

TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
STALL = float(os.environ.get("STALL", "1e-3"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
GN_BUDGET = int(os.environ.get("GN_BUDGET", "30"))
TIME_REPS = int(os.environ.get("TIME_REPS", "3"))

CKPT_CACHE = os.environ.get("CKPT_CACHE", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
OUT_TAG = os.environ.get("OUT_TAG", "")

log = b1.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B1D-SCREEN N={N} K={K} "
        f"R={R} M={M} node_steps={NODE_STEPS}")
    t_all = time.time()
    dx = 1.0 / (N - 1)
    interior = b1.interior_indices_1d(N)
    n_i = interior.size
    coords_int = b1.grid_coords_1d(N)[interior]               # (n_i, 1)
    T = b1.NUM_STEPS + 1
    m_tight = M
    m_half = max(K, M // 2)
    m_gen = M_GEN if M_GEN > 0 else min(4 * M, n_i)
    tag = OUT_TAG or f"N{N}_K{K}_R{R}_M{M}"
    OUT = f"{OUT_PREFIX}sep_b1d_{tag}.json"

    report = dict(config=dict(
        pde="burgers1d", kind="b1d_screen", N=N, k=K, r=R, M=M,
        m_tight=m_tight, m_half=m_half, m_gen=m_gen, n_interior=int(n_i),
        n_train=N_TRAIN, n_test=N_TEST, max_snaps=MAX_SNAPS,
        train_steps=TRAIN_STEPS, node_steps=NODE_STEPS, lr_nodes=LR_NODES,
        refit_every=REFIT_EVERY, refit_jac_states=REFIT_JAC_STATES,
        samp_rel=SAMP_REL, jac_rel=JAC_REL, pert_sigma=PERT_SIGMA,
        n_fit_states=N_FIT_STATES, n_heldout=N_HELDOUT, eq_snaps=EQ_SNAPS,
        dt=b1.DT, num_steps=b1.NUM_STEPS, weak_alpha=b1.WEAK_ALPHA,
        newton_iters=b1.NEWTON_ITERS, seed=SEED0, data_seed=DATA_SEED,
        test_seed=TEST_SEED, tr_factor=TR_FACTOR, step_tol=STEP_TOL,
        stall=STALL, extrap=EXTRAP, gn_budget=GN_BUDGET,
        time_reps=TIME_REPS, x64=True,
        matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        jax_version=jax.__version__,
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        gates={}, hist=[], variants={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ---------------- data ---------------------------------------------------
    data = b1.build_data_1d(N, N_TRAIN, N_TEST, DATA_SEED, TEST_SEED)
    U, nu_tr = data["train"]["U"], data["train"]["nu"]
    U_test, nu_test = data["test"]["U"], data["test"]["nu"]
    report["data"] = dict(
        train_worst_res=data["train"]["worst_res"],
        test_worst_res=data["test"]["worst_res"],
        train_sum=float(np.sum(U)), train_sumsq=float(np.sum(U * U)))
    S_flat = U.reshape(N_TRAIN * T, -1)
    n_states = N_TRAIN * T
    rng = np.random.default_rng(SEED0)
    if n_states > MAX_SNAPS:
        pick = np.sort(rng.choice(n_states, MAX_SNAPS, replace=False))
    else:
        pick = np.arange(n_states)
    save()

    # ---------------- decoder train (then FROZEN) ----------------------------
    ck_name = f"{OUT_PREFIX}sep_b1d_{tag}.pkl"
    if CKPT_CACHE and os.path.exists(CKPT_CACHE):
        params, Z_tr, ck_cfg = b1.load_pkl(CKPT_CACHE)
        assert int(ck_cfg["N"]) == N and int(ck_cfg["k"]) == K
        report["train"] = dict(cache=CKPT_CACHE, **ck_cfg.get("train", {}))
        log(f"  ckpt cache hit: {CKPT_CACHE}")
    else:
        U_tr = S_flat[pick][:, interior]
        params, Z_tr, tinfo = b1.train_autodecoder_1d(
            jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
            steps=TRAIN_STEPS, tag=f"N{N}")
        report["train"] = tinfo
        b1.save_pkl(ck_name, params, Z_tr,
                    dict(N=N, k=K, r=R, max_snaps=MAX_SNAPS, seed=SEED0,
                         data_seed=DATA_SEED, train=tinfo))
    row_of = {int(s): i for i, s in enumerate(pick)}
    save()

    h_fn = jax.jit(lambda z: b1.head(params, z))
    G_int = jnp.asarray(b1.features(params, jnp.asarray(coords_int)))
    kx, Phi_np, lam_np = b1.test_modes_1d(N, M)
    Phi_j = jnp.asarray(Phi_np)
    lam_j = jnp.asarray(lam_np, dtype=F64)
    kx_f = jnp.asarray(kx, dtype=F64)
    A_j = Phi_j.T @ G_int                                     # (M, R) exact
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    TR_DELTA = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(TR_DELTA)
    zbar = Z_tr.mean(0)

    def upwind(u):
        return b1.upwind_adv_field_1d(u, N)

    # ---------------- gate E: eigen identity ---------------------------------
    grng = np.random.default_rng(SEED0 + 50)
    gE = []
    for _ in range(4):
        z = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                        + 0.05 * grng.standard_normal(K))
        u = G_int @ h_fn(z)
        Upad = jnp.pad(u, 1)
        lap = (Upad[2:] - 2 * Upad[1:-1] + Upad[:-2]) / dx ** 2
        gE.append(rel(np.asarray(Phi_j.T @ (-lap)),
                      np.asarray(lam_j * (Phi_j.T @ u))))
    report["gates"]["gateE"] = float(np.max(gE))
    log(f"  GATE E (sine modes are exact eigenvectors): {np.max(gE):.2e}")
    assert np.max(gE) < 1e-10

    # ---------------- full-grid (oracle) weak ops ----------------------------
    def make_full_ops():
        def r_w(z, prev_m, nu):
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            hz = h_fn(z)
            Ah = A_j @ hz
            Nu = upwind(G_int @ hz)
            lin = (Ah - prev_m) + b1.DT * nu * lam_j * Ah
            return wt * (lin + b1.DT * (Phi_j.T @ Nu))

        def prev_of(z):
            return A_j @ h_fn(z)
        return dict(rJ=jax.jit(lambda z, p, nu: (
                        r_w(z, p, nu), jax.jacfwd(r_w)(z, p, nu))),
                    rn=jax.jit(lambda z, p, nu: jnp.linalg.norm(
                        r_w(z, p, nu))),
                    prev_of=jax.jit(prev_of), m=int(n_i))

    full_ops = make_full_ops()

    # gate F: full ops vs direct projection of the FOM interior residual form
    gF = []
    for _ in range(4):
        z = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                        + 0.05 * grng.standard_normal(K))
        zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
        nu = float(np.median(nu_tr))
        wt = (1.0 + b1.DT * nu * lam_np) ** (-b1.WEAK_ALPHA)
        u = np.asarray(G_int @ h_fn(z))
        up = np.asarray(G_int @ h_fn(zp))
        r_direct = wt * np.asarray(
            Phi_j.T @ jnp.asarray(
                np.asarray(b1.fom_residual_int(jnp.asarray(u), jnp.asarray(up),
                                               nu, N))))
        r_ops, _ = full_ops["rJ"](z, full_ops["prev_of"](zp), nu)
        gF.append(rel(np.asarray(r_ops), r_direct))
    report["gates"]["gateF"] = float(np.max(gF))
    log(f"  GATE F (weak ops == Phi^T FOM residual): {np.max(gF):.2e}")
    assert np.max(gF) < 1e-10

    # ---------------- state selection (fit / held-out, disjoint) -------------
    cands = [i for i, s in enumerate(pick)
             if (s % T) >= 1 and int(s - 1) in row_of]
    sel = np.random.default_rng(SEED0 + 130)
    base_pick = sel.choice(len(cands), N_FIT_STATES + N_HELDOUT, replace=False)
    prng = np.random.default_rng(SEED0 + 131)

    def expand(rows, n_pert):
        zs, zps, nus = [], [], []
        for ci in rows:
            i = cands[ci]
            s = int(pick[i])
            zp = Z_tr[row_of[s - 1]]
            nu = float(nu_tr[s // T])
            for p_ in range(1 + n_pert):
                z = Z_tr[i] if p_ == 0 else \
                    Z_tr[i] + PERT_SIGMA * prng.standard_normal(K)
                zs.append(z)
                zps.append(zp)
                nus.append(nu)
        return np.asarray(zs), np.asarray(zps), np.asarray(nus)

    Zb, Zbp, nu_b = expand(base_pick[:N_FIT_STATES], PERT_PER)
    Zh, Zhp, nu_h = expand(base_pick[N_FIT_STATES:], PERT_PER)
    S, Sh = len(Zb), len(Zh)
    WT_b = (1.0 + b1.DT * nu_b[:, None] * lam_np[None, :]) ** (-b1.WEAK_ALPHA)
    WT_h = (1.0 + b1.DT * nu_h[:, None] * lam_np[None, :]) ** (-b1.WEAK_ALPHA)
    Zb_j, Zh_j = jnp.asarray(Zb), jnp.asarray(Zh)
    Zhp_j = jnp.asarray(Zhp)
    WTb_j, WTh_j = jnp.asarray(WT_b), jnp.asarray(WT_h)
    report["config"]["n_fit_total"] = int(S)
    report["config"]["n_held_total"] = int(Sh)

    # held-out reconstruction reference (decoder frozen -> same for all arms)
    hrng = np.random.default_rng(SEED0 + 132)
    a_held = hrng.choice(len(Z_tr), 64, replace=False)
    pred = np.asarray(h_fn(jnp.asarray(Z_tr[a_held])) @ G_int.T)
    ut = S_flat[pick[a_held]][:, interior]
    held_recon = float(np.mean(np.linalg.norm(pred - ut, axis=1)
                               / np.linalg.norm(ut, axis=1)))
    report["held_recon_rel"] = held_recon
    log(f"  held-out reconstruction (frozen decoder): {held_recon:.3e}")

    # frozen teacher quantities at fit/held states (decoder frozen -> const)
    def teacher(Zc):
        Hb = h_fn(jnp.asarray(Zc))
        Uf = Hb @ G_int.T
        t = jax.vmap(upwind)(Uf) @ Phi_j                      # (S, M)

        def t_one(z):
            return Phi_j.T @ upwind(G_int @ h_fn(z))
        Jt = jax.vmap(jax.jacfwd(t_one))(jnp.asarray(Zc))     # (S, M, K)
        return np.asarray(Hb), np.asarray(t), np.asarray(Jt)

    Hb_fit, t_fit, Jt_fit = teacher(Zb)
    Hb_hld, t_hld, Jt_hld = teacher(Zh)
    den_s_b = np.sum((WT_b * t_fit) ** 2, 1) + 1e-300
    den_j_b = np.sum((WT_b[:, :, None] * Jt_fit) ** 2, (1, 2)) + 1e-300
    den_s_h = np.sum((WT_h * t_hld) ** 2, 1) + 1e-300
    den_j_h = np.sum((WT_h[:, :, None] * Jt_hld) ** 2, (1, 2)) + 1e-300

    # ---------------- node machinery (continuous, 1D) ------------------------
    LO, HI = float(dx - dx / 16.0), float(1.0 - dx + dx / 16.0)
    OFF = jnp.asarray(np.array([0.0, dx, -dx]))
    Hb_fit_j = jnp.asarray(Hb_fit)
    t_fit_j = jnp.asarray(t_fit)
    Jt_fit_j = jnp.asarray(Jt_fit)
    DSb, DJb = jnp.asarray(den_s_b), jnp.asarray(den_j_b)
    MINSEP_D = 0.5 * dx
    MINSEP_W = 1.0 / (dx * dx)

    def x_of(theta):
        return LO + (HI - LO) * jax.nn.sigmoid(theta)

    def theta_of(X):
        p = np.clip((X - LO) / (HI - LO), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    def phi_at(X):
        nrm = np.sqrt((N - 1) / 2.0)
        return jnp.sin(jnp.pi * kx_f[None, :] * X[:, None]) / nrm  # (m, M)

    def g3_of(X):
        Xs = (X[:, None] + OFF[None, :]).reshape(-1, 1)
        return b1.features(params, Xs).reshape(-1, 3, R)      # frozen bank

    def adv_nodes(G3, Hb):
        U3 = jnp.einsum("mfr,sr->smf", G3, Hb)                # (S, m, 3)
        c, xp, xm = U3[..., 0], U3[..., 1], U3[..., 2]
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        return c * ux                                         # (S, m)

    def term_samp(X, w, Hb, WTc, tt, den):
        a = (w[None, :] * adv_nodes(g3_of(X), Hb)) @ phi_at(X)
        d = WTc * (a - tt)
        return jnp.mean(jnp.sum(d * d, 1) / den)

    def term_jac(X, w, Zc, WTc, Jt, den):
        G3, PhiX = g3_of(X), phi_at(X)

        def a_one(z):
            Hb1 = h_fn(z[None])
            return ((w[None, :] * adv_nodes(G3, Hb1)) @ PhiX)[0]

        Ja = jax.vmap(jax.jacfwd(a_one))(Zc)                  # (S, M, K)
        d = WTc[:, :, None] * (Ja - Jt)
        return jnp.mean(jnp.sum(d * d, (1, 2)) / den)

    def minsep(X):
        d = jnp.abs(X[:, None] - X[None, :]) + jnp.eye(X.shape[0]) * 1e9
        return jnp.sum(jax.nn.relu(MINSEP_D - d) ** 2)

    # ---------------- NNLS baseline fits -------------------------------------
    u_full_of = jax.jit(lambda z: G_int @ h_fn(z))
    cand_pos = np.arange(n_i)                                 # all interior
    eq_pick = np.sort(np.random.default_rng(SEED0).choice(
        len(Z_tr), min(EQ_SNAPS, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]

    fits = {}
    for name, m_want in (("tight", m_tight), ("half", m_half), ("gen", m_gen)):
        keep, wq, info = b1.eq_fit_adv_1d(
            u_full_of, Phi_np, cand_pos, Z_eq, m_want, N,
            f"b1d N={N} M={M} m={m_want} adv-only")
        X_nodes = coords_int[cand_pos[keep], 0].astype(np.float64)
        fits[name] = dict(X=X_nodes, w=np.asarray(wq, dtype=np.float64),
                          info=info, pos=cand_pos[keep])
        report[f"eq_fit_{name}"] = info
    save()

    X0 = fits["tight"]["X"]
    w0 = fits["tight"]["w"]

    # gate C: continuous machinery at the grid init == indexed grid advection
    a_cont = np.asarray(adv_nodes(g3_of(jnp.asarray(X0)),
                                  jnp.asarray(Hb_fit[:3])))
    Uf3 = Hb_fit[:3] @ np.asarray(G_int).T
    N_full3 = np.stack([np.asarray(upwind(jnp.asarray(Uf3[s])))
                        for s in range(3)])
    a_grid = N_full3[:, fits["tight"]["pos"]]
    gateC = float(np.max(np.abs(a_cont - a_grid))
                  / (np.max(np.abs(a_grid)) + 1e-300))
    report["gates"]["gateC"] = gateC
    log(f"  GATE C (continuous machinery at grid init): {gateC:.2e}")
    assert gateC < 1e-12

    # ---------------- node training (theta only; decoder frozen) -------------
    import optax

    def learn_nodes(X_init, w_init, tag_n, do_gate_d=False):
        """Arm-n node learning at one budget: only the m continuous positions
        train; weights re-solved by NNLS on the exact loss-form rows every
        REFIT_EVERY steps (variable projection).  Decoder and teacher frozen
        throughout (anti-collusion by construction)."""
        norm_s = max(float(term_samp(jnp.asarray(X_init), jnp.asarray(w_init),
                                     Hb_fit_j, WTb_j, t_fit_j, DSb)), 1e-300)
        norm_j = max(float(term_jac(jnp.asarray(X_init), jnp.asarray(w_init),
                                    Zb_j, WTb_j, Jt_fit_j, DJb)), 1e-300)
        report.setdefault("init_terms", {})[tag_n] = dict(samp=norm_s,
                                                          jac=norm_j)
        log(f"  [{tag_n}] init terms: samp={norm_s:.3e} jac={norm_j:.3e}")

        def loss_fn(theta, w):
            X = x_of(theta)
            samp = term_samp(X, w, Hb_fit_j, WTb_j, t_fit_j, DSb)
            L = SAMP_REL * samp / norm_s
            jac = term_jac(X, w, Zb_j, WTb_j, Jt_fit_j, DJb)
            if JAC_REL > 0:
                L = L + JAC_REL * jac / norm_j
            L = L + MINSEP_W * minsep(X)
            return L, dict(samp=samp, jac=jac)

        vgrad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        theta0 = jnp.asarray(theta_of(X_init))
        w0_j = jnp.asarray(w_init)

        if do_gate_d:
            (_, _), gr0 = vgrad(theta0, w0_j)
            d_rng = np.random.default_rng(SEED0 + 7)
            vdir = jnp.asarray(d_rng.standard_normal(theta0.shape))
            vdir = vdir / jnp.linalg.norm(vdir)
            loss_j = jax.jit(lambda th: loss_fn(th, w0_j)[0])
            ad = float(jnp.sum(gr0 * vdir))

            def fd_probe(eps):
                return (float(loss_j(theta0 + eps * vdir))
                        - float(loss_j(theta0 - eps * vdir))) / (2 * eps)

            gateD = min(abs(fd_probe(e) - ad) / (abs(ad) + 1e-300)
                        for e in (1e-7, 3e-7, 1e-8))
            report["gates"]["gateD"] = gateD
            log(f"  GATE D (FD vs autodiff, kink-free window): ad={ad:.6e} "
                f"rel={gateD:.2e}")
            assert gateD < 1e-4

        def refit_w(theta):
            X = jnp.asarray(x_of(theta))
            PhiX = np.asarray(phi_at(X))                      # (m, M)
            Nn = np.asarray(adv_nodes(g3_of(X), Hb_fit_j))    # (S, m)
            Gr, br = [], []
            s_samp = np.sqrt(SAMP_REL / (S * norm_s))
            for si in range(S):
                base = PhiX.T * Nn[si][None, :]               # (M, m)
                sw = (s_samp / np.sqrt(den_s_b[si])) * WT_b[si][:, None]
                Gr.append(sw * base)
                br.append(sw[:, 0] * t_fit[si])
            if JAC_REL > 0:
                G3 = g3_of(X)

                def a_rows_one(z):
                    return adv_nodes(G3, h_fn(z[None]))[0]    # (m,)

                j_states = np.random.default_rng(SEED0 + 901).choice(
                    S, min(REFIT_JAC_STATES, S), replace=False)
                Jn = np.asarray(jax.vmap(jax.jacfwd(a_rows_one))(
                    jnp.asarray(Zb[j_states])))               # (Sj, m, K)
                s_jac = np.sqrt(JAC_REL / (S * norm_j))
                for jj, si in enumerate(j_states):
                    for k_ in range(K):
                        rows = PhiX.T * Jn[jj][:, k_][None, :]
                        sw = (s_jac / np.sqrt(den_j_b[si])) \
                            * WT_b[si][:, None]
                        Gr.append(sw * rows)
                        br.append(sw[:, 0] * Jt_fit[si][:, k_])
            Gr = np.concatenate(Gr, axis=0)
            br = np.concatenate(br)
            m_ = Gr.shape[1]
            pad_score = np.abs(Nn).mean(0)
            ww, _, _ = b1.nnls_capped(Gr, br, max_support=m_)
            supp = np.nonzero(ww > 0)[0]
            if len(supp) < m_:
                rest = np.setdiff1d(np.arange(m_), supp)
                ww[rest] = 1e-8 * max(ww.max(), 1e-300) * \
                    (pad_score[rest] / (pad_score.max() + 1e-300) + 1e-6)
            return jnp.asarray(ww)

        theta = theta0
        w_cur = w0_j
        if NODE_STEPS > 0:
            opt = optax.adam(LR_NODES)
            opt_state = opt.init(theta)
            best = None
            t_opt = time.time()
            for it in range(NODE_STEPS + 1):
                if it % REFIT_EVERY == 0 and it > 0:
                    w_cur = refit_w(theta)
                (L, aux), gr = vgrad(theta, w_cur)
                Lf = float(L)
                if best is None or Lf < best["loss"]:
                    best = dict(loss=Lf, theta=np.asarray(theta), step=it)
                if it % EVAL_EVERY == 0 or it == NODE_STEPS:
                    hs = float(term_samp(x_of(theta), w_cur,
                                         jnp.asarray(Hb_hld), WTh_j,
                                         jnp.asarray(t_hld),
                                         jnp.asarray(den_s_h)))
                    hj = float(term_jac(x_of(theta), w_cur, Zh_j, WTh_j,
                                        jnp.asarray(Jt_hld),
                                        jnp.asarray(den_j_h)))
                    report["hist"].append(dict(
                        arm=tag_n, step=it, loss=Lf, samp=float(aux["samp"]),
                        jac=float(aux["jac"]), held_samp=hs, held_jac=hj))
                    log(f"  [{tag_n}] step {it:5d}  loss {Lf:.4e}  samp "
                        f"{float(aux['samp']):.3e}  held samp {hs:.3e}  "
                        f"held jac {hj:.3e}  [{time.time()-t_opt:.0f}s]")
                if it == NODE_STEPS:
                    break
                upd, opt_state = opt.update(gr, opt_state)
                theta = optax.apply_updates(theta, upd)
            theta = jnp.asarray(best["theta"])
            log(f"  [{tag_n}] node train DONE: best loss {best['loss']:.4e} "
                f"at step {best['step']} [{time.time()-t_opt:.0f}s]")
        w_fin = np.asarray(refit_w(theta) if NODE_STEPS > 0 else w0_j)
        X_fin = np.asarray(x_of(theta))
        report[f"node_stats_{tag_n}"] = dict(
            mean_move=float(np.mean(np.abs(X_fin - X_init))),
            max_move=float(np.max(np.abs(X_fin - X_init))),
            w_nonzero=int(np.sum(w_fin > 0)))
        save()
        return X_fin, w_fin

    X_nt, w_nt = learn_nodes(X0, w0, "tight", do_gate_d=True)
    X_nh, w_nh = learn_nodes(fits["half"]["X"], fits["half"]["w"], "half")
    np.savez(OUT.replace(".json", "_nodes.npz"),
             X_tight=X_nt, w_tight=w_nt, X0_tight=X0, w0_tight=w0,
             X_half=X_nh, w_half=w_nh,
             X0_half=fits["half"]["X"], w0_half=fits["half"]["w"])
    save()

    # ---------------- certification (same instrument for every arm) ----------
    def make_sampled_ops(X_v, w_v):
        G3 = g3_of(jnp.asarray(X_v))
        Phi_q = phi_at(jnp.asarray(X_v)) * jnp.asarray(w_v)[:, None]

        def r_w(z, prev_m, nu):
            wt = (1.0 + b1.DT * nu * lam_j) ** (-b1.WEAK_ALPHA)
            hz = h_fn(z)
            Ah = A_j @ hz
            Nu = adv_nodes(G3, hz[None])[0]
            lin = (Ah - prev_m) + b1.DT * nu * lam_j * Ah
            return wt * (lin + b1.DT * (Phi_q.T @ Nu))

        def prev_of(z):
            return A_j @ h_fn(z)
        return dict(rJ=jax.jit(lambda z, p, nu: (
                        r_w(z, p, nu), jax.jacfwd(r_w)(z, p, nu))),
                    rn=jax.jit(lambda z, p, nu: jnp.linalg.norm(
                        r_w(z, p, nu))),
                    prev_of=jax.jit(prev_of), m=int(len(X_v)))

    def banked_ic_fit(u0_int):
        tgt = jnp.asarray(u0_int, dtype=F64)
        tn = float(np.linalg.norm(u0_int))

        def r_of(z):
            return (G_int @ h_fn(z) - tgt) / tn
        rJ = jax.jit(lambda z: (r_of(z), jax.jacfwd(r_of)(z)))
        orng = np.random.default_rng(SEED0 + 11)
        z0s = [zbar] + [Z_tr[orng.integers(len(Z_tr))] for _ in range(8)]
        best = None
        for z0 in z0s:
            z = jnp.asarray(z0, dtype=F64)
            lam_lm = 1e-6
            r, J = rJ(z)
            val = float(jnp.linalg.norm(r))
            for _ in range(200):
                H = J.T @ J
                g = J.T @ r
                dz = jnp.linalg.solve(
                    H + lam_lm * jnp.diag(jnp.diag(H))
                    + 1e-30 * jnp.eye(K, dtype=F64), -g)
                z_new = z + dz
                r_new = r_of(z_new)
                v_new = float(jnp.linalg.norm(r_new))
                if np.isfinite(v_new) and v_new < val:
                    z, val = z_new, v_new
                    r, J = rJ(z)
                    lam_lm = max(lam_lm / 3.0, 1e-12)
                    if val < 1e-14:
                        break
                else:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
            if best is None or val < best[1]:
                best = (np.asarray(z), val)
        return best

    def run_rollout(ops, ti):
        """One full 50-step rollout of test trajectory ti.  Returns errors
        and iteration counts; deterministic, so timed repetitions reproduce
        the same outputs (asserted)."""
        nu = float(nu_test[ti])
        u0 = U_test[ti, 0]
        z0, v0 = ic_cache[ti]
        u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
        tol_abs = STEP_TOL * u_scale * float(np.sqrt(n_i))
        zs = [z0]
        errs = []
        njacs = []
        uf = np.asarray(u_full_of(jnp.asarray(z0)))
        errs.append(rel(uf, u0[interior]))
        for t in range(1, T):
            z_prev = zs[-1]
            z_init = z_prev if len(zs) < 2 or EXTRAP == 0 else \
                z_prev + EXTRAP * (zs[-1] - zs[-2])
            prev_c = ops["prev_of"](jnp.asarray(z_prev, dtype=F64))
            z = jnp.asarray(z_init, dtype=F64)
            r, J = ops["rJ"](z, prev_c, nu)
            rn = float(jnp.linalg.norm(r))
            lam_lm = 1e-6
            nj = 1
            for attempt in range(1, GN_BUDGET + 1):
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
                if ndz > TR_DELTA:
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
                    r, J = ops["rJ"](z, prev_c, nu)
                    nj += 1
                    lam_lm = max(lam_lm / 3.0, 1e-12)
                    if rel_dec < STALL:
                        break
                else:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
            zs.append(np.asarray(z))
            njacs.append(nj)
            uf = np.asarray(u_full_of(jnp.asarray(zs[-1])))
            ut_ = U_test[ti, t][interior]
            errs.append(rel(uf, ut_))
        return dict(errs=errs, njacs=njacs, ic_resid=v0,
                    z_last=np.asarray(zs[-1]))

    # IC fits are shared by every arm (decoder frozen, same instrument)
    ic_cache = []
    for ti in range(N_TEST):
        z0, v0 = banked_ic_fit(U_test[ti, 0][interior])
        ic_cache.append((z0, v0))
    report["ic_resids"] = [float(v) for _, v in ic_cache]

    def certify(vname, ops, X_v=None, w_v=None, fit_info=None):
        out = dict(m=int(ops["m"]))
        if fit_info is not None:
            out["eq_rel_fit"] = fit_info["rel_fit"]
        # held-out rungs vs the full-grid ops
        recs = []
        for si in range(Sh):
            zj = Zh_j[si]
            pv = full_ops["prev_of"](Zhp_j[si])
            Rs, Js = [np.asarray(v) for v in
                      ops["rJ"](zj, pv, float(nu_h[si]))]
            Rf, Jf = [np.asarray(v) for v in
                      full_ops["rJ"](zj, pv, float(nu_h[si]))]
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
        out["held_recon_rel"] = held_recon

        # rollouts: burn-in rep (also compiles), then TIME_REPS timed reps per
        # trajectory; accuracy read from the LAST TIMED rep; determinism
        # asserted via the final latent.
        rolls = []
        for ti in range(N_TEST):
            r_burn = run_rollout(ops, ti)                     # burn / compile
            times, last = [], None
            for _ in range(TIME_REPS):
                t1 = time.perf_counter()
                last = run_rollout(ops, ti)
                times.append(time.perf_counter() - t1)
            assert np.allclose(last["z_last"], r_burn["z_last"], atol=0.0), \
                "rollout is not deterministic"
            errs = last["errs"]
            rolls.append(dict(
                traj=ti, nu=float(nu_test[ti]), ic_resid=last["ic_resid"],
                err_mean=float(np.mean(errs)), err_t0=errs[0],
                err_t1=errs[1], err_last=errs[-1],
                mean_njac=float(np.mean(last["njacs"])),
                times_s=[float(x) for x in times]))
            log(f"  [{vname}] traj {ti}: rollout err {np.mean(errs):.3e} "
                f"(t0 {errs[0]:.2e}, t50 {errs[-1]:.2e})  "
                f"median {np.median(times)*1e3:.0f} ms")
        out["rollout"] = rolls
        out["rollout_err_mean"] = float(np.mean([r_["err_mean"]
                                                 for r_ in rolls]))
        out["rollout_ms_median"] = float(np.median(
            [t for r_ in rolls for t in r_["times_s"]]) * 1e3)
        ho = out["heldout"]
        log(f"  [{vname}] SUMMARY m={out['m']}: held b {ho['b']:.3e} c1 "
            f"{ho['c1']:.3e} cos {ho['c1_cos']:.4f}  rollout "
            f"{out['rollout_err_mean']:.3e}  ({out['rollout_ms_median']:.0f} "
            f"ms/traj median)")
        report["variants"][vname] = out
        save()

    certify("oracle", full_ops)
    certify("base_tight", make_sampled_ops(X0, w0), X0, w0,
            fits["tight"]["info"])
    certify("nodes_tight", make_sampled_ops(X_nt, w_nt), X_nt, w_nt)
    certify("base_half", make_sampled_ops(fits["half"]["X"],
                                          fits["half"]["w"]),
            fits["half"]["X"], fits["half"]["w"], fits["half"]["info"])
    certify("nodes_half", make_sampled_ops(X_nh, w_nh), X_nh, w_nh)
    certify("base_gen", make_sampled_ops(fits["gen"]["X"], fits["gen"]["w"]),
            fits["gen"]["X"], fits["gen"]["w"], fits["gen"]["info"])

    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    with open(OUT, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    log(f"DONE -> {OUT} sha256={digest} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
