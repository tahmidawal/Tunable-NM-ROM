"""Experiment (1): multi-stage training of the PARAMETRIC FiLM coord decoder,
TRUE z given — the control for the auto-decoder experiment — plus the shared
building blocks (family, FOM, FiLM nets, staging schedule, metrics) used by
ms_autodecoder.py and ms_diag.py.

Wang & Lai staging on a FAMILY u(x, y; z): each stage is a fresh z-conditioned
FiLM net fitting the previous stages' residual over the training samples,
combined decoder u = sum_k eps_k net_k(x; z), eps_k = RMS of the residual.

Review-driven design (2026-08-15 consolidated review, F2/F4/F5/F6/F8):
  * UNITS: coordinate features are sin/cos(pi*j*x), j = 1..n_freq — j is a
    HALF-CYCLE index (j/2 cycles per unit length).  The residual frequency
    probe returns f_d in cycles/unit (radial MEAN amplitude per FFT annulus,
    DC dropped, over the whole training residual set), and the schedule is
    n_freq = ceil(2*f_d) + margin, capped at N-1 (the half-cycle Nyquist of an
    N-point grid).  Unit test: ms_freq_test.py.
  * LOSS: per-sample inverse-energy weights w_i = (1/mean u_i^2)/mean_j(...) —
    the family's snapshot energies span >100x, so an unweighted MSE fits the
    loud samples only.  Every arm trains on RELATIVE error.
  * METRICS: both the global Frobenius relative error and the mean per-sample
    rel-L2, on TRAIN and VAL, plus val error per amplitude quartile.
  * Control arms for the budget-vs-representation question (F6): BATCH may
    equal N_TRAIN (full-batch), P_SUB=0 (all points), CONST_LR=1 (constant
    LR, 2x steps arm), LBFGS_STEPS>0 (full-batch L-BFGS polish after Adam).
  * Optional Z_FF=m: Fourier features on the latent/parameter input
    (sin/cos(pi*j*z), j<=m) — the "spectral bias in z" control.
  * Provenance: report + pkl carry the full config manifest and a completion
    flag; consumers assert on it.

Usage: [N=64] [N_TRAIN=512] [N_VAL=64] [N_STAGES=3] [STEPS=20000] [BATCH=32]
       [P_SUB=1024] [HIDDEN=128] [CONST_LR=0] [LBFGS_STEPS=0] [Z_FF=0]
       [SEED=0] python ms_parametric.py [outdir]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

N = int(os.environ.get("N", "64"))
N_TRAIN = int(os.environ.get("N_TRAIN", "512"))
N_VAL = int(os.environ.get("N_VAL", "64"))
N_STAGES = int(os.environ.get("N_STAGES", "3"))
STEPS = int(os.environ.get("STEPS", "20000"))
BATCH = min(int(os.environ.get("BATCH", "32")), N_TRAIN)
P_SUB = int(os.environ.get("P_SUB", "1024"))       # points/sample/step; 0 = all
HIDDEN = int(os.environ.get("HIDDEN", "128"))
N_LAYERS = int(os.environ.get("N_LAYERS", "4"))
PEAK_LR = float(os.environ.get("PEAK_LR", "2e-3"))
CONST_LR = int(os.environ.get("CONST_LR", "0"))
LBFGS_STEPS = int(os.environ.get("LBFGS_STEPS", "0"))
Z_FF = int(os.environ.get("Z_FF", "0"))
FREQ_MARGIN = int(os.environ.get("FREQ_MARGIN", "4"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))
CG_TOL, CG_MAXITER = 1e-13, 100_000
F64 = jnp.float64

CONFIG = dict(N=N, n_train=N_TRAIN, n_val=N_VAL, n_stages=N_STAGES, steps=STEPS,
              batch=BATCH, p_sub=P_SUB, hidden=HIDDEN, n_layers=N_LAYERS,
              peak_lr=PEAK_LR, const_lr=CONST_LR, lbfgs_steps=LBFGS_STEPS,
              z_ff=Z_FF, freq_margin=FREQ_MARGIN, seed=SEED)


# --------------------------- family + FOM (f64) ---------------------------

def neg_lap_interior(u_int, n):
    dx = 1.0 / (n - 1)
    u = jnp.pad(u_int, 1)                       # ghost-zero Dirichlet
    lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
           - 4.0 * u[1:-1, 1:-1]) / dx**2
    return -lap


def sample_params(seed=SEED, m=None):
    rng = np.random.default_rng(seed)
    m = m or (N_TRAIN + N_VAL)
    cx = rng.uniform(0.15, 0.85, m)
    cy = rng.uniform(0.15, 0.85, m)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), m))
    a = rng.uniform(0.5, 2.0, m)
    z = np.stack([(cx - 0.5) / 0.35, (cy - 0.5) / 0.35,
                  (np.log(w) - np.log(0.045)) / 0.8, (a - 1.25) / 0.75], axis=1)
    return cx, cy, w, a, z


def source_interior(n, cx, cy, w, a):
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    return a * np.exp(-((Xi - cx) ** 2 + (Yi - cy) ** 2) / (2 * w ** 2))


def build_snapshots(n, seed=SEED, m=None, chunk=256):
    """FD/CG f64 fields for the seed's family. Returns (U (m, n^2), z (m,4),
    coords (n^2, 2), max rel residual over ALL samples)."""
    cx, cy, w, a, z = sample_params(seed, m)
    m = len(cx)
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    t0 = time.time()
    U = np.zeros((m, n, n))
    op = lambda v: neg_lap_interior(v, n)
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        op, F, tol=CG_TOL, maxiter=CG_MAXITER)[0])
    resid_all = jax.jit(jax.vmap(lambda u, F: jnp.linalg.norm(op(u) - F)
                                 / jnp.linalg.norm(F)))
    res_max = 0.0
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        F = jnp.asarray(np.stack([source_interior(n, cx[i], cy[i], w[i], a[i])
                                  for i in range(s, e)]))
        U_int = jax.lax.map(solve_one, F)
        U[s:e, 1:-1, 1:-1] = np.asarray(U_int)
        r = np.asarray(resid_all(U_int, F))
        cm = float(np.max(r))
        if not np.isfinite(cm) or cm > res_max:      # NaN-propagating
            res_max = cm
    print(f"  FOM: {m} CG solves in {time.time()-t0:.0f}s, max rel residual "
          f"over all samples {res_max:.2e}", flush=True)
    assert np.isfinite(res_max) and res_max < 1e-10, "FOM not converged"
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return (jnp.asarray(U.reshape(m, n * n)), jnp.asarray(z),
            jnp.asarray(coords), res_max)


# ------------------------- FiLM stage nets (f64) -------------------------

def coord_features(xy, n_freq):
    """sin/cos(pi*j*x), j=1..n_freq: j is a HALF-CYCLE index (j/2 cycles/unit)."""
    j = jnp.arange(1, n_freq + 1, dtype=F64)
    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)
    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1])], axis=1)


def z_features(z, z_ff):
    if z_ff <= 0:
        return z
    j = jnp.arange(1, z_ff + 1, dtype=F64)
    return jnp.concatenate([z, jnp.sin(jnp.pi * j[:, None] * z[None, :]).reshape(-1),
                            jnp.cos(jnp.pi * j[:, None] * z[None, :]).reshape(-1)])


def init_dense(key, d_in, d_out):
    W = jax.random.normal(key, (d_in, d_out), dtype=F64) * np.sqrt(1.0 / d_in)
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F64)}


def init_film_net(key, n_freq, k_lat=4, z_ff=0):
    d_in = 2 * (2 * n_freq + 1)
    d_z = k_lat * (1 + 2 * z_ff)
    keys = jax.random.split(key, N_LAYERS + 4)
    trunk = [init_dense(keys[0], d_in, HIDDEN)]
    for i in range(1, N_LAYERS):
        trunk.append(init_dense(keys[i], HIDDEN, HIDDEN))
    out = init_dense(keys[N_LAYERS], HIDDEN, 1)
    z_embed = init_dense(keys[N_LAYERS + 1], d_z, 64)
    film = init_dense(keys[N_LAYERS + 2], 64, N_LAYERS * 2 * HIDDEN)
    film["W"] = film["W"] * 0.01
    return {"trunk": trunk, "out": out, "z_embed": z_embed, "film": film}


def film_apply(params, z, xy, n_freq, z_ff=0):
    g = jax.nn.swish(z_features(z, z_ff) @ params["z_embed"]["W"]
                     + params["z_embed"]["b"])
    film = (g @ params["film"]["W"] + params["film"]["b"]).reshape(
        N_LAYERS, 2, HIDDEN)
    h = coord_features(xy, n_freq)
    for i, lyr in enumerate(params["trunk"]):
        h = h @ lyr["W"] + lyr["b"]
        h = h * (1.0 + film[i, 0]) + film[i, 1]
        h = jax.nn.swish(h)
    return (h @ params["out"]["W"] + params["out"]["b"])[:, 0]


def combined_apply(stages, z, xy):
    tot = 0.0
    for s in stages:
        tot = tot + s["eps"] * film_apply(s["params"], z, xy, s["n_freq"],
                                          s.get("z_ff", 0))
    return tot


def stages_to_np(stages):
    return [{"params": jax.tree_util.tree_map(np.asarray, s["params"]),
             "n_freq": int(s["n_freq"]), "eps": float(s["eps"]),
             "z_ff": int(s.get("z_ff", 0))} for s in stages]


def stages_from_np(raw):
    return [{"params": jax.tree_util.tree_map(jnp.asarray, s["params"]),
             "n_freq": s["n_freq"], "eps": s["eps"], "z_ff": s.get("z_ff", 0)}
            for s in raw]


# --------------------------- residual frequency probe ---------------------------

def dominant_radial_freq(e_fields, n):
    """Dominant radial frequency (CYCLES per unit length) of the mean amplitude
    spectrum over the given fields: radial MEAN per unit-width annulus, DC
    dropped.  (Summing per annulus biases toward the Nyquist ring — the ring
    area grows with radius; white noise would return the last bin.)"""
    E = np.asarray(e_fields).reshape(-1, n, n)
    A = np.abs(np.fft.fft2(E)).mean(0)
    fr = np.fft.fftfreq(n, d=1.0 / (n - 1))            # cycles/unit
    FX, FY = np.meshgrid(fr, fr, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2).reshape(-1)
    which = np.floor(R + 0.5).astype(int)              # annulus index ~ round(R)
    amp = np.bincount(which, weights=A.reshape(-1))
    cnt = np.bincount(which).astype(float)
    mean_amp = amp / np.maximum(cnt, 1.0)
    mean_amp[0] = 0.0                                  # drop DC only
    return float(np.argmax(mean_amp))


def freq_schedule(f_d, prev_n_freq, n, margin=FREQ_MARGIN):
    """Half-cycle feature index covering f_d cycles/unit: j = 2 f_d (+margin),
    non-decreasing across stages, capped at the grid's half-cycle Nyquist n-1."""
    return int(min(max(int(np.ceil(2.0 * f_d)) + margin, prev_n_freq), n - 1))


# --------------------------- metrics + weights ---------------------------

def sample_weights(U):
    """Inverse-energy weights normalized to mean 1 (train on relative error)."""
    inv = 1.0 / jnp.mean(U ** 2, axis=1)
    return inv / jnp.mean(inv)


def rel_metrics(pred, U):
    """(global Frobenius rel, mean per-sample rel-L2, per-sample rel array)."""
    d = jnp.linalg.norm(pred - U, axis=1)
    nrm = jnp.linalg.norm(U, axis=1)
    per = np.asarray(d / nrm)
    glob = float(jnp.linalg.norm(pred - U) / jnp.linalg.norm(U))
    return glob, float(per.mean()), per


def quartile_errors(per, U):
    """Mean per-sample rel-L2 within amplitude quartiles (by ||u_i||)."""
    nrm = np.asarray(jnp.linalg.norm(U, axis=1))
    q = np.quantile(nrm, [0.25, 0.5, 0.75])
    idx = np.digitize(nrm, q)
    return [float(per[idx == k].mean()) if np.any(idx == k) else float("nan")
            for k in range(4)]


def predict_all(stages, Z, coords, chunk=64):
    preds = []
    for s in range(0, Z.shape[0], chunk):
        preds.append(jax.vmap(lambda zi: combined_apply(stages, zi, coords))(
            Z[s:s + chunk]))
    return jnp.concatenate(preds, axis=0)


# --------------------------- one stage fit ---------------------------

def make_lr_schedule(steps):
    if CONST_LR:
        return PEAK_LR
    return optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, max(1, steps // 20), steps, end_value=1e-9)


def fit_stage(key, np_rng, coords, target, weights, n_freq, Z, *, k_lat=4,
              z_ff=0, learn_latents=False, lat_lr=5e-3, lat_reg=1e-4, steps=STEPS,
              tag=""):
    """Fit one FiLM stage to `target` (n_s, n^2) (RMS ~ 1) with per-sample
    weights (mean 1).  learn_latents: joint (weights, latents) with LAZY per-row
    Adam on the latents (rows outside the batch keep moments/step untouched, so
    no drift on stale momentum).  Returns (params, Z, final_loss, secs)."""
    params = init_film_net(key, n_freq, k_lat, z_ff)
    opt = optax.adamw(make_lr_schedule(steps), weight_decay=1e-6)
    state = opt.init(params)
    n_s, n_pts = target.shape

    def loss_fn(ps, z_b, t_b, w_b, pidx):
        pred = jax.vmap(lambda zi: film_apply(ps, zi, coords[pidx], n_freq, z_ff))(z_b)
        se = jnp.mean((pred - t_b[:, pidx]) ** 2, axis=1)         # (B,)
        return jnp.mean(w_b * se)

    @jax.jit
    def step_w(ps, st, z_b, t_b, w_b, pidx):
        val, g = jax.value_and_grad(loss_fn)(ps, z_b, t_b, w_b, pidx)
        up, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, up), st, val

    # lazy per-row Adam for latents
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    lat_sched = (lambda t: lat_lr) if CONST_LR else optax.warmup_cosine_decay_schedule(
        0.0, lat_lr, max(1, steps // 20), steps, end_value=1e-9)

    @jax.jit
    def step_wz(ps, st, Z, m, v, cnt, gstep, bi, t_b, w_b, pidx):
        z_b = Z[bi]
        def lz(ps_, z_b_):
            return (loss_fn(ps_, z_b_, t_b, w_b, pidx)
                    + lat_reg * jnp.mean(z_b_ ** 2))
        val, (gp, gz) = jax.value_and_grad(lz, argnums=(0, 1))(ps, z_b)
        up, st = opt.update(gp, st, ps)
        ps = optax.apply_updates(ps, up)
        m_b = b1 * m[bi] + (1 - b1) * gz
        v_b = b2 * v[bi] + (1 - b2) * gz ** 2
        c_b = cnt[bi] + 1.0
        mhat = m_b / (1 - b1 ** c_b[:, None])
        vhat = v_b / (1 - b2 ** c_b[:, None])
        lr = lat_sched(gstep)
        z_new = z_b - lr * mhat / (jnp.sqrt(vhat) + eps_a)
        return (ps, st, Z.at[bi].set(z_new), m.at[bi].set(m_b),
                v.at[bi].set(v_b), cnt.at[bi].set(c_b), val)

    m = jnp.zeros_like(Z); v = jnp.zeros_like(Z)
    cnt = jnp.zeros((Z.shape[0],), dtype=F64)
    all_pts = jnp.arange(n_pts)
    t0 = time.time()
    for it in range(steps):
        bi = (np.arange(n_s) if BATCH >= n_s else
              np_rng.choice(n_s, size=BATCH, replace=False))
        pidx = (all_pts if P_SUB <= 0 or P_SUB >= n_pts else
                jnp.asarray(np_rng.choice(n_pts, size=P_SUB, replace=False)))
        if learn_latents:
            params, state, Z, m, v, cnt, val = step_wz(
                params, state, Z, m, v, cnt, it, jnp.asarray(bi), target[bi],
                weights[bi], pidx)
        else:
            params, state, val = step_w(params, state, Z[bi], target[bi],
                                        weights[bi], pidx)
        if it % 5000 == 0:
            print(f"  [{tag}] step {it:6d}  loss {float(val):.3e}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    adam_loss = float(val)
    print(f"  [{tag}] Adam {steps} steps in {time.time()-t0:.0f}s "
          f"(final batch loss {adam_loss:.3e})", flush=True)

    if LBFGS_STEPS > 0 and not learn_latents:
        # full-batch (all samples, all points) L-BFGS polish of the weights
        def full_loss(ps):
            def one(zi, ti, wi):
                pred = film_apply(ps, zi, coords, n_freq, z_ff)
                return wi * jnp.mean((pred - ti) ** 2)
            return jnp.mean(jax.lax.map(lambda a: one(*a), (Z, target, weights)))
        lopt = optax.lbfgs()
        lstate = lopt.init(params)
        vg = optax.value_and_grad_from_state(full_loss)

        @jax.jit
        def lstep(ps, st):
            val, g = vg(ps, state=st)
            up, st = lopt.update(g, st, ps, value=val, grad=g, value_fn=full_loss)
            return optax.apply_updates(ps, up), st, val

        snap = params
        t1 = time.time()
        for it in range(LBFGS_STEPS):
            params, lstate, lval = lstep(params, lstate)
            if not np.isfinite(float(lval)):
                print(f"  [{tag}] L-BFGS non-finite at {it}; keeping Adam params",
                      flush=True)
                params = snap
                break
        else:
            print(f"  [{tag}] L-BFGS {LBFGS_STEPS} its: full loss "
                  f"{float(full_loss(snap)):.3e} -> {float(lval):.3e} "
                  f"[{time.time()-t1:.0f}s]", flush=True)
    return params, Z, adam_loss, time.time() - t0


# --------------------------- main: true-z control ---------------------------

def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}",
          flush=True)
    print("CONFIG " + json.dumps(CONFIG), flush=True)
    U, z, coords, fom_res = build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]
    w_tr = sample_weights(U_tr)
    np_rng = np.random.default_rng(SEED)
    key = jax.random.PRNGKey(SEED)
    stages, report = [], []
    e_tr = U_tr
    n_freq = freq_schedule(dominant_radial_freq(U_tr, N), 0, N)
    os.makedirs(OUTDIR, exist_ok=True)
    for k in range(N_STAGES):
        eps = float(jnp.sqrt(jnp.mean(e_tr ** 2)))
        f_d = dominant_radial_freq(e_tr, N)
        if k > 0:
            n_freq = freq_schedule(f_d, n_freq, N)
        print(f"stage {k}: eps={eps:.3e}  f_d~{f_d:.1f} cyc/unit  n_freq={n_freq}"
              f" (half-cycle idx)", flush=True)
        key, sub = jax.random.split(key)
        params, _, adam_loss, secs = fit_stage(
            sub, np_rng, coords, e_tr / eps, w_tr, n_freq, z_tr, z_ff=Z_FF,
            tag=f"s{k}")
        stages.append({"params": params, "n_freq": n_freq, "eps": eps, "z_ff": Z_FF})
        pred_tr = predict_all(stages, z_tr, coords)
        e_tr = U_tr - pred_tr
        g_tr, m_tr, _ = rel_metrics(pred_tr, U_tr)
        pred_va = predict_all(stages, z_va, coords)
        g_va, m_va, per_va = rel_metrics(pred_va, U_va)
        row = {"stage": k, "eps_in": eps, "f_d_cyc": f_d, "n_freq": n_freq,
               "adam_final_batch_loss": adam_loss, "secs": secs,
               "train_global_rel": g_tr, "train_mean_rel_l2": m_tr,
               "val_global_rel": g_va, "val_mean_rel_l2": m_va,
               "val_rel_l2_by_amp_quartile": quartile_errors(per_va, U_va)}
        report.append(row)
        print(f"  after stage {k}: TRAIN global {g_tr:.3e} / mean-rel {m_tr:.3e}"
              f"   VAL global {g_va:.3e} / mean-rel {m_va:.3e}   "
              f"val by amp quartile {['%.2e' % q for q in row['val_rel_l2_by_amp_quartile']]}",
              flush=True)
        with open(os.path.join(OUTDIR, "ms_parametric_stages.pkl"), "wb") as f:
            pickle.dump({"config": CONFIG, "stages": stages_to_np(stages)}, f)
        with open(os.path.join(OUTDIR, "ms_parametric_report.json"), "w") as f:
            json.dump({"config": CONFIG, "fom_max_rel_residual": fom_res,
                       "stages": report, "complete": k == N_STAGES - 1}, f, indent=2)
    print("RESULT " + "  ".join(
        f"s{r['stage']}: train={r['train_global_rel']:.2e}/{r['train_mean_rel_l2']:.2e} "
        f"val={r['val_global_rel']:.2e}/{r['val_mean_rel_l2']:.2e}" for r in report),
        flush=True)


if __name__ == "__main__":
    main()
