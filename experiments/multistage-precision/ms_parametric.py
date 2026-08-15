"""Experiment B: multi-stage training of the PARAMETRIC FiLM coord decoder.

Does Wang & Lai staging survive the jump from fitting one function to fitting
a solution FAMILY u(x, y; z)? Each stage is a fresh FiLM net (z-conditioned)
fitting the previous stages' residual over the TRAINING samples, normalized by
the global residual RMS eps_k; combined decoder u = sum_k eps_k net_k(x; z).

The scientific split this measures (absent in the paper, which fits fixed
targets with no generalization gap):
  - TRAIN fit error vs stage  -> how far staging pushes the optimization floor
  - VAL error vs stage        -> where the generalization floor caps the gain

Same Poisson-2D bump family / f64 FD-CG truth as the coord-decoder testbed
(sample_params ranges identical; n_train reduced for the local GB10 budget).
f64 data + f64 nets throughout. Adam only per stage (full-family L-BFGS is a
different beast; noted in README).

Usage: [N=64] [N_TRAIN=512] [N_VAL=64] [N_STAGES=3] [STEPS=25000]
       python ms_parametric.py [outdir]
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
STEPS = int(os.environ.get("STEPS", "25000"))
BATCH = int(os.environ.get("BATCH", "32"))
HIDDEN = int(os.environ.get("HIDDEN", "128"))
N_LAYERS = 4
PEAK_LR = float(os.environ.get("PEAK_LR", "2e-3"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))
CG_TOL, CG_MAXITER = 1e-13, 100_000

F64 = jnp.float64


# --------------------------- family + FOM (f64) ---------------------------

def neg_lap_interior(u_int, n):
    dx = 1.0 / (n - 1)
    u = jnp.pad(u_int, 1)
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
    z = np.stack([
        (cx - 0.5) / 0.35,
        (cy - 0.5) / 0.35,
        (np.log(w) - np.log(0.045)) / 0.8,
        (a - 1.25) / 0.75,
    ], axis=1)
    return cx, cy, w, a, z


def build_snapshots(n, chunk=256):
    cx, cy, w, a, z = sample_params()
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    m = len(cx)
    t0 = time.time()
    U = np.zeros((m, n, n))
    op = lambda v: neg_lap_interior(v, n)
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        op, F, tol=CG_TOL, maxiter=CG_MAXITER)[0])
    res_max = 0.0
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        F = np.stack([a[i] * np.exp(-((Xi - cx[i]) ** 2 + (Yi - cy[i]) ** 2)
                                    / (2 * w[i] ** 2)) for i in range(s, e)])
        U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(F)))
        U[s:e, 1:-1, 1:-1] = U_int
        r = float(jnp.linalg.norm(neg_lap_interior(jnp.asarray(U_int[0]), n)
                                  - F[0]) / jnp.linalg.norm(F[0]))
        # NaN-propagating accumulate (builtin max(x, nan) keeps x)
        if not np.isfinite(r) or r > res_max:
            res_max = r
    print(f"  FOM: {m} CG solves in {time.time()-t0:.0f}s, "
          f"spot rel residual {res_max:.2e}", flush=True)
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return (jnp.asarray(U.reshape(m, n * n)), jnp.asarray(z),
            jnp.asarray(coords))


# ------------------------- FiLM stage nets (f64) -------------------------

def coord_features(xy, n_freq):
    j = jnp.arange(1, n_freq + 1, dtype=F64)
    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)
    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1])], axis=1)


def init_dense(key, d_in, d_out):
    W = jax.random.normal(key, (d_in, d_out), dtype=F64) * np.sqrt(1.0 / d_in)
    return {"W": W, "b": jnp.zeros((d_out,), dtype=F64)}


def init_film_net(key, n_freq):
    d_in = 2 * (2 * n_freq + 1)
    keys = jax.random.split(key, N_LAYERS + 4)
    trunk = [init_dense(keys[0], d_in, HIDDEN)]
    for i in range(1, N_LAYERS):
        trunk.append(init_dense(keys[i], HIDDEN, HIDDEN))
    out = init_dense(keys[N_LAYERS], HIDDEN, 1)
    z_embed = init_dense(keys[N_LAYERS + 1], 4, 64)
    film = init_dense(keys[N_LAYERS + 2], 64, N_LAYERS * 2 * HIDDEN)
    film["W"] = film["W"] * 0.01
    return {"trunk": trunk, "out": out, "z_embed": z_embed, "film": film}


def film_apply(params, z, xy, n_freq):
    g = jax.nn.swish(z @ params["z_embed"]["W"] + params["z_embed"]["b"])
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
        tot = tot + s["eps"] * film_apply(s["params"], z, xy, s["n_freq"])
    return tot


# --------------------------- residual frequency probe ---------------------------

def dominant_radial_freq(e_fields, n):
    """Mean amplitude spectrum over sample fields -> peak radial bin."""
    A = np.abs(np.fft.fft2(np.asarray(e_fields).reshape(-1, n, n))).mean(0)
    fr = np.fft.fftfreq(n, d=1.0 / (n - 1))
    FX, FY = np.meshgrid(fr, fr, indexing="ij")
    R = np.sqrt(FX**2 + FY**2)
    bins = np.arange(0.0, R.max() + 1.0, 1.0)
    which = np.digitize(R.reshape(-1), bins)
    amp = np.zeros(len(bins) + 1)
    np.add.at(amp, which, A.reshape(-1))
    amp[0:2] = 0.0
    return float(np.argmax(amp))


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}",
          flush=True)
    nyq = (N - 1) // 2
    U, z, coords = build_snapshots(N)
    U_tr, U_va = U[:N_TRAIN], U[N_TRAIN:]
    z_tr, z_va = z[:N_TRAIN], z[N_TRAIN:]
    u_norm_tr = float(jnp.sqrt(jnp.mean(U_tr**2)))
    va_norms = jnp.linalg.norm(U_va, axis=1)

    def val_rel(stages):
        def one(zu):
            z_i, u_i = zu
            pred = combined_apply(stages, z_i, coords)
            return jnp.linalg.norm(pred - u_i)
        d = jax.lax.map(one, (z_va, U_va))
        return float(jnp.mean(d / va_norms))

    def train_resid(stages):
        """e = U_tr - combined prediction, full grid (chunked over samples)."""
        preds = []
        for s in range(0, N_TRAIN, 64):
            e = min(s + 64, N_TRAIN)
            preds.append(jax.vmap(
                lambda zi: combined_apply(stages, zi, coords))(z_tr[s:e]))
        return U_tr - jnp.concatenate(preds, axis=0)

    np_rng = np.random.default_rng(SEED)
    stages, report = [], []
    e_tr = U_tr
    n_freq = 16
    key = jax.random.PRNGKey(SEED)
    for k in range(N_STAGES):
        eps = float(jnp.sqrt(jnp.mean(e_tr**2)))
        f_d = dominant_radial_freq(e_tr[:32], N)
        if k > 0:
            n_freq = int(min(max(np.ceil(1.5 * f_d) + 4, n_freq), nyq))
        print(f"stage {k}: eps={eps:.3e}  f_d~{f_d:.0f}  n_freq={n_freq}",
              flush=True)
        key, sub = jax.random.split(key)
        params = init_film_net(sub, n_freq)
        target = e_tr / eps                       # (n_train, n^2), RMS ~ 1

        sched = optax.warmup_cosine_decay_schedule(
            0.0, PEAK_LR, max(1, STEPS // 20), STEPS, end_value=1e-9)
        opt = optax.adamw(sched, weight_decay=1e-6)
        state = opt.init(params)

        def loss_fn(ps, z_b, t_b):
            pred = jax.vmap(lambda zi: film_apply(ps, zi, coords, n_freq))(z_b)
            return jnp.mean((pred - t_b) ** 2)

        @jax.jit
        def step(ps, st, z_b, t_b):
            val, g = jax.value_and_grad(loss_fn)(ps, z_b, t_b)
            up, st = opt.update(g, st, ps)
            return optax.apply_updates(ps, up), st, val

        t0 = time.time()
        for it in range(STEPS):
            bi = np_rng.choice(N_TRAIN, size=BATCH, replace=False)
            params, state, val = step(params, state, z_tr[bi], target[bi])
            if it % 5000 == 0:
                print(f"  step {it:6d}  loss {float(val):.3e}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"  stage {k} trained {STEPS} steps in {time.time()-t0:.0f}s "
              f"(final batch loss {float(val):.3e})", flush=True)

        stages.append({"params": params, "n_freq": n_freq, "eps": eps})
        e_tr = train_resid(stages)
        fit_rel = float(jnp.sqrt(jnp.mean(e_tr**2))) / u_norm_tr
        v_rel = val_rel(stages)
        report.append({"stage": k, "eps_in": eps, "f_d": f_d, "n_freq": n_freq,
                       "train_fit_rel_rms": fit_rel, "val_rel_l2": v_rel})
        print(f"  after stage {k}: TRAIN fit rel {fit_rel:.3e}   "
              f"VAL rel-L2 {v_rel:.3e}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "ms_parametric_report.json"), "w") as f:
        json.dump({"N": N, "n_train": N_TRAIN, "n_val": N_VAL, "steps": STEPS,
                   "hidden": HIDDEN, "n_layers": N_LAYERS, "seed": SEED,
                   "stages": report}, f, indent=2)
    with open(os.path.join(OUTDIR, "ms_parametric_stages.pkl"), "wb") as f:
        pickle.dump([{"params": jax.tree_util.tree_map(np.asarray, s["params"]),
                      "n_freq": s["n_freq"], "eps": s["eps"]}
                     for s in stages], f)
    print("RESULT " + "  ".join(
        f"s{r['stage']}: train={r['train_fit_rel_rms']:.2e} "
        f"val={r['val_rel_l2']:.2e}" for r in report), flush=True)


if __name__ == "__main__":
    main()
