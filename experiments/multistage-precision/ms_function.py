"""Experiment A: multi-stage NN fitting of ONE Poisson-2D solution field.

Validates our implementation of Wang & Lai (JCP 504 (2024) 112865,
"Multi-stage neural networks: function approximator of machine precision")
before touching the parametric decoder. Algorithm 1 of the paper, with the
frequency-matching implemented via per-stage Fourier-feature bandwidth (the
paper's Fig. 3 shows a Fourier-feature first layer is equivalent to their
kappa-scaled sin first layer for capturing the residue's dominant frequency).

Per stage k:
  target_k = e_k / eps_k,  eps_k = RMS(e_k),  e_0 = u (the field), and
  e_{k+1} = e_k - eps_k * net_k.  Combined model: u ~= sum_k eps_k net_k.
  Stage bandwidth n_freq_k is set from the dominant radial frequency of e_k
  (2D FFT peak), with margin, capped at the grid Nyquist (N-1)//2.
  f64 throughout (the whole point is going below the f32 floor ~1e-7).
  Optimizer per stage: full-batch Adam (cosine decay) + L-BFGS polish.

Target field: FD/CG f64 solution of -lap u = gaussian bump (cx,cy,w,a) =
(0.4, 0.6, 0.05, 1.5) on the unit square, u=0 walls, N=128 grid — the same
FOM as the Poisson coord-decoder testbed.

Usage: [N=128] [N_STAGES=4] [ADAM_STEPS=20000] [LBFGS_STEPS=2000]
       python ms_function.py [outdir]
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

N = int(os.environ.get("N", "128"))
N_STAGES = int(os.environ.get("N_STAGES", "4"))
ADAM_STEPS = int(os.environ.get("ADAM_STEPS", "20000"))
LBFGS_STEPS = int(os.environ.get("LBFGS_STEPS", "2000"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
N_LAYERS = 3
PEAK_LR = float(os.environ.get("PEAK_LR", "2e-3"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.abspath(__file__))
CG_TOL, CG_MAXITER = 1e-13, 100_000

F64 = jnp.float64

SRC = dict(cx=0.4, cy=0.6, w=0.05, a=1.5)


# --------------------------- FOM (f64, from testbed) ---------------------------

def neg_lap_interior(u_int, n):
    dx = 1.0 / (n - 1)
    u = jnp.pad(u_int, 1)
    lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
           - 4.0 * u[1:-1, 1:-1]) / dx**2
    return -lap


def solve_field(n):
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    F = SRC["a"] * np.exp(-((Xi - SRC["cx"]) ** 2 + (Yi - SRC["cy"]) ** 2)
                          / (2 * SRC["w"] ** 2))
    op = lambda v: neg_lap_interior(v, n)
    u_int, _ = jax.scipy.sparse.linalg.cg(op, jnp.asarray(F), tol=CG_TOL,
                                          maxiter=CG_MAXITER)
    res = float(jnp.linalg.norm(neg_lap_interior(u_int, n) - F)
                / jnp.linalg.norm(F))
    U = np.zeros((n, n))
    U[1:-1, 1:-1] = np.asarray(u_int)
    print(f"  FOM: N={n} CG rel residual {res:.2e}", flush=True)
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return jnp.asarray(coords), jnp.asarray(U.reshape(-1))


# ------------------------------ stage networks ------------------------------

def coord_features(xy, n_freq):
    j = jnp.arange(1, n_freq + 1, dtype=F64)
    def ff(c):
        return jnp.concatenate(
            [c[:, None], jnp.sin(jnp.pi * j * c[:, None]),
             jnp.cos(jnp.pi * j * c[:, None])], axis=1)
    return jnp.concatenate([ff(xy[:, 0]), ff(xy[:, 1])], axis=1)


def init_net(key, n_freq):
    d_in = 2 * (2 * n_freq + 1)
    dims = [d_in] + [HIDDEN] * N_LAYERS + [1]
    keys = jax.random.split(key, len(dims))
    layers = []
    for i in range(len(dims) - 1):
        W = jax.random.normal(keys[i], (dims[i], dims[i + 1]), dtype=F64) \
            * np.sqrt(1.0 / dims[i])
        layers.append({"W": W, "b": jnp.zeros((dims[i + 1],), dtype=F64)})
    return layers


def net_apply(layers, xy, n_freq):
    h = coord_features(xy, n_freq)
    for lyr in layers[:-1]:
        h = jnp.tanh(h @ lyr["W"] + lyr["b"])
    return (h @ layers[-1]["W"] + layers[-1]["b"])[:, 0]


# --------------------------- residual frequency probe ---------------------------

def dominant_radial_freq(e_grid):
    """Peak of the radially-binned 2D amplitude spectrum, cycles per unit."""
    n = e_grid.shape[0]
    A = np.abs(np.fft.fft2(np.asarray(e_grid)))
    fr = np.fft.fftfreq(n, d=1.0 / (n - 1))       # cycles per unit length
    FX, FY = np.meshgrid(fr, fr, indexing="ij")
    R = np.sqrt(FX**2 + FY**2)
    bins = np.arange(0.0, R.max() + 1.0, 1.0)
    which = np.digitize(R.reshape(-1), bins)
    amp = np.zeros(len(bins) + 1)
    np.add.at(amp, which, A.reshape(-1))
    amp[0:2] = 0.0                                 # drop DC bin
    return float(np.argmax(amp))                   # bin index ~ cycles/unit


# ------------------------------ per-stage training ------------------------------

def train_stage(key, coords, target, n_freq, tag):
    """Fit target (RMS ~ 1) with a fresh net; full batch; Adam + L-BFGS."""
    layers = init_net(key, n_freq)

    def loss_fn(ps):
        return jnp.mean((net_apply(ps, coords, n_freq) - target) ** 2)

    sched = optax.warmup_cosine_decay_schedule(
        0.0, PEAK_LR, max(1, ADAM_STEPS // 20), ADAM_STEPS, end_value=1e-9)
    opt = optax.adam(sched)
    state = opt.init(layers)

    @jax.jit
    def astep(ps, st):
        val, g = jax.value_and_grad(loss_fn)(ps)
        up, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, up), st, val

    t0 = time.time()
    for it in range(ADAM_STEPS):
        layers, state, val = astep(layers, state)
    adam_loss = float(val)
    adam_layers = layers                     # snapshot for the non-finite fallback

    lopt = optax.lbfgs()
    lstate = lopt.init(layers)
    vg = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def lstep(ps, st):
        val, g = vg(ps, state=st)
        up, st = lopt.update(g, st, ps, value=val, grad=g, value_fn=loss_fn)
        return optax.apply_updates(ps, up), st, val

    for it in range(LBFGS_STEPS):
        layers, lstate, val = lstep(layers, lstate)
        if not np.isfinite(float(val)):
            print(f"  [{tag}] L-BFGS went non-finite at iter {it}; "
                  f"keeping Adam result", flush=True)
            return adam_layers, adam_loss, adam_loss, time.time() - t0
    print(f"  [{tag}] adam loss {adam_loss:.3e} -> lbfgs loss {float(val):.3e} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return layers, adam_loss, float(val), time.time() - t0


def main():
    print(f"jax_backend={jax.default_backend()}  x64={jax.config.jax_enable_x64}",
          flush=True)
    coords, u = solve_field(N)
    nyq = (N - 1) // 2

    eps0 = float(jnp.sqrt(jnp.mean(u**2)))
    e = u
    stages, report = [], []
    key = jax.random.PRNGKey(SEED)
    n_freq = 8                                      # stage-0 bandwidth
    for k in range(N_STAGES):
        eps = float(jnp.sqrt(jnp.mean(e**2)))
        f_d = dominant_radial_freq(e.reshape(N, N))
        if k > 0:
            n_freq = int(min(max(np.ceil(1.5 * f_d) + 4, n_freq), nyq))
        key, sub = jax.random.split(key)
        print(f"stage {k}: eps={eps:.3e}  f_d~{f_d:.0f} cyc  n_freq={n_freq}",
              flush=True)
        layers, adam_l, final_l, secs = train_stage(
            sub, coords, e / eps, n_freq, f"stage{k}")
        stages.append({"layers": layers, "n_freq": n_freq, "eps": eps})
        e = e - eps * net_apply(layers, coords, n_freq)
        rms = float(jnp.sqrt(jnp.mean(e**2)))
        rel = rms / eps0
        report.append({"stage": k, "eps_in": eps, "f_d": f_d, "n_freq": n_freq,
                       "adam_loss": adam_l, "final_loss": final_l,
                       "resid_rms": rms, "resid_rel": rel, "secs": secs})
        print(f"  after stage {k}: residual RMS {rms:.3e} (rel {rel:.3e})",
              flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "ms_function_report.json"), "w") as f:
        json.dump({"N": N, "hidden": HIDDEN, "layers": N_LAYERS,
                   "adam_steps": ADAM_STEPS, "lbfgs_steps": LBFGS_STEPS,
                   "seed": SEED, "eps0": eps0, "stages": report}, f, indent=2)
    with open(os.path.join(OUTDIR, "ms_function_stages.pkl"), "wb") as f:
        pickle.dump([{"layers": jax.tree_util.tree_map(np.asarray, s["layers"]),
                      "n_freq": s["n_freq"], "eps": s["eps"]}
                     for s in stages], f)
    print("RESULT " + "  ".join(
        f"s{r['stage']}:rel={r['resid_rel']:.2e}" for r in report), flush=True)


if __name__ == "__main__":
    main()
