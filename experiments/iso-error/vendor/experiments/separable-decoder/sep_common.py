"""Separable (two-track) nonlinear decoder for the weak NM-ROM (2026-08-22).

    u(x; z) = bc(x) * ( g(x)^T h(z) ),   g: R^2 -> R^r (Fourier-feature MLP),
                                         h: R^k -> R^r (nonlinear MLP head).

PURE NEURAL: no POD basis, no POD initialisation, no linear corrector anywhere.
The nonlinearity in z lives in h; the manifold is the k-dimensional image of h
embedded in span{g_i}.  Everything x-dependent factors through g, so at any
fixed point set the decoder restricted to those points is a cached (m x r)
matrix times h(z) -- the "sampling built in" property.  Design and gates:
reports/2026-08-22-separable-eq-decoder-design.md on main.

GOVERNING INVARIANT: the cached fast path must be verified identical (<=1e-12
relative) to the meshfree path evaluated through the INCUMBENT weak operators;
the discretization is never changed, only how it is evaluated.

All f64.  Import this before other project modules is NOT required -- it
bootstraps sys.path for the sibling-experiment / staged-deps layouts itself.
"""
from __future__ import annotations

import os
import sys
import time
import pickle

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

F64 = jnp.float64
HERE = os.path.dirname(os.path.abspath(__file__))


def _bootstrap():
    """sys.path for both layouts: worktree siblings (local) and stage deps
    (cluster: everything under HERE/deps/<experiment-name>/ or beside us)."""
    cands = []
    for name in ("burgers2d-rom-latent-stepping", "cost-to-tolerance",
                 "poisson2d-rom-objective", "nonlinear-decoder-architecture"):
        cands += [os.path.join(HERE, "deps", name),
                  os.path.join(os.path.dirname(HERE), name)]
    cands.append(HERE)          # staged flat modules (ctol_eq.py, ctol_tol.py)
    for d in cands:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


_bootstrap()


def log(*a):
    print(*a, flush=True)


# ------------------------------- model --------------------------------------

def bc_poly(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def init_mlp(key, sizes):
    params = []
    for i in range(len(sizes) - 1):
        key, k1 = jax.random.split(key)
        w = jax.random.normal(k1, (sizes[i], sizes[i + 1]), dtype=F64) \
            * jnp.sqrt(2.0 / sizes[i])
        params.append((w, jnp.zeros((sizes[i + 1],), dtype=F64)))
    return params


def apply_mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def init_separable(key, k_lat, r_feat, n_ff=64, ff_scale=4.0,
                   g_hidden=128, g_layers=2, h_hidden=128, h_layers=2,
                   out_scale=1.0, ff_scales=None, h_ff=0, h_ff_scale=1.0):
    """ff_scales (round-2 lever, default OFF): a list of band scales, e.g.
    [1, 4, 16] -- the n_ff frequencies are split evenly across the bands and
    each band's B columns are drawn at its own scale (multi-scale Fourier
    features).  ff_scales=None reproduces the original single-scale init
    bit-for-bit (same rng draw, same multiplier)."""
    kb, kg, kh, kl = jax.random.split(key, 4)
    if ff_scales:
        n_b = len(ff_scales)
        split = [n_ff // n_b + (1 if i < n_ff % n_b else 0) for i in range(n_b)]
        scale_vec = jnp.concatenate([jnp.full((s,), float(sc_), dtype=F64)
                                     for s, sc_ in zip(split, ff_scales)])
        B = jax.random.normal(kb, (2, n_ff), dtype=F64) * scale_vec[None, :]
    else:
        B = jax.random.normal(kb, (2, n_ff), dtype=F64) * ff_scale  # fixed freqs
    g_mlp = init_mlp(kg, [2 * n_ff] + [g_hidden] * g_layers + [r_feat])
    h_in = k_lat + 2 * int(h_ff)          # latent Fourier features (default 0)
    h_mlp = init_mlp(kh, [h_in] + [h_hidden] * h_layers + [r_feat])
    h_lin = jax.random.normal(kl, (k_lat, r_feat), dtype=F64) * 0.3
    # fixed data-scale constant, folded into the FEATURES so every cached bank
    # carries it (it is x-independent but keeps u ~ O(data) at init)
    out = dict(B=B, g=g_mlp, h=h_mlp, h_lin=h_lin,
               out_scale=jnp.asarray(float(out_scale), dtype=F64))
    if h_ff:
        key, khb = jax.random.split(key)
        out["hB"] = (jax.random.normal(khb, (k_lat, int(h_ff)), dtype=F64)
                     * float(h_ff_scale))
    return out


def features(params, xy):
    """bc(x) * g~(x): (n_pts, r).  ALL x-dependence of the decoder."""
    ang = 2.0 * jnp.pi * (xy @ params["B"])
    ff = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return (params["out_scale"] * bc_poly(xy))[..., None] * apply_mlp(params["g"], ff)


def head(params, z):
    """h(z): (..., r).  NONLINEAR in z (MLP + linear skip).

    ROUND 5 option, default OFF and bit-identical when unused: if params
    carries an `hB` matrix, the MLP sees random Fourier features of the LATENT
    as well as the latent itself.  This is a z-side change only -- the bank
    G(x) and therefore the cached-collapse property and gate 0 are untouched --
    and it exists because round 5 measured that the codes are already converged
    and h's own FUNCTION CLASS is what limits the fit."""
    zf = z
    if "hB" in params:
        ang = 2.0 * jnp.pi * (z @ params["hB"])
        zf = jnp.concatenate([z, jnp.sin(ang), jnp.cos(ang)], axis=-1)
    return apply_mlp(params["h"], zf) + z @ params["h_lin"]


class SeparableDecoder:
    """Meshfree callable, drop-in for the incumbent coord-decoder interface:
    dec(z, xy) -> (n_pts,).  kind='coord' so blat_common.make_weak_ops and
    ctol_tol.lm_tau_poisson treat it exactly like the FiLM decoder."""

    def __init__(self, params, k_lat, r_feat):
        self.params = params
        self.k = int(k_lat)
        self.r = int(r_feat)
        self.kind = "coord"

    def __call__(self, z, xy):
        return features(self.params, xy) @ head(self.params, z)

    def feat_at(self, xy, chunk=0):
        """Cached-bank builder: numpy -> device (n_pts, r), z-independent.

        chunk>0 evaluates the g-track in row blocks and concatenates.  The
        RESULT IS IDENTICAL (features acts pointwise in x); chunking only
        bounds the peak activation footprint, which at N=1024 (n^2 ~ 1.05e6)
        and g_hidden=1024 would otherwise be ~8.6 GB per hidden layer."""
        xy = jnp.asarray(xy, dtype=F64)
        if not chunk or xy.shape[0] <= chunk:
            return jnp.asarray(features(self.params, xy))
        out = [features(self.params, xy[s:s + chunk])
               for s in range(0, xy.shape[0], chunk)]
        return jnp.concatenate(out, axis=0)

    def head_fn(self):
        p = self.params
        return lambda z: head(p, z)


# ------------------------------ training ------------------------------------

def train_autodecoder(key, coords, U, k_lat, r_feat, steps=30000, lr=1e-3,
                      lam_orth=1e-4, log_every=2000, tag="", **arch):
    """Joint Adam over (g, h, per-snapshot codes Z).  U: (S, n_pts) f64
    snapshot values at coords (n_pts, 2).  Loss: global relative MSE plus a
    small feature-Gram orthonormality regulariser (conditioning only -- it
    cannot and does not linearise h).  Full batch: the separable form makes
    one step  S x r  +  n_pts x r  network work plus one (S,r)x(r,n_pts)
    matmul, so there is nothing to mini-batch at these sizes."""
    coords = jnp.asarray(coords, dtype=F64)
    U = jnp.asarray(U, dtype=F64)
    S = U.shape[0]
    key, kz, kp = jax.random.split(key, 3)
    u_rms = float(jnp.sqrt(jnp.mean(U * U)))
    params = init_separable(kp, k_lat, r_feat, out_scale=u_rms, **arch)
    Z = 0.1 * jax.random.normal(kz, (S, k_lat), dtype=F64)
    u_ms = jnp.mean(U * U)

    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-2)
    opt = optax.adam(sched)
    state = opt.init((params, Z))

    def loss_fn(pz):
        p, z = pz
        G = features(p, coords)                       # (n_pts, r)
        H = head(p, z)                                # (S, r)
        err = H @ G.T - U
        rel = jnp.mean(err * err) / u_ms
        C = (G.T @ G) / (G.shape[0] * p["out_scale"] ** 2)
        orth = jnp.mean((C - jnp.eye(C.shape[0], dtype=F64)) ** 2)
        return rel + lam_orth * orth, rel

    @jax.jit
    def step(pz, st):
        (val, rel), grads = jax.value_and_grad(loss_fn, has_aux=True)(pz)
        grads[0]["out_scale"] = jnp.zeros_like(grads[0]["out_scale"])
        upd, st = opt.update(grads, st)
        return optax.apply_updates(pz, upd), st, val, rel

    pz = (params, Z)
    t0 = time.time()
    rel = jnp.inf
    for i in range(steps):
        pz, state, val, rel = step(pz, state)
        if (i + 1) % log_every == 0 or i == 0:
            log(f"   train[{tag}] step {i+1:6d}/{steps}  rel-MSE {float(rel):.3e}"
                f"  [{time.time()-t0:.0f}s]")
    params, Z = pz
    # per-snapshot relative L2 at the end, the project's reporting metric
    G = features(params, coords)
    Uh = head(params, Z) @ G.T
    per = jnp.linalg.norm(Uh - U, axis=1) / jnp.linalg.norm(U, axis=1)
    info = dict(final_rel_mse=float(rel), steps=steps, lr=lr,
                lam_orth=lam_orth, seconds=time.time() - t0,
                recon_rel_l2_mean=float(jnp.mean(per)),
                recon_rel_l2_max=float(jnp.max(per)),
                n_snapshots=int(S), n_points=int(coords.shape[0]))
    log(f"   train[{tag}] done: recon rel-L2 mean {info['recon_rel_l2_mean']:.3e} "
        f"max {info['recon_rel_l2_max']:.3e}  [{info['seconds']:.0f}s]")
    return params, np.asarray(Z), info


def oracle_fit(dec, coords, u_true, z_inits, budget=200):
    """Direct latent fit to a HELD-OUT field (representation oracle): damped LM
    on the data misfit from each init, best result kept."""
    coords = jnp.asarray(coords, dtype=F64)
    ut = jnp.asarray(u_true, dtype=F64)
    tn = jnp.linalg.norm(ut)

    def r_of(z):
        return (dec(z, coords) - ut) / tn

    rJ = jax.jit(lambda z: (r_of(z), jax.jacfwd(r_of)(z)))

    def solve(z0):
        z = jnp.asarray(z0, dtype=F64)
        lam = 1e-6
        r, J = rJ(z)
        val = float(jnp.linalg.norm(r))
        for _ in range(budget):
            H = J.T @ J
            g = J.T @ r
            dz = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H))
                                  + 1e-30 * jnp.eye(z.shape[0], dtype=F64), -g)
            z_new = z + dz
            r_new = jax.jit(r_of)(z_new)
            v_new = float(jnp.linalg.norm(r_new))
            if np.isfinite(v_new) and v_new < val:
                z, val = z_new, v_new
                r, J = rJ(z)
                lam = max(lam / 3.0, 1e-12)
                if val < 1e-14:
                    break
            else:
                lam = min(lam * 10.0, 1e12)
                if lam >= 1e12:
                    break
        return z, val

    best = None
    for z0 in z_inits:
        z, val = solve(z0)
        if best is None or val < best[1]:
            best = (z, val)
    return best


def save_pkl(path, params, Z_tr, cfg):
    host = jax.tree_util.tree_map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(dict(params=host, Z_tr=np.asarray(Z_tr), cfg=cfg), f)


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    return params, d["Z_tr"], d["cfg"]


def time_fn(fn, reps=7, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), ts


def arch_from_env():
    """Architecture knobs of init_separable, read from env (HPO search space:
    HANDOFF.md).  Only returns keys that differ from the defaults so cells that
    do not set them reproduce the N=64 recipe exactly."""
    spec = dict(n_ff=("N_FF", int, 64), ff_scale=("FF_SCALE", float, 4.0),
                g_hidden=("G_HIDDEN", int, 128), g_layers=("G_LAYERS", int, 2),
                h_hidden=("H_HIDDEN", int, 128), h_layers=("H_LAYERS", int, 2),
                h_ff=("H_FF", int, 0), h_ff_scale=("H_FF_SCALE", float, 1.0))
    out = {}
    for k, (env, typ, dflt) in spec.items():
        v = typ(os.environ.get(env, str(dflt)))
        if v != dflt:
            out[k] = v
    if os.environ.get("FF_SCALES"):                # e.g. "1,4,16" (multi-scale)
        out["ff_scales"] = [float(v)
                            for v in os.environ["FF_SCALES"].split(",")]
    return out


def time_pair(fa, fb, reps=4, warm=2):
    """Head-to-head PAIRED timing of exactly two subjects, the n1024-arm
    protocol adopted as the project standard: warm both, then alternate
    AB | BA | AB | BA ... so each side is first exactly half the time (with
    reps=3 the realised order is 'abbaab').  ALL raw repetitions are returned;
    medians are derived from them, never stored alone."""
    for _ in range(warm):
        fa()
        fb()
    ta, tb, order = [], [], []
    for i in range(reps):
        for side in ("ab" if i % 2 == 0 else "ba"):
            t0 = time.perf_counter()
            (fa if side == "a" else fb)()
            dt = time.perf_counter() - t0
            (ta if side == "a" else tb).append(dt)
            order.append(side)
    return dict(a_ms=float(np.median(ta) * 1e3),
                b_ms=float(np.median(tb) * 1e3),
                a_raw_ms=[t * 1e3 for t in ta],
                b_raw_ms=[t * 1e3 for t in tb],
                order="".join(order), reps=int(reps), warm=int(warm))


def balanced_time(subjects, reps=7, warm=2, capture=True):
    """MANDATORY-MEASUREMENT-RULES timing harness (HANDOFF.md rule 3):
    subjects = [(name, fn)], where fn() runs one complete invocation and
    BLOCKS before returning its outputs.  Every subject is warmed `warm`
    times, then timed `reps` times in a BALANCED order: the full subject list
    is swept once per repetition, forward on even sweeps and reversed on odd
    sweeps (AB/BA), so no subject is systematically first or last.  ALL raw
    repetition times are returned (never only medians), plus each subject's
    captured outputs from its final timed invocation."""
    raw = {name: [] for name, _ in subjects}
    results = {}
    for name, fn in subjects:
        for _ in range(warm):
            results[name] = fn()
    for rep in range(reps):
        order = subjects if rep % 2 == 0 else list(reversed(subjects))
        for name, fn in order:
            t0 = time.perf_counter()
            res = fn()
            raw[name].append(time.perf_counter() - t0)
            if capture:
                results[name] = res
    return raw, results
