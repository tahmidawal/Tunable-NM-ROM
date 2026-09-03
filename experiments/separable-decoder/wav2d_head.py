"""Wave 2D phase 2 — the head h: R^K -> R^R (three training arms), the oracle, and the manifold gates.

Reuses the Stokes head (stk2d_head: SiLU MLP + linear skip, coefficient-space training, multi-start
LM oracle).  Because the bank is M-orthonormal everything lives in coefficient space:

    ||u - G h(z)||_M^2 = ||c - h(z)||_2^2 + ||u - G c||_M^2,   c = G^T M u.

THREE ARMS (the experimental variable of the cell):
  'sup'      z = (mu~, t~) in [-1,1]^6, supervised regression (mu, t) -> c.  K = 6.
  'auto'     one free code per (trajectory, time) row, initialised from the top-K POD coefficients
             (the 08-16 recipe), joint Adam.  K = 8.
  'auto+vc'  'auto' plus the velocity-consistency term  || J_h(z_s) zdot_s - c^v_s ||^2  with a free
             per-row zdot_s and c^v = G^T M v the FOM velocity's coefficients: the tangent space is
             asked to contain the velocity.  K = 8.

LOSS WEIGHTING: per-trajectory inverse mean-square weights on the coefficient (M-norm) error for u and,
in the vc arm, separately for v (wave snapshot norms pass near zero during kinetic/potential exchange,
so per-snapshot normalisation is never used).  The ENERGY norm is a reporting metric, not the loss;
WAVE2D-NOTES records this as amendment 5 to the design's 'energy norm everywhere' sentence.

MANIFOLD GATES (design phase 2): D1 held-out oracle vs POD-K; D2 J_h conditioning; G0a train/held-out
gap; G0b tangent-space velocity residual vs POD-K on the same states.  G0c (stepdiag) needs the ROM
arm and lives in wav2d_rom_gates.py.
"""
from __future__ import annotations

import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                          # noqa: E402
import optax                                                     # noqa: E402

import stk2d_head as sh                                          # noqa: E402
from stk2d_head import init_head, apply_head, param_count, oracle_fit, save_head, load_head_jax  # noqa

F64 = jnp.float64


# ----------------------------- latents for 'sup' -----------------------------

def sup_latents(mu, n_time=None):
    """(m, 5) family parameters in [-1,1]^5  ->  (m * n_time, 6) rows (mu~, t~) with t~ = 2t/T - 1"""
    import wav2d_common as wc
    T1 = (n_time or wc.NUM_STEPS + 1)
    tt = 2.0 * np.arange(T1) / (T1 - 1) - 1.0
    Z = np.concatenate([np.repeat(np.asarray(mu), T1, axis=0), np.tile(tt, len(mu))[:, None]], axis=1)
    return Z


def traj_weights(C, traj):
    """per-trajectory inverse mean-square weights, normalised to mean 1 over rows"""
    C = np.asarray(C); w = np.zeros(len(traj))
    for t in np.unique(traj):
        idx = traj == t
        w[idx] = 1.0 / max(float(np.mean(np.sum(C[idx] ** 2, axis=1))), 1e-300)
    return w / w.mean()


# ----------------------------- training -----------------------------

def train_head(C, traj, mode, K, MU=None, CV=None, hidden=128, layers=3, steps=40000, lr=3e-3,
               batch=2048, seed=0, vc_weight=1.0, log_every=0, tag=""):
    """C (S,R) coefficient targets, traj (S,) trajectory index of each row, mode in
    {'sup','auto','auto+vc'}; MU (S,K) latents for 'sup'; CV (S,R) velocity coefficients for 'auto+vc'.
    Returns a spec dict (params, Z, Zdot for vc, scale, config)."""
    C = np.asarray(C, dtype=float); S, R = C.shape
    scale = float(np.sqrt((C ** 2).mean()))
    Y = jnp.asarray(C / scale)
    W = jnp.asarray(traj_weights(C, traj))
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    params, sizes = init_head(k1, K, R, hidden, layers, 0, 1.0)
    vc = mode == "auto+vc"
    if mode == "sup":
        assert MU is not None and MU.shape == (S, K)
        Z = jnp.asarray(np.asarray(MU, dtype=float)); free = False
    elif mode in ("auto", "auto+vc"):
        Z = jnp.asarray(C[:, :K] / scale); free = True         # top-K POD coefficients (08-16 recipe)
    else:                                                    # pragma: no cover
        raise ValueError(mode)
    if vc:
        assert CV is not None and CV.shape == (S, R)
        YV = jnp.asarray(np.asarray(CV) / scale)
        WV = jnp.asarray(traj_weights(CV, traj))
        # initial zdot: finite difference of the initial codes along each trajectory
        Zn = np.asarray(Z); Zd = np.zeros_like(Zn)
        import wav2d_common as wc
        T1 = wc.NUM_STEPS + 1
        for t in np.unique(traj):
            idx = np.where(traj == t)[0]
            zz = Zn[idx]
            Zd[idx] = np.gradient(zz, wc.DT_SNAP, axis=0)
        Zdot = jnp.asarray(Zd)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-3)
    opt = optax.adam(sched)

    if free and vc:
        tgt = (params, Z, Zdot)

        def loss_fn(t_, y, yv, w, wv, idx):
            p, z, zd = t_
            zi, zdi = z[idx], zd[idx]
            e = apply_head(p, zi) - y
            h_dot = jax.vmap(lambda a, b: jax.jvp(lambda q: apply_head(p, q[None, :])[0], (a,), (b,))[1])(zi, zdi)
            ev = h_dot - yv
            return jnp.mean(w * jnp.sum(e * e, 1)) / jnp.mean(w * jnp.sum(y * y, 1)) + \
                vc_weight * jnp.mean(wv * jnp.sum(ev * ev, 1)) / jnp.mean(wv * jnp.sum(yv * yv, 1))
    elif free:
        tgt = (params, Z)

        def loss_fn(t_, y, w, idx):
            p, z = t_
            e = apply_head(p, z[idx]) - y
            return jnp.mean(w * jnp.sum(e * e, 1)) / jnp.mean(w * jnp.sum(y * y, 1))
    else:
        tgt = params

        def loss_fn(p, y, w, z):
            e = apply_head(p, z) - y
            return jnp.mean(w * jnp.sum(e * e, 1)) / jnp.mean(w * jnp.sum(y * y, 1))

    state = opt.init(tgt)

    @jax.jit
    def step(t_, st, *args):
        v, gr = jax.value_and_grad(loss_fn)(t_, *args)
        upd, st = opt.update(gr, st)
        return optax.apply_updates(t_, upd), st, v

    rng = np.random.default_rng(seed + 1)
    nb = int(batch) if batch and batch < S else S
    t0 = time.time(); v = np.inf
    for i in range(steps):
        idx = rng.integers(0, S, nb) if nb < S else np.arange(S)
        ji = jnp.asarray(idx)
        if free and vc:
            tgt, state, v = step(tgt, state, Y[ji], YV[ji], W[ji], WV[ji], ji)
        elif free:
            tgt, state, v = step(tgt, state, Y[ji], W[ji], ji)
        else:
            tgt, state, v = step(tgt, state, Y[ji], W[ji], Z[ji])
        if log_every and ((i + 1) % log_every == 0 or i == 0):
            print(f"   train[{tag}] {i+1:6d}/{steps} loss {float(v):.3e} [{time.time()-t0:.0f}s]", flush=True)
    if free and vc:
        params, Zf, Zdf = tgt
    elif free:
        params, Zf = tgt; Zdf = None
    else:
        params, Zf, Zdf = tgt, Z, None
    return dict(params=params, scale=scale, K=int(K), R=int(R), mode=mode, hidden=int(hidden),
                layers=int(layers), ff=0, ff_scale=1.0, steps=int(steps), lr=float(lr), batch=int(nb),
                seed=int(seed), n_fit=int(S), sizes=sizes, final_loss=float(v), seconds=float(time.time() - t0),
                n_params=param_count(params), Z=np.asarray(Zf), Zdot=(np.asarray(Zdf) if Zdf is not None else None),
                vc_weight=float(vc_weight) if vc else 0.0)


# ----------------------------- head evaluation (numpy + jax) -----------------------------

def head_jac(spec, Z):
    """J_h at rows Z (S,K) -> (S, R, K), in COEFFICIENT units (scale applied)"""
    p = spec["params"]; s = spec["scale"]
    f = lambda z: apply_head(p, z[None, :])[0] * s
    return np.asarray(jax.vmap(jax.jacfwd(f))(jnp.asarray(Z)))


def head_eval(spec, Z):
    return np.asarray(apply_head(spec["params"], jnp.asarray(Z))) * spec["scale"]


def jac_condition(spec, Z):
    """sigma_min / sigma_max of J_h at every row of Z (gate D2)"""
    J = head_jac(spec, Z)
    s = np.linalg.svd(J, compute_uv=False)
    return s[:, -1] / s[:, 0]


# ----------------------------- oracle & manifold gates -----------------------------

def oracle(spec, C_target, n_starts=8, iters=400, seed=11):
    """min_z ||c - h(z)|| per row; returns (Z* (S,K), coefficient residual (S,))"""
    return oracle_fit(spec, C_target, n_starts=n_starts, iters=iters, seed=seed)


def traj_rms_from_coeffs(res_coef, perp2, C_ref, traj):
    """traj-RMS of the reconstruction error per trajectory, from coefficient residuals and the
    truncation perp (M-norm squared):  sqrt(mean_t (r_t^2 + perp_t^2)) / sqrt(mean_t ||u_t||_M^2),
    with ||u_t||_M^2 = ||c_t||^2 + perp_t^2."""
    out = []
    for t in np.unique(traj):
        idx = traj == t
        num = np.mean(res_coef[idx] ** 2 + perp2[idx])
        den = np.mean(np.sum(C_ref[idx] ** 2, axis=1) + perp2[idx])
        out.append(np.sqrt(num / den))
    return np.array(out)


def pod_k_traj_rms(C, perp2, traj, K):
    """POD-K floor on the same rows: residual = coefficients beyond K plus the perp"""
    res = np.sqrt(np.sum(C[:, K:] ** 2, axis=1))
    return traj_rms_from_coeffs(res, perp2, C, traj)


def tangent_velocity_residual(spec, Zstar, CV, perpv2, K):
    """G0b: || (I - P_T) v ||_M / ||v||_M at oracle points, P_T the M-orthogonal projector onto
    range(G J_h) = G . P_J . G^T M  (G is M-orthonormal); and the POD-K value on the same states.
    CV = G^T M v (S,R), perpv2 = ||v - G CV||_M^2 (S,)."""
    J = head_jac(spec, Zstar)                                  # (S,R,K)
    out = np.zeros(len(Zstar)); ranks = np.zeros(len(Zstar), int)
    for s in range(len(Zstar)):
        Q, Rr = np.linalg.qr(J[s])
        # rank-revealing: drop columns whose R diagonal is below 1e-10 of the largest
        d = np.abs(np.diag(Rr)); keep = d >= 1e-10 * d.max()
        Qk = Q[:, keep]; ranks[s] = keep.sum()
        cv = CV[s]
        out[s] = np.sqrt(np.sum((cv - Qk @ (Qk.T @ cv)) ** 2) + perpv2[s])
    vnorm = np.sqrt(np.sum(CV ** 2, axis=1) + perpv2)
    pod = np.sqrt(np.sum(CV[:, K:] ** 2, axis=1) + perpv2) / vnorm
    return out / vnorm, pod, ranks


# ----------------------------- persistence (this cell's own format) -----------------------------

def save_spec(path, spec, extra=None):
    import json
    p = spec["params"]
    d = {f"W{i}": np.asarray(w) for i, (w, _) in enumerate(p["mlp"])}
    d.update({f"b{i}": np.asarray(b) for i, (_, b) in enumerate(p["mlp"])})
    d["skip"] = np.asarray(p["skip"]); d["Z"] = np.asarray(spec["Z"])
    if spec.get("Zdot") is not None:
        d["Zdot"] = np.asarray(spec["Zdot"])
    meta = {k: spec[k] for k in ("scale", "K", "R", "mode", "hidden", "layers", "steps", "lr", "batch",
                                 "seed", "n_fit", "final_loss", "seconds", "n_params", "vc_weight")}
    meta.update(dict(extra or {}))
    np.savez(path, meta=json.dumps(meta), **d)
    return meta


def load_spec(path, expect=None):
    """load a saved head iff every field in `expect` matches its recorded meta (it is a cache)"""
    import json, os
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    for k, v in (expect or {}).items():
        if meta.get(k) != v:
            return None
    nl = meta["layers"] + 1
    mlp = [(jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(nl)]
    spec = dict(params=dict(mlp=mlp, skip=jnp.asarray(d["skip"])), Z=d["Z"],
                Zdot=(d["Zdot"] if "Zdot" in d.files else None), from_cache=True)
    spec.update(meta)
    return spec


def duplicated_coordinate_control(spec):
    """a head with one extra, UNUSED latent coordinate: J_h has a zero column, so its
    conditioning must read 0 (negative control for gate D2)"""
    p = spec["params"]
    W0, b0 = p["mlp"][0]
    mlp = [(jnp.concatenate([W0, jnp.zeros((1, W0.shape[1]), dtype=F64)], 0), b0)] + list(p["mlp"][1:])
    skip = jnp.concatenate([p["skip"], jnp.zeros((1, p["skip"].shape[1]), dtype=F64)], 0)
    return dict(spec, params=dict(mlp=mlp, skip=skip), K=spec["K"] + 1)


def random_tangent_residual(CV, perpv2, K, seed=0):
    """G0b negative control: a random Gaussian K-dim tangent space must leave ~all of v unexplained"""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(CV))
    for s in range(len(CV)):
        Q, _ = np.linalg.qr(rng.normal(size=(CV.shape[1], K)))
        cv = CV[s]
        out[s] = np.sqrt(np.sum((cv - Q @ (Q.T @ cv)) ** 2) + perpv2[s])
    return out / np.sqrt(np.sum(CV ** 2, axis=1) + perpv2)
