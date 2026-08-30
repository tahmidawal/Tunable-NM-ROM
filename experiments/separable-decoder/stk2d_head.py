"""PHASE 2b: the nonlinear head h: R^K -> R^R, its training, and the
RECONSTRUCTION ORACLE.  JAX, f64, GPU.

The decoder is  u(z) = ubar + G h(z)  with G the phase-2a M_u-orthonormal
divergence-free bank.  Because G is orthonormal in the mass metric the whole
training problem lives in COEFFICIENT space:

    ||u - ubar - G h(z)||_M^2 = ||c - h(z)||_2^2 + ||perp||_M^2,

so the head is trained against the coefficient vectors c(mu) = Cd theta(mu) -
cbar and the POD-R truncation floor ||perp||_M is added back, unchanged, when
the field error is reported.  Nothing about the metric is approximated by
this: it is an identity, and gate S-METRIC asserts it against a direct field
computation rather than assuming it.

TWO TRAINING FORMS, both run, the primary chosen on a VALIDATION cohort that is
disjoint from both the fit cohort and the frozen held-out cohort:

  "auto"  AUTODECODER -- joint Adam over the head parameters and one free
          latent code per training sample, no encoder.  This is the convention
          this project already uses (`sep_common.train_autodecoder`).
  "sup"   SUPERVISED -- the latent is pinned to the (affinely rescaled) family
          parameter, z = 2 mu - 1, and the head is fitted as a regression
          mu -> c.  Legitimate because the ROM never uses mu at solve time:
          the head is a fixed map R^K -> R^R and z is recovered by minimising
          the residual.  Pinning the latent removes the code/parameter
          coupling, which is what limits "auto" here.

OPTIONAL LATENT FOURIER FEATURES.  zf = [z, sin(2 pi z B), cos(2 pi z B)].  The
precedent is this project's own round-5 finding that "the codes are already
converged and h's own FUNCTION CLASS is what limits the fit"
(`sep_common.init_separable`, `h_ff`).  Whether they help HERE is measured on
the validation cohort, not assumed.

RECONSTRUCTION ORACLE: for a held-out c, the oracle is min_z ||c - h(z)||,
solved by multi-start damped Levenberg-Marquardt, vmapped over (case x restart)
and scanned over iterations so the whole cohort is one jitted call.  It is the
best the decoder can do on that sample, so it upper-bounds what any ROM built
on this decoder can achieve, and the predeclared stop gate is stated in it.

The training array is passed as an explicit `jit` ARGUMENT everywhere.  Closing
a jit over it embeds it as a compile-time constant, which has cost this project
a job before (CLAUDE.md).
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                          # noqa: E402
import optax                                                     # noqa: E402

F64 = jnp.float64


# ------------------------------------------------------------------ model ---

def init_mlp(key, sizes):
    ps = []
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (sizes[i], sizes[i + 1]), dtype=F64)
                   * jnp.sqrt(2.0 / sizes[i]),
                   jnp.zeros((sizes[i + 1],), dtype=F64)))
    return ps


def init_head(key, K, R, hidden=128, layers=3, ff=0, ff_scale=1.0,
              skip_scale=0.3):
    k1, k2, k3 = jax.random.split(key, 3)
    K_in = K + 2 * int(ff)
    sizes = [K_in] + [hidden] * layers + [R]
    p = dict(mlp=init_mlp(k1, sizes),
             skip=jax.random.normal(k2, (K, R), dtype=F64) * skip_scale)
    if ff:
        p["B"] = jax.random.normal(k3, (K, int(ff)), dtype=F64) * float(ff_scale)
    return p, sizes


def feat(params, z):
    if "B" in params:
        a = 2.0 * jnp.pi * (z @ params["B"])
        return jnp.concatenate([z, jnp.sin(a), jnp.cos(a)], axis=-1)
    return z


def apply_head(params, z):
    x = feat(params, z)
    for w, b in params["mlp"][:-1]:
        x = jax.nn.silu(x @ w + b)
    w, b = params["mlp"][-1]
    return x @ w + b + z @ params["skip"]


def param_count(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# --------------------------------------------------------------- training ---

def train_head(Cfit, K, mode="sup", MU=None, hidden=128, layers=3,
               steps=40000, lr=3e-3, batch=2048, ff=0, ff_scale=1.0, seed=0,
               log_every=0, tag=""):
    """Fit the head.  Cfit is (S, R) f64 in coefficient space; MU is (S, K) in
    [0,1]^K and is REQUIRED for mode='sup'.  Returns a HeadSpec dict."""
    Cfit = np.asarray(Cfit, dtype=float)
    S, R = Cfit.shape
    scale = float(np.sqrt((Cfit ** 2).mean()))
    Y = jnp.asarray(Cfit / scale)
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    params, sizes = init_head(k1, K, R, hidden, layers, ff, ff_scale)
    if mode == "sup":
        assert MU is not None and MU.shape == (S, K), "mode='sup' needs MU"
        Z = jnp.asarray(2.0 * np.asarray(MU, dtype=float) - 1.0)
        free_codes = False
    elif mode == "auto":
        Z = 0.1 * jax.random.normal(k2, (S, K), dtype=F64)
        free_codes = True
    else:                                                   # pragma: no cover
        raise ValueError(mode)

    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, min(500, steps // 10 + 1), steps, lr * 1e-3)
    opt = optax.adam(sched)
    tgt = (params, Z) if free_codes else params
    state = opt.init(tgt)

    if free_codes:
        def loss_fn(pz, y, idx):                # y, idx are ARGUMENTS
            p, z = pz
            e = apply_head(p, z[idx]) - y
            return jnp.mean(e * e) / jnp.mean(y * y)
    else:
        def loss_fn(p, y, z):                   # y, z are ARGUMENTS
            e = apply_head(p, z) - y
            return jnp.mean(e * e) / jnp.mean(y * y)

    @jax.jit
    def step(t, st, a, b):
        v, gr = jax.value_and_grad(loss_fn)(t, a, b)
        upd, st = opt.update(gr, st)
        return optax.apply_updates(t, upd), st, v

    rng = np.random.default_rng(seed + 1)
    nb = int(batch) if batch and batch < S else S
    t0 = time.time()
    v = np.inf
    for i in range(steps):
        if nb < S:
            idx = rng.integers(0, S, nb)
            if free_codes:
                tgt, state, v = step(tgt, state, Y[idx], jnp.asarray(idx))
            else:
                tgt, state, v = step(tgt, state, Y[idx], Z[idx])
        else:
            if free_codes:
                tgt, state, v = step(tgt, state, Y, jnp.arange(S))
            else:
                tgt, state, v = step(tgt, state, Y, Z)
        if log_every and ((i + 1) % log_every == 0 or i == 0):
            print(f"   train[{tag}] {i+1:6d}/{steps} rel-MSE {float(v):.3e} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    params, Zf = (tgt if free_codes else (tgt, Z))
    spec = dict(params=params, scale=scale, K=int(K), R=int(R), mode=mode,
                hidden=int(hidden), layers=int(layers), ff=int(ff),
                ff_scale=float(ff_scale), steps=int(steps), lr=float(lr),
                batch=int(nb), seed=int(seed), n_fit=int(S), sizes=sizes,
                final_rel_mse=float(v), seconds=float(time.time() - t0),
                n_params=param_count(params), Z=np.asarray(Zf))
    return spec


def save_head(path, spec, extra=None):
    d = {f"W{i}": np.asarray(w) for i, (w, _) in enumerate(spec["params"]["mlp"])}
    d.update({f"b{i}": np.asarray(b)
              for i, (_, b) in enumerate(spec["params"]["mlp"])})
    d["skip"] = np.asarray(spec["params"]["skip"])
    if "B" in spec["params"]:
        d["B"] = np.asarray(spec["params"]["B"])
    d["Z"] = np.asarray(spec["Z"])
    meta = {k: spec[k] for k in ("scale", "K", "R", "mode", "hidden", "layers",
                                 "ff", "ff_scale", "steps", "lr", "batch",
                                 "seed", "n_fit", "final_rel_mse", "seconds",
                                 "n_params")}
    # PROVENANCE.  A SMOKE run trains heads too, and a later certified run
    # that silently consumed one would be uncertified without saying so.  The
    # producing run's smoke flag and its whole training configuration travel
    # WITH the head, and stk2d_rom_gates.py asserts them.
    meta.update(dict(extra or {}))
    d["meta"] = np.array([repr(meta)], dtype=object)
    np.savez(path, **{k: v for k, v in d.items() if k != "meta"},
             meta=repr(meta))
    return meta


def load_head_jax(path, expect=None):
    """Load a saved head back into the JAX form `oracle_fit` needs, but ONLY if
    every recorded field matches `expect`.  Returns None otherwise.

    This is what makes a failed run cheap to resume: training a head is
    deterministic given its seed and configuration, so a head on disk whose
    entire configuration matches is the same object the run would have
    produced.  It is a CACHE, exactly like the snapshot cache -- deleting it
    reproduces everything from the seed -- and the driver records, per rung,
    whether the head was trained or loaded."""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        meta = eval(str(d["meta"]))
    except Exception:
        return None
    for k, v in (expect or {}).items():
        if meta.get(k) != v:
            return None
    nl = meta["layers"] + 1
    mlp = [(jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(nl)]
    params = dict(mlp=mlp, skip=jnp.asarray(d["skip"]))
    if "B" in d.files:
        params["B"] = jnp.asarray(d["B"])
    spec = dict(params=params, Z=d["Z"], from_cache=True)
    spec.update(meta)
    return spec


def load_head_np(path):
    """Load a saved head into the NUMPY form the timed paths use."""
    d = np.load(path, allow_pickle=True)
    meta = eval(str(d["meta"]))                 # a plain dict literal we wrote
    nl = meta["layers"] + 1
    return dict(layers=[d[f"W{i}"] for i in range(nl)],
                biases=[d[f"b{i}"] for i in range(nl)],
                skip=d["skip"], B=(d["B"] if "B" in d.files else None),
                scale=meta["scale"], meta=meta, Z=d["Z"])


def spec_to_np(spec):
    p = spec["params"]
    return dict(layers=[np.asarray(w) for w, _ in p["mlp"]],
                biases=[np.asarray(b) for _, b in p["mlp"]],
                skip=np.asarray(p["skip"]),
                B=(np.asarray(p["B"]) if "B" in p else None),
                scale=spec["scale"], meta={k: spec[k] for k in
                                           ("K", "R", "mode", "hidden",
                                            "layers", "ff", "n_params")},
                Z=np.asarray(spec["Z"]))


# --------------------------------------------------- the reconstruction oracle

def _lm_batch(params, Y, Z0, iters):
    def r_of(z, y):
        return apply_head(params, z[None, :])[0] - y

    def one(z, y, lam):
        r = r_of(z, y)
        J = jax.jacfwd(r_of)(z, y)
        H = J.T @ J
        g = J.T @ r
        K = z.shape[0]
        dz = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H))
                              + 1e-30 * jnp.eye(K, dtype=F64), -g)
        zn = z + dz
        rn = r_of(zn, y)
        ok = jnp.linalg.norm(rn) < jnp.linalg.norm(r)
        return (jnp.where(ok, zn, z),
                jnp.where(ok, jnp.maximum(lam / 3.0, 1e-14),
                          jnp.minimum(lam * 10.0, 1e14)))

    vone = jax.vmap(one, in_axes=(0, 0, 0))

    def body(carry, _):
        z, lam = carry
        return vone(z, Y, lam), None

    (Zf, _), _ = jax.lax.scan(
        body, (Z0, jnp.full((Z0.shape[0],), 1e-6, dtype=F64)), None,
        length=iters)
    res = jax.vmap(lambda z, y: jnp.linalg.norm(r_of(z, y)))(Zf, Y)
    return Zf, res


def oracle_fit(spec, Ctarget, n_starts=8, iters=400, seed=11,
               use_code_pool=True):
    """min_z ||c - h(z)|| for every row of Ctarget, by multi-start LM.

    Returns (best latent (S,K), best residual in COEFFICIENT units (S,)).  The
    restarts are a FIXED deterministic set -- the zero vector, uniform draws
    from the latent box the training codes occupy, and (optionally) randomly
    chosen training codes -- so the oracle is reproducible."""
    params, scale, K = spec["params"], spec["scale"], spec["K"]
    C = np.asarray(Ctarget, dtype=float) / scale
    S = C.shape[0]
    rng = np.random.default_rng(seed)
    Zp = np.asarray(spec["Z"])
    lo, hi = Zp.min(0), Zp.max(0)
    starts = [np.zeros((S, K))]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(lo + (hi - lo) * rng.random((S, K)))
    if use_code_pool and Zp.shape[0]:
        starts.append(Zp[rng.integers(0, Zp.shape[0], size=S)])
    Z0 = jnp.asarray(np.concatenate(starts, axis=0))
    Yb = jnp.asarray(np.tile(C, (len(starts), 1)))
    Zf, res = jax.jit(_lm_batch, static_argnums=(3,))(params, Yb, Z0, int(iters))
    Zf = np.asarray(Zf).reshape(len(starts), S, K)
    res = np.asarray(res).reshape(len(starts), S)
    j = res.argmin(axis=0)
    return Zf[j, np.arange(S), :], res[j, np.arange(S)] * scale


def np_to_jax(spec_np):
    """Rebuild the JAX parameter pytree from a loaded numpy head, so gate
    S-HEAD compares the two implementations rather than two copies of one."""
    mlp = [(jnp.asarray(w), jnp.asarray(b))
           for w, b in zip(spec_np["layers"], spec_np["biases"])]
    p = dict(mlp=mlp, skip=jnp.asarray(spec_np["skip"]))
    if spec_np.get("B") is not None:
        p["B"] = jnp.asarray(spec_np["B"])
    return p
