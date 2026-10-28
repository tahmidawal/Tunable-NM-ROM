#!/usr/bin/env python3
"""podmlp.py -- shared POD+MLP decoder library (VERBATIM COPY of
Tunable-NM-ROM-podmlp/experiments-pod-mlp-boundary/podmlp.py, the recipe that
passed on Heat and Poisson), with FOUR additive kwargs on train() and nothing
else changed: width= (was hardcoded WIDTH), seglen=, pod_cap=. The math,
initialisation, schedule, loss and optimiser are untouched.


The settled pod-floor-ae recipe, ported to flat finite-difference fields:

    x_hat(z) = m + z @ V^T + MLP(z_std)
               |___|  |____|   |________|
               mean   frozen   tiny-init pre-norm residual MLP correction
                      POD lift (starts at 0 -> model starts exactly at POD)

  - Encoder z = V^T (x - m): fixed top-k POD projection (f64 Gram).
  - Linear branch z @ V^T with V FROZEN at the POD optimum.
  - MLP: 3 pre-norm residual blocks (LayerNorm->Dense->GELU->Dense), width W,
    output tiny-init (1e-3/sqrt(W)) so training starts at exactly POD.

GN-regularity note: dX/dz|_z = V^T + dMLP/dz has full column rank k for ANY z
(V^T is orthonormal), so the decoder Jacobian is well-conditioned at cold start
z=0 -- the frozen POD lift IS the linear skip that plain-CP needed for GN.

Exposes:
  build_pod(X, k)                 -> (V (n,k) f32, m (n,) f32)
  init_dec / dec_corr             -> the MLP correction (params, forward)
  train(X, V, m, k, ...)          -> (params, znorm)   trains the correction
  make_decode(V, m, znorm, params)-> decode(z)->x  (pure, jittable, differentiable)
"""
import numpy as np
import jax, jax.numpy as jnp

EPS = 1e-7
WIDTH, DEPTH = 256, 3

# ---------------- POD basis (f64 Gram, snapshot method) -----------------------
def build_pod(X, k, max_snap=6000, seed=0):
    """Top-k POD via the f64 snapshot-covariance (M x M) method. X: (M, n) f32."""
    M, n = X.shape
    if M > max_snap:                       # a low-dim manifold needs few snapshots
        idx = np.random.default_rng(seed).choice(M, max_snap, replace=False)
        Xs = X[idx].astype(np.float64)
    else:
        Xs = X.astype(np.float64)
    m64 = Xs.mean(0)
    Xc = Xs - m64                          # (Ms, n) f64
    G = Xc @ Xc.T                          # (Ms, Ms) f64 Gram
    evals, evecs = np.linalg.eigh(G)       # ascending
    order = evals[::-1][:k]                # top-k
    vecs = evecs[:, ::-1][:, :k]
    order = np.maximum(order, 0.0)
    V = Xc.T @ vecs / (np.sqrt(order) + 1e-30)   # (n, k) f64, orthonormal columns
    # orthonormality check (should be ~1e-8 with f64 Gram)
    dev = float(np.abs(V.T @ V - np.eye(k)).max())
    return V.astype(np.float32), m64.astype(np.float32), dev

# ---------------- MLP correction (identical to dec2d) -------------------------
def _he(key, fi, fo, extra=1.0):
    return jax.random.normal(key, (fi, fo)) * (jnp.sqrt(2.0 / fi) * extra)

def _init_block(key, W, depth):
    k1, k2 = jax.random.split(key)
    return {"gamma": jnp.ones((W,)), "beta": jnp.zeros((W,)),
            "W1": _he(k1, W, W), "b1": jnp.zeros((W,)),
            "W2": _he(k2, W, W, extra=1.0/jnp.sqrt(2.0*depth)), "b2": jnp.zeros((W,))}

def _layernorm(ln, x):
    mu = jnp.mean(x, -1, keepdims=True); var = jnp.mean((x-mu)**2, -1, keepdims=True)
    return ln["gamma"]*((x-mu)/jnp.sqrt(var+1e-6)) + ln["beta"]

def _res_block(b, h):
    z = _layernorm({"gamma": b["gamma"], "beta": b["beta"]}, h)
    z = z @ b["W1"] + b["b1"]; z = jax.nn.gelu(z); z = z @ b["W2"] + b["b2"]
    return h + z

def init_dec(seed, k, n, W=WIDTH, depth=DEPTH):
    keys = jax.random.split(jax.random.PRNGKey(seed), 60); ki = [0]
    def nk():
        o = keys[ki[0]]; ki[0] += 1; return o
    return {"in": (_he(nk(), k, W), jnp.zeros((W,))),
            "blocks": [_init_block(nk(), W, depth) for _ in range(depth)],
            "final_ln": {"gamma": jnp.ones((W,)), "beta": jnp.zeros((W,))},
            "out": (jax.random.normal(nk(), (W, n))*(1e-3/jnp.sqrt(W)), jnp.zeros((n,)))}

def dec_corr(p, zstd):
    w, b = p["in"]; h = zstd @ w + b
    for bl in p["blocks"]: h = _res_block(bl, h)
    h = _layernorm(p["final_ln"], h); w, b = p["out"]; return h @ w + b

# ---------------- decode: pure function of z (for the solver) -----------------
def make_decode(V, m, znorm, params):
    """Return decode(z)->x_flat. z in raw POD coords. Differentiable & jittable."""
    Vt = jnp.asarray(V.T); mj = jnp.asarray(m)
    zmu = jnp.asarray(znorm[0]); zsd = jnp.asarray(znorm[1])
    def decode(z):                          # z: (k,) or (B,k)
        zstd = (z - zmu) / zsd
        return mj + z @ Vt + dec_corr(params, zstd)
    return decode

# ---------------- training the correction -------------------------------------
def _rel_l2(X, Xh):
    return float(np.mean(np.sqrt(np.sum((X-Xh)**2, 1)) / (np.sqrt(np.sum(X**2, 1)) + EPS)))

def train(X, V, m, k, steps=100000, batch=256, seed=0, lr0=1e-3, wd=1e-4, log=None,
          width=WIDTH, seglen=20000, pod_cap=3000):
    """Train the MLP correction (V, m frozen). X: (M,n) f32 training snapshots.
    Returns (params, znorm=(zmu,zsd)). Reports POD@k and comp rel-L2 on X."""
    import optax
    n = X.shape[1]
    Vj = jnp.asarray(V); mj = jnp.asarray(m); Vt = jnp.asarray(V.T)
    Xj = jnp.asarray(X)
    # POD projections + normalization
    Z = np.asarray((Xj - mj) @ Vj)                      # (M,k)
    zmu = Z.mean(0); zsd = Z.std(0) + 1e-8
    Zj = jnp.asarray(Z); Zsj = jnp.asarray((Z - zmu) / zsd)
    zmuj = jnp.asarray(zmu); zsdj = jnp.asarray(zsd)
    # POD reconstruction error (diagnostic) -- chunked so we never materialize the full
    # (M, n) rebuild (6.7GB at n=256^2; the un-chunked numpy version OOM-kills the box).
    Vt_np = np.asarray(V.T)
    def _chunked_rel_err(recon_chunk, cap=None):
        M_ = X.shape[0] if cap is None else min(X.shape[0], cap); num = []
        for i0 in range(0, M_, 500):
            Xb = X[i0:i0+500]; Xh = recon_chunk(i0, i0+500)
            num.append(np.sqrt((np.asarray(Xb-Xh)**2).sum(1)) / (np.sqrt((np.asarray(Xb)**2).sum(1))+EPS))
        return float(np.mean(np.concatenate(num)))
    pod_err = _chunked_rel_err(lambda a, b: Z[a:b] @ Vt_np + m, cap=pod_cap)

    p = init_dec(seed + k, k, n, W=width)
    def recon(p, Z, Zs): return mj + Z @ Vt + dec_corr(p, Zs)
    def loss(p, Z, Zs, Xb):
        xh = recon(p, Z, Zs)
        return jnp.mean(jnp.sqrt(jnp.sum((Xb-xh)**2, 1)) / (jnp.sqrt(jnp.sum(Xb**2, 1)) + EPS))
    M = Zj.shape[0]
    def sched(step):
        warm = jnp.minimum(step/1000.0, 1.0)
        frac = jnp.clip((step-1000)/jnp.maximum(steps-1000, 1), 0.0, 1.0)
        return warm * (1e-5 + 0.5*(lr0-1e-5)*(1+jnp.cos(jnp.pi*frac)))
    opt = optax.inject_hyperparams(optax.adamw)(learning_rate=lr0, weight_decay=wd)
    st = opt.init(p)
    # data passed as explicit jit ARGUMENTS, never closed over: a closure embeds the
    # training array as a compile-time constant (6.7GB at n=256^2 -> lowering chokes).
    def astep(p, st, rng, i, Zja, Zsja, Xja):
        rng, bk = jax.random.split(rng); idx = jax.random.randint(bk, (batch,), 0, M)
        st.hyperparams["learning_rate"] = sched(i)
        l, g = jax.value_and_grad(loss)(p, Zja[idx], Zsja[idx], Xja[idx])
        upd, st = opt.update(g, st, p); p = optax.apply_updates(p, upd); return p, st, rng, l
    import functools
    @functools.partial(jax.jit, static_argnums=(3,))
    def seg(p, st, rng, nseg, start, Zja, Zsja, Xja):
        def body(c, i):
            p, st, rng = c; p, st, rng, l = astep(p, st, rng, start+i, Zja, Zsja, Xja); return (p, st, rng), l
        (p, st, rng), ls = jax.lax.scan(body, (p, st, rng), jnp.arange(nseg)); return p, st, rng, ls
    rng = jax.random.PRNGKey(seed+11); seglen = min(seglen, steps); start = 0
    for _ in range(max(steps//seglen, 1)):
        p, st, rng, ls = seg(p, st, rng, seglen, start, Zj, Zsj, Xj)
        p = jax.tree_util.tree_map(lambda a: a.block_until_ready(), p); start += seglen
        if log: log(f"    step {start} loss {float(np.mean(np.asarray(ls[-200:]))):.3e}")
    # comp (reconstruction) error on X -- chunk-accumulated, never a full (M,n) materialization
    comp = _chunked_rel_err(lambda a, b: np.asarray(recon(p, Zj[a:b], Zsj[a:b])))
    return p, (zmu, zsd), {"pod": pod_err, "comp": comp}
