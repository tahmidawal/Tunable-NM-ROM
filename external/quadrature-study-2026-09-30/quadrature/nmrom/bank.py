"""Random-Fourier-feature coordinate-network bank with exact Dirichlet factor.

G(x) = mu(x) * MLP([cos(2 pi B x), sin(2 pi B x)]) @ T,   mu = 16 x(1-x) y(1-y).

Because G takes coordinates, it can be decoded at any point of the domain (partial
decoding): off-mesh quadrature rules only need G and grad G at their m nodes.
Trained by variable projection: minimise || U_S - Q Q^T U_S ||^2 over the network
weights, with Q an orthonormal basis of the bank restricted to a node subset S.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax


def dirichlet_factor(X):
    """mu(x) = prod_j 4 x_j (1 - x_j): vanishes on the boundary of the unit square / cube."""
    return jnp.prod(4.0 * X * (1.0 - X), axis=-1)


def init_bank(key, R, n_feat=32, scale=1.5, width=256, dim=2):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d_in = 2 * n_feat
    def lin(k, a, b):
        return jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), jnp.zeros(b)
    W1, b1 = lin(k2, d_in, width)
    W2, b2 = lin(k3, width, width)
    W3, b3 = lin(k4, width, R)
    return dict(B=scale * jax.random.normal(k1, (dim, n_feat)), W1=W1, b1=b1, W2=W2, b2=b2,
                W3=W3 * 0.1, b3=b3, T=jnp.eye(R))


def bank_eval(params, X):
    """Decode the bank at points X (P, dim) -> (P, R).  Dispatches on the parameter keys:
    'B' random-Fourier-feature MLP (paper's), 'omega' SIREN, 'centers' Gaussian RBF bank,
    'modes' fixed spectral sine bank, 'grid' POD modes with cubic-convolution interpolation,
    'grid_lin' POD modes with bilinear interpolation."""
    if "omega" in params:
        return _siren_eval(params, X)
    if "centers" in params:
        return _rbf_eval(params, X)
    if "modes" in params:
        return _spectral_eval(params, X)
    if "grid" in params:
        return _interp_eval(params, X, cubic=True)
    if "grid_lin" in params:
        return _interp_eval(params, X, cubic=False)
    return _rff_eval(params, X)


def _rff_eval(params, X):
    """X: (P, 2) -> (P, R)."""
    X = jnp.asarray(X)
    proj = 2 * jnp.pi * X @ params["B"]
    g = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
    hdn = jax.nn.silu(g @ params["W1"] + params["b1"])
    hdn = jax.nn.silu(hdn @ params["W2"] + params["b2"])
    out = (hdn @ params["W3"] + params["b3"]) @ params["T"]
    return dirichlet_factor(X)[:, None] * out


def bank_eval_grad(params, X):
    """Values and spatial gradients at points X: value (P,R) followed by one (P,R) array per
    coordinate (2 in 2D, 3 in 3D)."""
    X = jnp.asarray(X)
    dim = X.shape[1]
    f = lambda x: bank_eval(params, x[None])[0]
    def one(x):
        outs = []
        for j in range(dim):
            v, g = jax.jvp(f, (x,), (jnp.eye(dim)[j],))
            outs.append(g)
        return (v, *outs)
    return jax.vmap(one)(X)


def bank_eval_chunked(params, X, chunk=65536, grad=False):
    fn = jax.jit(bank_eval_grad if grad else bank_eval)
    outs = [fn(params, X[i:i + chunk]) for i in range(0, X.shape[0], chunk)]
    if grad:
        return tuple(np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(1 + X.shape[1]))
    return np.concatenate([np.asarray(o) for o in outs])


def train_bank(params, X_nodes, snaps, steps=8000, S=8192, B=128, lr=1e-3, seed=0, log_every=500):
    """snaps: (n_snap, n_nodes) float32; X_nodes: (n_nodes, 2)."""
    rng = np.random.default_rng(seed)

    def loss_fn(params, Xs, Us):
        G = bank_eval(params, Xs)
        Gram = G.T @ G + 1e-10 * jnp.eye(G.shape[1])
        Cf = jax.scipy.linalg.solve(Gram, G.T @ Us, assume_a="pos")
        Res = Us - G @ Cf
        return jnp.sum(Res**2) / jnp.sum(Us**2)

    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.02)
    opt = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def update(params, state, Xs, Us):
        l, g = jax.value_and_grad(loss_fn)(params, Xs, Us)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l

    n_nodes, n_snap = X_nodes.shape[0], snaps.shape[0]
    hist = []
    for it in range(steps):
        nodes = rng.choice(n_nodes, S, replace=False)
        batch = rng.choice(n_snap, min(B, n_snap), replace=False)
        Us = jnp.asarray(snaps[batch][:, nodes].T.astype(np.float64))
        params, state, l = update(params, state, jnp.asarray(X_nodes[nodes]), Us)
        if it % log_every == 0 or it == steps - 1:
            hist.append((it, float(l)))
            print(f"  bank step {it} rel. proj. loss {float(l):.3e}", flush=True)
    return params, hist


def orthonormalise(params, X_nodes):
    """Fold a right multiplier into params so that the bank is orthonormal on X_nodes."""
    G = bank_eval_chunked(params, X_nodes)
    _, Rq = np.linalg.qr(G)
    Tinv = np.linalg.inv(Rq)
    params = dict(params)
    params["T"] = jnp.asarray(np.asarray(params["T"]) @ Tinv)
    return params


# ----------------------------------------------------------------------------- alternatives
def init_siren(key, R, width=256, omega=30.0, dim=2):
    """SIREN coordinate network (Sitzmann et al. 2020): sine activations, first-layer scale omega."""
    k0, k1, k2 = jax.random.split(key, 3)
    W0 = jax.random.uniform(k0, (dim, width), minval=-1.0 / dim, maxval=1.0 / dim)
    b = np.sqrt(6.0 / width) / omega
    W1 = jax.random.uniform(k1, (width, width), minval=-b, maxval=b)
    W2 = jax.random.uniform(k2, (width, R), minval=-b, maxval=b) * 0.1
    return dict(W0=W0, b0=jnp.zeros(width), W1=W1, b1=jnp.zeros(width), W2=W2, b2=jnp.zeros(R),
                omega=jnp.asarray(omega), T=jnp.eye(R))


def _siren_eval(params, X):
    X = jnp.asarray(X)
    h = jnp.sin(params["omega"] * (X @ params["W0"] + params["b0"]))
    h = jnp.sin(params["omega"] * (h @ params["W1"] + params["b1"]))
    out = (h @ params["W2"] + params["b2"]) @ params["T"]
    return dirichlet_factor(X)[:, None] * out


def init_rbf(key, R, n_per_axis=16, sigma=0.06, dim=2):
    """Gaussian radial-basis bank: fixed centres on a grid, learnable width, learnable mixing."""
    c1 = (np.arange(n_per_axis) + 0.5) / n_per_axis
    C = np.stack(np.meshgrid(*([c1] * dim), indexing="ij"), -1).reshape(-1, dim)
    W = jax.random.normal(key, (C.shape[0], R)) / np.sqrt(C.shape[0])
    return dict(centers=jnp.asarray(C), log_sigma=jnp.asarray(np.log(sigma)), W=W, T=jnp.eye(R))


def _rbf_eval(params, X):
    X = jnp.asarray(X)
    d2 = jnp.sum((X[:, None, :] - params["centers"][None, :, :]) ** 2, -1)
    phi = jnp.exp(-0.5 * d2 / jnp.exp(2.0 * params["log_sigma"]))
    return dirichlet_factor(X)[:, None] * ((phi @ params["W"]) @ params["T"])


def init_spectral(R, dim=2):
    """Fixed bank of the R lowest tensor sine modes (no training; exact off-mesh evaluation)."""
    from .grid import mode_list
    modes = mode_list(R) if dim == 2 else None
    return dict(modes=jnp.asarray(modes, dtype=jnp.float64), T=jnp.eye(R))


def _spectral_eval(params, X):
    X = jnp.asarray(X)
    m = params["modes"]
    out = 2.0 * jnp.sin(m[None, :, 0] * jnp.pi * X[:, 0:1]) * jnp.sin(m[None, :, 1] * jnp.pi * X[:, 1:2])
    return out @ params["T"]


def init_pod_interp(snaps, N, R, cubic=True):
    """POD modes of the training snapshots on the training mesh, evaluated off-mesh by
    cubic-convolution (Keys, a = -1/2, C^1) or bilinear (C^0) interpolation with zero
    Dirichlet padding.  snaps: (S, (N-1)^2) float32."""
    A = snaps.astype(np.float64)
    G = A @ A.T
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:R]
    modes = (A.T @ V[:, idx]) / np.sqrt(w[idx])[None, :]          # (n, R), orthonormal
    grid = modes.T.reshape(R, N - 1, N - 1)
    key = "grid" if cubic else "grid_lin"
    return {key: jnp.asarray(grid), "N": jnp.asarray(float(N)), "T": jnp.eye(R)}


def _keys_weights(t):
    """Cubic convolution weights for offsets -1, 0, 1, 2 at fractional position t (a = -1/2)."""
    a = -0.5
    def k(s):
        s = jnp.abs(s)
        return jnp.where(s <= 1, (a + 2) * s**3 - (a + 3) * s**2 + 1,
                         jnp.where(s < 2, a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a, 0.0))
    return jnp.stack([k(t + 1), k(t), k(t - 1), k(t - 2)], -1)


def _interp_eval(params, X, cubic=True):
    X = jnp.asarray(X)
    grid = params["grid"] if cubic else params["grid_lin"]        # (R, N-1, N-1), interior values
    N = params["N"]
    pad = 2
    gp = jnp.pad(grid, ((0, 0), (pad, pad), (pad, pad)))          # zero Dirichlet outside
    s = X * N - 1.0                                                # interior index coordinates
    i0 = jnp.floor(s)
    t = s - i0
    i0 = i0.astype(jnp.int32) + pad
    if cubic:
        wx, wy = _keys_weights(t[:, 0]), _keys_weights(t[:, 1])
        offs = jnp.arange(-1, 3)
    else:
        wx = jnp.stack([1 - t[:, 0], t[:, 0]], -1); wy = jnp.stack([1 - t[:, 1], t[:, 1]], -1)
        offs = jnp.arange(0, 2)
    ii = i0[:, 0:1] + offs[None, :]                                # (P, 4)
    jj = i0[:, 1:2] + offs[None, :]
    vals = gp[:, ii[:, :, None], jj[:, None, :]]                   # (R, P, 4, 4)
    out = jnp.einsum("rpij,pi,pj->pr", vals, wx, wy)
    return out @ params["T"]


BANK_ARCHS = ("rff", "siren", "rbf", "spectral", "pod_cubic", "pod_linear")
