"""Latent head h(z) = MLP(z) + W^T z and nested correction directions."""
import numpy as np
import jax
import jax.numpy as jnp
import optax


def init_head(key, k, R, width=128):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    def lin(kk, a, b):
        return jax.random.normal(kk, (a, b)) * np.sqrt(2.0 / a), jnp.zeros(b)
    W1, b1 = lin(k1, k, width); W2, b2 = lin(k2, width, width); W3, b3 = lin(k3, width, R)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3 * 0.1, b3=b3,
                Wskip=jax.random.normal(k4, (k, R)) * np.sqrt(1.0 / k))


def head(params, z):
    h = jax.nn.silu(z @ params["W1"] + params["b1"])
    h = jax.nn.silu(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"] + z @ params["Wskip"]


def head_jac(params, z):
    return jax.jacfwd(lambda zz: head(params, zz))(z)


def train_head(params, codes, k, steps=20000, B=256, lr=2e-3, seed=0, reg=1e-4, log_every=2000):
    """Auto-decoder fit of h and per-snapshot latent codes Z to coefficients `codes` (S, R)."""
    rng = np.random.default_rng(seed)
    S, R = codes.shape
    mean = codes.mean(0)
    U, s, Vt = np.linalg.svd(codes - mean, full_matrices=False)
    Z0 = (codes - mean) @ Vt[:k].T
    Z0 = Z0 / Z0.std(0, keepdims=True).clip(1e-8)
    Z = jnp.asarray(Z0)
    cn = jnp.asarray(codes)
    norm = float(np.mean(np.sum(codes**2, 1)))

    def loss_fn(params, Z, idx):
        pred = head(params, Z[idx])
        return jnp.mean(jnp.sum((pred - cn[idx]) ** 2, 1)) / norm + reg * jnp.mean(jnp.sum(Z[idx] ** 2, 1))

    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.02)
    opt = optax.multi_transform({"p": optax.adam(sched), "z": optax.adam(sched)},
                                {"p": "p", "z": "z"})
    both = {"p": params, "z": Z}
    state = opt.init(both)

    @jax.jit
    def update(both, state, idx):
        l, g = jax.value_and_grad(lambda b: loss_fn(b["p"], b["z"], idx))(both)
        upd, state = opt.update(g, state, both)
        return optax.apply_updates(both, upd), state, l

    for it in range(steps):
        idx = jnp.asarray(rng.choice(S, min(B, S), replace=False))
        both, state, l = update(both, state, idx)
        if it % log_every == 0 or it == steps - 1:
            print(f"  head step {it} rel. loss {float(l):.3e}", flush=True)
    return both["p"], np.asarray(both["z"])


def correction_directions(params, Z, codes):
    """Nested directions: left singular vectors of the coefficient residuals."""
    E = codes - np.asarray(jax.vmap(lambda z: head(params, z))(jnp.asarray(Z)))
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    return Vt.T, s
