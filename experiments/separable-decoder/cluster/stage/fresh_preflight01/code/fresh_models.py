"""Fresh learned coordinate bank and generic coefficient heads.

Quadratic formula independently reimplemented from the approved non-wave reference;
source provenance is recorded in ARCHITECTURE-SOURCE.json. No wave/Burgers weights.
"""
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np


def dense_init(key, nin, nout, scale=1.):
    return {"w": scale*jax.random.normal(key, (nin, nout), dtype=jnp.float64)/np.sqrt(nin), "b": jnp.zeros(nout, dtype=jnp.float64)}


def dense(p, x):
    return x@p["w"]+p["b"]


def bank_init(key, rank=64, width=128):
    k1, k2, k3 = jax.random.split(key, 3)
    # Fixed coordinate lift, not a data-derived spatial subspace.
    freq = np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4) if i > 0 or (i == 0 and j > 0)], dtype=np.float64)
    nin = 2+2*len(freq)
    return {"l1": dense_init(k1, nin, width), "l2": dense_init(k2, width, width), "out": dense_init(k3, width, rank)}, jnp.asarray(freq)


def bank_apply(p, frequency, xy, reflective):
    phase = 2*jnp.pi*(xy@frequency.T)
    lift = jnp.concatenate((2*xy-1, jnp.sin(phase), jnp.cos(phase)), axis=-1)
    h = jax.nn.silu(dense(p["l1"], lift))
    h = jax.nn.silu(dense(p["l2"], h))
    g = dense(p["out"], h)
    if reflective:
        g = g*(jnp.sin(jnp.pi*xy[..., 0])*jnp.sin(jnp.pi*xy[..., 1]))[..., None]
    return g


def head_init(key, linear, center, output_scale, kind, width=128):
    r, k = linear.shape
    p = {"linear": jnp.asarray(linear), "bias": jnp.asarray(center)}
    if kind == "quadratic":
        p["quadratic"] = jnp.zeros((k*(k+1)//2, r), dtype=jnp.float64)
    elif kind == "mlp":
        k1, k2 = jax.random.split(key)
        p["l1"] = dense_init(k1, k, width)
        p["l2"] = dense_init(k2, width, width)
        p["out"] = {"w": jnp.zeros((width, r), dtype=jnp.float64), "b": jnp.zeros(r, dtype=jnp.float64)}
    else:
        raise ValueError(kind)
    return p, {"output_scale": jnp.asarray(output_scale)}


def head_apply(p, frozen, z, kind):
    value = p["bias"]+z@p["linear"].T
    if kind == "quadratic":
        ii, jj = jnp.triu_indices(z.shape[-1])
        value = value+frozen["output_scale"]*((z[..., ii]*z[..., jj])@p["quadratic"])
    elif kind == "mlp":
        h = jax.nn.silu(dense(p["l1"], z))
        h = jax.nn.silu(dense(p["l2"], h))
        value = value+frozen["output_scale"]*dense(p["out"], h)
    else:
        raise ValueError(kind)
    return value


def head_geometry(p, frozen, z, w, kind):
    fun = lambda zz: head_apply(p, frozen, zz, kind)
    a, b = jax.jvp(fun, (z,), (w,))
    jac = jax.jacfwd(fun)(z)
    # w is held constant in the inner JVP; this is H_h(z)[w,w].
    curvature = jax.jvp(lambda zz: jax.jvp(fun, (zz,), (w,))[1], (z,), (w,))[1]
    return a, b, jac, curvature


def tree_to_npz(path, tree):
    leaves = {}
    def visit(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, prefix+"/"+k if prefix else k)
        else:
            leaves[prefix] = np.asarray(obj)
    visit(tree, "")
    np.savez_compressed(path, **leaves)


def tree_from_npz(path):
    tree = {}
    with np.load(path) as f:
        for name in f.files:
            bits = name.split("/")
            obj = tree
            for bit in bits[:-1]:
                obj = obj.setdefault(bit, {})
            obj[bits[-1]] = jnp.asarray(f[name])
    return tree


def count_parameters(tree):
    return int(sum(x.size for x in jax.tree.leaves(tree)))
