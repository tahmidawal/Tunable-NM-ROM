"""Parity of the new head `validate` (matmul + top_k) against the hires-heat form at small random shapes."""
import sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core as C, train as T
S, V, R, K, W = 300, 40, 24, 4, 32
ks = jax.random.split(jax.random.PRNGKey(1), 4)
p = dict(net=T.mlp_init(ks[0], [K, W, W, R]), skip=jax.random.normal(ks[1], (K, R)) * .1)
z = jax.random.normal(ks[2], (S, K)); vt = C.mlp_head(p, jax.random.normal(ks[3], (V, K))) + 1e-2
vn = jnp.sum(vt * vt, 1) * 1.01; vp = vn * .01 / 1.01
a = np.asarray(T.make_validate(K, R, True)(p, z, vt, vn, vp)); b = np.asarray(T.make_validate(K, R, False)(p, z, vt, vn, vp))
print('max abs diff', float(np.max(np.abs(a - b))), 'max', float(a.max()))
