"""Compile-only smoke at the real valR256 shapes (S=12288 training codes, V=1536 validation targets, R=256, K=16, width 512).
Job 4141159 stalled >17 min in ptxas with the GPU idle right after the bank finished, i.e. while compiling train.train_head.
Usage: head_compile_smoke.py old|new  -> prints lower/compile seconds of the head `validate` function (nothing is executed)."""
import sys, time
from pathlib import Path
import jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core as C, train as T

S, V, R, K, W = 12288, 1536, 256, 16, 512
p = dict(net=T.mlp_init(jax.random.PRNGKey(0), [K, W, W, R]), skip=jnp.zeros((K, R)))
z = jnp.zeros((S, K)); vt = jnp.zeros((V, R)); vn = jnp.ones(V); vp = jnp.zeros(V)
fn = T.make_validate(K, R, sys.argv[1] == 'old')
t = time.time(); lo = fn.lower(p, z, vt, vn, vp); print('lowered', round(time.time() - t, 1), flush=True)
t = time.time(); lo.compile(); print(sys.argv[1], 'compiled', round(time.time() - t, 1), 's', flush=True)
