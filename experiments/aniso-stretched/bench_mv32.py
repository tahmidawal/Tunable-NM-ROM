import time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import aniso3d as a3, solve3d as s3
from bench_mv import tm
for dt in (jnp.float32, jnp.float64):
    for p, N in ((1, 64), (2, 32), (4, 16), (2, 24), (4, 8), (4, 12)):
        S = s3.Solver(N, p, dtype=dt); n = S.n; U = jnp.asarray(np.random.default_rng(0).standard_normal((n, n, n)), dt); K = jnp.asarray(a3.ktensor(a3.sample_params("dev", 1)[0]), dt)
        mv = jax.jit(S.matvec); pc = jax.jit(lambda K, r: S._prec("fd", K)(r)); pj = jax.jit(lambda K, r: S._prec("jacobi", K)(r))
        print(dt.__name__, p, N, n, "matvec %.3f ms  fdprec %.3f jac %.3f" % (tm(mv, K, U), tm(pc, K, U), tm(pj, K, U)), flush=True)
