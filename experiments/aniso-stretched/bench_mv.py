import time, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import aniso3d as a3, solve3d as s3
def tm(f, *a, rep=20):
    r = f(*a); jax.block_until_ready(r); t = time.perf_counter()
    for _ in range(rep): r = f(*a)
    jax.block_until_ready(r); return 1e3 * (time.perf_counter() - t) / rep
if __name__ == "__main__":
  for p, N in ((1, 64), (2, 32), (4, 16), (1, 128)):
      S = s3.Solver(N, p); n = S.n; U = jnp.asarray(np.random.default_rng(0).standard_normal((n, n, n))); K = jnp.asarray(a3.ktensor(a3.sample_params("dev", 1)[0]))
      mv = jax.jit(S.matvec); mvd = jax.jit(S.matvec_dense); print("conv vs dense %.1e" % float(jnp.abs(mv(K, U) - mvd(K, U)).max() / jnp.abs(mvd(K, U)).max())); pc = jax.jit(lambda K, r: S._prec("fd", K)(r))
      x1 = jax.jit(lambda U: S._x(S.M, U)); y1 = jax.jit(lambda U: S._y(S.M, U)); z1 = jax.jit(lambda U: S._z(S.M, U))
      ax = jax.jit(lambda U: U + 2 * U)
      print(p, N, n, "matvec %.3f ms  fdprec %.3f  x %.3f y %.3f z %.3f  axpy %.3f" % (tm(mv, K, U), tm(pc, K, U), tm(x1, U), tm(y1, U), tm(z1, U), tm(ax, U)), flush=True)
