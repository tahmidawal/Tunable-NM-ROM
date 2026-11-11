import numpy as np, scipy.sparse.linalg as spla, time, sys
import aniso3d as a3, solve3d as s3, jax
print("backend", jax.default_backend())
mu = a3.sample_params("dev", 16)[0]; print("mu", mu)
for p, N in ((1, 8), (2, 6), (4, 3)):
    S = s3.Solver(N, p); A = S.sp.matrix(mu); b = S.sp.load(mu)
    u = spla.spsolve(A.tocsc(), b)
    U = np.random.default_rng(0).standard_normal((S.n,) * 3)
    mv = np.asarray(S.matvec(np.asarray(a3.ktensor(mu)), U)).ravel()
    for prec in ("none", "jacobi", "fd"):
        x, k, res = S.solve(mu, tol=1e-10, prec=prec)
        print(p, N, prec, "matvec err %.1e" % (abs(mv - A @ U.ravel()).max() / abs(mv).max()), "it", k, "sol err %.1e" % (abs(np.asarray(x).ravel() - u).max() / abs(u).max()))
