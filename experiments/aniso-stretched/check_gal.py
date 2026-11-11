import numpy as np, scipy.sparse.linalg as spla
import aniso3d as a3, solve3d as s3, gal3d
S = s3.Solver(4, 2); n = S.n
Phi = np.linalg.qr(np.random.default_rng(0).standard_normal((n ** 3, n ** 3)))[0]
G = gal3d.Galerkin(S, Phi, m_load=n)
for mu in a3.sample_params("dev", 3):
    u = spla.spsolve(S.sp.matrix(mu).tocsc(), S.sp.load(mu)); c, t = G.timed(mu, 20)
    print("full-rank Galerkin vs direct: %.1e" % (abs(Phi @ c - u).max() / abs(u).max()), "load trunc", G.load_trunc, t)
