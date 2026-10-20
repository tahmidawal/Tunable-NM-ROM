# jcp-mechanism diagnostic (2026-10-08, local GB10 via jaxrun): FD convergence of the 2D bank's analytic x+y derivative
# at 32 interior nodes of 1024^2, acc (R'=384). Output recorded in diagnostics/g1_fd_convergence_2d.txt. Motivates DESIGN A6-1b.
import sys, numpy as np, jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/q2d')
import qcore as Q
mdl = Q.Model()
X, w = Q.offmesh_rule('nodes', 1024)
sel = np.linspace(0, len(X) - 1, 32).astype(int)
Rp = 384
_, gx, gy = mdl.values_grads(X[sel], Rp)
b = np.asarray(gx + gy)
for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
    fd = 0.
    for e_ in (np.array([1., 0.]), np.array([0., 1.])):
        fd = fd + (np.asarray(mdl.values_grads(X[sel] + h * e_, Rp)[0]) - np.asarray(mdl.values_grads(X[sel] - h * e_, Rp)[0])) / (2 * h)
    err = np.abs(fd - b)
    i = np.unravel_index(err.argmax(), err.shape)
    print(h, err.max() / np.abs(b).max(), i, X[sel][i[0]])
