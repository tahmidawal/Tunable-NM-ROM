"""Patch a STAGED burgers2d_film.py: precondition the truth generator's inner
BiCGStab with the exact sine-basis Helmholtz inverse (I + dt*nu*(-lap_h))^-1
on the interior (boundary rows of the Jacobian are identity).

WHY: at N=1024 the unpreconditioned BiCGStab (LIN_TOL 1e-10, maxiter 2000)
fails/stalls (job 2825735: max Newton rel residual 8.67e-2 -- data gen
aborts).  The DISCRETE RESIDUAL, Newton acceptance guard, and the <=1e-8
truth-convergence check are unchanged: the residual check remains the sole
arbiter of what counts as truth, the preconditioner only changes how the
linear solve reaches it.  Same landmine class the project hit at N=2048
(exact-Helmholtz reference routes).
"""
import sys

p = sys.argv[1]
src = open(p).read()

anchor_def = "def make_rollout(n):"
i = src.index(anchor_def)
anchor_dx = '    dx = 1.0 / (n - 1)\n'
j = src.index(anchor_dx, i)
setup = '''    dx = 1.0 / (n - 1)
    # exact-Helmholtz preconditioner setup (see patch_bf_precond.py):
    # orthonormal interior sine basis S_pc and Dirichlet Laplacian eigenvalues
    _pp = np.arange(1, n - 1)
    S_pc = jnp.asarray(np.sqrt(2.0 / (n - 1))
                       * np.sin(np.pi * np.outer(_pp, _pp) / (n - 1)))
    _l1 = (4.0 / dx**2) * np.sin(np.pi * _pp / (2 * (n - 1))) ** 2
    lam_pc = jnp.asarray(_l1[:, None] + _l1[None, :])
'''
src = src[:j] + setup + src[j + len(anchor_dx):]

old_call = '''            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER)'''
new_call = '''            def Minv(v):
                V = v.reshape(n, n)
                C = S_pc.T @ V[1:-1, 1:-1] @ S_pc
                out = V.at[1:-1, 1:-1].set(
                    S_pc @ (C / (1.0 + DT * nu * lam_pc)) @ S_pc.T)
                return out.reshape(-1)
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=LIN_TOL, maxiter=LIN_MAXITER, M=Minv)'''
assert old_call in src, "bicgstab anchor missing"
assert src.count(old_call) == 1
src = src.replace(old_call, new_call)
open(p, "w").write(src)
print(f"patched exact-Helmholtz preconditioner into {p}")
