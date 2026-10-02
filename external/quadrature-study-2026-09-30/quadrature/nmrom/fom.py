"""Full-order solver: backward Euler / steady, Newton with BiCGStab preconditioned
by the exact Helmholtz inverse (I + dt (nu A - r)) via DST-I.  Matrix free in JAX."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab
from .grid import laplacian, dst1_2d, idst1_2d, interior_coords


def _helm_diag(N, alpha, beta):
    h = 1.0 / N
    k = jnp.arange(1, N)
    lam1 = (4.0 / h**2) * jnp.sin(k * jnp.pi * h / 2) ** 2
    return 1.0 + beta + alpha * (lam1[:, None] + lam1[None, :])


def make_fom(pde, N):
    """Returns (step, residual) jitted for mesh N."""
    steady = pde.steady

    def residual(u, u_n, dt, nu, r, f, p):
        Nl = pde.nonlinear_grid(u, N, p)
        if steady:
            return -nu * laplacian(u, N) + Nl - f
        return u - u_n + dt * (Nl - nu * laplacian(u, N) - r * u - f)

    def newton_iter(u, u_n, dt, nu, r, f, p, tol_lin):
        R = residual(u, u_n, dt, nu, r, f, p)
        Jv = lambda v: jax.jvp(lambda uu: residual(uu, u_n, dt, nu, r, f, p), (u,), (v,))[1]
        if steady:
            diag = _helm_diag(N, nu, 0.0) - 1.0   # nu*A (no identity)
        else:
            diag = _helm_diag(N, dt * nu, -dt * r)
        Minv = lambda v: idst1_2d(dst1_2d(v) / diag)
        d, _ = bicgstab(Jv, -R, tol=tol_lin, atol=0.0, maxiter=400, M=Minv)
        nR = jnp.linalg.norm(R)
        # backtracking on the residual norm (monotone safeguard)
        def body(carry):
            alpha, _ = carry
            return alpha * 0.5, jnp.linalg.norm(residual(u + 0.5 * alpha * d, u_n, dt, nu, r, f, p))
        def cond(carry):
            alpha, nRt = carry
            return jnp.logical_and(nRt > nR, alpha > 1e-3)
        alpha, _ = jax.lax.while_loop(cond, body, (1.0, jnp.linalg.norm(residual(u + d, u_n, dt, nu, r, f, p))))
        return u + alpha * d, nR, alpha

    newton_iter = jax.jit(newton_iter, static_argnames=())

    def step(u_n, dt, nu, r, f, p, tol=1e-9, maxit=12, tol_lin=1e-4, u_guess=None):
        u = u_n if u_guess is None else u_guess
        scale = max(float(jnp.linalg.norm(u_n)), 1e-12) if not steady else max(float(jnp.linalg.norm(f)), 1e-12)
        nits = 0
        for it in range(maxit):
            u_new, nR, nd = newton_iter(u, u_n, dt, nu, r, f, p, tol_lin)
            nits += 1
            if float(nR) <= tol * scale:
                return u, nits, float(nR) / scale
            u = u_new
        return u, nits, float(nR) / scale

    return step, jax.jit(residual)


def solve_case(pde, N, p, dt=None, tol=1e-9, verbose=False, store_every=1):
    """Solve one case on mesh N. Returns dict with states at output times (and
    all stored steps for time-dependent problems)."""
    X = interior_coords(N)
    f = jnp.asarray(pde.source(X, p).reshape(N - 1, N - 1))
    nu, r = pde.nu(p), pde.react(p)
    step, _ = make_fom(pde, N)
    t0 = time.time()
    if pde.steady:
        u0 = jnp.zeros((N - 1, N - 1))
        u, nits, res = step(u0, 1.0, nu, r, f, p, tol=tol, maxit=30, tol_lin=1e-6)
        return dict(u_out=np.asarray(u)[None], t_out=np.array([0.0]), states=np.asarray(u)[None],
                    t_states=np.array([0.0]), newton_its=[nits], time=time.time() - t0, p=p, N=N)
    dt = pde.dt if dt is None else dt
    nsteps = int(round(pde.T / dt))
    u = jnp.asarray(pde.u0(X, p).reshape(N - 1, N - 1))
    u_prev = None
    states, t_states = [np.asarray(u)], [0.0]
    u_out, t_out = [], []
    its = []
    out_steps = {int(round(t / dt)): t for t in pde.out_times}
    for n in range(1, nsteps + 1):
        guess = u if u_prev is None else 2 * u - u_prev
        u_new, nits, res = step(u, dt, nu, r, f, p, tol=tol, u_guess=guess)
        its.append(nits)
        u_prev, u = u, u_new
        if n % store_every == 0:
            states.append(np.asarray(u)); t_states.append(n * dt)
        if n in out_steps:
            u_out.append(np.asarray(u)); t_out.append(out_steps[n])
        if verbose and n % 10 == 0:
            print(f"  step {n}/{nsteps} newton its {nits} res {res:.2e}", flush=True)
    return dict(u_out=np.stack(u_out), t_out=np.array(t_out), states=np.stack(states),
                t_states=np.array(t_states), newton_its=its, time=time.time() - t0, p=p, N=N)
