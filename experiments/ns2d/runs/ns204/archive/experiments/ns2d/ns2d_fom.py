"""ns2d_fom.py -- 2D incompressible Navier-Stokes, vorticity-streamfunction form,
periodic unit torus.  The certified full-order model of lane ns2d (DESIGN.md).

    omega_t = J(psi, omega) + nu Lap omega + f,     Lap psi = -omega,
    u = psi_y,  v = -psi_x                          (u . grad omega = -J(psi, omega))

DISCRETISATION (frozen, DESIGN.md "Discretisation"):
    grid       N x N nodes x_i = i h, h = 1/N, axis 0 = x (index i), axis 1 = y (index j)
    Laplacian  5-point periodic; exact eigenvalues on the 2D DFT
                   lambda_k = N^2 (4 - 2 cos(2 pi kx/N) - 2 cos(2 pi ky/N))   (of -Lap_h)
    Poisson    psi = -Lap_h^{-1} omega by FFT, mean mode zero (mean-zero gauge)
    Jacobian   Arakawa (1966)  J_A = (J++ + J+x + Jx+)/3: bilinear, antisymmetric,
               sum J_A = sum omega J_A = sum psi J_A = 0 identically
    time       implicit midpoint:  (w1 - w0)/dt = J_A(psi_m, w_m) + nu Lap_h w_m + f(t_mid),
               w_m = (w0 + w1)/2, psi_m = -Lap_h^{-1} w_m
    solver     tolerance-terminated Newton, matrix-free BiCGStab, preconditioned by the
               exact FFT Helmholtz inverse (I - dt nu/2 Lap_h)^{-1}   (engines.make_fom pattern)

Backward Euler (scheme='be') is provided ONLY as the negative control of gate F-BUDGET
(it must NOT conserve energy/enstrophy at nu = 0).  jac_sign=-1 is provided ONLY as the
negative control of gate F-MMS (the flipped advection sign must fail at O(1)).

All arrays f64.  Every large array is a jit ARGUMENT, never a closure constant.
"""
from __future__ import annotations

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                              # noqa: E402

F64 = jnp.float64
PI = np.pi


# ------------------------------------------------------------------ operators --

def lam_grid(N):
    """Eigenvalues of -Lap_h on the 2D DFT modes, shape (N, N), lam[0,0] = 0."""
    k = jnp.arange(N, dtype=F64)
    l1 = (2.0 - 2.0 * jnp.cos(2.0 * PI * k / N)) * (N * N)
    return l1[:, None] + l1[None, :]


def laplacian(w, N):
    """5-point periodic Laplacian of a (N, N) field."""
    return (N * N) * (jnp.roll(w, 1, 0) + jnp.roll(w, -1, 0)
                      + jnp.roll(w, 1, 1) + jnp.roll(w, -1, 1) - 4.0 * w)


def poisson(w, N):
    """psi = -Lap_h^{-1} w with zero mean (exact FFT inverse)."""
    lam = lam_grid(N).at[0, 0].set(1.0)
    W = jnp.fft.fft2(w) / lam
    W = W.at[0, 0].set(0.0)
    return jnp.real(jnp.fft.ifft2(W))


def velocity(psi, N):
    """u = psi_y, v = -psi_x by centred differences (reporting / CFL / energy)."""
    u = 0.5 * N * (jnp.roll(psi, -1, 1) - jnp.roll(psi, 1, 1))
    v = -0.5 * N * (jnp.roll(psi, -1, 0) - jnp.roll(psi, 1, 0))
    return u, v


def arakawa(psi, z, N):
    """Arakawa's J_A(psi, z) = psi_x z_y - psi_y z_x, second order, conserving.

    Shifts: p_x = psi[i+1, j] (roll -1 on axis 0), m_x = psi[i-1, j], etc."""
    r = jnp.roll
    px, mx, py, my = r(psi, -1, 0), r(psi, 1, 0), r(psi, -1, 1), r(psi, 1, 1)
    zx, zmx, zy, zmy = r(z, -1, 0), r(z, 1, 0), r(z, -1, 1), r(z, 1, 1)
    zpp, zpm = r(zx, -1, 1), r(zx, 1, 1)          # z[i+1, j+1], z[i+1, j-1]
    zmp, zmm = r(zmx, -1, 1), r(zmx, 1, 1)        # z[i-1, j+1], z[i-1, j-1]
    ppp, ppm = r(px, -1, 1), r(px, 1, 1)
    pmp, pmm = r(mx, -1, 1), r(mx, 1, 1)
    jpp = (px - mx) * (zy - zmy) - (py - my) * (zx - zmx)
    jpx = px * (zpp - zpm) - mx * (zmp - zmm) - py * (zpp - zmp) + my * (zpm - zmm)
    jxp = zy * (ppp - pmp) - zmy * (ppm - pmm) - zx * (ppp - ppm) + zmx * (pmp - pmm)
    return (N * N / 4.0) * (jpp + jpx + jxp) / 3.0


def jac_centred(psi, z, N):
    """Plain centred-difference Jacobian J++ alone: the NEGATIVE CONTROL of gate
    F-JAC (it does not conserve enstrophy)."""
    r = jnp.roll
    px, mx, py, my = r(psi, -1, 0), r(psi, 1, 0), r(psi, -1, 1), r(psi, 1, 1)
    zx, zmx, zy, zmy = r(z, -1, 0), r(z, 1, 0), r(z, -1, 1), r(z, 1, 1)
    return (N * N / 4.0) * ((px - mx) * (zy - zmy) - (py - my) * (zx - zmx))


def enstrophy(w, N):
    return 0.5 * jnp.sum(w * w) / (N * N)


def energy(w, N):
    """E = 1/2 h^2 sum psi omega, the quadratic invariant of the semi-discrete system."""
    return 0.5 * jnp.sum(poisson(w, N) * w) / (N * N)


def coords(N):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    return X, Y


# ------------------------------------------------------------------ stepper ----

def make_fom(N, dt, nsteps, out_every, f_fn=None, max_newton=20, jac_sign=1.0,
             scheme='midpoint', maxiter_lin=200):
    """Returns (run, step).

    run(w0, nu, ntol, ltol, t0=0.) -> (states (nsteps//out_every + 1, N, N),
                                      newton_iters (nsteps,), rel_resid (nsteps,))
    step(w, t, nu, ntol, ltol) -> (w_new, (iters, rel_resid))

    `f_fn(t)`, if given, returns the (N, N) source at time t (traced, closed form).
    Newton stops at ||F|| <= ntol ||w||; BiCGStab at relative `ltol`.
    """
    assert nsteps % out_every == 0
    assert scheme in ('midpoint', 'be')
    lam = lam_grid(N)
    half = 0.5 if scheme == 'midpoint' else 1.0
    nout = nsteps // out_every

    def rhs(wm, nu, t_eval):
        psi = poisson(wm, N)
        out = jac_sign * arakawa(psi, wm, N) + nu * laplacian(wm, N)
        if f_fn is not None:
            out = out + f_fn(t_eval)
        return out

    def residual(w1, w0, nu, t0):
        wm = half * w1 + (1.0 - half) * w0
        t_eval = t0 + half * dt
        return w1 - w0 - dt * rhs(wm, nu, t_eval)

    def helm(v, nu):
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(v) / (1.0 + dt * nu * half * lam)))

    def step(w0, t0, nu, ntol, ltol):
        scale = jnp.maximum(jnp.linalg.norm(w0), 1e-300)

        def cond(s):
            return (s[2] > ntol * scale) & (s[1] < max_newton) & jnp.isfinite(s[2])

        def body(s):
            w, it, rn = s
            r = residual(w, w0, nu, t0)

            def jv(v):
                return jax.jvp(lambda q: residual(q, w0, nu, t0), (w,), (v,))[1]

            delta, _ = jax.scipy.sparse.linalg.bicgstab(
                jv, -r, tol=ltol, maxiter=maxiter_lin, M=lambda v: helm(v, nu))
            cand = w + delta
            rn2 = jnp.linalg.norm(residual(cand, w0, nu, t0))
            return cand, it + 1, rn2

        r0 = jnp.linalg.norm(residual(w0, w0, nu, t0))
        w, it, rn = jax.lax.while_loop(cond, body, (w0, jnp.int32(0), r0))
        return w, (it, rn / scale)

    def run(w0, nu, ntol, ltol, t0=0.0):
        def block(carry, _):
            w, t = carry

            def inner(c, _):
                w, t = c
                w, aux = step(w, t, nu, ntol, ltol)
                return (w, t + dt), aux

            (w, t), (it, rn) = jax.lax.scan(inner, (w, t), None, length=out_every)
            return (w, t), (w, it, rn)

        _, (states, it, rn) = jax.lax.scan(block, (w0, jnp.asarray(t0, F64)), None,
                                           length=nout)
        return (jnp.concatenate((w0[None], states)), it.reshape(-1), rn.reshape(-1))

    return jax.jit(run), jax.jit(step)


# ---------------------------------------------------------------- the family ---

MODES = np.array([(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)], dtype=float)
NU_LO, NU_HI = 1e-3, 1e-2


def params_draw(seed, count):
    """Column-by-column draws (the params_draw landmine: drawing 4 cases != extending 2).
    Columns 0..5 = a_k, 6..11 = b_k (unit normals), 12 = nu (log-uniform)."""
    r = np.random.default_rng(seed)
    cols = [r.standard_normal(count) for _ in range(12)]
    cols.append(np.exp(r.uniform(np.log(NU_LO), np.log(NU_HI), count)))
    return np.stack(cols, axis=1)


def initial(N, phys):
    """omega_0 on the N-grid, rescaled analytically so U_rms = 1 (mesh-independent).
    U_rms^2 = sum_k (a_k^2 + b_k^2) / (8 pi^2 |k|^2)  for psi_k = omega_k / (4 pi^2 |k|^2)."""
    a, b = np.asarray(phys[:6], float), np.asarray(phys[6:12], float)
    X, Y = coords(N)
    w = np.zeros((N, N))
    for (kx, ky), ak, bk in zip(MODES, a, b):
        ph = 2.0 * PI * (kx * X + ky * Y)
        w += ak * np.cos(ph) + bk * np.sin(ph)
    k2 = (MODES ** 2).sum(1)
    urms = np.sqrt(np.sum((a * a + b * b) / (8.0 * PI * PI * k2)))
    return w / urms


def cfl_of(states, N, dt):
    """max |u| dt / h over the saved states (velocity from the discrete psi)."""
    def one(w):
        u, v = velocity(poisson(w, N), N)
        return jnp.max(jnp.sqrt(u * u + v * v))
    umax = jax.vmap(one)(jnp.asarray(states))
    return float(jnp.max(umax) * dt * N), np.asarray(umax)


# ------------------------------------------------------- analytic test cases ---

def taylor_green(N, amp=1.0):
    """omega_0 = amp sin(2 pi x) sin(2 pi y); psi = omega / lambda.  Returns
    (omega_0, lambda_h of this mode, continuum lambda 8 pi^2)."""
    X, Y = coords(N)
    w0 = amp * np.sin(2 * PI * X) * np.sin(2 * PI * Y)
    lam_h = 4.0 * N * N * (1.0 - np.cos(2 * PI / N))
    return w0, float(lam_h), float(8 * PI * PI)


def tg_discrete(w0, lam_h, nu, dt, n, scheme='midpoint'):
    """Closed-form FULLY discrete TG solution after n steps."""
    a = dt * nu * lam_h
    fac = (1 - a / 2) / (1 + a / 2) if scheme == 'midpoint' else 1.0 / (1 + a)
    return w0 * fac ** n


def mms_fields():
    """The manufactured solution of gate F-MMS (J != 0), as closed-form JAX scalars.

        omega = sin(2 pi x) cos(2 pi y) (1 + 0.5 sin(2 pi t)) + 0.5 cos(4 pi x) sin(2 pi y)
        psi   = -Lap^{-1} omega  (each mode divided by 4 pi^2 |k|^2)

    Returns (omega(x,y,t), psi(x,y,t), source(x,y,t,nu)) with the source
    f = omega_t - J(psi, omega) - nu Lap omega computed by forward-mode AD of the
    closed forms, so no derivative is hand-typed."""
    def omega(x, y, t):
        return (jnp.sin(2 * PI * x) * jnp.cos(2 * PI * y) * (1.0 + 0.5 * jnp.sin(2 * PI * t))
                + 0.5 * jnp.cos(4 * PI * x) * jnp.sin(2 * PI * y))

    def psi(x, y, t):
        return (jnp.sin(2 * PI * x) * jnp.cos(2 * PI * y) * (1.0 + 0.5 * jnp.sin(2 * PI * t))
                / (8 * PI * PI)
                + 0.5 * jnp.cos(4 * PI * x) * jnp.sin(2 * PI * y) / (20 * PI * PI))

    def source(x, y, t, nu):
        wt = jax.grad(omega, 2)(x, y, t)
        wx = jax.grad(omega, 0)(x, y, t)
        wy = jax.grad(omega, 1)(x, y, t)
        px = jax.grad(psi, 0)(x, y, t)
        py = jax.grad(psi, 1)(x, y, t)
        wxx = jax.grad(jax.grad(omega, 0), 0)(x, y, t)
        wyy = jax.grad(jax.grad(omega, 1), 1)(x, y, t)
        return wt - (px * wy - py * wx) - nu * (wxx + wyy)

    return omega, psi, source


def mms_on_grid(N, nu):
    """Grid-evaluated closed forms: omega_fn(t) -> (N,N), f_fn(t) -> (N,N)."""
    omega, psi, source = mms_fields()
    X, Y = coords(N)
    Xj, Yj = jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())
    vsrc = jax.vmap(source, in_axes=(0, 0, None, None))
    vom = jax.vmap(omega, in_axes=(0, 0, None))

    def f_fn(t):
        return vsrc(Xj, Yj, t, nu).reshape(N, N)

    def omega_fn(t):
        return vom(Xj, Yj, t).reshape(N, N)

    return omega_fn, f_fn


def rel(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def observed_order(errs):
    """Richardson-style observed orders from errors on a 2x-refined ladder."""
    e = np.asarray(errs, float)
    return (np.log(e[:-1] / e[1:]) / np.log(2.0)).tolist()
