"""fom_ext.py -- the lane's certified vorticity FOM (ns2d_fom.py, imported READ-ONLY for its operators:
lam_grid / laplacian / poisson / arakawa / velocity / energy / enstrophy) extended by two terms that the
candidate families need and the lane FOM cannot express (its mean-zero psi gauge has no mean flow):

    omega_t = J_A(psi, omega) + nu Lap_h omega  - (U omega_x + V omega_y)  + f(x, y, t; fp)

  * uniform background advection (U, V), centred differences (skew-symmetric: conserves sum omega^2)
  * Kolmogorov forcing in the vorticity equation, f = A cos(2 pi KF (y - V t) + phase), i.e. the
    forcing stripes move with the frame (a Galilean drift of a forced Kolmogorov flow).

Everything else (implicit midpoint, tolerance-terminated Newton, matrix-free BiCGStab with the exact
FFT Helmholtz preconditioner, output stride) is the lane's make_fom verbatim.  With uv = 0 and no
forcing the residual is the lane's residual with exact zeros added (checked in smoke.log against the
head investigation's data64.npz).  Per-trajectory parameters (nu, uv, fp) are jit/vmap ARGUMENTS."""
import sys
LANE = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d'
if LANE not in sys.path:
    sys.path.insert(0, LANE)
import numpy as np
import jax
import ns2d_fom as F                       # sets jax_enable_x64 = True
import jax.numpy as jnp

F64 = jnp.float64
PI = np.pi
KF = 4                                     # Kolmogorov forcing wavenumber (family C)


def make_fom_ext(N, dt, nsteps, out_every, forced=False, max_newton=20, maxiter_lin=200):
    """run(w0, nu, uv(2,), fp(3,)=[A, phase, Vdrift], ntol, ltol) ->
       (states (nsteps//out_every+1, N, N), newton_iters (nsteps,), rel_resid (nsteps,))"""
    assert nsteps % out_every == 0
    lam = F.lam_grid(N)
    half = 0.5
    nout = nsteps // out_every
    _, Y = F.coords(N)
    Yj = jnp.asarray(Y)

    def rhs(wm, nu, uv, fp, t_eval):
        psi = F.poisson(wm, N)
        out = F.arakawa(psi, wm, N) + nu * F.laplacian(wm, N)
        wx = 0.5 * N * (jnp.roll(wm, -1, 0) - jnp.roll(wm, 1, 0))     # axis 0 = x
        wy = 0.5 * N * (jnp.roll(wm, -1, 1) - jnp.roll(wm, 1, 1))     # axis 1 = y
        out = out - (uv[0] * wx + uv[1] * wy)
        if forced:
            out = out + fp[0] * jnp.cos(2.0 * PI * KF * (Yj - fp[2] * t_eval) + fp[1])
        return out

    def residual(w1, w0, nu, uv, fp, t0):
        wm = half * w1 + (1.0 - half) * w0
        return w1 - w0 - dt * rhs(wm, nu, uv, fp, t0 + half * dt)

    def helm(v, nu):
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(v) / (1.0 + dt * nu * half * lam)))

    def step(w0, t0, nu, uv, fp, ntol, ltol):
        scale = jnp.maximum(jnp.linalg.norm(w0), 1e-300)

        def cond(s):
            return (s[2] > ntol * scale) & (s[1] < max_newton) & jnp.isfinite(s[2])

        def body(s):
            w, it, rn = s
            r = residual(w, w0, nu, uv, fp, t0)

            def jv(v):
                return jax.jvp(lambda q: residual(q, w0, nu, uv, fp, t0), (w,), (v,))[1]

            delta, _ = jax.scipy.sparse.linalg.bicgstab(
                jv, -r, tol=ltol, maxiter=maxiter_lin, M=lambda v: helm(v, nu))
            cand = w + delta
            rn2 = jnp.linalg.norm(residual(cand, w0, nu, uv, fp, t0))
            return cand, it + 1, rn2

        r0 = jnp.linalg.norm(residual(w0, w0, nu, uv, fp, t0))
        w, it, rn = jax.lax.while_loop(cond, body, (w0, jnp.int32(0), r0))
        return w, (it, rn / scale)

    def run(w0, nu, uv, fp, ntol, ltol):
        def block(carry, _):
            w, t = carry

            def inner(c, _):
                w, t = c
                w, aux = step(w, t, nu, uv, fp, ntol, ltol)
                return (w, t + dt), aux

            (w, t), (it, rn) = jax.lax.scan(inner, (w, t), None, length=out_every)
            return (w, t), (w, it, rn)

        _, (states, it, rn) = jax.lax.scan(block, (w0, jnp.asarray(0.0, F64)), None, length=nout)
        return jnp.concatenate((w0[None], states)), it.reshape(-1), rn.reshape(-1)

    return jax.jit(run), jax.jit(jax.vmap(run, in_axes=(0, 0, 0, 0, None, None)))


# ---------------------------------------------------------------- numpy invariants (analysis) --
def energy_enstrophy_np(U, N):
    """U (S, N*N) -> (E (S,), Z (S,)) with the FOM's discrete definitions (numpy FFT Poisson)."""
    k = np.arange(N)
    l1 = (2.0 - 2.0 * np.cos(2.0 * np.pi * k / N)) * (N * N)
    lam = l1[:, None] + l1[None, :]
    lam[0, 0] = 1.0
    W = np.fft.fft2(U.reshape(-1, N, N)) / lam
    W[:, 0, 0] = 0.0
    psi = np.real(np.fft.ifft2(W)).reshape(U.shape[0], -1)
    E = 0.5 * (psi * U).sum(1) / (N * N)
    Z = 0.5 * (U * U).sum(1) / (N * N)
    return E, Z
