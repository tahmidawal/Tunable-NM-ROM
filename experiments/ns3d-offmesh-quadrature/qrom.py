"""ns3d-offmesh-quadrature: the co-moving NS3D ROM with the advection contraction injected.

Everything here is a line-for-line copy of the ns3d-shift / ns3d-shift-head drivers used by the
ns3d-pod-match development ROM (`SR.make_frozen_step`, `H.make_head_run`, `SR.make_frozen_run`),
except that the Jacobian of the advection, J(c) = d a / d c (M_t, R), comes from `jac_adv`:

  kind="tensor"  J(c) = sym c,  sym = (T + T^T_23) / pre                     (the incumbent)
  kind="quad"    J(c) = sum_q P_q^T [ U_q x (W_q c) + (U_q c) x W_q ],  P_q = n^{-3/2} w_q psi(x_q) / pre

with a(c) = J(c) c / 2 in both (the residual only ever uses symmid @ mid / 2). U_q / W_q are the
continuum-normalised bank columns / curls at the rule points (3, R); psi the analytic tests.

Table builders:
  net_tables      U, W from the coordinate network (autodiff), times T_rot S_n  (mesh-free)
  interp_tables   U, W from the exact trigonometric interpolant of the mesh bank (direct NUDFT)
  shifted_grid_tables  the same interpolant on a shifted n^3 grid, by FFT phase shift (positive control)
  test_values     psi_m(x_q), (Mq, 3, M_t)

All large arrays are explicit jit arguments (never closed over).
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ns3d_fom as F
import shift_rom as SR
import head_rom as H
import coordnet as CN


# ------------------------------------------------------------------ the contraction --

def cross1(a, b):
    """Cross product over axis 1 (the 3-component axis) with broadcasting."""
    return jnp.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                      a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                      a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), axis=1)


def fold_adv(kind, Q, pre):
    if kind == "tensor":
        return (Q + jnp.swapaxes(Q, 1, 2)) / pre[:, None, None]
    U, W, P = Q
    return (U, W, P / pre[None, None, :])


def jac_adv(kind, adv, mid):
    if kind == "tensor":
        return jnp.einsum("msk,k->ms", adv, mid)
    U, W, P = adv
    u = jnp.einsum("qcr,r->qc", U, mid)
    om = jnp.einsum("qcr,r->qc", W, mid)
    X = cross1(U, om[:, :, None]) + cross1(u[:, :, None], W)          # (Mq, 3, R)
    return jnp.einsum("qcm,qcr->mr", P, X)


def advection(kind, Q, c):
    """a(c) (M_t,) without the CN preconditioner (pre = 1)."""
    pre = jnp.ones(Q.shape[0] if kind == "tensor" else Q[2].shape[2])
    return 0.5 * jac_adv(kind, fold_adv(kind, Q, pre), c) @ c


def jacobian(kind, Q, c):
    pre = jnp.ones(Q.shape[0] if kind == "tensor" else Q[2].shape[2])
    return jac_adv(kind, fold_adv(kind, Q, pre), c)


# ------------------------------------------------------- SR.make_frozen_step, injected --

def make_frozen_step(dt, n, r, iters, damping, kind):
    def prepare(nu, A, Q, lam, Dd):
        pre = 1.0 + 0.5 * dt * nu * lam
        Apre = A / pre[:, None]
        adv = fold_adv(kind, Q, pre)
        lamApre = (nu * lam / pre)[:, None] * A
        Ddpre = Dd / pre[None, :, None]
        return Apre, adv, Apre + 0.5 * dt * lamApre, lamApre, Ddpre

    def residual_and_jacobian(w, prev, ops):
        Apre, adv, jconst, lamApre, Ddpre = ops
        a, delta = w[:r], w[r:]
        mid = 0.5 * (a + prev)
        symmid = jac_adv(kind, adv, mid)
        shift_cols = jnp.einsum("dmr,r->md", Ddpre, mid)
        res = (Apre @ (a - prev) - 0.5 * dt * (symmid @ mid) + dt * (lamApre @ mid)
               - shift_cols @ delta)
        jac_a = jconst - 0.5 * dt * symmid - 0.5 * jnp.einsum("dmr,d->mr", Ddpre, delta)
        return res, jnp.concatenate((jac_a, -shift_cols), axis=1)

    def solve(w0, prev, ops):
        def body(w, _):
            res, jac = residual_and_jacobian(w, prev, ops)
            hessian = jac.T @ jac
            factor = jnp.linalg.cholesky(
                hessian + jnp.diag(damping * jnp.diagonal(hessian) + 1e-300))
            step = jax.scipy.linalg.cho_solve((factor, True), -(jac.T @ res))
            return w + step, jnp.linalg.norm(res)
        w, norms = jax.lax.scan(body, w0, None, length=iters, unroll=iters)
        return w, norms[-1]
    return solve, prepare, residual_and_jacobian


# ------------------------------------------------------------ the two queries --

def make_head_run(dt, nsteps, out_every, n, R, k, iters, damping, ic_iters, kind,
                  history=False, extrapolate=True):
    """H.make_head_run (q = 0, frame free) with the contraction injected.
    history=True also returns the head coefficients a at t0 and after every step (nsteps+1, R)."""
    assert nsteps % out_every == 0
    q = 0
    _, prepare, residual_and_jacobian = make_frozen_step(dt, n, R, iters, damping, kind)
    code_fit = H.make_code_fit(k, ic_iters)

    def coeff(p, w, C):
        return H.head_apply(p, w[:k])

    def residual_jac(w, a_prev, ops, p, C):
        a = coeff(p, w, C)
        res, jfull = residual_and_jacobian(jnp.concatenate((a, w[k + q:])), a_prev, ops)
        ja = jfull[:, :R]
        blocks = [ja @ H.head_jacobian(p, w[:k]), jfull[:, R:]]
        return res, jnp.concatenate(blocks, axis=1)

    def step_solve(w0, a_prev, ops, p, C):
        def body(w, _):
            res, jac = residual_jac(w, a_prev, ops, p, C)
            hess = jac.T @ jac
            L = jnp.linalg.cholesky(hess + jnp.diag(damping * jnp.diagonal(hess) + 1e-300))
            stepv = jax.scipy.linalg.cho_solve((L, True), -(jac.T @ res))
            return w + stepv, jnp.linalg.norm(res)
        w, norms = jax.lax.scan(body, w0, None, length=iters, unroll=iters)
        return w, norms[-1]

    @jax.jit
    def run(u0, nu, basis, A, Q, lam, Dd, p, C, Hcand, Hnorm, Zcand):
        ops = prepare(nu, A, Q, lam, Dd)
        c0 = SR.grid_centroid(u0)
        a0 = basis.T @ SR.shift_field(u0, -c0 * n).ravel()
        idx = jnp.argmin(Hnorm - 2 * (Hcand @ a0))
        z0, y0 = code_fit(p, Zcand[idx], a0, C)
        w0 = jnp.concatenate((z0, y0, jnp.zeros(3)))

        def step(carry, _):
            w, previous, c = carry
            guess = w + (w - previous) if extrapolate else w
            new, rn = step_solve(guess, coeff(p, w, C), ops, p, C)
            centre = (c + new[k + q:]) % 1.0
            return (new, w, centre), (coeff(p, new, C) if history else None)

        def block(carry, _):
            carry, info = jax.lax.scan(step, carry, None, length=out_every)
            w, _, c = carry
            field = SR.shift_field((basis @ coeff(p, w, C)).reshape(3, n, n, n), c * n)
            return carry, (field, info)

        frame0 = SR.shift_field((basis @ coeff(p, w0, C)).reshape(3, n, n, n), c0 * n)
        _, (fields, info) = jax.lax.scan(block, (w0, w0, c0), None,
                                         length=nsteps // out_every)
        fields = jnp.concatenate((frame0[None], fields))
        if not history:
            return fields
        hist = jnp.concatenate((coeff(p, w0, C)[None], info.reshape(nsteps, R)))
        return fields, hist
    return run


def make_span_run(dt, nsteps, out_every, n, r, iters, damping, kind, history=False,
                  extrapolate=True):
    """SR.make_frozen_run(diagnose=False) with the contraction injected."""
    assert nsteps % out_every == 0
    solve, prepare, _ = make_frozen_step(dt, n, r, iters, damping, kind)

    @jax.jit
    def run(u0, nu, basis, A, Q, lam, Dd):
        ops = prepare(nu, A, Q, lam, Dd)
        c0 = SR.grid_centroid(u0)
        a0 = basis.T @ SR.shift_field(u0, -c0 * n).ravel()
        w0 = jnp.concatenate((a0, jnp.zeros(3)))

        def step(carry, _):
            w, previous, c = carry
            guess = w + (w - previous) if extrapolate else w
            new, rn = solve(guess, w[:r], ops)
            centre = (c + new[r:]) % 1.0
            return (new, w, centre), (new[:r] if history else None)

        def block(carry, _):
            carry, info = jax.lax.scan(step, carry, None, length=out_every)
            w, _, c = carry
            return carry, (shift_field_out(basis, w[:r], c, n), info)

        frame0 = shift_field_out(basis, a0, c0, n)
        _, (fields, info) = jax.lax.scan(block, (w0, w0, c0), None,
                                         length=nsteps // out_every)
        fields = jnp.concatenate((frame0[None], fields))
        if not history:
            return fields
        return fields, jnp.concatenate((a0[None], info.reshape(nsteps, r)))
    return run


def shift_field_out(basis, a, c, n):
    return SR.shift_field((basis @ a).reshape(3, n, n, n), c * n)


# ------------------------------------------------------------------- tables --

def ids_arrays(ids):
    wave = np.asarray([row["wave"] for row in ids], dtype=np.float64)          # (Mt, 3)
    pol = np.asarray([row["polarization"] for row in ids], dtype=np.float64)    # (Mt, 3)
    cosine = np.asarray([row["kind"] == "cos" for row in ids])
    return wave, pol, cosine


@jax.jit
def _tests(X, wave, pol, cosine):
    angle = 2 * jnp.pi * (X @ wave.T)                                           # (P, Mt)
    s = jnp.where(cosine[None, :], jnp.cos(angle), jnp.sin(angle)) * np.sqrt(2.0)
    return s[:, None, :] * pol.T[None, :, :]                                    # (P, 3, Mt)


def test_values(X, ids, w, n):
    """P (Mq, 3, Mt) = n^{-3/2} w_q psi_m(x_q)."""
    wave, pol, cosine = ids_arrays(ids)
    psi = _tests(jnp.asarray(X), jnp.asarray(wave), jnp.asarray(pol), jnp.asarray(cosine))
    return psi * (jnp.asarray(w) * n ** -1.5)[:, None, None]


def _curl_from_gradient(g):
    """g (3e, P, 3c, R) with g[e, :, c] = d_e u_c  ->  curl (P, 3, R)."""
    return jnp.stack((g[1, :, 2] - g[2, :, 1], g[2, :, 0] - g[0, :, 2],
                      g[0, :, 1] - g[1, :, 0]), axis=1)


@jax.jit
def _net_block(p, X, TS):
    u = CN.velocity(p, X)                                                       # (P, 3, Rraw)
    om = _curl_from_gradient(CN.velocity_gradient(p, X))
    return jnp.einsum("pcr,rs->pcs", u, TS), jnp.einsum("pcr,rs->pcs", om, TS)


def net_tables(p, TS, X, chunk=8192):
    """U, W (Mq, 3, R) from the network: g(x) T_rot S_n and its curl (continuum normalised)."""
    TSj = jnp.asarray(TS)
    Us, Ws = [], []
    for s in range(0, len(X), chunk):
        u, om = _net_block(p, jnp.asarray(X[s:s + chunk]), TSj)
        Us.append(u)
        Ws.append(om)
    return jnp.concatenate(Us), jnp.concatenate(Ws)


def interp_spectrum(G, n):
    """Half-space masked Fourier coefficients of the continuum-normalised mesh bank and of its
    curl: waves (K, 3) and C (K, 6R) complex with f(x) = 2 Re sum_k C_k e^{2 pi i k.x}
    (the k = 0 mode is outside the mask). Columns 0..3R-1 are u (component-major), 3R.. the curl."""
    R = G.shape[1]
    geom = F.geometry(n)
    k, _, mask = (np.asarray(a) for a in geom)
    modes = np.round(k / (2 * np.pi)).astype(int)                               # (3, n, n, n)
    half = mask & ((modes[0] > 0) | ((modes[0] == 0) & (modes[1] > 0))
                   | ((modes[0] == 0) & (modes[1] == 0) & (modes[2] > 0)))
    idx = np.nonzero(half)
    waves = np.stack([modes[d][idx] for d in range(3)], axis=1).astype(np.float64)
    cols = jnp.asarray(G.T.reshape(R, 3, n, n, n))
    spec = F.fft(cols) * n ** 1.5                                               # continuum scale
    kj = jnp.asarray(k)
    curl = jax.vmap(lambda c: 1j * F.cross(kj, c))(spec)                       # (R, 3, n, n, n)
    sel = np.asarray(spec)[:, :, idx[0], idx[1], idx[2]]                        # (R, 3, K)
    selc = np.asarray(curl)[:, :, idx[0], idx[1], idx[2]]
    # outside-mask energy must be zero for a mesh bank (it lives in the FOM space)
    full = float(np.sum(np.abs(np.asarray(spec)) ** 2))
    kept = 2.0 * float(np.sum(np.abs(sel) ** 2))
    C = np.concatenate((sel.transpose(2, 1, 0).reshape(-1, 3 * R),
                        selc.transpose(2, 1, 0).reshape(-1, 3 * R)), axis=1)    # (K, 6R) comp-major
    return waves, C, dict(modes_half=int(len(waves)), energy_outside_mask=max(0.0, 1 - kept / full))


@jax.jit
def _nudft_block(X, waves, C):
    E = jnp.exp(2j * jnp.pi * (X @ waves.T))                                    # (P, K)
    return 2.0 * jnp.real(E @ C)                                                # (P, 6R)


def interp_tables(waves, C, X, R, chunk=512):
    wj, Cj = jnp.asarray(waves), jnp.asarray(C)
    out = []
    for s in range(0, len(X), chunk):
        out.append(_nudft_block(jnp.asarray(X[s:s + chunk]), wj, Cj))
    V = jnp.concatenate(out)                                                    # (Mq, 6R)
    U = V[:, :3 * R].reshape(-1, 3, R)
    W = V[:, 3 * R:].reshape(-1, 3, R)
    return U, W


def shifted_grid_points(n, shift):
    s = np.arange(n, dtype=np.float64) / n
    X = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3) + np.asarray(shift)[None]
    return X % 1.0


def shifted_grid_tables(G, n, shift):
    """The mesh-bank interpolant (and its spectral curl) on the grid shifted by `shift` (3,),
    by FFT phase shift: values at x_i + shift, x_i the mesh nodes (ij order). Continuum scale."""
    R = G.shape[1]
    geom = F.geometry(n)
    off = jnp.asarray(-np.asarray(shift) * n)

    @jax.jit
    def one(col, kvec, off):          # geometry and offset are explicit arguments (not captured)
        u = col.reshape(3, n, n, n)
        om = F.ifft(1j * F.cross(kvec, F.fft(u)))
        return SR.shift_field(u, off), SR.shift_field(om, off)
    Us, Ws = [], []
    for r in range(R):
        u, om = one(jnp.asarray(G[:, r]), geom[0], off)
        Us.append(u.reshape(3, -1).T)
        Ws.append(om.reshape(3, -1).T)
    U = jnp.stack(Us, axis=-1) * n ** 1.5
    W = jnp.stack(Ws, axis=-1) * n ** 1.5
    return U, W


def pad_tables(U, W, P, size):
    """Zero-weight padding to `size` points (rows of P are exactly zero)."""
    extra = size - U.shape[0]
    if extra <= 0:
        return U, W, P
    z = lambda a: jnp.concatenate((a, jnp.zeros((extra,) + a.shape[1:], a.dtype)))
    return z(U), z(W), z(P)


# ------------------------------------------------------- batched advection for rho --

@jax.jit
def adv_batch(U, W, P, Cs):
    """a(c) for a batch Cs (S, R) -> (S, Mt), pre = 1."""
    u = jnp.einsum("qcr,sr->sqc", U, Cs)
    om = jnp.einsum("qcr,sr->sqc", W, Cs)
    x = jnp.stack((u[..., 1] * om[..., 2] - u[..., 2] * om[..., 1],
                   u[..., 2] * om[..., 0] - u[..., 0] * om[..., 2],
                   u[..., 0] * om[..., 1] - u[..., 1] * om[..., 0]), axis=-1)
    return jnp.einsum("qcm,sqc->sm", P, x)


@jax.jit
def jac_batch(U, W, P, Cs):
    return jax.vmap(lambda c: jac_adv("quad", (U, W, P), c))(Cs)


@jax.jit
def tensor_adv_batch(T, Cs):
    return jnp.einsum("mjk,sj,sk->sm", T, Cs, Cs)


@jax.jit
def tensor_jac_batch(T, Cs):
    sym = T + jnp.swapaxes(T, 1, 2)
    return jnp.einsum("msk,tk->tms", sym, Cs)
