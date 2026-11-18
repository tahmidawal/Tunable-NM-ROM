"""Dense original-upwind weak model. Large arrays are always JIT arguments."""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import engines as e


def mode_order(n, count):
    k = np.arange(1, n)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4*n*n*(np.sin(np.pi*kx/(2*n))**2 + np.sin(np.pi*ky/(2*n))**2)
    ix = np.argsort(lam.ravel(), kind='stable')[:count]
    return np.stack((kx.ravel()[ix], ky.ravel()[ix]), axis=1), lam.ravel()[ix]


def project(value, modes, n):
    """Physical sine moment: psi^T W value, psi=2 sin sin, W=I/n^2."""
    vector = value.ndim == 1
    v = value[:, None] if vector else value
    transformed = e.dst2(v.T.reshape(v.shape[1], n-1, n-1))
    result = transformed[:, modes[:, 0]-1, modes[:, 1]-1].T/n
    return result[:, 0] if vector else result


def explicit_project(value, modes, n):
    """Independent NumPy sine sums, no FFT and no shared projection code."""
    kmax = int(np.max(modes))
    sine = np.sin(np.pi*np.arange(1, n)[:, None]*np.arange(1, kmax+1)[None, :]/n)
    v = np.asarray(value).reshape(n-1, n-1)
    all_modes = (2/n**2) * (sine.T @ v @ sine)
    return all_modes[modes[:, 0]-1, modes[:, 1]-1]


def weak_all(c, previous, nu, data, n, dt):
    G, A, lam, modes = data['G'], data['A'], data['lam'], data['modes']
    u = G @ c
    adv, _ = e.spatial(u, n)
    return (A @ (c-previous) + dt*(project(adv, modes, n)+nu*lam*(A @ c)))/(1+dt*nu*lam)


def value_jac(c, previous, nu, data, n, dt):
    """Analytic directional derivative of the unchanged sign-upwind stencil."""
    G, A, lam, modes = data['G'], data['A'], data['lam'], data['modes']
    u = G @ c
    ug = jnp.pad(u.reshape(n-1, n-1), 1)
    bg = jnp.pad(G.reshape(n-1, n-1, G.shape[1]), ((1, 1), (1, 1), (0, 0)))
    uc = ug[1:-1, 1:-1]
    positive = uc > 0
    dx = jnp.where(positive, uc-ug[:-2, 1:-1], ug[2:, 1:-1]-uc)
    dy = jnp.where(positive, uc-ug[1:-1, :-2], ug[1:-1, 2:]-uc)
    bc = bg[1:-1, 1:-1]
    dbx = jnp.where(positive[..., None], bc-bg[:-2, 1:-1], bg[2:, 1:-1]-bc)
    dby = jnp.where(positive[..., None], bc-bg[1:-1, :-2], bg[1:-1, 2:]-bc)
    adv = (uc*n*(dx+dy)).reshape(-1)
    jac_adv = (n*((dx+dy)[..., None]*bc + uc[..., None]*(dbx+dby))).reshape(G.shape)
    scale = 1/(1+dt*nu*lam)
    residual = scale*(A @ (c-previous)+dt*(project(adv, modes, n)+nu*lam*(A @ c)))
    jac = scale[:, None]*(A+dt*(project(jac_adv, modes, n)+nu*lam[:, None]*A))
    return residual, jac


def make_solver(n, dt, steps, budget):
    """Same residual for all prefixes; zero rows are an implementation convenience.

    This diagnostic retains the max-row kernel for every M and makes no production
    timing claim. Actual residual rows are always reported as M, not buffer size.
    """
    def step(previous, nu, data, m, gtol):
        mask = (jnp.arange(data['A'].shape[0]) < m).astype(jnp.float64)
        def evaluate(c):
            r, J = value_jac(c, previous, nu, data, n, dt)
            return r*mask, J*mask[:, None]
        def stationarity(r, J):
            return jnp.linalg.norm(J.T @ r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300)
        r, J = evaluate(previous)
        rn, gn = jnp.linalg.norm(r), stationarity(r, J)
        history = jnp.full((budget+1, 6), jnp.nan)
        history = history.at[0].set(jnp.array([rn, gn, 1e-3, 0., 0., 0.]))
        reason = jnp.where(gn <= gtol, 1, 0).astype(jnp.int32)
        # reason: 1 stationary, 2 tiny step, 3 damping/nonfinite, 4 budget.
        def body(s):
            c, r, J, rn, gn, damping, it, reason, rejected, hist = s
            H, g = J.T @ J, J.T @ r
            delta = jnp.linalg.solve(H+jnp.diag(damping*(jnp.diag(H)+1e-30)), -g)
            candidate = c+delta
            r2, J2 = evaluate(candidate)
            rn2 = jnp.linalg.norm(r2)
            finite = jnp.all(jnp.isfinite(delta)) & jnp.isfinite(rn2)
            accept = finite & (rn2 < rn)
            c = jnp.where(accept, candidate, c)
            r = jnp.where(accept, r2, r)
            J = jnp.where(accept, J2, J)
            rn = jnp.where(accept, rn2, rn)
            gn = stationarity(r, J)
            tiny = jnp.linalg.norm(delta) <= 1e-14*(1+jnp.linalg.norm(c))
            reason = jnp.where(gn <= gtol, 1,
                      jnp.where(tiny, 2, jnp.where((~finite)|((~accept)&(damping >= 1e14)), 3, 0))).astype(jnp.int32)
            damping = jnp.where(accept, jnp.maximum(damping/3, 1e-12), jnp.minimum(damping*10, 1e14))
            hist = hist.at[it+1].set(jnp.array([rn, gn, damping, accept, jnp.linalg.norm(delta), rn2]))
            return c, r, J, rn, gn, damping, it+1, reason, rejected+(~accept), hist
        state = jax.lax.while_loop(lambda s: (s[6] < budget)&(s[7] == 0), body,
            (previous, r, J, rn, gn, jnp.float64(1e-3), jnp.int32(0), reason, jnp.int32(0), history))
        c, r, J, rn, gn, damping, it, reason, rejected, history = state
        reason = jnp.where(reason == 0, 4, reason)
        stats = jnp.array([it, reason, gn, rn, rejected, damping])
        return c, (c, stats, history)
    def rollout(c0, nu, data, m, gtol):
        _, (coeff, stats, histories) = jax.lax.scan(lambda c, _: step(c, nu, data, m, gtol), c0, None, length=steps)
        return jnp.concatenate((c0[None], coeff)), stats, histories
    return jax.jit(rollout)


def physical_tangent(G, n, threshold=1e-12):
    gram = np.asarray(jax.device_get(G.T @ G))/n**2
    eigenvalues, V = np.linalg.eigh(gram)
    keep = eigenvalues > threshold*eigenvalues[-1]
    transform = V[:, keep]/np.sqrt(eigenvalues[keep])[None, :]
    return gram, eigenvalues, transform


def field_metrics(fields, reference, n):
    """All metrics use the same-grid target and initial norm."""
    den = jnp.linalg.norm(reference[0])
    error = jnp.linalg.norm(fields-reference, axis=1)/den
    grad = lambda v: jnp.stack(jnp.gradient(jnp.pad(v.reshape(n-1, n-1), 1), 1/n)).reshape(-1)
    gs = jax.vmap(grad)(fields)
    gr = jax.vmap(grad)(reference)
    gradient_error = jnp.linalg.norm(gs-gr, axis=1)/jnp.maximum(jnp.linalg.norm(gr, axis=1), 1e-300)
    aa, la = jax.vmap(lambda u: e.spatial(u, n))(fields)
    ar, lr = jax.vmap(lambda u: e.spatial(u, n))(reference)
    adv_error = jnp.linalg.norm(aa-ar, axis=1)/jnp.maximum(jnp.linalg.norm(ar, axis=1), 1e-300)
    lap_error = jnp.linalg.norm(la-lr, axis=1)/jnp.maximum(jnp.linalg.norm(lr, axis=1), 1e-300)
    energy = .5*jnp.sum(fields**2, axis=1)/n**2
    return jnp.stack((error, gradient_error, adv_error, lap_error, energy), axis=1)


def reference_residual(u, previous, nu, dt, n, stable_diffusion):
    """Same original upwind and five-point diffusion; optional cancellation repair."""
    if not stable_diffusion:
        return e.residual(u, previous, nu, dt, n)
    adv, _ = e.spatial(u, n)
    p = jnp.pad(u.reshape(n-1, n-1), 1)
    c = p[1:-1, 1:-1]
    lap = n**2*((p[:-2, 1:-1]-c)+(p[2:, 1:-1]-c)+(p[1:-1, :-2]-c)+(p[1:-1, 2:]-c))
    return u-previous+dt*(adv-nu*lap.reshape(-1))


def make_reference(n, dt, steps, stable_diffusion):
    """Instrumented original Newton/BiCGSTAB reference, unchanged stopping rules."""
    def step(previous, nu, ntol, ltol):
        scale = jnp.maximum(jnp.linalg.norm(previous), 1e-300)
        residual = lambda u: reference_residual(u, previous, nu, dt, n, stable_diffusion)
        rn = jnp.linalg.norm(residual(previous))
        trace = jnp.full((21,), jnp.nan).at[0].set(rn/scale)
        def body(state):
            u, it, rn, trace = state
            r = residual(u)
            def jv(v):
                return jax.jvp(residual, (u,), (v,))[1]
            delta, _ = jax.scipy.sparse.linalg.bicgstab(jv, -r, tol=ltol, maxiter=200,
                M=lambda v: e.helmholtz(v, nu, dt, n))
            candidate = u+delta
            rn2 = jnp.linalg.norm(residual(candidate))
            return candidate, it+1, rn2, trace.at[it+1].set(rn2/scale)
        u, it, rn, trace = jax.lax.while_loop(
            lambda s: (s[2] > ntol*scale)&(s[1] < 20)&jnp.isfinite(s[2]), body,
            (previous, jnp.int32(0), rn, trace))
        return u, (u, it, rn/scale, trace)
    def query(initial, nu, ntol, ltol):
        u0 = initial[1:-1, 1:-1].reshape(-1)
        _, (fields, iterations, residuals, traces) = jax.lax.scan(lambda u, _: step(u, nu, ntol, ltol), u0, None, length=steps)
        return jnp.concatenate((u0[None], fields)), iterations, residuals, traces
    return jax.jit(query)
