"""Post-reset wave operators, symmetric CN solve, and weak test functions.

The only inherited numerical dependency is the independently verified fresh
September wave spatial operator. Arrays here use its active degrees of freedom.
"""
from functools import partial
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fresh-wave-head'))
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import numpy as np
from fresh_fom import (Grid, positive_laplacian, damping_ratio, localized_initial,
                       integrate, integrate_balance, energy, provenance)


def parameter_rows(seed, count):
    """Exactly the approved fresh Gaussian-core family, new recorded seeds."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        sx, sy = rng.uniform(.36, .42, 2)
        cx = rng.uniform(sx+.025, 1-sx-.025)
        cy = rng.uniform(sy+.025, 1-sy-.025)
        amp, c = rng.uniform(.7, 1.3), rng.uniform(.85, 1.15)
        dx, dy = rng.uniform(-.5, .5, 2)
        if i % 4 == 0:
            dx = dy = 0.
        sigx, sigy = rng.uniform(.12, .16, 2)
        rows.append((cx, cy, sx, sy, amp, c, dx, dy, sigx, sigy))
    return np.asarray(rows, dtype=np.float64)


def full_fields(u, grid):
    return jnp.pad(u, [(0, 0)]*(u.ndim-2)+[(1, 1), (1, 1)]) if grid.bx == 'dirichlet' else u


def active_fields(u, grid):
    return u[..., 1:-1, 1:-1] if grid.bx == 'dirichlet' else u


def cn_matrix_action(u, c, dt, grid):
    mass = jnp.asarray(grid.mass())
    return mass*((1+dt*damping_ratio(grid, c)/2)*u + dt*dt*c*c*positive_laplacian(u, grid)/4)


def pcg(action, b, x, diagonal, tol, maxiter):
    """PCG with true residual verification; no misleading implicit info flag."""
    r = b-action(x)
    z = r/diagonal
    p = z
    rho = jnp.vdot(r, z).real
    bnorm = jnp.linalg.norm(b)
    threshold = tol*jnp.maximum(bnorm, jnp.finfo(b.dtype).tiny)
    carry = (x, r, p, rho, jnp.asarray(0), jnp.asarray(True))
    def cond(s):
        _, rr, _, _, i, valid = s
        return (i < maxiter) & (jnp.linalg.norm(rr) > threshold) & valid
    def body(s):
        xx, rr, pp, rz, i, valid = s
        ap = action(pp)
        curvature = jnp.vdot(pp, ap).real
        good = (curvature > 0) & jnp.isfinite(curvature) & (rz >= 0)
        alpha = rz/jnp.where(good, curvature, 1.)
        xx = xx+alpha*pp
        rr = rr-alpha*ap
        zz = rr/diagonal
        rznew = jnp.vdot(rr, zz).real
        pp = zz+(rznew/jnp.maximum(rz, jnp.finfo(b.dtype).tiny))*pp
        return xx, rr, pp, rznew, i+1, valid & good & jnp.all(jnp.isfinite(xx))
    x, _, _, _, count, valid = jax.lax.while_loop(cond, body, carry)
    true_relative = jnp.linalg.norm(b-action(x))/jnp.maximum(bnorm, jnp.finfo(b.dtype).tiny)
    valid = valid & jnp.isfinite(true_relative) & (true_relative <= tol*1.05+1e-13)
    return x, count, true_relative, valid


def cn_step(u, v, c, dt, grid, tol, maxiter, direct=False):
    mass = jnp.asarray(grid.mass())
    dr = damping_ratio(grid, c)
    rhs = mass*((1+dt*dr/2)*u-dt*dt*c*c*positive_laplacian(u, grid)/4+dt*v)
    if direct:
        eig = sine_eigenvalues(grid.n)
        un = dst2(dst2(rhs/mass)/(1+dt*dt*c*c*eig/4))
        count = jnp.asarray(0)
        residual = jnp.linalg.norm(rhs-cn_matrix_action(un, c, dt, grid))/jnp.maximum(jnp.linalg.norm(rhs), 1e-300)
        valid = jnp.isfinite(residual) & (residual < 1e-10)
    else:
        diagonal = mass*(1+dt*dr/2+dt*dt*c*c/grid.h**2)
        un, count, residual, valid = pcg(lambda x: cn_matrix_action(x, c, dt, grid), rhs,
                                        u+dt*v, diagonal, tol, maxiter)
    vn = 2*(un-u)/dt-v
    return un, vn, count, residual, valid


@partial(jax.jit, static_argnames=('grid', 'steps', 'stride', 'maxiter', 'direct'))
def cn_rollout(u0, v0, c, dt, tol, *, grid, steps, stride, maxiter=1000, direct=False):
    if steps % stride:
        raise ValueError('steps must be divisible by observation stride')
    if direct and grid.bx != 'dirichlet':
        raise ValueError('sine direct solve requires reflective Dirichlet boundaries')
    def block(carry, _):
        u, v, alive = carry
        def one(s, _):
            u, v, alive = s
            un, vn, count, residual, valid = cn_step(u, v, c, dt, grid, tol, maxiter, direct)
            alive = alive & valid & jnp.all(jnp.isfinite(vn))
            return (un, vn, alive), (count, residual, alive)
        (u, v, alive), audit = jax.lax.scan(one, (u, v, alive), None, length=stride)
        return (u, v, alive), (u, v, audit)
    _, (u, v, audits) = jax.lax.scan(block, (u0, v0, jnp.asarray(True)), None, length=steps//stride)
    return {'u': jnp.concatenate((u0[None], u)), 'v': jnp.concatenate((v0[None], v)),
            'iterations': audits[0].reshape(-1), 'true_relative_residual': audits[1].reshape(-1),
            'completed': audits[2].reshape(-1)}


def dst1(a, axis):
    a = jnp.moveaxis(a, axis, -1)
    n = a.shape[-1]+1
    zero = jnp.zeros((*a.shape[:-1], 1), dtype=a.dtype)
    odd = jnp.concatenate((zero, a, zero, -a[..., ::-1]), axis=-1)
    result = -jnp.fft.rfft(odd, axis=-1).imag[..., 1:n]/jnp.sqrt(2*n)
    return jnp.moveaxis(result, -1, axis)


def dst2(a):
    return dst1(dst1(a, -1), -2)


def sine_eigenvalues(n):
    modes = jnp.arange(1, n, dtype=jnp.float64)
    eig = 4*n*n*jnp.sin(jnp.pi*modes/(2*n))**2
    return eig[:, None]+eig[None, :]


@jax.jit
def spectral_propagate(u0, v0, c, times):
    """Exact semidiscrete reflective reference; includes every requested field."""
    omega = c*jnp.sqrt(sine_eigenvalues(u0.shape[-1]+1))
    a, b = dst2(u0), dst2(v0)
    def one(t):
        co, si = jnp.cos(omega*t), jnp.sin(omega*t)
        return dst2(co*a+si/omega*b), dst2(-omega*si*a+co*b)
    return jax.lax.map(one, times)


def smooth_tests(grid, count):
    """Mass-orthonormal sine/cosine tests, lowest radial frequencies first."""
    first = 1 if grid.bx == 'dirichlet' else 0
    extent = int(np.ceil(np.sqrt(count)))+2
    while True:
        candidates = sorted(((i, j) for i in range(first, extent) for j in range(first, extent)),
                            key=lambda ij: (ij[0]**2+ij[1]**2, ij))
        if len(candidates) >= count and candidates[count-1][0]**2+candidates[count-1][1]**2 < extent**2:
            break
        extent *= 2
    modes = np.asarray(candidates[:count], dtype=int)
    if np.max(modes) >= grid.n:
        raise ValueError('Too many weak modes for this mesh')
    xy = grid.coordinates().reshape(-1, 2)
    phi = tests_at(xy, modes, grid.bx)
    lam = 4/grid.h**2*np.sum(np.sin(np.pi*modes/(2*grid.n))**2, axis=1)
    defect = np.max(abs(phi.T@(grid.mass().ravel()[:, None]*phi)-np.eye(count)))
    if defect > 1e-11:
        raise RuntimeError(f'Test mass orthogonality failure: {defect}')
    return phi, lam, modes


def tests_at(xy, modes, bc):
    xy, modes = np.asarray(xy), np.asarray(modes)
    if bc == 'dirichlet':
        return 2*np.sin(np.pi*xy[:, :1]*modes[:, 0])*np.sin(np.pi*xy[:, 1:]*modes[:, 1])
    norm = np.where(modes == 0, 1., np.sqrt(2.))
    return (norm[:, 0]*np.cos(np.pi*xy[:, :1]*modes[:, 0]) *
            norm[:, 1]*np.cos(np.pi*xy[:, 1:]*modes[:, 1]))


def face_data(grid, modes):
    """Four distinct physical face integrals; corner has TWO half-weights."""
    if grid.bx != 'absorbing':
        return []
    x = np.linspace(0., 1., grid.n+1)
    w = np.full(grid.n+1, grid.h)
    w[[0, -1]] *= .5
    result = []
    for axis, side in ((0, 0.), (0, 1.), (1, 0.), (1, 1.)):
        xy = np.stack((np.full_like(x, side), x), -1) if axis == 0 else np.stack((x, np.full_like(x, side)), -1)
        result.append((xy, w.copy(), tests_at(xy, modes, grid.bx)))
    return result


def metrics(u, v, ut, vt, grid, speed):
    """Full-mesh actual-output errors on fixed physical initial-state scales."""
    mass = jnp.asarray(grid.mass())
    l2 = lambda a: jnp.sqrt(jnp.maximum(jnp.sum(mass*a*a, axis=(-2, -1)), 0.))
    e0 = energy(ut[0], vt[0], grid, speed)
    u_scale, state_scale = l2(ut[0]), jnp.sqrt(2*e0)
    du, dv = u-ut, v-vt
    traces = {'displacement': l2(du)/u_scale, 'velocity': l2(dv)/state_scale,
              'energy_state': jnp.sqrt(jnp.maximum(2*energy(du, dv, grid, speed), 0.))/state_scale}
    return {**traces, 'maximum': jnp.max(jnp.stack([jnp.max(x) for x in traces.values()])),
            'u_scale': u_scale, 'state_scale': state_scale,
            'finite': jnp.all(jnp.isfinite(u)) & jnp.all(jnp.isfinite(v))}
