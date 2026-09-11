"""Direct linear evolution in the frozen learned bank, with the same sine tests.

The state has r free coefficients, not k nonlinear latent variables. C maps
orthonormal bank coordinates to M sine moments. All PDE residuals are weak.
Only small reduced evolution maps are cached; full input/output is charged.
"""
import scipy.linalg
import numpy as np
import jax
import jax.numpy as jnp


def reduced_maps(triangular, matrix, eigenvalues, times, nu, dt):
    r = np.asarray(triangular)
    c = scipy.linalg.solve_triangular(r.T, np.asarray(matrix).T, lower=True).T
    assert c.shape[0] >= 2*c.shape[1]
    left = scipy.linalg.lstsq(c, np.eye(c.shape[0]), cond=None)[0]
    generator = -nu * left @ (np.asarray(eigenvalues)[:, None]*c)
    eig = scipy.linalg.eigvals(generator)
    assert np.max(eig.real) < 0, eig
    factors = (1-dt*nu*eigenvalues/2)/(1+dt*nu*eigenvalues/2)
    step = left @ (factors[:, None]*c)
    steps = np.rint(np.asarray(times)/dt).astype(int)
    np.testing.assert_allclose(steps*dt, times, atol=1e-13)
    maps = {
        "linear_weak_exact": np.stack([scipy.linalg.expm(t*generator) for t in times]),
        "linear_weak_cn": np.stack([np.linalg.matrix_power(step, int(s)) for s in steps]),
    }
    return maps, dict(weak_coordinates=c, generator=generator, cn_step=step,
                      weak_singular_values=scipy.linalg.svdvals(c),
                      generator_eigenvalues=eig)


@jax.jit
def query(projection, maps, u0):
    y0 = projection.T @ u0.reshape(-1)
    ys = jnp.einsum("tij,j->ti", maps, y0)
    return ys @ projection.T


def verify():
    """Analytic eigenmodes and independent stacked least-squares time stepping."""
    rng = np.random.default_rng(901001)
    m, r = 12, 4
    matrix = np.zeros((m, r)); matrix[:r] = np.diag([1., 2., 3., 4.])
    triangular = np.diag([1., 2., 3., 4.])
    lam = np.arange(1, m+1, dtype=np.float64)
    times = np.array([0., .1, .2]); dt = .025; nu = .02
    maps, evidence = reduced_maps(triangular, matrix, lam, times, nu, dt)
    expected = np.stack([np.diag(np.exp(-nu*t*lam[:r])) for t in times])
    analytic_error = float(np.max(np.abs(maps["linear_weak_exact"]-expected)))
    matrix = rng.normal(size=(m, r)); triangular = np.linalg.qr(rng.normal(size=(r, r)))[1]
    maps, evidence = reduced_maps(triangular, matrix, lam, times, nu, dt)
    y0 = rng.normal(size=r); y = y0.copy(); c = evidence["weak_coordinates"]
    factors = (1-dt*nu*lam/2)/(1+dt*nu*lam/2)
    for _ in range(8): y = scipy.linalg.lstsq(c, factors*(c@y))[0]
    cn_error = float(np.linalg.norm(y-maps["linear_weak_cn"][-1]@y0))
    assert analytic_error < 1e-13 and cn_error < 1e-12
    return dict(analytic_eigenmode_error=analytic_error, independent_cn_error=cn_error)
