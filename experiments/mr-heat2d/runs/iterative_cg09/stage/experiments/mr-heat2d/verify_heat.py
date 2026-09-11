"""Independent reference checks and regression tests, run before every pilot."""
import json
import unittest
from pathlib import Path

import heat_core as hc
import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import dstn


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/np.linalg.norm(np.asarray(b)))


def verification(cfg):
    results = {}
    n = 32
    rng = np.random.default_rng(123)
    field = rng.standard_normal((n-1, n-1))
    actual = hc.dst2(jnp.asarray(field))
    results["dst_vs_scipy"] = relative(actual, dstn(field, type=1, norm="ortho"))
    results["dst_involution"] = relative(hc.dst2(actual), field)
    results["laplacian_diagonalization"] = relative(hc.dst2(hc.negative_laplacian(jnp.asarray(field), n)), hc.eigenvalues(n)*actual)
    xy = jnp.asarray(hc.coords(n))
    mode = (jnp.sin(2*jnp.pi*xy[:, 0])*jnp.sin(3*jnp.pi*xy[:, 1])).reshape(n-1, n-1)
    times = jnp.asarray(cfg["times"])
    nu = cfg["diffusivity"]
    discrete = hc.propagate(mode, hc.eigenvalues(n), times, nu)
    decay = np.exp(-nu*np.asarray(times)*float(hc.eigenvalues(n)[1, 2]))
    results["discrete_eigenmode_decay"] = relative(discrete, decay[:, None, None]*mode)
    continuum = hc.propagate(mode, hc.eigenvalues(n, True), times, nu)
    analytic = np.exp(-nu*np.asarray(times)*13*np.pi**2)[:, None, None]*np.asarray(mode)
    results["continuum_eigenmode_decay"] = relative(continuum, analytic)
    results["initial_fidelity"] = relative(discrete[0], mode)

    # Same rollout and LM implementation, exactly representable nontrivial heat mode.
    linear_head = lambda params, z: z
    matrix = jnp.asarray([[1.0], [.5], [.25], [.125]])
    lam = jnp.full(4, 13*jnp.pi**2)
    factors = hc.cn_factor(lam, .025, nu)
    roll = hc.make_rollout(linear_head, 20, 1e-11, 20, 4)
    states, info = roll({}, matrix, factors, jnp.ones(1))
    expected = np.asarray(factors[0])**np.arange(0, 21, 4)
    results["linear_weak_rollout_exact_cn"] = relative(states[:, 0], expected)
    results["second_output_advancement"] = float(abs(states[2, 0]-states[1, 0])/abs(states[1, 0]))
    # Deliberately frozen carry, equivalent to the old satisfied-LHS defect.
    frozen = np.ones_like(expected)*expected[1]
    results["frozen_negative_control_error"] = relative(frozen[1:], expected[1:])
    results["linear_rollout_energy_monotone"] = bool(np.all(np.diff(np.asarray(states[:, 0])**2) < 0))
    exact = np.exp(-13*np.pi**2*nu*float(times[-1]))
    coarse = float(hc.cn_factor(lam, .025, nu)[0])**20
    fine = float(hc.cn_factor(lam, .0125, nu)[0])**40
    results["cn_time_refinement_ratio"] = abs(coarse-exact)/abs(fine-exact)

    # Weak projection identity on an arbitrary field independently using stencil.
    phi = hc.mode_matrix(n, 8)
    low_lam = hc.eigenvalues(n)[:8, :8].reshape(-1)
    results["weak_discrete_operator_identity"] = relative(phi.T@hc.negative_laplacian(jnp.asarray(field), n).reshape(-1), low_lam*(phi.T@field.reshape(-1)))

    draws = hc.sample_family(cfg["validation_seed"], cfg["n_validation"], cfg)
    nf = max(cfg["reference_intervals"])
    nc = min(cfg["reference_intervals"])
    reference_errors = []
    spatial_rows = []
    for draw in draws:
        refs = []
        for mesh in (nc, nf):
            u0 = hc.initial_field(jnp.asarray(hc.coords(mesh)), draw).reshape(mesh-1, mesh-1)
            refs.append(np.asarray(hc.propagate(u0, hc.eigenvalues(mesh, True), times, nu)))
        restricted = restrict(refs[1], nf, nc)
        reference_errors.append(max(hc.error_metrics(refs[0], restricted, nc)["relative_current"]))
        errs = []
        for mesh in (32, 64, 128):
            u0 = hc.initial_field(jnp.asarray(hc.coords(mesh)), draw).reshape(mesh-1, mesh-1)
            fields = hc.propagate(u0, hc.eigenvalues(mesh), times, nu)
            error = max(hc.error_metrics(fields, restrict(refs[1], nf, mesh), mesh)["relative_current"])
            errs.append(error)
        spatial_rows.append(dict(errors_32_64_128=errs, ratios=[errs[0]/errs[1], errs[1]/errs[2]]))
    results["reference_spectral_refinement_current_errors"] = reference_errors
    results["spatial_refinement"] = spatial_rows
    # Boundary factor is identically zero on all four sides, including corners.
    boundary = jnp.asarray([[0., .3], [1., .7], [.4, 0.], [.9, 1.], [0., 0.]])
    params = hc.sc.init_separable(jax.random.PRNGKey(77), 2, 4, n_ff=4, g_hidden=8, h_hidden=8)
    results["hard_boundary_max"] = float(jnp.max(jnp.abs(hc.sc.features(params, boundary))))
    assert max(results[k] for k in ("dst_vs_scipy", "dst_involution", "laplacian_diagonalization", "discrete_eigenmode_decay", "continuum_eigenmode_decay", "initial_fidelity", "weak_discrete_operator_identity")) < 1e-12
    assert results["linear_weak_rollout_exact_cn"] < 1e-8
    assert results["second_output_advancement"] > .01
    assert results["frozen_negative_control_error"] > .1
    assert results["linear_rollout_energy_monotone"]
    assert 3.9 < results["cn_time_refinement_ratio"] < 4.1
    assert max(reference_errors) < cfg["reference_uncertainty_budget"]
    assert min(min(row["ratios"]) for row in spatial_rows) > 3.5
    assert results["hard_boundary_max"] == 0
    results["passed"] = True
    return results


def restrict(fields, fine, coarse):
    assert fine % coarse == 0
    full = np.pad(np.asarray(fields), ((0, 0), (1, 1), (1, 1)))
    stride = fine//coarse
    return full[:, stride:fine:stride, stride:fine:stride]


class ComponentTests(unittest.TestCase):
    def test_transform_and_laplacian(self):
        n = 8
        field = jnp.arange((n-1)**2, dtype=jnp.float64).reshape(n-1, n-1)
        self.assertLess(relative(hc.dst2(hc.dst2(field)), field), 1e-12)
        self.assertLess(relative(hc.dst2(hc.negative_laplacian(field, n)), hc.eigenvalues(n)*hc.dst2(field)), 1e-12)

    def test_rollout_advances_and_matches_cn(self):
        roll = hc.make_rollout(lambda p, z: z, 20, 1e-11, 5, 1)
        matrix = jnp.ones((4, 1))
        zs, info = roll({}, matrix, jnp.full(4, .8), jnp.ones(1))
        np.testing.assert_allclose(zs[:, 0], .8**np.arange(6), atol=1e-9)
        self.assertGreater(float(abs(zs[-1, 0]-zs[1, 0])), .1)

    def test_independent_field_fit_and_stationarity(self):
        fit = jax.jit(hc.make_lm(lambda p, z: z, 30, 1e-10))
        matrix = jnp.asarray([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
        target = jnp.array([.5, -.2, .3, 1.2])
        z, info = fit({}, matrix, target, jnp.zeros(2))
        np.testing.assert_allclose(z, [.5, -.2], atol=1e-9)
        self.assertEqual(int(info[2]), 1)

    def test_transfer_rebuilds_operator(self):
        p = hc.sc.init_separable(jax.random.PRNGKey(9), 2, 4, n_ff=4, g_hidden=8, h_hidden=8)
        arrays = []
        for n in (8, 16):
            bank = hc.sc.features(p, jnp.asarray(hc.coords(n)))
            arrays.append(hc.mode_matrix(n, 2).T@bank)
        self.assertEqual(arrays[0].shape, arrays[1].shape)
        self.assertGreater(float(jnp.linalg.norm(arrays[0]-arrays[1])), 1e-5)


if __name__ == "__main__":
    unittest.main()
