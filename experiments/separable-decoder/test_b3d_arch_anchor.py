"""Sub-minute GPU geometry checks against independent NumPy differences."""
import os
import unittest

import numpy as np
import jax
import jax.numpy as jnp

import b3d_arch_anchor as model
import b3d_arch_baseline as control
import b3d_arch_bench as bench


def fixture(r=11, k=4, learned=True):
    rng = np.random.default_rng(26)
    u, _ = np.linalg.qr(rng.normal(size=(r, k)), mode='reduced')
    shared = dict(anchor=u, center=rng.normal(size=r),
                  latent_scale=np.geomspace(.3, 3.2, k), output_scale=np.asarray(1.3))
    p, frozen = model.init(jax.random.PRNGKey(44), shared)
    if learned:
        w, bias = p['residual'][-1]
        p['residual'][-1] = (jnp.asarray(rng.normal(size=w.shape)*.07),
                             jnp.asarray(rng.normal(size=bias.shape)*.1))
        p['bias'] = p['bias'] + .23  # recovery is relative to the trainable bias
    return p, frozen, shared


class ProtectedHeadTests(unittest.TestCase):
    def test_exact_control_initialization_at_campaign_dimensions(self):
        p, f, shared = fixture(r=128, k=32, learned=False)
        z = jnp.asarray(np.random.default_rng(8).normal(size=(7, 32)))
        expected = shared['center'] + np.asarray(z) @ shared['anchor'].T
        np.testing.assert_allclose(model.apply(p, f, z), expected, rtol=1e-13, atol=1e-13)
        pcontrol, fcontrol = control.init(jax.random.PRNGKey(44), shared)
        self.assertEqual(control.CONFIG['width'], 128)
        np.testing.assert_array_equal(model.apply(p, f, z), control.apply(pcontrol, fcontrol, z))
        self.assertEqual(set(p), {'bias', 'residual'})
        self.assertNotIn('anchor', p)
        np.testing.assert_array_equal(p['residual'][-1][0], 0.)

    def test_nonzero_residual_projection_and_omission_control(self):
        p, f, _ = fixture()
        z = np.random.default_rng(72).normal(size=(9, 4)) * np.asarray(f['latent_scale'])
        output = np.asarray(model.apply(p, f, jnp.asarray(z)))
        independent = model.apply_np(p, f, z)
        np.testing.assert_allclose(output, independent, rtol=1e-12, atol=1e-12)
        single = np.stack([np.asarray(model.apply(p, f, jnp.asarray(zi))) for zi in z])
        np.testing.assert_allclose(output, single, rtol=1e-12, atol=1e-12)
        u, b = np.asarray(f['anchor']), np.asarray(p['bias'])
        np.testing.assert_allclose((output-b) @ u, z, rtol=1e-12, atol=1e-12)
        nonlinear = float(f['output_scale']) * control.mlp_np(p['residual'], z / np.asarray(f['latent_scale']))
        omitted_projection = b + z @ u.T + nonlinear
        self.assertGreater(np.max(np.abs((omitted_projection-b) @ u-z)), .01)
        self.assertGreater(np.linalg.norm(output-(b+z @ u.T)), .1)
        field_distance = np.linalg.norm(output[1:]-output[:-1], axis=1)
        latent_distance = np.linalg.norm(z[1:]-z[:-1], axis=1)
        self.assertTrue(np.all(field_distance >= latent_distance-1e-12))

    def test_nonuniform_scale_jacobian_and_hessian(self):
        p, f, _ = fixture()
        z = np.asarray([.21, -.34, .43, -.18])
        u = np.asarray(f['anchor'])
        jac = np.asarray(jax.jacfwd(model.apply, argnums=2)(p, f, jnp.asarray(z)))
        hess = np.asarray(jax.jacfwd(jax.jacfwd(model.apply, argnums=2), argnums=2)(p, f, jnp.asarray(z)))
        eye = np.eye(len(z))
        eps = 2e-5
        fdjac = np.column_stack([(model.apply_np(p, f, z+eps*v)-model.apply_np(p, f, z-eps*v))/(2*eps) for v in eye])
        np.testing.assert_allclose(jac, fdjac, rtol=2e-7, atol=2e-8)
        eps = 3e-4
        fdhess = np.empty_like(hess)
        for i, vi in enumerate(eye):
            for j, vj in enumerate(eye):
                fdhess[:, i, j] = (model.apply_np(p,f,z+eps*vi+eps*vj)
                    - model.apply_np(p,f,z+eps*vi-eps*vj)
                    - model.apply_np(p,f,z-eps*vi+eps*vj)
                    + model.apply_np(p,f,z-eps*vi-eps*vj))/(4*eps**2)
        np.testing.assert_allclose(hess, fdhess, rtol=1e-5, atol=5e-6)
        np.testing.assert_allclose(u.T @ jac, eye, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.einsum('rk,rij->kij',u,hess), 0., atol=1e-12)
        np.testing.assert_allclose(hess, hess.swapaxes(1,2), rtol=1e-12, atol=1e-12)
        self.assertGreater(np.linalg.norm(hess), .1)
        self.assertGreaterEqual(np.linalg.eigvalsh(jac.T @ jac).min(), 1.-1e-12)
        probe = np.stack([z, -z, 2*z])
        diagnostics = model.diagnostics(p, f, probe)
        self.assertGreaterEqual(diagnostics['minimum_gram_eigenvalue'], 1.-1e-12)
        self.assertLess(diagnostics['coordinate_recovery_max_absolute'], 1e-12)
        self.assertLess(diagnostics['anchor_jacobian_identity_max_absolute'], 1e-12)
        bench.check_model(model, p, f, probe)

    def test_common_training_updates_bias_and_preserves_anchor(self):
        _, _, shared = fixture(r=7, k=2, learned=False)
        rng = np.random.default_rng(22)
        z = rng.normal(size=(24, 2)) * shared['latent_scale']
        u = shared['anchor']
        residual_direction = rng.normal(size=7)
        residual_direction -= u @ (u.T @ residual_direction)
        residual_direction /= np.linalg.norm(residual_direction)
        target = shared['center'] + z @ u.T + (.2+.3*z[:, 0]**2)[:, None]*residual_direction
        train = dict(target=target, norm2=np.sum(target**2, axis=1)+.1,
                     perpendicular2=np.full(24, .1))
        p, frozen, codes, metrics = bench.train_model(model, shared, z, train,
            steps=80, lr=.01, batch=24)
        self.assertLess(metrics['global_mse_final'], metrics['global_mse_initial'])
        np.testing.assert_array_equal(frozen['anchor'], shared['anchor'])
        self.assertGreater(np.linalg.norm(np.asarray(p['bias'])-shared['center']), 1e-4)
        np.testing.assert_allclose((np.asarray(model.apply(p,frozen,jnp.asarray(codes)))-np.asarray(p['bias'])) @ u,
                                   codes, rtol=1e-12, atol=1e-12)

    def test_invalid_shared_geometry_rejected(self):
        _, _, shared = fixture(learned=False)
        for change in [dict(anchor=shared['anchor']*1.001),
                       dict(latent_scale=np.zeros(4)), dict(output_scale=np.asarray(-1.)),
                       dict(center=np.full(11, np.nan)), dict(latent_scale=np.ones(3))]:
            with self.assertRaises(ValueError):
                model.init(jax.random.PRNGKey(0), dict(shared, **change))


if __name__ == '__main__':
    print('jax_backend='+jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu'
    assert jax.config.x64_enabled
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    unittest.main()
