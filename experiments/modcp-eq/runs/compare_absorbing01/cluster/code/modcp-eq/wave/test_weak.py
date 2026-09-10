"""Weak CN versus independent assembled equations and decoder derivatives."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from common.decoders import (DecoderConfig, init_decoder, add_modulation,
                             decode_points, decode_grid, prepare_points, decode_cached)
from physics import Grid, smooth_tests, face_data
from weak import numerical_rule, make_query
from test_fresh_fom import independent_matrices


def full_rule(grid, count):
    phi, eigen, modes = smooth_tests(grid, count)
    faces = face_data(grid, modes)
    return {'xy': grid.coordinates().reshape(-1, 2), 'active_ids': np.arange(np.prod(grid.shape)),
            'weight': grid.mass().ravel(), 'test': phi, 'eigen': eigen, 'modes': modes,
            'face_xy': np.concatenate([f[0] for f in faces]) if faces else np.empty((0, 2)),
            'face_weight': np.concatenate([f[1] for f in faces]) if faces else np.empty(0),
            'face_test': np.concatenate([f[2] for f in faces]) if faces else np.empty((0, count))}


class WeakTests(unittest.TestCase):
    def test_independent_weak_cn_and_tangent(self):
        cfg = {'observation_dt': .01, 'end_time': .02, 'initial_fit_cap': 10, 'initial_fit_tolerance': 1e-6}
        rng = np.random.default_rng(91061)
        for bc in ('dirichlet', 'absorbing'):
            grid = Grid(5, bc, bc)
            dc = DecoderConfig(k=3, rank=4, outputs=2, intervals=5, boundary='dirichlet' if bc == 'dirichlet' else 'free', head_width=8)
            p = init_decoder(jax.random.PRNGKey(17), dc)
            raw = full_rule(grid, 8)
            rule = numerical_rule(p, dc, raw)
            _, _, _, residual = make_query(dc, grid, cfg, 10, .01)
            z = jnp.asarray(rng.normal(size=dc.k))
            old = rng.normal(size=(np.prod(grid.shape), 2))
            m, k, c = independent_matrices(grid, 1.07)
            phi = raw['test']
            oldmass = phi.T@m@old
            oldface = phi.T@(c/1.07)@old
            scale = jnp.asarray([.2, 1.7])
            args = (p, rule, jnp.asarray(oldmass), jnp.asarray(oldface), jnp.asarray(1.07), scale)
            actual = residual(z, *args)
            field = np.asarray(decode_points(p, z, jnp.asarray(raw['xy']), dc))
            ru = phi.T@(m@(field[:, 0]-old[:, 0])-.01*m@(field[:, 1]+old[:, 1])/2)/scale[0]
            rv = phi.T@(m@(field[:, 1]-old[:, 1])+.01*c@(field[:, 1]+old[:, 1])/2+.01*k@(field[:, 0]+old[:, 0])/2)/scale[1]
            np.testing.assert_allclose(actual, np.r_[ru, rv], atol=2e-13)
            direction = jnp.asarray(rng.normal(size=dc.k))
            derivative = jax.jvp(lambda zz: residual(zz, *args), (z,), (direction,))[1]
            step = 1e-5
            finite_difference = (residual(z+step*direction, *args)-residual(z-step*direction, *args))/(2*step)
            np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=2e-10)

    def test_zero_modulation_dense_sample_cache_and_boundary_transfer(self):
        dc = DecoderConfig(k=3, rank=4, outputs=2, intervals=8, head_width=8, width=8, inr_width=12)
        z = jnp.asarray([.3, -.2, .1])
        p = init_decoder(jax.random.PRNGKey(91062), dc)
        dm = replace(dc, architecture='modcp')
        pm = add_modulation(p, dm)
        xy = jnp.asarray([[0., .3], [.02, .48], [.2, .7], [.98, .52], [1., .4]])
        np.testing.assert_array_equal(decode_points(p, z, xy, dc), decode_points(pm, z, xy, dm))
        for conf, params in ((dc, p), (dm, pm), (replace(dc, architecture='film'), init_decoder(jax.random.PRNGKey(91063), replace(dc, architecture='film')))):
            for n in (8, 16):
                axis = jnp.linspace(0., 1., n+1)
                coords = jnp.stack(jnp.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
                dense = decode_grid(params, z, n, conf)
                point = decode_points(params, z, coords, conf).reshape(n+1, n+1, 2)
                cached = decode_cached(params, z, prepare_points(params, coords, conf), conf).reshape(n+1, n+1, 2)
                np.testing.assert_allclose(dense, point, atol=5e-15)
                np.testing.assert_allclose(dense, cached, atol=5e-15)
            # Reflective numerical rule includes no physical damping faces.
            rule = numerical_rule(params, conf, full_rule(Grid(8), 8))
            self.assertEqual(rule['face_projection'].shape[1], 0)
        mutated = dict(p)
        mutated['factors'] = p['factors'].at[..., 0].add(30.).at[..., -1].add(-20.)
        for n in (8, 16):
            np.testing.assert_array_equal(decode_grid(p, z, n, dc), decode_grid(mutated, z, n, dc))


if __name__ == '__main__':
    unittest.main()
