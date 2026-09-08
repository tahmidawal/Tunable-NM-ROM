"""Small GPU fixtures: physical interpolation, passthrough, analytic DST evolution."""
import unittest

import heat_core as hc
import jax
import jax.numpy as jnp
import numpy as np
from transfer_core import interpolation_tables, interpolate, restrict_input, make_fom, host_outputs, unique_solvers


class TransferTests(unittest.TestCase):
    def test_physical_tent_interpolation(self):
        for n, s in ((16, 4), (32, 8), (16, 16)):
            tent = lambda m: np.outer(1-abs(2*np.arange(1, m)/m-1), 1-abs(2*np.arange(1, m)/m-1))
            indices, weights = map(jnp.asarray, interpolation_tables(n, s))
            actual = np.asarray(jax.jit(interpolate)(jnp.asarray(tent(s))[None], indices, weights))[0]
            np.testing.assert_array_equal(actual, tent(n))

    def test_input_output_policy_and_decay(self):
        n, s = 32, 8
        xy = hc.coords(n)
        # Contains information between coarse samples; t0 cannot be interpolated.
        u0 = (np.sin(np.pi*xy[:, 0])*np.sin(2*np.pi*xy[:, 1])+.1*np.sin(9*np.pi*xy[:, 0])*np.sin(np.pi*xy[:, 1])).reshape(n-1, n-1)
        coarse = restrict_input(u0, n, s)
        np.testing.assert_array_equal(coarse, u0[3::4, 3::4])
        times = jnp.array([.1, .2])
        indices, weights = map(jnp.asarray, interpolation_tables(n, s))
        actual = make_fom(n, s)(jnp.asarray(coarse), hc.eigenvalues(s), times, .02, indices, weights)
        out = host_outputs(u0, actual)
        np.testing.assert_array_equal(out[0], u0)
        self.assertTrue(out.flags.c_contiguous)
        # Independent NumPy sine matrices and physical interpolation.
        nodes = np.arange(1, s)
        transform = np.sqrt(2/s)*np.sin(np.pi*np.outer(nodes, nodes)/s)
        lam = 4*s*s*np.sin(np.pi*nodes/(2*s))**2
        spectrum = transform@coarse@transform
        for i, t in enumerate(np.asarray(times)):
            evolved = transform@(spectrum*np.exp(-.02*t*(lam[:, None]+lam[None, :])))@transform
            padded = np.pad(evolved, 1)
            x = np.arange(n+1)/n; xc = np.arange(s+1)/s
            tmp = np.stack([np.interp(x, xc, column) for column in padded.T], axis=1)
            expected = np.stack([np.interp(x, xc, row) for row in tmp])[1:-1, 1:-1]
            np.testing.assert_allclose(out[i+1], expected, atol=1e-14, rtol=1e-13)
        self.assertGreater(np.linalg.norm(out[2]-out[1]), .01)

    def test_aliases_and_invocation_count(self):
        grids = [64, 128, 256, 512, 1024]
        self.assertEqual(sum(2+len(unique_solvers(n, [16, 32, 64, 128])) for n in grids)*12*3, 1152)
        self.assertEqual(unique_solvers(64, [16, 32, 64, 128]), [16, 32, 64])


if __name__ == "__main__": unittest.main()
