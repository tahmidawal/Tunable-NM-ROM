"""Small GPU regression for compiled composition and parity failure detection."""
import unittest

import heat_core as hc
from runtime_paths import build_paths, parity
import jax
import jax.numpy as jnp
import numpy as np


class RuntimeTests(unittest.TestCase):
    def test_compiled_query_retains_all_stages(self):
        cfg = dict(k=2, fit_budget=30, step_budget=20, gradient_tolerance=1e-7,
                   times=[0., .1, .2], modes_per_axis=2, diffusivity=.02)
        params = hc.sc.init_separable(jax.random.PRNGKey(18), 2, 4, n_ff=4,
                                     g_hidden=8, h_hidden=8)
        n = 8
        bank = hc.sc.features(params, jnp.asarray(hc.coords(n)))
        projection, triangular = jnp.linalg.qr(bank, mode="reduced")
        matrix = hc.mode_matrix(n, 2).T@bank
        codes = jnp.array([[.1, .2], [.3, -.2], [-.1, .5]])
        library = hc.sc.head(params, codes)@triangular.T
        u0 = (bank@hc.sc.head(params, codes[0])).reshape(n-1, n-1)
        factor = hc.cn_factor(hc.eigenvalues(n)[:2, :2].reshape(-1), .025, .02)
        paths = build_paths(cfg, .025)
        z0, initial = paths["initialize"](params, projection, triangular, library, codes, u0)
        zs, steps = paths["rollout"](params, matrix, factor, z0)
        modular = (paths["readout"](params, bank, zs), initial, steps, zs)
        compiled = paths["compiled"](params, projection, triangular, library, codes, bank, matrix,
                                     hc.eigenvalues(n)[:2, :2].reshape(-1), u0)
        jax.block_until_ready(compiled)
        gate = parity(modular, compiled, 1e-10, 1e-9)
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(compiled[0].shape, (3, (n-1)**2))

    def test_parity_rejects_changed_counter_even_with_equal_fields(self):
        a = (np.ones((3, 4)), np.ones((2, 5)), np.ones((8, 5)), np.ones((3, 2)))
        b = tuple(value.copy() for value in a)
        b[2][0, 0] += 1
        gate = parity(a, b, 1e-10, 1e-9)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["step_counter_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
