import unittest

import heat_core as hc
import head_refine as hr
import jax
import jax.numpy as jnp
import numpy as np


class HeadTests(unittest.TestCase):
    def test_exact_loss_and_head_gradient_match_full_field(self):
        params = hc.sc.init_separable(jax.random.PRNGKey(31), 2, 4, n_ff=4, g_hidden=8, h_hidden=8)
        bank = hc.sc.features(params, jnp.asarray(hc.coords(8)))
        q, triangular = jnp.linalg.qr(bank, mode="reduced")
        truth = jax.random.normal(jax.random.PRNGKey(1), (5, len(bank)), dtype=jnp.float64)
        target, norm2, perpendicular2 = hr.compression(truth, q)
        pz = hr.split_head(params), jnp.ones((5, 2))*.1
        idx = jnp.arange(5)
        compressed = lambda pz: hr.compressed_loss(pz, triangular, target, norm2, perpendicular2, idx)
        full = lambda pz: jnp.mean(jnp.sum((hc.sc.head(pz[0], pz[1])@bank.T-truth)**2, axis=1)/norm2)
        np.testing.assert_allclose(compressed(pz), full(pz), rtol=1e-12)
        a, b = jax.grad(compressed)(pz), jax.grad(full)(pz)
        for aa, bb in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(aa, bb, atol=1e-12)

    def test_training_freezes_spatial_parameters_and_shapes(self):
        params = hc.sc.init_separable(jax.random.PRNGKey(31), 2, 4, n_ff=4, g_hidden=8, h_hidden=8)
        bank = hc.sc.features(params, jnp.asarray(hc.coords(8)))
        q, triangular = jnp.linalg.qr(bank, mode="reduced")
        codes = jnp.array([[.1, .2], [.3, -.2], [-.1, .4]])
        truth = hc.sc.head(params, codes)@bank.T
        target, norm2, perpendicular2 = hr.compression(truth, q)
        updated, zs, info = hr.train(params, codes*.8, triangular, target, norm2, perpendicular2, 33,
                                    dict(updates=20, batch_size=3, learning_rate=.001))
        for key in ("B", "g", "out_scale"):
            for a, b in zip(jax.tree.leaves(params[key]), jax.tree.leaves(updated[key])):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(jax.tree.structure(params), jax.tree.structure(updated))
        self.assertEqual(zs.shape, codes.shape)
        self.assertEqual(info["sampled_snapshots"], 60)


if __name__ == "__main__":
    unittest.main()
