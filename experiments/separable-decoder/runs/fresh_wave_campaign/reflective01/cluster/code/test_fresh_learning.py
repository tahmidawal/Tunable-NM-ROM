"""Numerical controls for learned-bank orthogonalization and latent fitting."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from fresh_learning import orthonormalize_bank,project_data,fit_batch
from fresh_models import head_init,head_apply


class FreshLearningTests(unittest.TestCase):
    def test_zero_and_nan_banks_rejected(self):
        for raw in (np.zeros((8,3)),np.full((8,3),np.nan)):
            with self.assertRaises(RuntimeError):
                orthonormalize_bank(raw,np.ones(8),1e-8)

    def test_weighted_qr_preserves_span_and_near_member_floor(self):
        rng=np.random.default_rng(699911)
        mass=rng.uniform(.2,.8,9)
        raw=rng.normal(size=(9,3))
        g,r,s=orthonormalize_bank(raw,mass,1e-8)
        np.testing.assert_allclose(g@r,raw,atol=1e-13)
        np.testing.assert_allclose(g.T@(mass[:,None]*g),np.eye(3),atol=1e-13)
        exact=g@np.array([.2,-.3,.4])
        outside=rng.normal(size=9)
        outside-=g@(g.T@(mass*outside))
        target=exact+1e-8*outside
        data={"u":target[None,None],"v":target[None,None],"scales":np.ones((1,2))}
        projected=project_data({"validation":data},{"mass":mass,"g":g})["validation"]
        expected=1e-16*np.sum(mass*outside*outside)
        self.assertGreater(float(projected["u_floor_squared"][0,0]),0.)
        self.assertAlmostEqual(float(projected["u_floor_squared"][0,0])/expected,1.,places=6)

    def test_nonorthogonal_linear_fit_and_latent_rescaling(self):
        linear=np.array([[1.,.2],[.3,.8],[.4,-.2],[.1,.3]])
        target=np.array([.4,-.1,.2,.3])
        reference=linear@np.linalg.lstsq(linear,target,rcond=None)[0]
        for scale in (np.ones(2),np.array([1e-3,1e3])):
            p,f=head_init(jax.random.PRNGKey(1),linear*scale,np.zeros(4),1.,"quadratic")
            fitted=fit_batch(p,f,jnp.asarray(target[None]),jnp.ones(1),jnp.zeros((1,2)),kind="quadratic",iterations=80)
            predicted=np.asarray(head_apply(p,f,fitted[0][0],"quadratic"))
            np.testing.assert_allclose(predicted,reference,atol=1e-8)
            self.assertLess(float(fitted[5][0]),1e-6)
            self.assertTrue(bool(fitted[7][0]))


if __name__=="__main__":
    print({"jax_backend":jax.default_backend(),"x64":jax.config.jax_enable_x64},flush=True)
    unittest.main()
