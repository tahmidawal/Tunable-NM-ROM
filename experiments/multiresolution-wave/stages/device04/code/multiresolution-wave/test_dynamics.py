"""Small independent mathematical controls; safe for the local smoke budget."""
import unittest
import numpy as np
import scipy.linalg
import dynamics as d
from dynamics import jax, jnp
from fresh_models import head_init
from fresh_rom import weak_acceleration

class DynamicsTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7090721)
        self.basis=rng.normal(size=(9,3))*np.array([.1,.4,1.1])
        self.center=rng.normal(size=9)*.2
        raw=rng.normal(size=(9,9)); self.k=raw.T@raw+.1*np.eye(9)
        raw=rng.normal(size=(9,5)); self.d=raw@raw.T*.04
        self.q=np.array([.2,-.4,.1]);self.w=np.array([-.1,.7,.3]);self.c=1.13

    def test_nonzero_offset_damped_propagation(self):
        generator=np.asarray(d.affine_generator(jnp.asarray(self.basis),jnp.asarray(self.center),jnp.asarray(self.k),jnp.asarray(self.d),self.c))
        mass=self.basis.T@self.basis
        expected=np.zeros((7,7));expected[:3,3:6]=np.eye(3)
        expected[3:6,:3]=np.linalg.solve(mass,-self.c**2*self.basis.T@self.k@self.basis)
        expected[3:6,3:6]=np.linalg.solve(mass,-self.c*self.basis.T@self.d@self.basis)
        expected[3:6,-1]=np.linalg.solve(mass,-self.c**2*self.basis.T@self.k@self.center)
        np.testing.assert_allclose(generator,expected,atol=5e-13)
        states,prop=d.affine_evolve(jnp.asarray(generator),jnp.asarray(self.q),jnp.asarray(self.w),.05,observations=7)
        initial=np.r_[self.q,self.w,1.]
        truth=np.stack([scipy.linalg.expm(expected*i*.05)@initial for i in range(7)])
        np.testing.assert_allclose(states,truth,atol=2e-12)
        self.assertGreater(np.linalg.norm(expected[3:6,-1]),.1)
        dropped=expected.copy();dropped[3:6,-1]=0
        self.assertGreater(np.linalg.norm(scipy.linalg.expm(dropped*.3)@initial-truth[-1]),.01)

    def test_zero_nonlinearity_matches_manifold_acceleration(self):
        p,f=head_init(jax.random.PRNGKey(4),self.basis,self.center,.2,"mlp",width=8)
        acceleration,*_=weak_acceleration(p,f,jnp.asarray(self.q),jnp.asarray(self.w),jnp.asarray(self.c**2*self.k),jnp.asarray(self.c*self.d),"mlp")
        generator=d.affine_generator(jnp.asarray(self.basis),jnp.asarray(self.center),jnp.asarray(self.k),jnp.asarray(self.d),self.c)
        expected=generator@jnp.asarray(np.r_[self.q,self.w,1.])
        np.testing.assert_allclose(acceleration,expected[3:6],atol=4e-12)

    def test_zero_representability_and_coordinate_gauge(self):
        q=np.linalg.pinv(self.basis)@(-self.center)
        zero_fit=self.center+self.basis@q
        np.testing.assert_allclose(self.basis.T@zero_fit,0,atol=1e-14)
        self.assertGreater(np.linalg.norm(zero_fit),.1)
        qq,rr=np.linalg.qr(self.basis)
        gen1=d.affine_generator(jnp.asarray(self.basis),jnp.asarray(self.center),jnp.asarray(self.k),jnp.asarray(self.d),self.c)
        gen2=d.affine_generator(jnp.asarray(qq),jnp.asarray(self.center),jnp.asarray(self.k),jnp.asarray(self.d),self.c)
        s1,_=d.affine_evolve(gen1,jnp.asarray(self.q),jnp.asarray(self.w),.05,observations=7)
        s2,_=d.affine_evolve(gen2,jnp.asarray(rr@self.q),jnp.asarray(rr@self.w),.05,observations=7)
        a1,b1=d.affine_coefficients(s1,jnp.asarray(self.basis),jnp.asarray(self.center))
        a2,b2=d.affine_coefficients(s2,jnp.asarray(qq),jnp.asarray(self.center))
        np.testing.assert_allclose(a1,a2,atol=2e-13);np.testing.assert_allclose(b1,b2,atol=2e-12)

if __name__=="__main__":
    print({"jax_backend":jax.default_backend(),"x64":jax.config.jax_enable_x64},flush=True)
    unittest.main()
