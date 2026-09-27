"""Small numerical controls, no training or historical final-cohort access."""
import unittest
import numpy as np
from scipy.fft import dstn
import jax
import jax.numpy as jnp
from core import *
from scipy.interpolate import RegularGridInterpolator

class Controls(unittest.TestCase):
    def test_dst_scipy_and_roundtrip(self):
        a=np.random.default_rng(31).normal(size=(15,15))
        b=np.asarray(dst2(jnp.asarray(a)))
        self.assertLess(relative(b,dstn(a,type=1,norm='ortho')),1e-14)
        self.assertLess(relative(dst2(jnp.asarray(b)),a),1e-14)

    def test_discrete_sine_and_cg(self):
        n=16
        x=np.arange(1,n)/n
        u=np.sin(np.pi*2*x[:,None])*np.sin(np.pi*3*x[None,:])
        f=eigenvalues(n)[1,2]*u
        got=np.asarray(dst_solve(jnp.pad(jnp.asarray(f),1),jnp.asarray(eigenvalues(n))))[1:-1,1:-1]
        cg=jax.scipy.sparse.linalg.cg(lambda a:mp.neg_lap_interior(a,n+1),jnp.asarray(f),tol=1e-13)[0]
        self.assertLess(relative(got,u),1e-14)
        self.assertLess(relative(got,cg),1e-13)
        self.assertLess(relative(mp.neg_lap_interior(jnp.asarray(got),n+1),f),1e-13)

    def test_random_cg_parity(self):
        n=16
        f=np.random.default_rng(39).normal(size=(n-1,n-1))
        got=np.asarray(dst_solve(jnp.pad(jnp.asarray(f),1),jnp.asarray(eigenvalues(n))))[1:-1,1:-1]
        cg=jax.scipy.sparse.linalg.cg(lambda a:mp.neg_lap_interior(a,n+1),jnp.asarray(f),tol=1e-13)[0]
        self.assertLess(relative(got,cg),1e-12)

    def test_coarse_interpolation(self):
        source=full_source(32,source_params(7,1)[0])
        got=np.asarray(dst_coarse_solve(jnp.asarray(source),jnp.asarray(eigenvalues(16))))
        coarse=np.asarray(dst_solve(jnp.asarray(source[::2,::2]),jnp.asarray(eigenvalues(16))))
        xy=np.linspace(0,1,33)
        points=np.stack(np.meshgrid(xy,xy,indexing='ij'),axis=-1)
        expected=RegularGridInterpolator((np.linspace(0,1,17),)*2,coarse)(points)
        self.assertLess(relative(got,expected),1e-13)

    def test_checkpoint_weak_operator(self):
        ck=Path('in/model.pkl')
        if not ck.exists():
            ck=Path(__file__).resolve().parents[1]/'separable-decoder/runs/inherited_qf/sep_poisson_N256_K16_R64.pkl'
        weights,codes,cfg=sc.load_pkl(ck)
        ops=assemble(weights,codes,16,64,150)
        z=jnp.asarray(codes[0])
        u=ops['decode'](z,ops['bank'],weights)
        direct=ops['project'](u,ops['S'],ops['I'],ops['J'],jnp.ones_like(ops['W']))
        self.assertLess(relative(ops['B']@sc.head(weights,z),direct),1e-12)
        self.assertEqual(ops['info']['retained_bank_rank'],64)
        src=full_source(16,source_params(3,1)[0])
        field,info=rom_query(src,ops,0.1)
        self.assertTrue(np.isfinite(field).all())
        self.assertGreater(info['total_seconds'],0)
        self.assertAlmostEqual(info['total_seconds'],sum(info[k] for k in ('input_seconds','projection_init_seconds','solver_seconds','output_seconds')))

    def test_family_nested_and_count(self):
        ps=source_params(7090703,6)
        np.testing.assert_array_equal(ps, source_params(7090703,6))
        # Do not rely on the prefix of a differently sized sample_params call.
        self.assertFalse(np.array_equal(ps[:2],source_params(7090703,2)))
        a,b=full_source(16,ps[0]),full_source(32,ps[0])
        np.testing.assert_array_equal(a, observation(b,32,16))

if __name__=='__main__':
    assert jax.default_backend()=='gpu'
    assert jax.config.jax_enable_x64
    print('jax_backend=gpu',jax.devices(),flush=True)
    unittest.main()
