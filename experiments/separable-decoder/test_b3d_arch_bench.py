"""Sub-minute common-evaluator controls; independent full-field references."""
from types import SimpleNamespace
import os
import unittest
import numpy as np
import jax
import jax.numpy as jnp

import b3d_arch_baseline as baseline
import b3d_arch_bench as bench
import b3d_arch_pilot as pilot
import b3d_common as b3


def setUpModule():
    print('jax_backend=' + jax.default_backend(), flush=True)
    assert jax.default_backend() == 'gpu'
    assert jax.config.x64_enabled
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'


class ArchitectureBenchTests(unittest.TestCase):
    def test_tangent_and_stationarity_against_full_field_lstsq(self):
        rng = np.random.default_rng(93)
        q, _ = np.linalg.qr(rng.normal(size=(23,7)), mode='reduced')
        jac = rng.normal(size=(7,3))
        velocity = rng.normal(size=23)
        residual = rng.normal(size=23)
        vc, rc = q.T@velocity, q.T@residual
        vperp = velocity-q@vc
        actual = bench.geometry_metrics(jac,rc,np.dot(residual,residual),vc,
                                         np.dot(vperp,vperp),np.dot(velocity,velocity))
        jf = q@jac
        dz = np.linalg.lstsq(jf,velocity,rcond=1e-10)[0]
        projection = jf@np.linalg.lstsq(jf,residual,rcond=1e-10)[0]
        self.assertAlmostEqual(actual['tangent_relative'],np.linalg.norm(velocity-jf@dz)/np.linalg.norm(velocity),places=12)
        self.assertAlmostEqual(actual['invariant_stationarity'],np.linalg.norm(projection)/np.linalg.norm(residual),places=12)
        # Same tangent range under a very different latent scaling.
        scaled = bench.geometry_metrics(jac@np.diag([.001,2.,1000.]),rc,np.dot(residual,residual),vc,
                                         np.dot(vperp,vperp),np.dot(velocity,velocity))
        self.assertAlmostEqual(actual['invariant_stationarity'],scaled['invariant_stationarity'],places=10)

    def test_rank_deficiency_and_zero_velocity(self):
        jac = np.asarray([[1.,2.],[0.,0.],[0.,0.]])
        metric = bench.geometry_metrics(jac,np.asarray([1.,1.,0.]),2.,np.zeros(3),0.,0.)
        self.assertEqual(metric['rank'],1)
        self.assertIsNone(metric['tangent_relative'])
        self.assertEqual(metric['tangent_absolute'],0.)
        empty = bench.geometry_metrics(np.zeros((3,2)),np.ones(3),3.,np.ones(3),1.,4.)
        self.assertEqual(empty['rank'],0)
        self.assertAlmostEqual(empty['tangent_relative'],1.)

    def test_rhs_matches_independent_signed_stencil(self):
        rng = np.random.default_rng(7)
        n, nu = 5,.03
        u = rng.normal(size=(3,3,3)); dx=1/(n-1)
        padded = np.pad(u,1)
        expected = np.zeros_like(u)
        for i,j,k in np.ndindex(u.shape):
            pos=np.asarray([i+1,j+1,k+1]); c=u[i,j,k]
            lap,adv=0.,0.
            for axis in range(3):
                lo=pos.copy(); hi=pos.copy(); lo[axis]-=1; hi[axis]+=1
                left,right=padded[tuple(lo)],padded[tuple(hi)]
                lap += (left-2*c+right)/dx**2
                adv += c*((c-left) if c>0 else (right-c))/dx
            expected[i,j,k]=nu*lap-adv
        actual=nu*b3.lap_3d(jnp.asarray(u.ravel()),n)-b3.upwind_adv_field_3d(jnp.asarray(u.ravel()),n)
        np.testing.assert_allclose(actual,expected.ravel(),rtol=1e-13,atol=1e-13)

    def test_training_control_and_bad_initialization(self):
        rng=np.random.default_rng(12)
        orth,_=np.linalg.qr(rng.normal(size=(5,5)))
        shared=dict(anchor=orth[:,:2],center=np.zeros(5),latent_scale=np.ones(2),output_scale=np.asarray(1.))
        z=rng.normal(size=(24,2))
        target=z@shared['anchor'].T + .4*z[:,0,None]**2*orth[:,2]
        train=dict(target=target,norm2=np.sum(target**2,axis=1)+.1,perpendicular2=np.full(24,.1))
        p,f,zz,info=bench.train_model(baseline,shared,z,train,steps=80,lr=.01,batch=24)
        self.assertLess(info['global_mse_final'],info['global_mse_initial'])
        bench.check_model(baseline,p,f,zz)
        def bad_init(key,shared_):
            pp,ff=baseline.init(key,shared_); pp['bias']=pp['bias']+.1
            return pp,ff
        bad=SimpleNamespace(NAME='bad',MODE='codes',init=bad_init,apply=baseline.apply)
        with self.assertRaises(AssertionError):
            bench.train_model(bad,shared,z,train,steps=2,batch=24)

    def test_duplicate_optimizer_seeds_rejected(self):
        self.assertEqual(bench.parse_seeds('200,201'),[200,201])
        with self.assertRaises(AssertionError):
            bench.parse_seeds('200,200')

    def test_pilot_adapter_preserves_fields_and_derivatives(self):
        rng = np.random.default_rng(61)
        g = rng.normal(size=(17, 5))
        q, r = np.linalg.qr(g, mode='reduced')
        u, _ = np.linalg.qr(rng.normal(size=(5, 2)), mode='reduced')
        shared = dict(anchor=u, center=rng.normal(size=5), latent_scale=np.ones(2), output_scale=np.asarray(1.))
        p, f = baseline.init(jax.random.PRNGKey(12), shared)
        # A nonzero nonlinear head catches an adapter that silently uses old MLP fields.
        weights, bias = p['residual'][-1]
        p['residual'][-1] = (weights + .02, bias + .03)
        payload = dict(kind='b3d_arch_checkpoint', bank_params={}, trainable=p, frozen=f, r=r)
        adapted, head, head_np = pilot.adapt(payload, baseline)
        z = jnp.asarray([.3, -.4])
        expected = q @ np.asarray(baseline.apply(p, f, z))
        np.testing.assert_allclose(g @ head(adapted, z), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(head(adapted, z), head_np(adapted, z), rtol=1e-12, atol=1e-12)
        jac = jax.jacfwd(lambda zz: g @ head(adapted, zz))(z)
        direct_jac = q @ jax.jacfwd(lambda zz: baseline.apply(p, f, zz))(z)
        np.testing.assert_allclose(jac, direct_jac, rtol=1e-12, atol=1e-12)

    def test_promotion_rejects_failed_or_missing_control(self):
        names = ['F5_nonnegativity_train_val', 'D1_bank_vs_meshfree_numpy',
                 'D2_lineage_rlite', 'D3_rank_of_A', 'D4_heldout_oracle_validation']
        report = dict(complete=True, pilot_passed=True, gates={
            name: dict(passed=True, control_fired=True, M_stability_pass=True) for name in names})
        self.assertTrue(pilot.promotion_passed(report))
        report['gates'][names[-1]]['control_fired'] = False
        self.assertFalse(pilot.promotion_passed(report))
        del report['gates'][names[-1]]['control_fired']
        self.assertFalse(pilot.promotion_passed(report))


if __name__ == '__main__':
    print('jax_backend='+jax.default_backend(),flush=True)
    assert jax.default_backend()=='gpu'
    unittest.main()
