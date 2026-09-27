"""Short guarded checks of full-field loss denominators and continuation state."""
import unittest,json
from training_core import *

class TrainingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck=Path('in/model.pkl')
        if not ck.exists():ck=Path(__file__).resolve().parents[1]/'separable-decoder/runs/inherited_qf/sep_poisson_N256_K16_R64.pkl'
        cls.p,cls.z,_=sc.load_pkl(ck)
        cls.U,cls.coords,_=field_dataset(source_params(7030401,8),16)
        cls.cfg=json.loads(Path(__file__).with_name('config04.json').read_text())

    def test_full_field_normalizers(self):
        p,z=self.p,jnp.asarray(self.z[:8]);truth=self.U[:,:19];coords=self.coords[:19]
        den=jnp.mean(self.U*self.U,axis=1);norm=jnp.mean(self.U*self.U)
        field=sc.head(p,z)@sc.features(p,coords).T
        per=np.asarray(jnp.mean((field-truth)**2,axis=1))
        for rel in (False,True):
            value,(rec,orth)=loss_components(p,z,truth,coords,den,norm,rel,self.cfg['orthogonality_weight'])
            expected=np.mean(per/np.asarray(den)) if rel else np.mean(per)/float(norm)
            self.assertAlmostEqual(float(rec),expected,places=10)
            self.assertAlmostEqual(float(value),expected+self.cfg['orthogonality_weight']*float(orth),places=10)
        # The full-grid denominator is deliberately different from the sampled one.
        self.assertGreater(float(jnp.linalg.norm(den-jnp.mean(truth*truth,axis=1))),0.)

    def test_reset_continuation_and_seed_prefix(self):
        c={**self.cfg,'training_steps_each':4,'warmup_steps':1,'point_batch':16,'source_batch':4}
        p=self.p;z=jnp.asarray(self.z[:8]);normalizer=jnp.mean(self.U*self.U);den=jnp.mean(self.U*self.U,axis=1)
        for relative_loss in (False,True):
            optimizer,step=training_step(c,relative_loss);state=optimizer.init((p,z));key=jax.random.PRNGKey(c['training_rng_seed'])
            initial_hash=weights_sha(p);args=((p,z),state,key,self.U,self.coords,den,normalizer)
            a=step(*args);b=step(*args);jax.block_until_ready(a)
            self.assertEqual(weights_sha(a[0][0]),weights_sha(b[0][0]))
            self.assertEqual(initial_hash,weights_sha(p))
            # Warmup starts at zero, so exercise the next actual optimizer update.
            cstate=step(a[0],a[1],a[2],self.U,self.coords,den,normalizer);jax.block_until_ready(cstate)
            self.assertNotEqual(initial_hash,weights_sha(cstate[0][0]))
            np.testing.assert_array_equal(cstate[0][0]['out_scale'],p['out_scale'])
            self.assertTrue(np.isfinite(float(cstate[3])))
        full=source_params(0,576)[:512];short=source_params(0,512)
        np.testing.assert_array_equal(full[:,0],short[:,0]);self.assertFalse(np.array_equal(full[:,1:],short[:,1:]))

    def test_training_only_code_initialization(self):
        cfg={**self.cfg,'new_code_fit_budget':3}
        codes,record=initialize_codes(self.p,self.z[:4],self.U,self.coords,cfg)
        np.testing.assert_array_equal(codes[:4],self.z[:4])
        self.assertEqual(codes.shape,(8,16));self.assertEqual(len(record['records']),4)
        self.assertTrue(all(r['finite'] for r in record['records']))
        self.assertTrue(all(r['attempts']<=3 for r in record['records']))

if __name__=='__main__':
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    print('jax_backend=gpu',jax.devices(),flush=True);unittest.main()
