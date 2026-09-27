"""Short GPU controls for DST coefficients, charged lookup and solved fields."""
import unittest
from speed_core import *
from kernel_solver import specialized_kernel,specialized_query,solution_agreement

class SpeedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck=Path('in/model.pkl')
        if not ck.exists():ck=Path(__file__).resolve().parents[1]/'separable-decoder/runs/inherited_qf/sep_poisson_N256_K16_R64.pkl'
        cls.params,cls.codes,_=sc.load_pkl(ck)
        cls.ops=assemble(cls.params,cls.codes,32,64,150)
        cls.cache=weak_code_cache(cls.ops,cls.codes)
        cls.source=full_source(32,source_params(7090703,6)[0])
        cls.limits=dict(field_relative=1e-6,latent_relative=1e-4,objective_initial_scaled=1e-5)

    def test_coefficients_and_lookup(self):
        gate=projection_agreement(self.source,self.ops,self.cache)
        self.assertLess(gate['coefficient_relative'],1e-11);self.assertTrue(gate['lookup_index_matches'])
        fm=project_source_dst(jnp.asarray(self.source),self.ops['I'],self.ops['J'],self.ops['W'])
        z,index,distance,gap=nearest_code(fm,self.cache['predictions'],self.cache['codes'])
        cpu=np.sum((np.asarray(self.cache['predictions'])-np.asarray(fm)[None,:])**2,axis=1)
        self.assertEqual(int(index),int(np.argmin(cpu)))
        np.testing.assert_array_equal(z,self.codes[int(index)])
        self.assertAlmostEqual(float(distance),float(np.sqrt(cpu.min())),places=12)
        self.assertGreater(float(gap),0.);print('projection_and_lookup',gate,flush=True)

    def test_tied_cache_candidates_are_visible(self):
        predictions=jnp.array([[1.,2.],[1.,2.],[4.,5.]])
        codes=jnp.arange(48,dtype=jnp.float64).reshape(3,16)
        z,index,distance,gap=nearest_code(jnp.array([1.,2.]),predictions,codes)
        self.assertEqual(int(index),0);self.assertEqual(float(distance),0.);self.assertEqual(float(gap),0.)
        np.testing.assert_array_equal(z,codes[0])

    def test_complete_query_and_original_control(self):
        ops=self.ops;cache=self.cache
        old=specialized_kernel(ops,make_lm_kernel(ops,150,True,False))
        kernels={(p,i):make_speed_kernel(ops,150,p,i) for p in ('skinny_sine_products','forward_dst_and_gather') for i in ('mean_training_code','nearest_cached_scaled_weak_prediction')}
        for tau in (.01,0.):
            answers={(p,i):speed_query(self.source,ops,cache,tau,k,p,i) for (p,i),k in kernels.items()}
            control=specialized_query(self.source,ops,tau,old)
            check=solution_agreement(control,answers[('skinny_sine_products','mean_training_code')],self.limits)
            self.assertTrue(check['passed'])
            for initialization in ('mean_training_code','nearest_cached_scaled_weak_prediction'):
                thin=answers[('skinny_sine_products',initialization)];fft=answers[('forward_dst_and_gather',initialization)]
                check=solution_agreement(thin,fft,self.limits);self.assertTrue(check['passed'])
                for _,row in (thin,fft):
                    self.assertAlmostEqual(row['total_seconds'],row['input_seconds']+row['fused_device_seconds']+row['output_seconds'],places=10)
                    self.assertLessEqual(row['max_linear_backward_error'],1e-12)
                    self.assertEqual(row['absolute_tau_threshold'],tau*row['initial_residual'])
                    if initialization=='nearest_cached_scaled_weak_prediction':
                        np.testing.assert_array_equal(row['initial_latent'],self.codes[row['selected_training_code_index']])
                print('field_projection_agreement',tau,initialization,check,flush=True)

if __name__=='__main__':
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    print('jax_backend=gpu',jax.devices(),flush=True);unittest.main()
