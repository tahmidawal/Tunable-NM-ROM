"""Small controls for dimension-generic training and fine-control accounting."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import heads32 as h
from fresh_models import head_init,head_apply
from fresh_learning import train_head

class Heads32Tests(unittest.TestCase):
    def test_original_training_objective_and_endpoint_shape(self):
        rng=np.random.default_rng(7090741);r,k,width=64,32,16
        linear=np.linalg.qr(rng.normal(size=(r,k)))[0]*.02;center=rng.normal(size=r)*.01;codes=rng.normal(size=(12,k))*.1
        a=codes@linear.T+center+rng.normal(size=(12,r))*.001;b=rng.normal(size=(12,r))*.01
        scales=np.linspace(.1,.3,12);seed=7090742;output_scale=.03
        cfg=dict(rank=r,head_width=width,head_steps=2,head_batch=4,head_lr=.001,code_lr=.003,code_penalty=1e-6)
        p,f=head_init(h.jax.random.PRNGKey(seed),linear,center,output_scale,'mlp',width)
        indices=np.random.default_rng(seed).integers(0,len(a),cfg['head_batch'])
        pred=np.asarray(head_apply(p,f,h.jnp.asarray(codes[indices]),'mlp'))
        expected=np.mean(np.sum((pred-a[indices])**2,axis=1)/scales[indices]**2)+cfg['code_penalty']*np.mean(codes[indices]**2)
        projected=dict(a=a.reshape(3,4,r),b=b.reshape(3,4,r),u_scale=scales,v_scale=scales)
        with tempfile.TemporaryDirectory() as temp:
            trained,frozen,newcodes,history=train_head(cfg,projected,(linear,center,codes,output_scale),'mlp',0.,seed,Path(temp))
            self.assertAlmostEqual(history[0][1],expected,places=14)
            self.assertEqual(trained['linear'].shape,(64,32));self.assertEqual(newcodes.shape,(12,32))
            self.assertEqual(float(frozen['output_scale']),output_scale)
            self.assertTrue((Path(temp)/'head.npz').is_file())

    def test_predeclared_panel_and_fine_timing_ineligibility(self):
        import json
        cfg=json.loads((Path(__file__).resolve().parent/'heads32-config.json').read_text())
        per_reflective=1+2*len(cfg['nonlinear_dts'])+1+1
        per_absorbing=1+2*len(cfg['nonlinear_dts'])+1+len(cfg['fom_cfls'])
        expected=(per_reflective+per_absorbing)*len(cfg['meshes'])*len(cfg['validation_indices'])*cfg['repetitions']
        self.assertEqual(expected,cfg['expected_timed_invocations'])
        self.assertFalse(cfg['accuracy_only_comparison_eligible']);self.assertEqual(cfg['accuracy_only_repetitions'],1)
        self.assertNotIn(cfg['accuracy_only_dt'],cfg['nonlinear_dts'])
        self.assertLess(cfg['new_latent_dimension'],cfg['weak_bank_equations'])

if __name__=='__main__':
    print({'jax_backend':h.jax.default_backend(),'x64':h.jax.config.jax_enable_x64},flush=True)
    unittest.main()
