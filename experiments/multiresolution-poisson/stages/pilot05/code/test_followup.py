"""Small frozen-checkpoint fusion and exact QR controls."""
import unittest
from followup import *

class FollowupControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck=Path('in/model.pkl')
        if not ck.exists():ck=Path(__file__).resolve().parents[1]/'separable-decoder/runs/inherited_qf/sep_poisson_N256_K16_R64.pkl'
        cls.weights,cls.codes,cls.cfg=sc.load_pkl(ck)
        cls.ops=assemble(cls.weights,cls.codes,32,64,150)
        cls.source=full_source(32,source_params(7090703,6)[0])

    def test_fused_parity(self):
        kernel=fused_kernel(self.ops)
        for tau in [.01,0.]:
            modular=rom_query(self.source,self.ops,tau)
            fused=fused_query(self.source,self.ops,tau,kernel)
            check=parity(modular,fused)
            print('parity',tau,check,flush=True)
            self.assertTrue(check['passed'])
            r=fused[1]
            self.assertAlmostEqual(r['total_seconds'],r['input_seconds']+r['fused_device_seconds']+r['output_seconds'])

    def test_qr_full_field_identity(self):
        q,r,info=qr_bank(self.ops)
        self.assertLess(info['relative_reconstruction'],1e-13)
        self.assertLess(info['orthogonality_frobenius'],1e-12)
        field=np.asarray(dst_solve(jnp.asarray(self.source),jnp.asarray(eigenvalues(32))))
        solvers=oracle_solvers(self.weights,r,self.cfg['k'],[4],self.ops['info']['trust_delta'])
        diagnostic,bank,best=oracle_fit(field,self.ops,q,r,solvers,[self.codes.mean(0)])
        self.assertLess(diagnostic['rows'][0]['qr_identity_absolute'],1e-12)
        self.assertLessEqual(diagnostic['full_bank_same_grid_error'],diagnostic['best_same_grid_error'])

if __name__=='__main__':
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    print('jax_backend=gpu',jax.devices(),flush=True)
    unittest.main()
