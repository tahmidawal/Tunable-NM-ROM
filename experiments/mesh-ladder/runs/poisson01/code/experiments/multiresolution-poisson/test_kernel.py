"""Guarded GPU smoke: conditioned SPD systems and actual checkpoint LM parity."""
import json,unittest
from kernel_solver import *
from followup import fused_kernel,fused_query


def synthetic_checks(config):
    rng=np.random.default_rng(7030316);rows=[]
    solve=jax.jit(lambda A,b:guarded_gj(A,b,config['linear_backward_limit']))
    for cond,limit in zip(config['synthetic_condition_numbers'],config['synthetic_forward_limits']):
        for repeat in range(3):
            q,_=np.linalg.qr(rng.normal(size=(16,16)))
            A=(q*np.geomspace(1.,1./cond,16))@q.T
            exact=rng.normal(size=16);b=A@exact
            x,eta,fallback,proposal=solve(jnp.asarray(A),jnp.asarray(b));x=np.asarray(x)
            relative_error=np.linalg.norm(x-exact)/np.linalg.norm(exact)
            generic=np.linalg.solve(A,b)
            row=dict(requested_condition=cond,actual_condition=float(np.linalg.cond(A)),repeat=repeat,
                known_solution_relative_error=float(relative_error),generic_solution_relative_error=relative(x,generic),
                backward_error=float(eta),fallback_count=int(fallback),proposal_backward_error=float(proposal),
                forward_limit=limit)
            assert relative_error<=limit and float(eta)<=config['linear_backward_limit'],row
            rows.append(row)
    # A non-SPD matrix is outside the specialized contract and must fall back.
    A=np.eye(16);A[0,0]=-1.;b=np.ones(16)
    x,eta,fallback,_=solve(jnp.asarray(A),jnp.asarray(b))
    assert int(fallback)==1 and relative(x,np.linalg.solve(A,b))<1e-14
    return rows


class KernelControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config=Path(__file__).with_name('config03.json');cls.config=json.loads(config.read_text())
        ck=Path('in/model.pkl')
        if not ck.exists():ck=Path(__file__).resolve().parents[1]/'separable-decoder/runs/inherited_qf/sep_poisson_N256_K16_R64.pkl'
        cls.weights,cls.codes,cls.cfg=sc.load_pkl(ck)
        cls.ops=assemble(cls.weights,cls.codes,32,64,150)
        cls.source=full_source(32,source_params(7090703,6)[0])

    def test_conditioned_and_fallback(self):
        rows=synthetic_checks(self.config)
        print('synthetic_max_forward',max(r['known_solution_relative_error'] for r in rows),flush=True)
        print('synthetic_max_backward',max(r['backward_error'] for r in rows),flush=True)

    def test_actual_trajectory_and_agreement(self):
        ops=self.ops;fm=ops['project'](jnp.asarray(self.source),ops['S'],ops['I'],ops['J'],ops['W'])
        generic=make_lm_kernel(ops,150,specialized=False,trace=True)
        special=make_lm_kernel(ops,150,specialized=True,trace=True)
        timed=make_lm_kernel(ops,150,specialized=True,trace=False)
        kernel=specialized_kernel(ops,timed)
        for tau in [.01,0.]:
            old=ops['solve'](ops['z0'],fm,jnp.asarray(tau))
            control,cs,ct=generic(ops['z0'],fm,jnp.asarray(tau))
            self.assertLess(relative(control[0],old[0]),1e-12)
            np.testing.assert_array_equal(np.asarray(control[3:]),np.asarray(old[3:]))
            candidate,diag,trace=special(ops['z0'],fm,jnp.asarray(tau))
            count=int(candidate[5]);A,b,x=map(np.asarray,trace[:3])
            for mat,rhs,got in zip(A[:count],b[:count],x[:count]):
                eta=np.linalg.norm(mat@got-rhs)/(np.linalg.norm(mat)*np.linalg.norm(got)+np.linalg.norm(rhs)+1e-300)
                self.assertLessEqual(eta,self.config['linear_backward_limit'])
            ref=rom_query(self.source,ops,tau)
            got=specialized_query(self.source,ops,tau,kernel)
            check=solution_agreement(ref,got,self.config['agreement_limits'])
            print('actual_agreement',tau,check,'fallbacks',got[1]['fallback_count'],flush=True)
            self.assertTrue(check['passed'])
            self.assertLess(relative(candidate[0],got[1]['latent']),1e-10)

if __name__=='__main__':
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    print('jax_backend=gpu',jax.devices(),flush=True);unittest.main()
