"""Sub-minute loss/gradient parity and multires diagnostic integration smoke."""
import json
import pickle
from pathlib import Path
import heat_core as hc
import head_refine as hr
import accuracy_training as at
from run_pilot import assemble
import jax
import jax.numpy as jnp
import numpy as np

def main():
    root=Path(__file__).parent
    cfg=json.loads((root/'config-pilot.json').read_text())
    settings=json.loads((root/'config-accuracy.json').read_text())
    prior=root/'runs/accuracy10/smoke02/outputs/checkpoints'
    models={name:(jax.tree.map(jnp.asarray,(c:=pickle.loads((prior/f'{name}.pkl').read_bytes()))['params']),jnp.asarray(c['codes']))
            for name in ['frozen','uniform','initial_only','initial_tail']}
    params,codes=models['frozen']; arrays,_=assemble(params,codes,64,cfg)
    draw=hc.sample_family(790711,1,cfg)[0]
    fields=hc.propagate(hc.initial_field(jnp.asarray(hc.coords(64)),draw).reshape(63,63),hc.eigenvalues(64),jnp.asarray(cfg['times']),cfg['diffusivity'])
    target,norm2,perp=hr.compression(fields.reshape(6,-1),arrays['projection'])
    pz=hr.split_head(params),codes[:6]; ix=jnp.arange(6)
    for tail in [0.,.5]:
        def full(pz):
            residual=hc.sc.head(pz[0],pz[1])@arrays['bank'].T-fields.reshape(6,-1)
            errors2=jnp.sum(residual**2,axis=1)/norm2
            return (1-tail)*jnp.mean(errors2)+tail*jnp.sqrt(jnp.mean(errors2**2)+1e-30)
        left,lg=jax.value_and_grad(full)(pz)
        right,rg=jax.value_and_grad(at.objective)(pz,arrays['triangular'],target,norm2,perp,ix,tail)
        np.testing.assert_allclose(left,right,rtol=1e-11,atol=1e-12)
        for a,b in zip(jax.tree.leaves(lg),jax.tree.leaves(rg)): np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
    saved=[]
    def save(a):
        saved.append(np.asarray(a)); return dict(index=len(saved)-1)
    diagnostic=at.fine_diagnostic(models,arrays,np.asarray(fields),np.asarray(fields),64,0,settings,save)
    assert len(diagnostic['models'])==4
    for fit in diagnostic['models']:
        assert np.asarray(fit['fits']).shape==(6,4,5)
        np.testing.assert_array_equal(np.argmin(np.asarray(fit['fits'])[:,:,3],axis=1),fit['best'])
    print(json.dumps(dict(passed=True,loss_and_gradient_parity=True,fine_models=4,all_start_records=96,
                         backend=jax.default_backend(),x64=jax.config.jax_enable_x64)))

if __name__=='__main__':main()
