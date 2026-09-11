"""Matched fixed-encoder field-only and phase-supervised MLP32 heads."""
import json
from pathlib import Path
import time
import numpy as np
import optax
import dynamics
import pilot as base
from pilot import jax,jnp
from fresh_models import head_init,head_apply,tree_to_npz


def loss_components(p,frozen,z,w,a,b,u_scale,phase_scale,speed,stiffness):
    def one(z,w):return jax.jvp(lambda x:head_apply(p,frozen,x,'mlp'),(z,),(w,))
    pred,velocity=jax.vmap(one)(z,w)
    du,dv=pred-a,velocity-b
    field=jnp.mean(jnp.sum(du*du,axis=1)/(u_scale*u_scale))
    energy=jnp.mean(speed*speed*jnp.einsum('bi,ij,bj->b',du,stiffness,du)/(phase_scale*phase_scale))
    tangent=jnp.mean(jnp.sum(dv*dv,axis=1)/(phase_scale*phase_scale))
    return jnp.array([field,energy,tangent])


def train_endpoints(inputs,out,cfg):
    tc=cfg['head_training'];bc='dirichlet'
    linear,center,manifest=dynamics.regenerate_ladder(inputs/bc,bc,out,cfg)
    assert manifest['data_hashes_match'] and manifest['saved_initialization_matched']
    with np.load(out/'training_ladder_dirichlet.npz') as f:
        a,b,scales,parameters=f['a'],f['b'],f['case_scales'],f['parameters']
    count,frames,_=a.shape
    original=json.loads((inputs/bc/'campaign-config.json').read_text())
    bank=base.rebuild(inputs/bc,base.Grid(original['n']))
    transform=jnp.asarray(bank['transform']);stiffness=transform.T@bank['k']@transform
    latent=tc.get('latent_dimension',32)
    inverse=np.linalg.pinv(linear[:,:latent]);aa,bb=a.reshape(-1,64),b.reshape(-1,64)
    z=(aa-center)@inverse.T;w=bb@inverse.T
    scale=float(np.sqrt(np.mean(np.sum(aa*aa,axis=1))/64))
    tensors=tuple(map(jnp.asarray,(z,w,aa,bb,np.repeat(scales[:,0],frames),np.repeat(scales[:,1],frames),np.repeat(parameters[:,5],frames))))
    np.savez_compressed(out/'fixed_encoder_training.npz',linear=linear[:,:latent],center=center,inverse=inverse,
        z=z,w=w,stiffness=np.asarray(stiffness),parameters=parameters,initial_scales=scales)
    initial,frozen=head_init(jax.random.PRNGKey(tc['seed']),linear[:,:latent],center,scale,'mlp',width=128)
    endpoints={};records=[]
    arms=tc.get('objective_arms',[dict(name='trained_field',weights=[1.,0.,0.]),dict(name='trained_phase',weights=[1.,1.,1.])])
    for arm in arms:
        name,weights=arm['name'],arm['weights']
        p=initial;optimizer=optax.adam(optax.cosine_decay_schedule(tc['learning_rate'],tc['steps'],alpha=.1));state=optimizer.init(p)
        objective_weights=jnp.asarray(weights)
        def objective(p,frozen,batch,stiffness,weights):
            components=loss_components(p,frozen,*batch,stiffness)
            return components@weights,components
        @jax.jit
        def update(p,state,batch,stiffness,weights):
            (loss,components),grad=jax.value_and_grad(objective,has_aux=True)(p,frozen,batch,stiffness,weights)
            updates,state=optimizer.update(grad,state,p)
            return optax.apply_updates(p,updates),state,loss,components
        rng=np.random.default_rng(tc['seed']+1);history=[];start=time.perf_counter()
        for step in range(tc['steps']):
            indices=rng.integers(0,len(aa),tc['batch_size']);batch=tuple(x[indices] for x in tensors)
            p,state,loss,components=update(p,state,batch,stiffness,objective_weights)
            if step%200==0 or step==tc['steps']-1:
                values=np.asarray(components)
                if not np.all(np.isfinite(values)):raise RuntimeError('Nonfinite phase head training')
                history.append(dict(step=step,objective=float(loss),field=float(values[0]),energy=float(values[1]),tangent=float(values[2])))
                print('head_accuracy_training',name,step,history[-1],flush=True)
        jax.block_until_ready(p)
        endpoint=dict(p=p,frozen=frozen,codes=jnp.asarray(z));dest=out/f'head_{name}.npz';tree_to_npz(dest,endpoint)
        endpoints[name]=endpoint
        records.append(dict(method=name,seed=tc['seed'],steps=tc['steps'],batch_size=tc['batch_size'],learning_rate=tc['learning_rate'],
            objective_weights=weights,history=history,seconds_including_compilation=time.perf_counter()-start,
            checkpoint_path=dest.name,checkpoint_sha256=base.sha(dest),internal_configuration_dimension=latent,internal_phase_dimension=2*latent,
            encoder='FixedPCAaffine leftinverse on original bank coefficients; codes and their time derivatives fixed consistently.',
            scale_definitions='Field uses original suppliedu0 massL2; energy and tangent use sqrt(2 original supplied initial physical energy); stiffness displacement factor includes querycase speed squared.',
            training_codes_sha256=base.array_sha(z),training_velocities_sha256=base.array_sha(w),training_only=True))
    return endpoints,linear,center,manifest,records
