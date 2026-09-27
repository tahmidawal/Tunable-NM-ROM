"""Sampled-field auto-decoder training; no physical descriptors or time inputs."""
import json
from pathlib import Path
import pickle
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .decoders import decode_points


def _host(tree):
    return jax.tree_util.tree_map(np.asarray,tree)


def save_checkpoint(path,params,Z,cfg,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    with temporary.open('wb') as f:
        pickle.dump(dict(params=_host(params),Z=_host(Z),config=cfg.to_dict(),extra=extra or {}),f,protocol=5)
    temporary.replace(path)


def train(params,Z,states,coords,config,*,steps,seed,scales=None,point_weights=None,
          batch_size=64,point_batch=256,lr=1e-3,code_lr_factor=3.,out_dir,stage_name,
          lr_total_steps=None,start_step=0):
    """Train existing parameter/code arrays; deterministic resumable sampling.

    states=(snapshots,points,components), coords=(points,2); scales is either a
    component vector or snapshot/component array. Weights are integration masses.
    Final checkpoints include optimizer state and random stream, not just weights.
    """
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    states=jnp.asarray(states,dtype=jnp.float64);coords=jnp.asarray(coords,dtype=jnp.float64)
    S,P,C=states.shape
    scales=jnp.ones((S,C)) if scales is None else jnp.broadcast_to(jnp.asarray(scales),(S,C))
    scales=jnp.maximum(scales,1e-12)
    weights=jnp.ones(P)/P if point_weights is None else jnp.asarray(point_weights)/jnp.sum(point_weights)
    schedule=optax.cosine_decay_schedule(lr,lr_total_steps or steps,alpha=.1)
    popt=optax.adam(schedule);zopt=optax.adam(lambda n:code_lr_factor*schedule(n))
    po=popt.init(params);zo=zopt.init(Z);key=jax.random.PRNGKey(seed)
    history=[];completed=0
    resume=out/(stage_name+'_trainstate.pkl')
    if resume.exists():
        with resume.open('rb') as f:saved=pickle.load(f)
        assert saved['config']==config.to_dict() and saved['steps']==steps and saved['seed']==seed
        params,Z,po,zo,key=jax.tree_util.tree_map(jnp.asarray,(saved['params'],saved['Z'],saved['po'],saved['zo'],saved['key']))
        completed=saved['completed'];history=saved['history']

    def loss(p,codes,si,pi):
        prediction=jax.vmap(lambda z:decode_points(p,z,coords[pi],config))(codes[si])
        residual=(prediction-states[si[:,None],pi[None,:]])/scales[si,None,:]
        # Uniform point sampling with explicit importance correction reproduces
        # the full integration-weighted loss in expectation.
        fit=jnp.mean(jnp.sum(jnp.mean(residual**2,axis=-1)*weights[pi][None,:]*P,axis=-1)/point_batch)
        return fit+1e-7*jnp.mean(codes[si]**2)

    @jax.jit
    def update(params,Z,po,zo,key,states_arg,coords_arg,scales_arg,weights_arg):
        # Arguments below deliberately retain large inputs in the executable API;
        # closed-over Python references are not constants in the compiled graph.
        def batch_loss(p,codes,si,pi):
            pred=jax.vmap(lambda z:decode_points(p,z,coords_arg[pi],config))(codes[si])
            r=(pred-states_arg[si[:,None],pi[None,:]])/scales_arg[si,None,:]
            return jnp.mean(jnp.mean(r**2,axis=-1)*weights_arg[pi][None,:]*P)+1e-7*jnp.mean(codes[si]**2)
        key,ks,kp=jax.random.split(key,3)
        si=jax.random.randint(ks,(batch_size,),0,S);pi=jax.random.randint(kp,(point_batch,),0,P)
        value,(pg,zg)=jax.value_and_grad(batch_loss,argnums=(0,1))(params,Z,si,pi)
        pu,po=popt.update(pg,po,params);zu,zo=zopt.update(zg,zo,Z)
        return optax.apply_updates(params,pu),optax.apply_updates(Z,zu),po,zo,key,value

    t0=time.perf_counter()
    for step in range(completed,steps):
        params,Z,po,zo,key,value=update(params,Z,po,zo,key,states,coords,scales,weights)
        if (step+1)%100==0 or step+1==steps:
            v=float(value)
            if not np.isfinite(v):raise FloatingPointError(f'{stage_name} training became nonfinite at {step+1}')
            history.append(dict(step=step+1,loss=v,seconds=time.perf_counter()-t0))
            print(f'TRAIN {stage_name} step={step+1}/{steps} loss={v:.8g} seconds={time.perf_counter()-t0:.1f}',flush=True)
        if (step+1)%1000==0 or step+1==steps:
            payload=dict(params=_host(params),Z=_host(Z),po=_host(po),zo=_host(zo),key=np.asarray(key),
                         completed=step+1,history=history,config=config.to_dict(),steps=steps,seed=seed)
            temp=resume.with_suffix('.part')
            with temp.open('wb') as f:pickle.dump(payload,f,protocol=5)
            temp.replace(resume)
            save_checkpoint(out/(stage_name+'.pkl'),params,Z,config,dict(steps=step+1,seed=seed,history=history))
            (out/(stage_name+'_history.json')).write_text(json.dumps(history,indent=2)+'\n')
    return params,Z,history
