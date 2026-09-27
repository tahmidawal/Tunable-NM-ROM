"""PDE-agnostic full-field relative-loss training with validation checkpoints."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from . import models3d as M


def train(train_x,train_y,valid_x,valid_y,spec,cfg,out,dump,checkpoint,train_denominators=None,valid_denominators=None,initial_params=None):
    """Arrays are BXYZC; output channels pack (time,component), component fastest.

    cfg declares steps, wall_seconds, batch_size, seed, learning_rate,
    validation_every, components_per_output. Saves actual curves and optimizer
    state. Every checkpoint selection uses worst validation time/case error;
    unopened test fields must never be passed here.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x=jnp.asarray(train_x,dtype=jnp.float64);y=jnp.asarray(train_y,dtype=jnp.float64)
    vx=jnp.asarray(valid_x,dtype=jnp.float64);vy=jnp.asarray(valid_y,dtype=jnp.float64)
    params=M.init_model(jax.random.PRNGKey(cfg['seed']),spec,x.shape[-1],y.shape[-1]) if initial_params is None else jax.device_put(initial_params)
    assert all(p.dtype==jnp.float64 for p in jax.tree_util.tree_leaves(params))
    channels=cfg.get('components_per_output',1);nt=y.shape[-1]//channels
    schedule=optax.cosine_decay_schedule(cfg['learning_rate'],cfg['steps'],alpha=.03)
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(schedule));state=opt.init(params)
    key=jax.random.PRNGKey(cfg['seed']+1);batch=min(cfg['batch_size'],len(x))
    def default_denominators(target):
        return jnp.sum(target.reshape(*target.shape[:-1],nt,channels)**2,axis=(1,2,3,5))
    dn=default_denominators(y) if train_denominators is None else jnp.asarray(train_denominators)
    vdn=default_denominators(vy) if valid_denominators is None else jnp.asarray(valid_denominators)
    assert dn.shape==(len(x),nt) and vdn.shape==(len(vx),nt)
    def errors(pred,target,denominator):
        pred=pred.reshape(*pred.shape[:-1],nt,channels);target=target.reshape(pred.shape)
        axes=(1,2,3,5)
        return jnp.sum((pred-target)**2,axis=axes)/jnp.maximum(denominator,1e-20)
    @jax.jit
    def step(p,state,key,x,y,denominators):
        indices=jax.random.randint(key,(batch,),0,len(x));a=x[indices];b=y[indices]
        def loss(p):
            e=errors(M.apply_model(p,a,spec),b,denominators[indices])
            return jnp.mean(e)+.1*jnp.mean(e*e)
        value,grads=jax.value_and_grad(loss)(p)
        update,state=opt.update(grads,state,p)
        return optax.apply_updates(p,update),state,value
    @jax.jit
    def evaluate(p,a,b,d):return errors(M.apply_model(p,a,spec),b,d)
    best=float('inf');best_params=None;curve=[];start=time.perf_counter();exit_reason='steps'
    for it in range(cfg['steps']):
        key,sub=jax.random.split(key);params,state,value=step(params,state,sub,x,y,dn)
        if it==0 or (it+1)%cfg.get('curve_every',50)==0:
            val=float(value)
            if not np.isfinite(val):raise FloatingPointError(f'nonfinite training loss at {it+1}')
            curve.append(dict(step=it+1,training_objective=val,seconds=time.perf_counter()-start))
        if it==0 or (it+1)%cfg['validation_every']==0 or it+1==cfg['steps']:
            e=np.concatenate([np.asarray(evaluate(params,vx[i:i+1],vy[i:i+1],vdn[i:i+1])) for i in range(len(vx))])
            if not np.isfinite(e).all():raise FloatingPointError(f'nonfinite validation error at {it+1}')
            worst=float(np.max(np.sqrt(e)));median=float(np.median(np.max(np.sqrt(e),axis=1)))
            selected=worst<best
            if selected:
                best=worst;best_params=params;best_step=it+1
                checkpoint(out/'best.pkl',dict(params=params,spec=spec,config=cfg,step=it+1,
                     validation_error_by_case_time=np.sqrt(e),selection='minimum worst validation error under declared denominator'))
            curve.append(dict(step=it+1,validation_worst=worst,validation_case_median=median,
                              validation_error_by_case_time=np.sqrt(e).tolist(),selected=selected,seconds=time.perf_counter()-start))
            dump(out/'curve.json',curve)
            checkpoint(out/'latest.pkl',dict(params=params,state=state,key=key,spec=spec,config=cfg,step=it+1))
            print('OPERATOR',spec['kind'],it+1,'val_worst',worst,'seconds',time.perf_counter()-start,flush=True)
            if time.perf_counter()-start>=cfg['wall_seconds']:
                exit_reason='wall_budget';break
    info=dict(spec=spec,config=cfg,parameter_count=M.parameter_count(params),steps_completed=it+1,
              initialization='fresh random' if initial_params is None else 'explicit retained pretrained parameters',
              best_step=best_step,best_validation_worst=best,seconds=time.perf_counter()-start,
              exit_reason=exit_reason,normalization=cfg.get('normalization','current-output norm' if train_denominators is None else 'explicit per-case/time squared norm'),converged_claim=False,parameter_dtype='float64',fft_dtype='complex128')
    dump(out/'training.json',info)
    return best_params,info
