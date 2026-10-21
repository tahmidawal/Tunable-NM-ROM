"""Field-metric PCA initialization and minibatch nonlinear coefficient fitting."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import ns3d_model as D


def initialize(params, C, Rb, k, seed, initial_variance, n):
    rank = Rb.shape[0]
    p = D.init(jax.random.PRNGKey(seed), k, rank, width=rank, n_ff=params['B'].shape[1])
    Y = jnp.asarray(C) @ jnp.asarray(Rb).T
    weight = 1 / jnp.asarray(initial_variance)
    center = jnp.sum(weight[:, None] * Y, axis=0) / jnp.sum(weight)
    centered = Y-center
    _, basis = jnp.linalg.eigh(centered.T @ (weight[:, None] * centered))
    basis = basis[:, -k:][:, ::-1]
    scores = centered @ basis
    code_scale = jnp.sqrt(jnp.mean(scores*scores)) / .25
    Z = scores / code_scale
    linear = jnp.linalg.solve(jnp.asarray(Rb), basis*code_scale).T
    bias = jnp.linalg.solve(jnp.asarray(Rb), center)
    layers = list(p['h']); layers[-1] = (jnp.zeros_like(layers[-1][0]), bias)
    candidate = {**params, 'h': layers, 'h_lin': linear}
    expected = center + scores @ basis.T
    actual = D.head(candidate, Z) @ jnp.asarray(Rb).T
    error = float(jnp.linalg.norm(actual-expected)/jnp.linalg.norm(expected))
    assert error < 1e-10
    return candidate, np.asarray(Z), dict(kind='weighted physical-metric affine PCA plus initially zero nonlinear residual',
        latent_dimension=k, head_width=rank, initialization_relative_parity=error,
        weighted_PCA='inverse trajectory initial mean-square velocity', code_scale=float(code_scale))


def train(params, Z, C, Rb, cfg, out, initial_variance, seed):
    out=Path(out); steps=cfg['capacity_head_steps']; seconds=cfg['capacity_head_seconds']
    target=jnp.asarray(C) @ jnp.asarray(Rb).T; metric=jnp.asarray(Rb)
    denominator=jnp.asarray(initial_variance)*(3*cfg['n']**3)/Rb.shape[0]
    theta={key:params[key] for key in ('h','h_lin')}; pair=(theta,jnp.asarray(Z))
    batch=min(cfg.get('head_batch',128),len(Z))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(optax.warmup_cosine_decay_schedule(0.,.001,min(200,steps//10+1),steps,.00002)))
    state=opt.init(pair); key=jax.random.PRNGKey(seed)
    @jax.jit
    def update(pair,state,key,target,metric,denominator):
        ids=jax.random.randint(key,(batch,),0,len(target))
        def loss(pair):
            theta,codes=pair
            residual=D.head(theta,codes[ids])@metric.T-target[ids]
            return jnp.mean(residual*residual/denominator[ids,None])
        value,gradient=jax.value_and_grad(loss)(pair)
        if not cfg.get('free_codes',True):
            gradient=(gradient[0],jnp.zeros_like(gradient[1]))
        delta,state=opt.update(gradient,state,pair)
        return optax.apply_updates(pair,delta),state,value
    start=time.monotonic(); curve=[]
    for step in range(steps):
        key,sub=jax.random.split(key); pair,state,value=update(pair,state,sub,target,metric,denominator)
        if step==0 or (step+1)%500==0 or step+1==steps:
            row=dict(step=step+1,relative_mse=float(value),seconds=time.monotonic()-start);curve.append(row)
            assert np.isfinite(row['relative_mse'])
            print('PCA_HEAD',row,flush=True)
            if (step+1)%5000==0 or step==0 or step+1==steps:
                D.checkpoint(out,{**params,**pair[0]},pair[1],cfg,dict(curve=curve,complete=False))
            if row['seconds']>=seconds:break
    fitted={**params,**pair[0]}
    info=dict(steps=step+1,requested_steps=steps,seconds=time.monotonic()-start,curve=curve,
        stopping='steps' if step+1==steps else 'wall_budget',batch=batch,
        objective='mean squared physical coefficient error / trajectory initial mean-square velocity',
        converged_claim=False,free_codes=cfg.get('free_codes',True))
    D.checkpoint(out,fitted,pair[1],cfg,info)
    return fitted,np.asarray(pair[1]),info
