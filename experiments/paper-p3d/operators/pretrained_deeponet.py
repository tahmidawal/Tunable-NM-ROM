"""Train-only POD supervision initializes an otherwise unchanged DeepONet.

The learned coordinate trunk remains trainable and is not replaced by POD at
inference. Branch supervision uses the actual learned-trunk field Gram metric.
No development or final targets enter either pretraining stage.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
import optax
from . import models3d as M
from . import extra_models3d as E


def summary(errors):
    return dict(mean=float(np.mean(errors)),median=float(np.median(errors)),worst=float(np.max(errors)))


def pretrain(train_x,train_y,spec,training_cfg,cfg,out,dump,checkpoint,denominators=None):
    assert spec['kind']=='deeponet3d' and jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    out=Path(out);out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    x=np.asarray(train_x);y=np.asarray(train_y);batch_count,*spatial,channels=y.shape;points=int(np.prod(spatial));rank=spec['rank']
    snapshots=np.moveaxis(y,-1,1).reshape(batch_count*channels,points)
    norms=np.linalg.norm(snapshots,axis=1)
    dn=norms**2 if denominators is None else np.asarray(denominators).reshape(-1)
    assert len(dn)==len(snapshots) and np.all(dn>0)
    weighted=snapshots/np.sqrt(dn[:,None])
    values,vectors=scipy.linalg.eigh(weighted@weighted.T,subset_by_index=[max(0,len(weighted)-rank),len(weighted)-1])
    order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    available=int(np.count_nonzero(values>values[0]*1e-12));assert available==rank,('training snapshot rank',available,rank)
    basis=weighted.T@vectors/np.sqrt(values)[None,:]
    assert np.max(np.abs(basis.T@basis-np.eye(rank)))<1e-6
    teacher=basis*np.sqrt(points);weights=values/np.sum(values)
    projection=snapshots@basis@basis.T
    teacher_errors=np.linalg.norm(projection-snapshots,axis=1)/np.sqrt(dn)
    np.savez_compressed(out/'training_teacher.npz',spatial_basis=basis,eigenvalues=values,
        denominators=dn,projection_errors=teacher_errors)
    params=M.init_model(jax.random.PRNGKey(training_cfg['seed']),spec,x.shape[-1],channels)
    checkpoint(out/'random_initialization.pkl',dict(params=params,spec=spec,seed=training_cfg['seed']))
    coordinate_indices=[i%x.shape[-1] for i in spec.get('coordinate_channels',[-3,-2,-1])]
    field_indices=[i for i in range(x.shape[-1]) if i not in coordinate_indices]
    coords=jnp.asarray(x[0][...,coordinate_indices]).reshape(points,3)
    assert np.max(np.abs(x[...,coordinate_indices]-x[0][...,coordinate_indices]))==0,'one shared coordinate mesh required'
    target=jnp.asarray(teacher);weights_j=jnp.asarray(weights)
    trunk=params['trunk'];steps=cfg['trunk_steps'];lr=optax.cosine_decay_schedule(cfg['trunk_learning_rate'],steps,alpha=.03)
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr));state=opt.init(trunk)
    @jax.jit
    def trunk_loss(p,c,t,w):return jnp.mean(jnp.sum((E.deeponet_trunk({'trunk':p},c,spec)-t)**2*w,axis=-1))
    @jax.jit
    def trunk_step(p,state,key,c,t,w):
        indices=jax.random.randint(key,(min(cfg['trunk_batch_points'],points),),0,points)
        value,gradient=jax.value_and_grad(trunk_loss)(p,c[indices],t[indices],w)
        update,state=opt.update(gradient,state,p);return optax.apply_updates(p,update),state,value
    key=jax.random.PRNGKey(training_cfg['seed']+20001);curve=[];best=float('inf');start=time.perf_counter()
    for it in range(steps):
        key,sub=jax.random.split(key);trunk,state,value=trunk_step(trunk,state,sub,coords,target,weights_j)
        if it==0 or (it+1)%cfg['checkpoint_every']==0 or it+1==steps:
            score=float(trunk_loss(trunk,coords,target,weights_j));assert np.isfinite(score)
            selected=score<best
            if selected:best=score;best_trunk=trunk;best_step=it+1
            curve.append(dict(step=it+1,training_weighted_trunk_loss=score,selected=selected,seconds=time.perf_counter()-start))
            dump(out/'trunk_curve.json',curve);checkpoint(out/'trunk_latest.pkl',dict(trunk=trunk,state=state,key=key,step=it+1))
            print('PODINIT_TRUNK',it+1,score,time.perf_counter()-start,flush=True)
            if time.perf_counter()-start>=cfg['trunk_wall_seconds']:break
    trunk_info=dict(steps_completed=it+1,best_step=best_step,best_training_loss=best,seconds=time.perf_counter()-start,
        curve=curve,selection='minimum full training weighted trunk loss')
    params={**params,'trunk':best_trunk};checkpoint(out/'trunk_selected.pkl',dict(params=params,spec=spec,info=trunk_info))
    matrix=np.asarray(E.deeponet_trunk(params,coords,spec))/np.sqrt(rank)
    bias=np.asarray(params['bias'])
    shifted=snapshots-np.tile(bias,batch_count)[:,None]
    coefficients,residuals,numerical_rank,singular=scipy.linalg.lstsq(matrix,shifted.T,lapack_driver='gelsd')
    assert numerical_rank==rank
    learned_projection=(matrix@coefficients).T+np.tile(bias,batch_count)[:,None]
    learned_errors=np.linalg.norm(learned_projection-snapshots,axis=1)/np.sqrt(dn)
    # Cholesky is a field metric, not an arbitrary coefficient Euclidean norm.
    gram=matrix.T@matrix;root=np.linalg.cholesky(gram)
    np.savez_compressed(out/'branch_teacher.npz',coefficients=coefficients.T.reshape(batch_count,channels,rank),
        gram=gram,learned_trunk_projection_errors=learned_errors)
    targets=jnp.asarray(coefficients.T.reshape(batch_count,channels,rank));root=jnp.asarray(root)
    fields=jnp.asarray(x[...,field_indices]);denom=jnp.asarray(dn.reshape(batch_count,channels))
    branch={k:params[k] for k in ['branch','branch_hidden','branch_read']}
    steps=cfg['branch_steps'];lr=optax.cosine_decay_schedule(cfg['branch_learning_rate'],steps,alpha=.03)
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(lr));state=opt.init(branch)
    @jax.jit
    def branch_error(p,x,t,d,r):
        difference=(E.deeponet_coefficients(p,x,spec)-t)@r
        return jnp.sum(difference**2,axis=-1)/d
    @jax.jit
    def branch_step(p,state,key,x,t,d,r):
        indices=jax.random.randint(key,(min(cfg['branch_batch_size'],batch_count),),0,batch_count)
        def loss(p):
            e=branch_error(p,x[indices],t[indices],d[indices],r)
            return jnp.mean(e)+.1*jnp.mean(e*e)
        value,gradient=jax.value_and_grad(loss)(p);update,state=opt.update(gradient,state,p)
        return optax.apply_updates(p,update),state,value
    best=float('inf');curve=[];start=time.perf_counter()
    for it in range(steps):
        key,sub=jax.random.split(key);branch,state,value=branch_step(branch,state,sub,fields,targets,denom,root)
        if it==0 or (it+1)%cfg['checkpoint_every']==0 or it+1==steps:
            errors=np.concatenate([np.asarray(branch_error(branch,fields[i:i+1],targets[i:i+1],denom[i:i+1],root)) for i in range(batch_count)])
            score=float(np.mean(errors));assert np.isfinite(score);selected=score<best
            if selected:best=score;best_branch=branch;best_step=it+1
            curve.append(dict(step=it+1,training_branch_relative_mse=score,selected=selected,seconds=time.perf_counter()-start))
            dump(out/'branch_curve.json',curve);checkpoint(out/'branch_latest.pkl',dict(branch=branch,state=state,key=key,step=it+1))
            print('PODINIT_BRANCH',it+1,score,time.perf_counter()-start,flush=True)
            if time.perf_counter()-start>=cfg['branch_wall_seconds']:break
    params={**params,**best_branch}
    info=dict(schema='train-only-deeponet-pod-initialization-v1',config=cfg,spec=spec,seed=training_cfg['seed'],
        training_cases=batch_count,training_snapshots=len(snapshots),training_array_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
        teacher='relative-field-error weighted POD of identical training outputs only',trunk=trunk_info,
        branch=dict(steps_completed=it+1,best_step=best_step,best_training_relative_mse=best,seconds=time.perf_counter()-start,
            selection='minimum full training coefficient error in actual learned-trunk field Gram metric'),
        teacher_projection_error=summary(teacher_errors),learned_trunk_projection_error=summary(learned_errors),
        learned_trunk_numerical_rank=int(numerical_rank),seconds=time.perf_counter()-begin,
        development_data_used=False,final_data_used=False,online_architecture_changed=False,
        following_stage='joint end-to-end fine-tuning with original field loss; all trunk and branch weights trainable')
    checkpoint(out/'initialized.pkl',dict(params=params,spec=spec,info=info));dump(out/'pretraining.json',info)
    return params,info
