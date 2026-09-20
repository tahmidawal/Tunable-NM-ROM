"""Two-stage learned bank and nonlinear coefficient head; no POD bank."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import common as C


def train_bank(fields, cfg, out, validation_fields=None):
    u=np.asarray(fields).reshape(-1,np.prod(fields.shape[-3:]))
    n=cfg['train_intervals']; x=jnp.asarray(C.coords(n)); data=jnp.asarray(u)
    k0,k1=jax.random.split(jax.random.PRNGKey(cfg['model_seed']))
    p=C.init_bank(k0,cfg,float(np.sqrt(np.mean(u*u))))
    eta=.1*jax.random.normal(k1,(len(u),cfg['bank_rank']),dtype=jnp.float64)
    norms=jnp.maximum(jnp.mean(data*data,axis=1),1e-20)
    schedule=optax.cosine_decay_schedule(cfg['bank_learning_rate'],cfg['bank_steps'],alpha=.03)
    opt=optax.adam(schedule); state=opt.init((p,eta))
    b=min(cfg['bank_batch_states'],len(u)); m=min(cfg['bank_batch_points'],u.shape[1])

    def objective(pz,values,points,rows,cols,norms):
        p,z=pz; g=C.features(p,points[cols]); pred=z[rows]@g.T
        loss=jnp.mean((pred-values[rows[:,None],cols[None,:]])**2/norms[rows,None])
        normalized=g/jnp.maximum(p['scale'],1e-20)
        gram=normalized.T@normalized/len(cols)
        return loss+1e-6*jnp.mean((gram-jnp.eye(g.shape[1]))**2),loss

    @jax.jit
    def step(pz,state,key,values,points,norms):
        a,bkey=jax.random.split(key)
        rows=jax.random.randint(a,(b,),0,values.shape[0]);cols=jax.random.randint(bkey,(m,),0,points.shape[0])
        (_,loss),grads=jax.value_and_grad(objective,has_aux=True)(pz,values,points,rows,cols,norms)
        grads[0]['freq']=jnp.zeros_like(grads[0]['freq']);grads[0]['scale']=jnp.zeros_like(grads[0]['scale'])
        updates,state=opt.update(grads,state,pz)
        return optax.apply_updates(pz,updates),state,loss

    key=jax.random.PRNGKey(cfg['bank_minibatch_seed']); pz=(p,eta); history=[]; begin=time.perf_counter()
    best=float('inf');best_params=None;best_step=None;exit_reason='steps'
    C.checkpoint(out/'bank_partial.pkl',dict(pz=pz,state=state,key=key,step=0,cfg=cfg))
    for it in range(cfg['bank_steps']):
        key,sub=jax.random.split(key);pz,state,loss=step(pz,state,sub,data,x,norms)
        if it==0 or (it+1)%100==0:
            history.append(dict(step=it+1,relative_mse=float(loss),seconds=time.perf_counter()-begin))
        if (it+1)%cfg['checkpoint_every']==0 or it+1==cfg['bank_steps']:
            C.checkpoint(out/'bank_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
            C.dump(out/'bank_curve.json',history)
            print('BANK',history[-1],flush=True)
            if validation_fields is not None:
                candidate=C.bank_at(pz[0],n,cfg['field_chunk']);vq,vr=np.linalg.qr(candidate,mode='reduced')
                vs=np.linalg.svd(vr,compute_uv=False);assert vs[-1]>vs[0]*1e-12
                vu=np.asarray(validation_fields).reshape(len(validation_fields),-1)
                ve=np.sqrt(np.maximum(np.sum(vu*vu,axis=1)-np.sum((vu@vq)**2,axis=1),0)/np.sum(vu*vu,axis=1))
                worst=float(max(ve));selected=worst<best
                if selected:
                    best=worst;best_step=it+1;best_params=jax.tree_util.tree_map(np.asarray,pz[0])
                    C.checkpoint(out/'bank_best_training.pkl',dict(params=best_params,step=best_step,validation_projection_errors=ve,cfg=cfg))
                history.append(dict(step=it+1,validation_projection_worst=worst,validation_projection_errors=ve.tolist(),selected=selected,seconds=time.perf_counter()-begin))
                C.dump(out/'bank_curve.json',history);print('BANK_VALID',history[-1],flush=True)
            if time.perf_counter()-begin>=cfg.get('bank_wall_seconds',1e100):exit_reason='wall_budget';break
    params=best_params if best_params is not None else pz[0];g=C.bank_at(params,n,cfg['field_chunk'])
    q,r=np.linalg.qr(g,mode='reduced');sv=np.linalg.svd(r,compute_uv=False)
    assert sv[-1]>sv[0]*1e-12, ('bank rank deficient',sv.tolist())
    rotation=np.linalg.solve(r,np.eye(len(r)))
    target=u@q; norm2=np.sum(u*u,axis=1)
    perpendicular=np.maximum(norm2-np.sum(target*target,axis=1),0.)
    info=dict(seconds=time.perf_counter()-begin,steps=it+1,best_step=best_step,exit_reason=exit_reason,
              validation_projection_worst=best if best_params is not None else None,basis_hash=C.sha(q),
              condition_number=float(sv[0]/sv[-1]),orthogonality=float(np.max(np.abs(q.T@q-np.eye(len(r))))),
              training_projection_mean=float(np.mean(np.sqrt(perpendicular/norm2))),
              training_projection_worst=float(np.max(np.sqrt(perpendicular/norm2))),
              training_matrix_hash=C.sha(u),converged_claim=False)
    C.checkpoint(out/'bank.pkl',dict(params=params,rotation=rotation,info=info,cfg=cfg))
    np.savez_compressed(out/'training_coordinates.npz',target=target,norm2=norm2,perpendicular2=perpendicular)
    return params,rotation,q,target,norm2,perpendicular,info


def train_head(target,norm2,perpendicular,k,cfg,out,validation=None):
    target=jnp.asarray(target);norm2=jnp.asarray(norm2);perpendicular=jnp.asarray(perpendicular)
    key=jax.random.PRNGKey(cfg['model_seed']+k);a,b=jax.random.split(key)
    p=C.init_head(a,k,target.shape[1],cfg['head_width'])
    z=.1*jax.random.normal(b,(len(target),k),dtype=jnp.float64)
    schedule=optax.cosine_decay_schedule(cfg['head_learning_rate'],cfg['head_steps'],alpha=.03)
    opt=optax.adam(schedule);state=opt.init((p,z));batch=min(cfg['head_batch_states'],len(target))
    def objective(pz,targets,norms,perps,idx):
        p,z=pz;err=jnp.sum((C.head(p,z[idx])-targets[idx])**2,axis=1)
        # Per-state full-field relative MSE, including the fixed perpendicular floor.
        e=(err+perps[idx])/jnp.maximum(norms[idx],1e-20)
        return jnp.mean(e)+.1*jnp.mean(e**2)
    @jax.jit
    def step(pz,state,key,targets,norms,perps):
        idx=jax.random.randint(key,(batch,),0,len(targets))
        value,grad=jax.value_and_grad(objective)(pz,targets,norms,perps,idx)
        update,state=opt.update(grad,state,pz)
        return optax.apply_updates(pz,update),state,value
    pz=(p,z);key=jax.random.PRNGKey(cfg['head_minibatch_seed']+k);history=[];begin=time.perf_counter()
    best=float('inf');best_pz=None;best_step=None;exit_reason='steps'
    if validation is not None:
        import shared_rom as S
        vt,vn,vp=map(jnp.asarray,validation);fit=S.lm(C.head,240,1e-8)
        @jax.jit
        def validate(params,codes,targets,norms,perps):
            library=C.head(params,codes);eye=jnp.eye(target.shape[1],dtype=jnp.float64)
            def one(t):
                ids=jnp.argsort(jnp.sum((library-t)**2,axis=1))[:4]
                zs,stats=jax.vmap(lambda z:fit(params,eye,t,z))(codes[ids]);which=jnp.argmin(stats[:,3])
                return jnp.sum((C.head(params,zs[which])-t)**2),stats[which]
            err,stats=jax.vmap(one)(targets)
            return jnp.sqrt((err+perps)/norms),stats
    for it in range(cfg['head_steps']):
        key,sub=jax.random.split(key);pz,state,value=step(pz,state,sub,target,norm2,perpendicular)
        if it==0 or (it+1)%100==0:history.append(dict(step=it+1,objective=float(value),seconds=time.perf_counter()-begin))
        if (it+1)%cfg['checkpoint_every']==0 or it+1==cfg['head_steps']:
            C.checkpoint(out/f'head_K{k}_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
            C.dump(out/f'head_K{k}_curve.json',history);print('HEAD',k,history[-1],flush=True)
            if validation is not None:
                ve,stats=jax.device_get(validate(*pz,vt,vn,vp));worst=float(max(ve));selected=worst<best
                if selected:
                    best=worst;best_step=it+1;best_pz=jax.tree_util.tree_map(np.asarray,pz)
                    C.checkpoint(out/f'head_K{k}_best_training.pkl',dict(pz=best_pz,step=best_step,validation_errors=ve,cfg=cfg))
                history.append(dict(step=it+1,validation_best_found_worst=worst,validation_errors=ve.tolist(),validation_stats=stats.tolist(),selected=selected,seconds=time.perf_counter()-begin))
                C.dump(out/f'head_K{k}_curve.json',history);print('HEAD_VALID',k,worst,flush=True)
            if time.perf_counter()-begin>=cfg.get('head_wall_seconds',1e100):exit_reason='wall_budget';break
    p,z=best_pz if best_pz is not None else pz;prediction=np.asarray(C.head(p,z));residual=np.asarray(target)-prediction
    _,sv,vt=np.linalg.svd(residual,full_matrices=False);directions=vt.T
    errors=np.sqrt((np.sum(residual**2,axis=1)+np.asarray(perpendicular))/np.asarray(norm2))
    info=dict(k=k,seconds=time.perf_counter()-begin,steps=it+1,best_step=best_step,exit_reason=exit_reason,
              validation_best_found_worst=best if best_pz is not None else None,converged_claim=False,
              training_error_mean=float(np.mean(errors)),training_error_median=float(np.median(errors)),
              training_error_worst=float(np.max(errors)),correction_singular_values=sv.tolist(),
              correction_rule='SVD of field-orthonormal training reconstruction residuals at jointly learned training codes',
              direction_hash=C.sha(directions))
    C.checkpoint(out/f'head_K{k}.pkl',dict(params=p,codes=z,directions=directions,info=info,cfg=cfg))
    return dict(params=p,codes=z,directions=directions,info=info)
