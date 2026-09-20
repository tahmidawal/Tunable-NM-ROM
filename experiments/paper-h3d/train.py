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
        per_state=jnp.mean((pred-values[rows[:,None],cols[None,:]])**2/norms[rows,None],axis=1)
        loss=jnp.mean(per_state)+cfg.get("bank_tail_weight",0.)*jnp.mean(per_state**2)
        normalized=g/jnp.maximum(p['scale'],1e-20)
        gram=normalized.T@normalized/len(cols)
        return loss+1e-6*jnp.mean((gram-jnp.eye(g.shape[1]))**2),loss

    @jax.jit
    def step(pz,state,key,values,points,norms):
        a,bkey=jax.random.split(key)
        rows=jax.random.randint(a,(b,),0,values.shape[0]);cols=jax.random.randint(bkey,(m,),0,points.shape[0])
        (_,loss),grads=jax.value_and_grad(objective,has_aux=True)(pz,values,points,rows,cols,norms)
        grads[0]['freq']=jnp.zeros_like(grads[0]['freq']);grads[0]['scale']=jnp.zeros_like(grads[0]['scale'])
        if cfg.get('bank_gradient_clip'):
            factor=jnp.minimum(1.,cfg['bank_gradient_clip']/jnp.maximum(optax.global_norm(grads),1e-30))
            grads=jax.tree_util.tree_map(lambda v:v*factor,grads)
        updates,state=opt.update(grads,state,pz)
        return optax.apply_updates(pz,updates),state,loss

    key=jax.random.PRNGKey(cfg['bank_minibatch_seed']); pz=(p,eta); history=[]; begin=time.perf_counter()
    selected=None; selected_error=float('inf'); selected_step=None
    C.checkpoint(out/'bank_partial.pkl',dict(pz=pz,state=state,key=key,step=0,cfg=cfg))
    for it in range(cfg['bank_steps']):
        key,sub=jax.random.split(key);pz,state,loss=step(pz,state,sub,data,x,norms)
        if it==0 or (it+1)%100==0 or (it+1)%cfg['checkpoint_every']==0 or it+1==cfg['bank_steps']:
            assert np.isfinite(float(loss)), f'nonfinite bank loss at {it+1}'
            history.append(dict(step=it+1,relative_mse=float(loss),seconds=time.perf_counter()-begin))
        if cfg.get('bank_coefficient_refit_every',0) and (it+1)%cfg['bank_coefficient_refit_every']==0:
            gfit=C.bank_at(pz[0],n,cfg['field_chunk']);qfit,rfit=np.linalg.qr(gfit,mode='reduced')
            fitted=np.linalg.solve(rfit,(u@qfit).T).T
            # Exact training-only free coefficients remove auto-decoder code lag.
            # Reset their Adam moments after this discontinuous update; retain
            # spatial-bank moments and the optimizer step count.
            pz=(pz[0],jnp.asarray(fitted));adam=state[0]
            adam=adam._replace(mu=(adam.mu[0],jnp.zeros_like(pz[1])),nu=(adam.nu[0],jnp.zeros_like(pz[1])))
            state=(adam,*state[1:])
            history[-1]['training_coefficient_refit']=True
            history[-1]['coefficient_refit_finite']=bool(np.isfinite(fitted).all())
            assert np.isfinite(fitted).all()
            del gfit,qfit,rfit,fitted
        if (it+1)%cfg['checkpoint_every']==0 or it+1==cfg['bank_steps']:
            C.checkpoint(out/'bank_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
            if validation_fields is not None:
                candidate=C.bank_at(pz[0],n,cfg['field_chunk']);qb,_=np.linalg.qr(candidate,mode='reduced')
                val=np.asarray(validation_fields).reshape(-1,len(candidate));vnorm=np.sum(val*val,axis=1)
                ve=np.sqrt(np.maximum(vnorm-np.sum((val@qb)**2,axis=1),0.)/vnorm)
                vby=ve.reshape(validation_fields.shape[:2]);worst=float(np.max(vby[:,1:]))
                history[-1].update(validation_projection_evolved_worst=worst,validation_projection_by_time=vby.tolist())
                if worst<selected_error:
                    selected=pz;selected_error=worst;selected_step=it+1
                    C.checkpoint(out/'bank_selected_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
                del candidate,qb
            C.dump(out/'bank_curve.json',history)
            print('BANK',history[-1],flush=True)
    if selected is not None:pz=selected
    params,eta=pz;g=C.bank_at(params,n,cfg['field_chunk'])
    q,r=np.linalg.qr(g,mode='reduced');sv=np.linalg.svd(r,compute_uv=False)
    assert sv[-1]>sv[0]*1e-12, ('bank rank deficient',sv.tolist())
    rotation=np.linalg.solve(r,np.eye(len(r)))
    target=u@q; norm2=np.sum(u*u,axis=1)
    perpendicular=np.maximum(norm2-np.sum(target*target,axis=1),0.)
    info=dict(seconds=time.perf_counter()-begin,steps=cfg['bank_steps'],selected_step=selected_step,
              validation_projection_evolved_worst=selected_error if selected is not None else None,basis_hash=C.sha(q),
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
    initialization=dict(kind=cfg.get('head_initialization','random_joint_codes'))
    if initialization['kind']=='weighted_pca_linear_skip':
        # PCA uses only training targets. Relative-field loss induces the row
        # weight 1 / norm2; centering and scale are retained for exact replay.
        y=np.asarray(target);weights=1/np.maximum(np.asarray(norm2),1e-20)
        mean=np.average(y,axis=0,weights=weights)
        _,singular,vt=np.linalg.svd((y-mean)*np.sqrt(weights[:,None]),full_matrices=False)
        axes=vt[:k].T;raw=(y-mean)@axes
        scale=np.maximum(np.sqrt(np.mean(raw*raw,axis=0)),1e-12)
        z=jnp.asarray(raw/scale);p['skip']=jnp.asarray(scale[:,None]*axes.T)
        p['net'][-1]=(jnp.zeros_like(p['net'][-1][0]),jnp.asarray(mean))
        linear=mean+raw@axes.T
        np.testing.assert_allclose(np.asarray(C.head(p,z)),linear,rtol=1e-12,atol=1e-12)
        initialization.update(training_only=True,mean=mean.tolist(),axes=axes.tolist(),code_scale=scale.tolist(),
            singular_values=singular.tolist(),training_target_sha256=C.sha(y),training_norm2_sha256=C.sha(np.asarray(norm2)),
            initial_reconstruction_sha256=C.sha(linear))
        C.checkpoint(out/f'head_K{k}_initial.pkl',dict(params=p,codes=z,initialization=initialization,cfg=cfg))
    else:
        assert initialization['kind']=='random_joint_codes',initialization['kind']
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
    selected=None;selected_error=float('inf');selected_step=None;selected_nonstationary=None
    if validation is not None:
        import rom as R
        solve=R.lm(C.head,cfg.get('head_validation_fit_budget',400),cfg.get('head_validation_fit_tolerance',1e-8))
        @jax.jit
        def validate(params,codes,matrix,targets,norms,perps):
            library=C.head(params,codes)@matrix.T
            def one(target,norm,perp):
                order=jnp.argsort(jnp.sum((library-target)**2,axis=1))[:cfg.get('head_validation_starts',4)]
                zs,stats=jax.vmap(lambda start:solve(params,matrix,target,start))(codes[order])
                selected=jnp.argmin(stats[:,3]);difference=matrix@C.head(params,zs[selected])-target
                error=jnp.sqrt((jnp.sum(difference**2)+perp)/jnp.maximum(norm,1e-20))
                return error,stats[selected]
            return jax.vmap(one)(targets,norms,perps)
        vargs=tuple(jnp.asarray(validation[name]) for name in ('matrix','target','norm2','perpendicular2'))
        if cfg.get('head_include_initial_checkpoint',False):
            errors,stats=validate(*pz,*vargs);errors=np.asarray(errors).reshape(validation['shape']);stats=np.asarray(stats)
            assert np.isfinite(errors).all() and np.isfinite(stats).all()
            selected=pz;selected_error=float(np.max(errors[:,1:]));selected_step=0
            selected_nonstationary=int(np.count_nonzero(stats[:,2]!=1))
            history.append(dict(step=0,objective=None,seconds=time.perf_counter()-begin,
                validation_current_error_by_case_time=errors.tolist(),validation_evolved_worst=selected_error,
                validation_nonstationary_fits=selected_nonstationary,validation_fit_stats=stats.tolist(),selected=True))
            C.checkpoint(out/f'head_K{k}_selected_partial.pkl',dict(pz=pz,state=state,key=key,step=0,cfg=cfg))
            C.dump(out/f'head_K{k}_curve.json',history);print('HEAD_INITIAL',k,history[-1],flush=True)
    for it in range(cfg['head_steps']):
        key,sub=jax.random.split(key);pz,state,value=step(pz,state,sub,target,norm2,perpendicular)
        if it==0 or (it+1)%100==0 or (it+1)%cfg['checkpoint_every']==0 or it+1==cfg['head_steps']:history.append(dict(step=it+1,objective=float(value),seconds=time.perf_counter()-begin))
        if (it+1)%cfg['checkpoint_every']==0 or it+1==cfg['head_steps']:
            C.checkpoint(out/f'head_K{k}_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
            if validation is not None:
                errors,stats=validate(*pz,*vargs);errors=np.asarray(errors).reshape(validation['shape']);stats=np.asarray(stats)
                assert np.isfinite(errors).all() and np.isfinite(stats).all()
                worst=float(np.max(errors[:,1:]));nonstationary=int(np.count_nonzero(stats[:,2]!=1))
                chosen=worst<selected_error
                history[-1].update(validation_current_error_by_case_time=errors.tolist(),validation_evolved_worst=worst,
                    validation_nonstationary_fits=nonstationary,validation_fit_stats=stats.tolist(),selected=chosen)
                if chosen:
                    selected=pz;selected_error=worst;selected_step=it+1;selected_nonstationary=nonstationary
                    C.checkpoint(out/f'head_K{k}_selected_partial.pkl',dict(pz=pz,state=state,key=key,step=it+1,cfg=cfg))
            C.dump(out/f'head_K{k}_curve.json',history);print('HEAD',k,history[-1],flush=True)
    if selected is not None:pz=selected
    p,z=pz;prediction=np.asarray(C.head(p,z));residual=np.asarray(target)-prediction
    _,sv,vt=np.linalg.svd(residual,full_matrices=False);directions=vt.T
    errors=np.sqrt((np.sum(residual**2,axis=1)+np.asarray(perpendicular))/np.asarray(norm2))
    info=dict(k=k,seconds=time.perf_counter()-begin,steps=cfg['head_steps'],converged_claim=False,
              initialization=initialization,
              training_error_mean=float(np.mean(errors)),training_error_median=float(np.median(errors)),
              training_error_worst=float(np.max(errors)),correction_singular_values=sv.tolist(),
              correction_rule='SVD of field-orthonormal training reconstruction residuals at jointly learned training codes',
              direction_hash=C.sha(directions))
    if validation is not None:
        info.update(selected_step=selected_step,validation_selection='minimum worst evolved current-relative best-found fit on fixed development fields',
            validation_evolved_worst=selected_error,validation_nonstationary_fits=selected_nonstationary,
            validation_fit_budget=cfg.get('head_validation_fit_budget',400),validation_fit_tolerance=cfg.get('head_validation_fit_tolerance',1e-8),
            validation_fit_starts=cfg.get('head_validation_starts',4))
    C.checkpoint(out/f'head_K{k}.pkl',dict(params=p,codes=z,directions=directions,info=info,cfg=cfg))
    return dict(params=p,codes=z,directions=directions,info=info)
