"""Fixed-bank initial-emphasis and tail-aware head/code continuation."""
import hashlib
import pickle
import time
from functools import partial

import heat_core as hc
import head_refine as hr
import jax
import jax.numpy as jnp
import numpy as np
import optax


def objective(pz, triangular, target, norm2, perpendicular2, indices, tail):
    params, codes = pz
    residual = hc.sc.head(params, codes[indices])@triangular.T-target[indices]
    errors2 = (jnp.sum(residual**2, axis=1)+perpendicular2[indices])/norm2[indices]
    mean = jnp.mean(errors2)
    return (1-tail)*mean+tail*jnp.sqrt(jnp.mean(errors2**2)+1e-30)


def train(base, codes0, triangular, target, norm2, perpendicular2, settings, arm):
    steps, batch = settings['updates'], settings['batch_size']
    assert batch % 2 == 0
    pz = hr.split_head(base), codes0
    schedule = optax.warmup_cosine_decay_schedule(0, settings['learning_rate'], min(200, steps//4),
                                                 steps, settings['learning_rate']*.01)
    opt = optax.adam(schedule); state = opt.init(pz)
    tail = .5 if arm == 'initial_tail' else 0.
    @jax.jit
    def step(pz, state, key, triangular, target, norm2, perpendicular2):
        if arm == 'uniform':
            indices = jax.random.randint(key, (batch,), 0, len(target))
        else:
            a,b,c = jax.random.split(key,3)
            first = 6*jax.random.randint(a,(batch//2,),0,len(target)//6)
            later = 6*jax.random.randint(b,(batch//2,),0,len(target)//6)+jax.random.randint(c,(batch//2,),1,6)
            indices = jnp.concatenate((first,later))
        value, gradient = jax.value_and_grad(objective)(pz,triangular,target,norm2,perpendicular2,indices,tail)
        update,state = opt.update(gradient,state)
        return optax.apply_updates(pz,update),state,value
    key=jax.random.PRNGKey(settings['minibatch_seed']); history=[]; begin=time.perf_counter()
    for i in range(steps):
        key,batch_key=jax.random.split(key)
        pz,state,value=step(pz,state,batch_key,triangular,target,norm2,perpendicular2)
        if i == 0 or (i+1)%2000 == 0:
            record=dict(update=i+1,loss=float(value),seconds=time.perf_counter()-begin)
            history.append(record); print('accuracy_train',arm,record,flush=True)
    jax.block_until_ready(pz)
    return hr.full_params(base,pz[0]),pz[1],dict(arm=arm,history=history,updates=steps,
        sampled_snapshots=steps*batch,seconds=time.perf_counter()-begin,
        full_relative_mse=float(objective(pz,triangular,target,norm2,perpendicular2,jnp.arange(len(target)),0.)),
        initial_relative_mse=float(objective(pz,triangular,target,norm2,perpendicular2,jnp.arange(0,len(target),6),0.)),
        later_relative_mse=float(objective(pz,triangular,target,norm2,perpendicular2,jnp.asarray([i for i in range(len(target)) if i%6]),0.)))


def prepare(base, original_codes, cfg, settings, arrays, cases, out, save_field, progress):
    """Current/opened and new development fields never enter training or lookup."""
    n=cfg['train_intervals']; times=jnp.asarray(cfg['times']); xy=jnp.asarray(hc.coords(n))
    def trajectories(draws):
        return jax.block_until_ready(jnp.stack([hc.propagate(hc.initial_field(xy,draw).reshape(n-1,n-1),
            hc.eigenvalues(n),times,cfg['diffusivity']) for draw in draws])).reshape(-1,len(xy))
    original=hc.sample_family(cfg['train_seed'],cfg['n_train'],cfg)
    added=hc.sample_family(790713,128,cfg)
    draws=np.concatenate((original,added)); assert len(original_codes)==len(draws)*len(times)
    assert not any(np.array_equal(draw,case['draw']) for draw in draws for case in cases)
    train_fields=trajectories(draws)
    target,norm2,perpendicular2=jax.block_until_ready(hr.compression(train_fields,arrays['projection']))
    evaluation=trajectories([c['draw'] for c in cases])
    projected=(evaluation@arrays['projection'])@arrays['projection'].T
    shape=(len(cases),len(times),n-1,n-1)
    bank_errors=[hc.error_metrics(np.asarray(projected).reshape(shape)[i],np.asarray(evaluation).reshape(shape)[i],n) for i in range(len(cases))]
    diagnostic=dict(intervals=n,truth=save_field(np.asarray(evaluation).reshape(shape)),
        bank_projection=save_field(np.asarray(projected).reshape(shape)),bank_errors=bank_errors,models=[])
    # This predeclared eligibility check only selects whether the head-only
    # experiment is appropriate. It never changes an arm or training schedule.
    gate=dict(threshold=settings['bank_gate'],worst_error=max(max(m['relative_current']) for m in bank_errors),
              norm='current-relative full-field L2 versus exact semidiscrete same-grid trajectory',
              failed_cases=[dict(case=cases[i]['case'],cohort=cases[i]['cohort'],errors=m['relative_current'])
                            for i,m in enumerate(bank_errors) if max(m['relative_current'])>=settings['bank_gate']])
    gate['passed']=not gate['failed_cases']
    baseline_pz=hr.split_head(base),original_codes
    baseline_metrics=dict(full_relative_mse=float(objective(baseline_pz,arrays['triangular'],target,norm2,perpendicular2,jnp.arange(len(target)),0.)),
        initial_relative_mse=float(objective(baseline_pz,arrays['triangular'],target,norm2,perpendicular2,jnp.arange(0,len(target),6),0.)),
        later_relative_mse=float(objective(baseline_pz,arrays['triangular'],target,norm2,perpendicular2,jnp.asarray([i for i in range(len(target)) if i%6]),0.)))
    progress(dict(reconstruction=diagnostic,bank_gate=gate,training_baseline=baseline_metrics))
    assert gate['passed'], gate
    np.savez_compressed(out/'training_compression.npz',target=np.asarray(target),norm2=np.asarray(norm2),
                        perpendicular2=np.asarray(perpendicular2),triangular=np.asarray(arrays['triangular']),
                        training_draws=draws,initial_codes=np.asarray(original_codes))
    model_data={'frozen':(base,original_codes)}; model_records=[]
    (out/'checkpoints').mkdir()
    for arm in ['frozen']+settings['training_arms']:
        if arm=='frozen': params,codes,info=base,original_codes,dict(arm='frozen',updates=0)
        else: params,codes,info=train(base,original_codes,arrays['triangular'],target,norm2,perpendicular2,settings,arm)
        for key in ('B','g','out_scale'):
            for a,b in zip(jax.tree.leaves(params[key]),jax.tree.leaves(base[key])): np.testing.assert_array_equal(a,b)
        path=out/'checkpoints'/f'{arm}.pkl'
        path.write_bytes(pickle.dumps(dict(params=jax.device_get(params),codes=np.asarray(codes),config=cfg,refinement=info)))
        model_records.append(dict(name=arm,path=str(path.relative_to(out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),details=info))
        model_data[arm]=(params,codes)
    # All scheduled training endpoints are saved before any head/rollout score.
    for name,(params,codes) in model_data.items():
        reconstructed,zs,infos,best,nearest=jax.block_until_ready(hr.fit_fields(params,codes,arrays['bank'],arrays['projection'],
                                              arrays['triangular'],evaluation,settings))
        infos,best,zs=np.asarray(infos),np.asarray(best),np.asarray(zs)
        metrics=[hc.error_metrics(np.asarray(reconstructed).reshape(shape)[i],np.asarray(evaluation).reshape(shape)[i],n) for i in range(len(cases))]
        diagnostic['models'].append(dict(name=name,reconstructed=save_field(np.asarray(reconstructed).reshape(shape)),
            latents=zs.tolist(),fits=infos.tolist(),best=best.tolist(),nearest_training_indices=np.asarray(nearest).tolist(),metrics=metrics))
        print('accuracy_reconstruction',name,'initial_worst',max(m['relative_current'][0] for m in metrics),
              'later_worst',max(max(m['relative_current'][1:]) for m in metrics),flush=True)
    return model_data,model_records,diagnostic


@partial(jax.jit,static_argnames=('budget','tolerance','starts'))
def fit_targets(params,codes,triangular,targets,*,budget,tolerance,starts):
    library=hc.sc.head(params,codes)@triangular.T
    nearest=jnp.argsort(jnp.sum((targets[:,None]-library[None])**2,axis=2),axis=1)[:,:starts-1]
    zs=jnp.concatenate((codes[nearest],jnp.broadcast_to(jnp.mean(codes,axis=0),(len(targets),1,codes.shape[1]))),axis=1)
    solve=hc.make_lm(hc.sc.head,budget,tolerance)
    all_zs,infos=jax.lax.map(lambda a:jax.vmap(lambda z:solve(params,triangular,a[0],z))(a[1]),(targets,zs))
    best=jnp.argmin(infos[:,:,3],axis=1)
    return all_zs[jnp.arange(len(targets)),best],all_zs,infos,best,nearest


def fine_diagnostic(models,arrays,fields,physical,n,case,settings,save_field):
    """Untimed truth-informed fits; no state flows back into an online query."""
    targets=jax.block_until_ready(jnp.asarray(fields).reshape(len(fields),-1)@arrays['projection'])
    projected=np.asarray(jax.block_until_ready(targets@arrays['projection'].T)).reshape(fields.shape)
    record=dict(intervals=n,case=case,reference='Exact semidiscrete heat evolution on the supplied grid; continuum-spectral errors separate.',
        target=save_field(np.asarray(targets)),bank_projection=save_field(projected),
        bank_same_grid=hc.error_metrics(projected,fields,n),bank_physical=hc.error_metrics(projected,physical,n),models=[])
    for name,(params,codes) in models.items():
        zs,all_zs,infos,best,nearest=jax.block_until_ready(fit_targets(params,codes,arrays['triangular'],targets,
            budget=settings['fit_budget'],tolerance=settings['fit_gradient_tolerance'],starts=settings['fit_starts']))
        predicted=np.asarray(jax.block_until_ready(hc.sc.head(params,zs)@arrays['bank'].T)).reshape(fields.shape)
        record['models'].append(dict(name=name,reconstructed=save_field(predicted),latents=np.asarray(zs).tolist(),
            all_latents=np.asarray(all_zs).tolist(),fits=np.asarray(infos).tolist(),best=np.asarray(best).tolist(),
            nearest_training_indices=np.asarray(nearest).tolist(),vs_same_grid=hc.error_metrics(predicted,fields,n),
            vs_physical=hc.error_metrics(predicted,physical,n)))
    return record
