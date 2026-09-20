"""Bounded Heat3D training + frozen-bank rank ladder on development data."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import common as C
import train as T
import rom as R


def run(cfg,out,smoke=False):
    out.mkdir(parents=True,exist_ok=True); (out/'fields').mkdir(exist_ok=True)
    assert jax.default_backend()=='gpu',jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
    print('jax_backend=gpu x64=True precision=highest',flush=True)
    begin=time.perf_counter()
    record=dict(schema='paper-heat3d-result-v1',config=cfg,source_commit=os.environ.get('SOURCE_COMMIT'),
                job_id=os.environ.get('SLURM_JOB_ID'),gpu=jax.devices()[0].device_kind,jax_version=jax.__version__,
                backend=jax.default_backend(),x64=True,matmul_precision='highest',smoke=smoke,complete=False,
                status='verification',final_cohort_opened=False,verification={},bank={},heads=[],meshes=[],invocations=[],
                data_contract='supplied full interior nodal initial field only; decoder and operators receive no generator descriptors',
                output_contract='all six interior fields, including reconstructed t=0 for ROMs and exact supplied t=0 for DST',
                timing_contract='synchronized device query and complete host query; input upload, cold fit, latent evolution, dense readout and host output copy; setup/training separate')
    save=lambda:C.dump(out/'result.json',record)
    save();record['verification']['heat']=C.verify(cfg);record['verification']['rom']=R.verify();save()
    train_p=C.family(cfg['train_seed'],cfg['train_count']);valid_p=C.family(cfg['validation_seed'],cfg['validation_count'])
    assert not any(np.array_equal(a,b) for a in train_p for b in valid_p)
    C.dump(out/'cohorts.json',dict(training_parameters=train_p.tolist(),validation_parameters=valid_p.tolist(),
             train_sha256=C.sha(train_p),validation_sha256=C.sha(valid_p),reserved_final_seed=cfg['reserved_final_seed'],
             final_cohort_opened=False))
    record['status']='generating_training_data';save()
    data=C.dataset(cfg['train_intervals'],train_p,cfg)
    record['status']='training_bank';save()
    validation_training_mesh=C.dataset(cfg['train_intervals'],valid_p,cfg)
    params,rotation,basis,target,norm2,perpendicular,bank_info=T.train_bank(data,cfg,out,
              validation_training_mesh if cfg.get('bank_validation_selection',False) else None)
    record['bank']=bank_info;save();models=[]
    for k in cfg['latent_dimensions']:
        record['status']=f'training_head_K{k}';save()
        model=T.train_head(target,norm2,perpendicular,k,cfg,out);models.append(model)
        record['heads'].append(model['info']);save()
    operator_models=[]
    if cfg.get('operators'):
        from operators import training as OT
        from operators import heat_adapter as OH
        scale=float(np.sqrt(np.mean(data[:,0]**2)))
        train_x,train_y=OH.arrays(data,scale);valid_x,valid_y=OH.arrays(validation_training_mesh,scale)
        record['operators']=[]
        for setting in cfg['operators']:
            name=setting['name'];spec=setting['model'];record['status']=f'training_{name}';save()
            op,info=OT.train(train_x,train_y,valid_x,valid_y,spec,setting['training'],out/'operators'/name,C.dump,C.checkpoint)
            info.update(name=name,physical_scale=scale,training_parameters_sha256=C.sha(train_p),validation_parameters_sha256=C.sha(valid_p),
                        input='supplied initial field divided by training RMS, plus three coordinate channels',
                        output='five evolved fields in physical units, supplied initial field prepended at inference')
            record['operators'].append(info);operator_models.append((name,spec,op,scale,info));save()
            C.checkpoint(out/'operators'/name/'adapter.pkl',dict(params=op,spec=spec,physical_scale=scale,info=info))
        del train_x,train_y,valid_x,valid_y
    del data,validation_training_mesh

    # Finer continuum-spectral references: empirical refinement remains visible.
    record['status']='reference_refinement';save()
    lo,hi=cfg['reference_intervals'];physical={};uncertainty=[]
    for case,p in enumerate(valid_p):
        low=C.dataset(lo,np.asarray([p]),cfg,True)[0]
        high=C.dataset(hi,np.asarray([p]),cfg,True)[0]
        uncertainty.append(C.metrics(low,C.restrict(high,hi,lo))['current_all'])
        for n in cfg['evaluation_intervals']:
            physical[(n,case)]=C.restrict(high,hi,n).copy()
        print('REFERENCE',case,uncertainty[-1],flush=True)
    record['reference']=dict(intervals=[lo,hi],current_relative_refinement=uncertainty,
             threshold=cfg['reference_budget'],passed=bool(max(uncertainty)<cfg['reference_budget']),
             status='empirical continuum spectral refinement; not a proved error bound')
    save()
    for n in cfg['evaluation_intervals']:
        record['status']=f'prepare_mesh_{n}';save();setup_begin=time.perf_counter()
        bank=C.bank_at(params,n,cfg['field_chunk'])@rotation
        test,a,lam,triples=R.assemble(bank,n,cfg['weak_tests'])
        truths=C.dataset(n,valid_p,cfg)
        training=C.dataset(n,train_p,cfg)
        ranks=sorted(set([cfg['bank_rank']]+[k+q for k in cfg['latent_dimensions'] for q in cfg['q_ladder']]))
        pod,pod_info=R.pod_models(training,n,ranks,cfg);del training
        methods=R.linear_weak(bank,a,lam,test,cfg);methods.update(pod)
        methods['linear_bank_galerkin_exact']=R.bank_galerkin(bank,n,cfg)
        metadata={name:dict(kind='linear_control') for name in methods}
        lam_full=C.eigenvalues(n);times=jnp.asarray(cfg['times']);nu=cfg['diffusivity']
        methods['dst_exact']=lambda u,lf=lam_full:C.propagate(u,lf,times,nu)
        metadata['dst_exact']=dict(kind='full_order')
        for name,spec,op,scale,info in operator_models:
            from operators import heat_adapter as OH
            methods[name]=OH.engine(op,spec,scale,n)
            metadata[name]=dict(kind='neural_operator',model=spec,parameter_count=info['parameter_count'],physical_scale=scale,
                                training_intervals=cfg['train_intervals'],resolution_transfer=n!=cfg['train_intervals'])
            if n!=cfg['train_intervals'] and n%cfg['train_intervals']==0:
                native_name=name+'_native_grid_interpolated'
                methods[native_name]=OH.native_engine(op,spec,scale,cfg['train_intervals'],n)
                metadata[native_name]=dict(kind='neural_operator',model=spec,parameter_count=info['parameter_count'],physical_scale=scale,
                    training_intervals=cfg['train_intervals'],resolution_transfer=False,
                    input_restriction='nested nodal sampling',readout='trilinear interpolation with explicit zero boundary nodes; dense output charged')
        mesh=dict(intervals=n,bank_sha256=C.sha(bank),weak_operator_sha256=C.sha(a),weak_tests=len(triples),
                  weak_singular_values=np.linalg.svd(a,compute_uv=False).tolist(),pod=pod_info,quadrature=[],representation=[])
        for model in models:
            k=model['info']['k'];decoded=np.asarray(C.head(model['params'],model['codes']))
            best,fit_stats=R.best_found_fields(model,bank,truths,cfg)
            for case,truth in enumerate(truths):
                mesh['representation'].append(dict(case=case,k=k,head_best_found=C.metrics(best[case],truth),
                         fit_stats=fit_stats[case].tolist(),nonstationary_fits=int(np.count_nonzero(fit_stats[case,:,2]!=1))))
            np.savez_compressed(out/'fields'/f'N{n}_K{k}_best_found.npz',prediction=best,stats=fit_stats)
            try:
                indices,weights,eq=R.fit_quadrature(bank,test,decoded,cfg)
                exact=np.asarray([u[0].reshape(-1)@test for u in truths]);sample=np.asarray([u[0].reshape(-1)[indices]@weights for u in truths])
                errors=np.linalg.norm(sample-exact,axis=1)/np.maximum(np.linalg.norm(exact,axis=1),1e-300)
                eq.update(k=k,validation_moment_errors=errors.tolist(),certificate_threshold=cfg['quadrature_certificate'],
                          certified=bool(max(errors)<cfg['quadrature_certificate']))
                np.savez_compressed(out/f'eq_N{n}_K{k}.npz',indices=indices,weighted_tests=weights)
            except (RuntimeError,ValueError) as exc:
                eq=dict(k=k,certified=False,error=str(exc));indices=None;weights=None
            mesh['quadrature'].append(eq)
            for q in cfg['q_ladder']:
                if k+q>cfg['bank_rank']:continue
                name=f'nmrom_K{k}_q{q}_dense'
                methods[name]=R.engine(model,bank,a,lam,test,np.arange(len(bank)),q,cfg)
                metadata[name]=dict(kind='nonlinear_rom',k=k,q=q,cold_start='dense diagnostic',quadrature_certified=None)
                if indices is not None:
                    name=f'nmrom_K{k}_q{q}_eq'
                    methods[name]=R.engine(model,bank,a,lam,weights,indices,q,cfg)
                    metadata[name]=dict(kind='nonlinear_rom',k=k,q=q,cold_start='NNLS sampled',quadrature_certified=eq['certified'])
            # Half-step diagnostic for the highest nonredundant correction rung.
            q=max(q for q in cfg['q_ladder'] if k+q<=cfg['bank_rank'])
            name=f'nmrom_K{k}_q{q}_dense_dt_half'
            methods[name]=R.engine(model,bank,a,lam,test,np.arange(len(bank)),q,cfg,dt=cfg['dt']/2)
            metadata[name]=dict(kind='nonlinear_rom',k=k,q=q,cold_start='dense diagnostic',dt=cfg['dt']/2)
            # Orthonormal bank projection is only a diagnostic, never a deployed neural model.
        qb,rb=np.linalg.qr(bank,mode='reduced')
        for case,truth in enumerate(truths):
            projection=(truth.reshape(len(truth),-1)@qb)@qb.T
            mesh['representation'].append(dict(case=case,bank_projection=C.metrics(projection,truth)))
        mesh['setup_seconds']=time.perf_counter()-setup_begin
        mesh['methods']=metadata;record['meshes'].append(mesh);save()
        # Compilation is outside timings and warmed queries are never selected as minima.
        record['status']=f'compile_mesh_{n}';save()
        sample=jnp.asarray(truths[0,0])
        for name,method in methods.items():
            compilation=time.perf_counter();jax.block_until_ready(method(sample))
            print('COMPILED',n,name,round(time.perf_counter()-compilation,3),flush=True)
        rng=np.random.default_rng(cfg['timing_seed']+n)
        record['status']=f'evaluate_mesh_{n}';save()
        for case,truth in enumerate(truths):
            np.savez_compressed(out/'fields'/f'N{n}_case{case}_reference.npz',same_grid=truth,physical=physical[(n,case)])
            for rep in range(cfg['repetitions']):
                for name in rng.permutation(list(methods)):
                    method=methods[name];C.burn(cfg['burn_seconds'])
                    start=time.perf_counter();u=jax.device_put(truth[0]);u.block_until_ready();input_end=time.perf_counter()
                    value=method(u);jax.block_until_ready(value);device_end=time.perf_counter()
                    value=jax.device_get(value);finish=time.perf_counter()
                    if isinstance(value,tuple):
                        pred,initial_stats,step_stats,coefficients=value
                        initial_stats=np.asarray(initial_stats);step_stats=np.asarray(step_stats)
                        chosen=int(np.argmin(initial_stats[:,3]))
                        nonstationary=int(initial_stats[chosen,2]!=1)+int(np.count_nonzero((step_stats[:,2]!=1)|(step_stats[:,5]>cfg['lm_tolerance'])))
                        counters=dict(initial_stats=initial_stats.tolist(),step_stats=step_stats.tolist(),
                                     selected_initial_start=chosen,initial_stationary=bool(initial_stats[chosen,2]==1),nonstationary_solves=nonstationary,
                                     full_gradient_worst=float(np.max(step_stats[:,5])),
                                     attempted_iterations=int(np.sum(initial_stats[:,0])+np.sum(step_stats[:,0])))
                    else:pred=value;counters=dict(nonstationary_solves=0)
                    pred=np.asarray(pred).reshape(truth.shape)
                    finite=bool(np.isfinite(pred).all())
                    row=dict(intervals=n,case=case,repetition=rep,method=str(name),
                             input_ms=(input_end-start)*1000,device_ms=(device_end-input_end)*1000,
                             total_ms=(finish-start)*1000,finite=finite,**counters)
                    if finite:
                        row.update(same_grid=C.metrics(pred,truth),physical=C.metrics(pred,physical[(n,case)]))
                    if rep==0:
                        path=out/'fields'/f'N{n}_case{case}_{name}.npz';np.savez_compressed(path,prediction=pred)
                        row['field_file']=str(path.relative_to(out));row['field_sha256']=C.sha(pred)
                    record['invocations'].append(row)
                save();print('CASE',n,case,'REP',rep,flush=True)
        del bank,test,truths,methods,pod
    record['status']='complete';record['complete']=True;record['elapsed_seconds']=time.perf_counter()-begin
    save();summarize(record,out)


def summarize(record,out):
    groups={}
    for row in record['invocations']:groups.setdefault((row['intervals'],row['method']),[]).append(row)
    table=[]
    for (n,name),rows in sorted(groups.items()):
        times=np.array([r['device_ms'] for r in rows]);med=float(np.median(times));cases={r['case']:r for r in rows}
        finite=[r for r in cases.values() if r['finite']]
        errors=[r['same_grid']['current_evolved'] for r in finite]
        table.append(dict(intervals=n,method=name,cases=len(cases),invocations=len(rows),device_ms_median=med,
                          device_ms_repetitions=times.tolist(),total_ms_median=float(np.median([r['total_ms'] for r in rows])),
                          time_outliers_above_1p5_median=int(np.count_nonzero(times>1.5*med)),
                          same_grid_current_evolved_median=float(np.median(errors)) if errors else None,
                          same_grid_current_evolved_worst=float(max(errors)) if errors else None,
                          physical_current_evolved_worst=max(r['physical']['current_evolved'] for r in finite) if finite else None,
                          nonfinite_cases=len(cases)-len(finite),
                          cases_with_nonstationary_solves=sum(r['nonstationary_solves']>0 for r in cases.values())))
    C.dump(out/'summary.json',dict(schema='paper-heat3d-summary-v1',source_commit=record['source_commit'],
             job_id=record['job_id'],gpu=record['gpu'],complete=record['complete'],smoke=record['smoke'],rows=table,
             final_cohort_opened=False,interpretation='development pilot; convergence, physical-reference qualification and EQ certification remain separate'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--verify-only',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.verify_only:
        print(json.dumps(dict(heat=C.verify(cfg),rom=R.verify()),indent=2));raise SystemExit(0)
    if args.smoke:
        cfg.update(train_intervals=8,evaluation_intervals=[8],reference_intervals=[8,16],train_count=4,validation_count=1,
                   bank_rank=8,latent_dimensions=[2],bank_steps=3,head_steps=3,weak_tests=32,q_ladder=[0,2],
                   bank_width=16,head_width=16,fourier_features=4,bank_batch_states=4,bank_batch_points=64,
                   quadrature_candidates=128,quadrature_fit_rows=64,quadrature_decoder_snapshots=4,
                   repetitions=1,burn_seconds=.001,lm_budget=8,times=[0.,.1,.2],field_chunk=1024,
                   bank_coefficient_refit_every=1,checkpoint_every=1)
        for setting in cfg.get('operators',[]):
            setting['model'].update(width=2,modes=[2,2,2],depth=1,padding=1,levels=2)
            if setting['model']['kind'] in ('deeponet3d','transolver3d'):
                setting['model'].update(rank=4,trunk_width=8,pool_bins=2,heads=2,slices=4,patch=2,reference_grid=2)
            setting['training'].update(steps=3,wall_seconds=20,batch_size=2,validation_every=2)
    run(cfg,Path(args.out),args.smoke)
