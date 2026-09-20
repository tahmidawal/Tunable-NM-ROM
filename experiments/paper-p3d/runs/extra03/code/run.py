"""Current learned Poisson3D training and frozen-model development panels."""
from __future__ import annotations
import argparse
import json
import os
import pickle
import hashlib
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import common as C
import train as T
import shared_rom as S
import poisson as P


def summarize(record,out):
    groups={}
    for row in record['invocations']:groups.setdefault((row['intervals'],row['method']),[]).append(row)
    rows=[]
    for (n,name),group in sorted(groups.items()):
        times=np.asarray([r['device_ms'] for r in group]);median=float(np.median(times));cases={r['case']:r for r in group}
        finite=[r for r in cases.values() if r['finite']]
        rows.append(dict(intervals=n,method=name,cases=len(cases),invocations=len(group),device_ms_median=median,
              device_ms_repetitions=times.tolist(),total_ms_median=float(np.median([r['total_ms'] for r in group])),
              timing_outliers_above_1p5_median=int(np.count_nonzero(times>1.5*median)),
              same_grid_error_mean=float(np.mean([r['same_grid_error'] for r in finite])) if finite else None,
              same_grid_error_median=float(np.median([r['same_grid_error'] for r in finite])) if finite else None,
              same_grid_error_worst=max(r['same_grid_error'] for r in finite) if finite else None,
              physical_error_worst=max(r['physical_error'] for r in finite) if finite else None,
              nonfinite_cases=len(cases)-len(finite),nonstationary_cases=sum(not r['stationary'] for r in cases.values()),
              cases_above_same_grid_target=sum(r['same_grid_error']>record['config']['same_grid_target'] for r in finite)))
    C.dump(out/'summary.json',dict(schema='paper-poisson3d-summary-v1',source_commit=record['source_commit'],
        job_id=record['job_id'],gpu=record['gpu'],complete=record['complete'],smoke=record['smoke'],rows=rows,
        reference=record.get('reference'),final_cohort_opened=False,
        interpretation='single-seed development pilot; training convergence, stationarity and physical-reference qualification remain separate'))


def run(cfg,out,smoke=False):
    out.mkdir(parents=True,exist_ok=True);(out/'fields').mkdir(exist_ok=True)
    assert jax.default_backend()=='gpu'
    assert jax.config.jax_enable_x64 and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')=='highest'
    assert cfg['bank_width']>=cfg['bank_rank']
    print('jax_backend=gpu x64=True precision=highest',flush=True)
    begin=time.perf_counter()
    record=dict(schema='paper-poisson3d-result-v1',config=cfg,source_commit=os.environ.get('SOURCE_COMMIT'),
        job_id=os.environ.get('SLURM_JOB_ID'),gpu=jax.devices()[0].device_kind,jax_version=jax.__version__,
        backend='gpu',x64=True,matmul_precision='highest',smoke=smoke,complete=False,status='verification',
        final_cohort_opened=False,bank={},heads=[],meshes=[],invocations=[],
        data_contract='supplied full interior nodal forcing only; no generator descriptors online',
        output_contract='complete interior solution field, zero boundary known to every method',
        timing_contract='same-invocation accuracy/device/host time; upload, cold solve and dense readout; offline training/setup separate')
    save=lambda:C.dump(out/'result.json',record)
    save();record['verification']=P.verify();save()
    train_p=C.family(cfg['train_seed'],cfg['train_count']);valid_p=C.family(cfg['validation_seed'],cfg['validation_count'])
    C.dump(out/'cohorts.json',dict(training_parameters=train_p.tolist(),validation_parameters=valid_p.tolist(),
        train_sha256=C.sha(train_p),validation_sha256=C.sha(valid_p),reserved_final_seed=cfg['reserved_final_seed'],final_cohort_opened=False))
    record['status']='generating_training_data';save()
    fields=P.dataset(cfg['train_intervals'],train_p)
    record['status']='training_bank';save()
    validation_fields=P.dataset(cfg['train_intervals'],valid_p)
    reuse=Path(cfg['reuse_checkpoint_directory']) if cfg.get('reuse_checkpoint_directory') else None
    prior=None
    if reuse:
        prior=json.loads((reuse/'result.json').read_text())
        assert prior['complete'] and not prior['final_cohort_opened']
        for key in ['train_seed','validation_seed','train_count','validation_count','train_intervals','bank_rank']:
            assert cfg[key]==prior['config'][key],('checkpoint cohort/architecture mismatch',key)
        old_cohorts=json.loads((reuse/'cohorts.json').read_text())
        assert old_cohorts['train_sha256']==C.sha(train_p) and old_cohorts['validation_sha256']==C.sha(valid_p)
        cached=pickle.loads((reuse/'bank.pkl').read_bytes())
        params,rotation,info=cached['params'],cached['rotation'],cached['info']
        basis=C.bank_at(params,cfg['train_intervals'],cfg['field_chunk'])@rotation
        u=fields.reshape(len(fields),-1);target=u@basis;norm2=np.sum(u*u,axis=1)
        perp=np.maximum(norm2-np.sum(target*target,axis=1),0.)
        assert C.sha(u)==info['training_matrix_hash']
        C.checkpoint(out/'bank.pkl',cached)
        record['reused_checkpoints']=dict(source_commit=prior['source_commit'],job_id=prior['job_id'],
            files={str(p.relative_to(reuse)):hashlib.sha256(p.read_bytes()).hexdigest() for p in reuse.rglob('*') if p.is_file()})
    else:
        params,rotation,basis,target,norm2,perp,info=T.train_bank(fields[:,None],cfg,out,validation_fields)
    record['bank']=info;save();models=[]
    vu=validation_fields.reshape(len(validation_fields),-1);vt=vu@basis;vn=np.sum(vu*vu,axis=1);vp=np.maximum(vn-np.sum(vt*vt,axis=1),0)
    for k in cfg['latent_dimensions']:
        record['status']=f'training_head_K{k}';save()
        if reuse:
            model=pickle.loads((reuse/f'head_K{k}.pkl').read_bytes())
            C.checkpoint(out/f'head_K{k}.pkl',model)
        else:model=T.train_head(target,norm2,perp,k,cfg,out,(vt,vn,vp))
        model={**model,'params':jax.device_put(model['params']),'codes':jax.device_put(model['codes'])}
        jax.block_until_ready((model['params'],model['codes']))
        models.append(model);record['heads'].append(model['info']);save()
    operator_models=[];record['operators']=[]
    if cfg.get('operators'):
        from operators import poisson_adapter as O
        from operators import training as OT
        train_sources=np.stack([np.asarray(P.source(cfg['train_intervals'],p)) for p in train_p])
        valid_sources=np.stack([np.asarray(P.source(cfg['train_intervals'],p)) for p in valid_p])
        scales=dict(input=float(np.sqrt(np.mean(train_sources**2))),output=float(np.sqrt(np.mean(fields**2))))
        tx,ty=O.arrays(train_sources,fields,scales);vx,vy=O.arrays(valid_sources,validation_fields,scales)
        for i,entry in enumerate(cfg['operators']):
            record['status']='training_'+entry['name'];save()
            ocfg=entry.get('training',{**cfg['operator_training'],'seed':cfg['operator_training']['seed']+i})
            if entry.get('reuse'):
                assert reuse is not None
                saved=pickle.loads((reuse/'operators'/entry['name']/'best.pkl').read_bytes())
                assert saved['spec']==entry['spec']
                op=jax.device_put(saved['params']);jax.block_until_ready(op)
                oinfo=next(dict(x) for x in prior['operators'] if x['name']==entry['name'])
                assert oinfo['training_input_sha256']==C.sha(train_sources) and oinfo['training_target_sha256']==C.sha(fields)
                C.checkpoint(out/'operators'/entry['name']/'best.pkl',saved)
                C.dump(out/'operators'/entry['name']/'training.json',oinfo)
            else:op,oinfo=OT.train(tx,ty,vx,vy,entry['spec'],ocfg,out/'operators'/entry['name'],C.dump,C.checkpoint)
            oinfo.update(name=entry['name'],scales=scales,training_input_sha256=C.sha(train_sources),training_target_sha256=C.sha(fields),
                validation_input_sha256=C.sha(valid_sources),validation_target_sha256=C.sha(validation_fields))
            record['operators'].append(oinfo);operator_models.append((entry['name'],op,entry['spec'],scales));save()
    record['status']='reference_refinement';save()
    lo,hi=cfg['reference_intervals'];physical={};uncertainty=[]
    for case,p in enumerate(valid_p):
        low=P.dataset(lo,np.asarray([p]),True)[0];high=P.dataset(hi,np.asarray([p]),True)[0]
        uncertainty.append(P.error(low,C.restrict(high,hi,lo)))
        for n in cfg['evaluation_intervals']:physical[(n,case)]=C.restrict(high,hi,n).copy()
        print('REFERENCE',case,uncertainty[-1],flush=True)
    record['reference']=dict(intervals=[lo,hi],relative_refinement=uncertainty,threshold=cfg['reference_budget'],
        passed=bool(max(uncertainty)<cfg['reference_budget']),status='empirical continuum-spectral refinement, not a proved error bound');save()
    for n in cfg['evaluation_intervals']:
        record['status']=f'prepare_mesh_{n}';save();setup=time.perf_counter()
        bank=C.bank_at(params,n,cfg['field_chunk'])@rotation
        test,a,projection,lam,triples=P.assemble(bank,n,cfg['weak_tests'])
        truths=P.dataset(n,valid_p);sources=np.stack([np.asarray(P.source(n,p)) for p in valid_p])
        training=fields if n==cfg['train_intervals'] else P.dataset(n,train_p)
        ranks=sorted(set([cfg['bank_rank']]+[k+q for k in cfg['latent_dimensions'] for q in cfg['q_ladder']]))
        pod,pod_info=P.pod_basis(training,max(ranks))
        linear,linfo=P.linear_weak(bank,a,projection)
        methods=dict(linear_bank_weak_qR=linear,linear_bank_galerkin=P.galerkin(bank,n))
        metadata={name:dict(kind='linear_control',rank=cfg['bank_rank']) for name in methods}
        for rank in ranks:
            if rank<=pod.shape[1]:
                name=f'pod{rank}_galerkin';methods[name]=P.galerkin(pod[:,:rank],n)
                metadata[name]=dict(kind='pod',rank=rank)
        full_lam=C.eigenvalues(n);methods['dst_exact']=lambda f,l=full_lam:P.solve_dst(f,l)
        metadata['dst_exact']=dict(kind='full_order')
        for name,op,spec,scales in operator_models:
            methods[name]=O.engine(op,spec,scales,n)
            metadata[name]=dict(kind='neural_operator',spec=spec,training_intervals=cfg['train_intervals'],frozen_mesh_transfer=n!=cfg['train_intervals'])
            if n!=cfg['train_intervals']:
                native=cfg['train_intervals'];alt=f'{name}_native{native}_interpolate'
                methods[alt]=O.native_interpolated(op,spec,scales,n,native)
                metadata[alt]=dict(kind='neural_operator',spec=spec,training_intervals=native,
                    query='restrict supplied nodal forcing, predict on native mesh, trilinear interpolate with zero boundary')
        mesh=dict(intervals=n,weak_tests=len(triples),bank_sha256=C.sha(bank),weak_operator_sha256=C.sha(a),
            weak_singular_values=np.linalg.svd(a,compute_uv=False).tolist(),pod=pod_info,linear_endpoint=linfo,quadrature=[],representation=[])
        qb,rb=np.linalg.qr(bank,mode='reduced')
        for case,truth in enumerate(truths):
            best=qb@(qb.T@truth.reshape(-1));mesh['representation'].append(dict(case=case,kind='bank_projection',error=P.error(best,truth)))
        for model in models:
            k=model['info']['k'];decoded=np.asarray(C.head(model['params'],model['codes']))
            try:
                inverse_tests=cfg.get('quadrature_use_inverse_tests',False)
                old_mesh=next((x for x in prior['meshes'] if x['intervals']==n),None) if prior else None
                quadrature_keys=['weak_tests','quadrature_seed','quadrature_candidates','quadrature_decoder_snapshots',
                    'quadrature_fit_rows','quadrature_row_floor','quadrature_use_inverse_tests']
                cache_valid=(old_mesh is not None and old_mesh['bank_sha256']==C.sha(bank)
                    and old_mesh['weak_operator_sha256']==C.sha(a)
                    and all(cfg.get(key)==prior['config'].get(key) for key in quadrature_keys))
                cached_eq=reuse/f'eq_N{n}_K{k}.npz' if reuse else None
                if cache_valid and cached_eq.exists():
                    z=np.load(cached_eq);indices=z['indices'];weighted=z['weighted_tests']
                    eq=dict(next(x for x in old_mesh['quadrature'] if x['k']==k))
                    eq.update(reused_from_job=prior['job_id'],exact_bank_and_operator_hashes_matched=True)
                else:indices,weighted,eq=S.fit_quadrature(bank,projection if inverse_tests else test,decoded,cfg)
                sampled=weighted if inverse_tests else weighted/lam[None,:]
                eq['fit_test_normalization']='inverse eigenvalue' if inverse_tests else 'unscaled sine'
                exact=sources.reshape(len(sources),-1)@projection
                approx=sources.reshape(len(sources),-1)[:,indices]@sampled
                errors=np.linalg.norm(approx-exact,axis=1)/np.maximum(np.linalg.norm(exact,axis=1),1e-300)
                eq.update(k=k,validation_forcing_moment_errors=errors.tolist(),threshold=cfg['quadrature_certificate'],
                    certified=bool(max(errors)<cfg['quadrature_certificate']))
                np.savez_compressed(out/f'eq_N{n}_K{k}.npz',indices=indices,weighted_tests=weighted,projection=sampled)
            except (RuntimeError,ValueError) as exc:indices=None;eq=dict(k=k,certified=False,error=str(exc))
            mesh['quadrature'].append(eq)
            for q in cfg['q_ladder']:
                if k+q>cfg['bank_rank']:continue
                name=f'nmrom_K{k}_q{q}_dense';methods[name]=P.engine(model,bank,a,projection,np.arange(len(bank)),q,cfg)
                metadata[name]=dict(kind='nonlinear_rom',k=k,q=q,cold_projection='dense grid-dependent diagnostic')
                if indices is not None:
                    name=f'nmrom_K{k}_q{q}_eq';methods[name]=P.engine(model,bank,a,sampled,indices,q,cfg)
                    metadata[name]=dict(kind='nonlinear_rom',k=k,q=q,cold_projection='NNLS sampled weak moments',quadrature_certified=eq['certified'])
                oracle_cfg={**cfg,'initial_starts':cfg['oracle_starts'],'lm_budget':max(240,cfg['lm_budget']),'lm_tolerance':min(1e-9,cfg['lm_tolerance'])}
                oracle=P.engine(model,bank,rb,qb,np.arange(len(bank)),q,oracle_cfg)
                for case,truth in enumerate(truths):
                    best,stats,coef,_=jax.device_get(oracle(jnp.asarray(truth)))
                    mesh['representation'].append(dict(case=case,k=k,q=q,kind='best_found_augmented',error=P.error(best,truth),
                        stats=np.asarray(stats).tolist(),stationary=bool(int(stats[2])==1 and stats[5]<=oracle_cfg['lm_tolerance'])))
                    if case==0:np.savez_compressed(out/'fields'/f'N{n}_K{k}_q{q}_oracle_case0.npz',prediction=best,stats=stats,coefficients=coef)
        mesh['setup_seconds']=time.perf_counter()-setup;mesh['methods']=metadata;record['meshes'].append(mesh);save()
        record['status']=f'compile_mesh_{n}';save();sample=jnp.asarray(sources[0])
        for name,method in methods.items():
            start=time.perf_counter();jax.block_until_ready(method(sample));print('COMPILED',n,name,round(time.perf_counter()-start,3),flush=True)
        rng=np.random.default_rng(cfg['timing_seed']+n);record['status']=f'evaluate_mesh_{n}';save()
        for case,(f,truth) in enumerate(zip(sources,truths)):
            np.savez_compressed(out/'fields'/f'N{n}_case{case}_reference.npz',forcing=f,same_grid=truth,physical=physical[(n,case)])
            for rep in range(cfg['repetitions']):
                for name in rng.permutation(list(methods)):
                    C.burn(cfg['burn_seconds']);start=time.perf_counter()
                    fj=jax.device_put(f);fj.block_until_ready();uploaded=time.perf_counter()
                    value=methods[name](fj);jax.block_until_ready(value);computed=time.perf_counter()
                    value=jax.device_get(value);finished=time.perf_counter()
                    counters=dict(stationary=True,iterations=0)
                    if isinstance(value,tuple):
                        pred,stats,coef,starts=value
                        counters=dict(stationary=bool(int(stats[2])==1 and stats[5]<=cfg['lm_tolerance']),iterations=int(stats[7]),
                            selected_stats=np.asarray(stats).tolist(),all_starts_stats=np.asarray(starts).tolist())
                    else:pred=value
                    pred=np.asarray(pred).reshape(truth.shape);finite=bool(np.isfinite(pred).all())
                    row=dict(intervals=n,case=case,repetition=rep,method=str(name),finite=finite,
                        input_ms=(uploaded-start)*1000,device_ms=(computed-uploaded)*1000,total_ms=(finished-start)*1000,**counters)
                    if finite:row.update(same_grid_error=P.error(pred,truth),physical_error=P.error(pred,physical[(n,case)]))
                    if rep==0:
                        path=out/'fields'/f'N{n}_case{case}_{name}.npz';np.savez_compressed(path,prediction=pred)
                        row.update(field_file=str(path.relative_to(out)),field_sha256=C.sha(pred))
                    record['invocations'].append(row)
                save();print('CASE',n,case,'REP',rep,flush=True)
    record.update(status='complete',complete=True,elapsed_seconds=time.perf_counter()-begin);save();summarize(record,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--out',required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.verify_only:print(json.dumps(P.verify(),indent=2));raise SystemExit(0)
    if args.smoke:
        cfg.update(train_intervals=8,evaluation_intervals=[8],reference_intervals=[8,16],train_count=12,validation_count=1,
            bank_rank=8,latent_dimensions=[2],bank_steps=3,head_steps=3,weak_tests=32,q_ladder=[0,2],bank_width=16,head_width=16,
            fourier_features=4,bank_batch_states=4,bank_batch_points=64,quadrature_candidates=128,quadrature_fit_rows=64,
            quadrature_decoder_snapshots=4,repetitions=1,burn_seconds=.001,lm_budget=8,field_chunk=1024,oracle_starts=2,
            checkpoint_every=3,operators=[])
    run(cfg,Path(args.out),args.smoke)
