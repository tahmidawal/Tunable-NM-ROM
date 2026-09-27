"""Fixed-rank staged neural training with full multiresolution development tests."""
import argparse,json,os,socket,gc
from pathlib import Path
from staged_training import *
from staged_oracles import prepare_head_oracle,head_oracle
from tuning_core import make_tuning_kernel,tuning_query
from iterative_core import make_cg,cg_query,verify_cg
from speed_core import weak_code_cache
from pilot import reference,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    if a.smoke:
        cfg.update(training_nodes=32,training_prefix_count=72,new_development_count=0,intervals=[64],reference_intervals=[64,128],repetitions=1,warmup=1,burn_seconds=.001)
        cfg['training_common'].update(timing_block_updates=1,log_every_updates=1,point_batch=32,source_batch=4,max_matched_steps=200,training_burn_seconds=.001)
        for phase in cfg['training_phases']:phase['steps']=2
        cfg['joint_matched']['steps']=6
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    original_p,original_z,ckcfg=sc.load_pkl(a.checkpoint);assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==cfg['checkpoint_sha256'] and (ckcfg['k'],ckcfg['r'])==(16,64)
    training=source_params(cfg['original_draw_seed'],cfg['original_draw_count'])[:cfg['training_prefix_count']]
    if a.smoke:original_z=original_z[:len(training)]
    else:assert len(training)==len(original_z) and sha(training)==ckcfg['training_draw_sha256']
    old=np.concatenate((source_params(cfg['existing_development_seed'],cfg['existing_development_count']),source_params(cfg['fresh_development_seed'],cfg['fresh_development_count'])))
    if not a.smoke:assert sha(old)==cfg['parameter_sha256']
    fresh=source_params(cfg['new_development_seed'],cfg['new_development_count']);draws=np.concatenate((old,fresh));groups=['existing_development']*len(old)+['new_development']*len(fresh)
    if a.smoke:draws=draws[:1];groups=groups[:1]
    d=dict(config=cfg,checkpoint_config=ckcfg,verification=verify_cg(),provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID','local'),hostname=socket.gethostname(),gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__),
        cohort=dict(parameters=draws.tolist(),groups=groups,parameter_sha256=sha(draws),existing_parameter_sha256=sha(old),new_parameter_sha256=sha(fresh),scope='development only; all draws fixed before training'),
        training=dict(parameters=training.tolist(),parameter_sha256=sha(training),phases=[]),checkpoints=[],setup=[],oracles=[],head_oracles=[],rows=[],references=[],warmup=[],complete=False)
    path=a.out/'result.json';persist=lambda:save(path,d);persist();(a.out/'checkpoints').mkdir();(a.out/'fields').mkdir();(a.out/'references').mkdir();stored=set()
    def field_store(field):
        field=np.asarray(field);digest=sha(field)
        if digest not in stored:np.savez_compressed(a.out/'fields'/(digest+'.npz'),field=field);stored.add(digest)
        return digest
    offline_start=time.perf_counter();U,coords,data_info=field_dataset(training,cfg['training_nodes']);d['training']['data']=data_info;norm=float(jnp.mean(U*U));models={'original_relative':(original_p,original_z)}
    def checkpoint(tag,p,z,record):
        dest=a.out/'checkpoints'/(tag+'.pkl');sc.save_pkl(dest,p,z,{**ckcfg,'staged_training':record,'training_draw_sha256':sha(training)})
        info=dict(id=tag,path=str(dest.relative_to(a.out)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),weights_sha256=weights_sha(p),codes_sha256=sha(z),training_metrics=reconstruction_metrics(p,z,U,coords,norm),record=record)
        d['checkpoints'].append(info);persist()
    checkpoint('original_relative',original_p,original_z,dict(phase='frozen_control',updates=0))
    G,Q,R,T,C,perp,initial_qr=bank_qr(original_p,U,coords);d['training']['initial_bank']=initial_qr;persist();assert initial_qr['rank_valid']
    P,pod_info=normalized_pod(U,ckcfg['r']);Udev,_,devdata=field_dataset(draws,cfg['training_nodes']);pod_projected=(Udev@P)@P.T
    np.savez_compressed(a.out/'pod_diagnostic.npz',basis=np.asarray(P),truth=np.asarray(Udev),prediction=np.asarray(pod_projected))
    d['training']['pod_diagnostic']={**pod_info,'development_data':devdata,'development_relative_errors':np.asarray(jnp.linalg.norm(pod_projected-Udev,axis=1)/jnp.linalg.norm(Udev,axis=1)).tolist(),'diagnostic_mesh_intervals':cfg['training_nodes']-1}
    p,z=original_p,original_z;free=np.asarray(C);head_metric=None;matched=0.;persist()
    for spec in cfg['training_phases']:
        options={**cfg['training_common'],**spec};phase=spec['phase'];latent=free if phase=='bank' else z
        kwargs={} if phase!='head' else dict(R=head_metric[0],T=head_metric[1],perpendicular=head_metric[2])
        p,trained,record=train_phase(p,latent,U,coords,options,phase,spec['tag'],**kwargs);d['training']['phases'].append(record);matched+=record['optimizer_seconds'];persist()
        assert record['finite'] and record['budget_reached'],'Training phase failed; preserve attempt before any modified rerun'
        if phase=='bank':
            free=trained;G,Q,R,T,C,perp,qrinfo=bank_qr(p,U,coords);head_metric=(R,T,perp);record['bank_qr']=qrinfo
            record['free_coefficient_relative_errors']=np.asarray(jnp.linalg.norm(jnp.asarray(free)@G.T-U,axis=1)/jnp.linalg.norm(U,axis=1)).tolist()
            np.savez_compressed(a.out/'bank_free_coefficients.npz',learned=free,optimal=np.asarray(C),R=np.asarray(R),target=np.asarray(T),perpendicular=np.asarray(perp))
        else:z=trained
        checkpoint(spec['tag'],p,z,record)
        if phase=='bank':assert qrinfo['rank_valid'],'Failed trained bank rank; truncated-SVD diagnostic saved'
        if spec['tag'] in cfg['trained_model_ids']:models[spec['tag']]=(p,z)
    matched_cfg={**cfg['training_common'],**cfg['joint_matched']};control_p,control_z,record=train_phase(original_p,original_z,U,coords,matched_cfg,'joint','joint_matched',matched_seconds=matched)
    d['training']['phases'].append(record);persist();assert record['finite'] and record['budget_reached'];models['joint_matched']=(control_p,control_z);checkpoint('joint_matched',control_p,control_z,record)
    d['training']['matched_optimizer_target_seconds']=matched;d['training']['control_optimizer_overshoot_seconds']=record['optimizer_seconds']-matched;d['training']['offline_seconds_including_data_diagnostics_qr_and_checkpoints']=time.perf_counter()-offline_start
    assert set(models)==set(cfg['trained_model_ids']);persist();del U,Udev,coords,G,Q,R,T,C,P,pod_projected;gc.collect();jax.clear_caches()
    refs={}
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        for n in cfg['intervals']:
            lo,hi=[x[::nr//n,::nr//n] for x,nr in zip(chain,cfg['reference_intervals'])];refs[n,case]=(hi.copy(),relative(lo,hi));np.savez_compressed(a.out/'references'/f'n{n}_case{case}.npz',coarse=lo,fine=hi)
            d['references'].append(dict(intervals=n,case=case,reference_delta=relative(lo,hi)))
        print('reference',case,flush=True)
    persist()
    for n in cfg['intervals']:
        operators={};caches={};kernels={};qr={};head_engines={};rank_validity={}
        for tag in cfg['trained_model_ids']:
            p,z=models[tag];ops=assemble(p,z,n,cfg['requested_modes'],cfg['lm_budget']);cache=weak_code_cache(ops,z);Q,R=jnp.linalg.qr(ops['bank'],mode='reduced');jax.block_until_ready((Q,R));left,sv,right=jnp.linalg.svd(R,full_matrices=False);singular=np.asarray(sv);threshold=singular[0]*max(ops['bank'].shape)*np.finfo(float).eps;rank=int((singular>threshold).sum());rank_validity[tag]=rank==R.shape[1]
            if not rank_validity[tag]:Q=Q@left[:,:rank];R=sv[:rank,None]*right[:rank]
            setup={**ops['info'],'model':tag,'cache':cache['info'],'R_sha256':sha(R),'condition_number':float(singular[0]/singular[-1]),'rank_valid':rank_validity[tag],'projection_rank':rank,'projection_kind':'full_rank_QR' if rank_validity[tag] else 'truncated_SVD_diagnostic_failed_rank'};d['setup'].append(setup)
            np.savez_compressed(a.out/f'cache_n{n}_{tag}.npz',codes=np.asarray(cache['codes']),predictions=np.asarray(cache['predictions']),B=np.asarray(ops['B']),R=np.asarray(R))
            operators[tag]=ops;caches[tag]=cache;kernels[tag]=make_tuning_kernel(ops,cfg['online_preset'],cfg['linear_backward_error_limit']);qr[tag]=(Q,R)
            if n==cfg['head_oracle']['intervals']:head_engines[tag]=prepare_head_oracle(ops,Q,R,z,cfg['head_oracle'],cfg['linear_backward_error_limit'])
        cg=make_cg(n,cfg['cg_maxiter']);lam=jnp.asarray(eigenvalues(n));methods=cfg['trained_model_ids']+[f'cg_{t:.0e}' for t in cfg['cg_tolerances']]+['dst']
        def query(method,source):
            if method in models:
                field,row=tuning_query(source,operators[method],caches[method],cfg['online_preset'],kernels[method],cfg)
                row['solver_valid']=row['solver_valid'] and rank_validity[method]
                row.update(model=method,bank_rank_valid=rank_validity[method],preset=cfg['online_preset'],requested_modes=cfg['requested_modes'],retained_modes=operators[method]['info']['retained_modes'],checkpoint_sha256=next(x['sha256'] for x in d['checkpoints'] if x['id']==method),operator_sha256=operators[method]['info']['operator_sha256']);return field,row
            if method=='dst':
                field,row=fom_query(source,lam);row['fused_device_seconds']=row['solver_seconds'];row['solver_valid']=bool(np.isfinite(field).all());return field,row
            return cg_query(source,cg,float(method.removeprefix('cg_')))
        for case,param in enumerate(draws):
            source=full_source(n,param);same=reference(param,n);fine,delta=refs[n,case]
            for tag in cfg['trained_model_ids']:
                Q,R=qr[tag];v=jnp.asarray(same[1:-1,1:-1].ravel());projected=jnp.pad((Q@(Q.T@v)).reshape(n-1,n-1),1);field=np.asarray(projected)
                d['oracles'].append(dict(model=tag,intervals=n,case=case,group=groups[case],same_grid_relative_error=relative(field,same),physical_error=relative(field,fine),field_sha256=field_store(field),rank_valid=rank_validity[tag],scope='full same-grid bank projection if rank valid; otherwise truncated-SVD diagnostic; never online initialization'))
                if tag in head_engines:
                    candidates,record=head_oracle(same,operators[tag],Q,R,head_engines[tag],cfg['head_oracle'],cfg['linear_backward_error_limit'])
                    record.update(model=tag,intervals=n,case=case,group=groups[case],bank_rank_valid=rank_validity[tag],candidates=[])
                    for candidate,candidate_row in candidates:
                        candidate_row.update(field_sha256=field_store(candidate),physical_error=relative(candidate,fine));candidate_row['solver_valid']=candidate_row['solver_valid'] and rank_validity[tag];record['candidates'].append(candidate_row)
                    valid=[i for i,x in enumerate(record['candidates']) if x['solver_valid']];record['best_found_stationary_index']=min(valid,key=lambda i:record['candidates'][i]['same_grid_relative_error']) if valid else None;d['head_oracles'].append(record)
            warm=time.perf_counter()
            for method in methods:
                for _ in range(cfg['warmup']):query(method,source)
            d['warmup'].append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm))
            for rep in range(cfg['repetitions']):
                for ordinal,method in enumerate(methods if rep%2==0 else methods[::-1]):
                    burn(cfg['burn_seconds']);field,row=query(method,source);error=relative(field,fine);stride=n//cfg['observation_intervals']
                    row.update(intervals=n,nodes_per_axis=n+1,case=case,group=groups[case],repetition=rep,order_index=ordinal,method=method,source_sha256=sha(source),field_sha256=field_store(field),physical_error=error,same_grid_error=relative(field,same),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]),finite=bool(np.isfinite(field).all()))
                    d['rows'].append(row)
            persist();print('evaluation',n,case,groups[case],flush=True)
        del operators,caches,kernels,qr,head_engines,cg;gc.collect();jax.clear_caches()
    assert len(d['rows'])==len(draws)*len(cfg['intervals'])*cfg['repetitions']*(len(models)+len(cfg['cg_tolerances'])+1)
    d['complete']=True;persist();print('STAGED ACCURACY COMPLETE',flush=True)


if __name__=='__main__':main()
