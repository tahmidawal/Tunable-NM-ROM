"""One frozen nested correction family; no additional network training or search."""
import argparse,json,os,socket,gc,shutil,hashlib
from pathlib import Path
from correction_core import *
from iterative_core import make_cg,cg_query,verify_cg
from pilot import reference,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--basis',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    if a.smoke:cfg.update(intervals=[32],reference_intervals=[32,64],observation_intervals=32,repetitions=1,warmup=1,burn_seconds=.001)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==cfg['checkpoint_sha256'] and hashlib.sha256(a.original.read_bytes()).hexdigest()==cfg['original_checkpoint_sha256'] and hashlib.sha256(a.basis.read_bytes()).hexdigest()==cfg['basis_sha256']
    p,z,ckcfg=sc.load_pkl(a.checkpoint);po,zo,cfg_o=sc.load_pkl(a.original);basis=dict(np.load(a.basis));directions=basis['coefficient_directions'];assert (ckcfg['k'],ckcfg['r'])==(16,128) and (cfg_o['k'],cfg_o['r'])==(16,64);np.testing.assert_array_equal(z,basis['training_latents']);assert directions.shape==(128,32)
    metric_error=float(np.linalg.norm(directions.T@basis['R'].T@basis['R']@directions-np.eye(32)));assert metric_error<1e-10
    old=np.concatenate((source_params(cfg['existing_development_seed'],cfg['existing_development_count']),source_params(cfg['fresh_development_seed'],cfg['fresh_development_count'])));later=source_params(cfg['new_development_seed'],cfg['new_development_count']);draws=np.concatenate((old,later));groups=['existing_development']*len(old)+['new_development']*len(later)
    if not a.smoke:assert sha(old)==cfg['parameter_sha256']
    if a.smoke:draws=draws[:1];groups=groups[:1]
    d=dict(config=cfg,provenance=dict(commit=os.environ.get('COMMIT','local'),job_id=os.environ.get('SLURM_JOB_ID','local'),hostname=socket.gethostname(),gpu=jax.devices()[0].device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__),verification=verify_cg(),
        cohort=dict(parameters=draws.tolist(),groups=groups,parameter_sha256=sha(draws),scope='all42 already-opened development sources; no final confirmation'),basis=dict(path='basis.npz',sha256=cfg['basis_sha256'],physical_metric_orthogonality_error=metric_error),checkpoints=[],setup=[],oracles=[],rows=[],references=[],warmup=[],complete=False)
    path=a.out/'result.json';persist=lambda:save(path,d);persist();(a.out/'checkpoints').mkdir();(a.out/'fields').mkdir();(a.out/'references').mkdir();shutil.copyfile(a.basis,a.out/'basis.npz');models={'original_relative':(po,zo),'r128_joint':(p,z)}
    for name,src in [('original_relative',a.original),('r128_joint',a.checkpoint)]:
        dest=a.out/'checkpoints'/(name+'.pkl');shutil.copyfile(src,dest);params,codes=models[name];d['checkpoints'].append(dict(id=name,path=str(dest.relative_to(a.out)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),codes_sha256=sha(codes),latent_dimension=codes.shape[1],feature_count=params['h_lin'].shape[1]))
    persist();stored=set()
    def field_store(field):
        field=np.asarray(field);digest=sha(field)
        if digest not in stored:np.savez_compressed(a.out/'fields'/(digest+'.npz'),field=field);stored.add(digest)
        return digest
    refs={}
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        for n in cfg['intervals']:
            lo,hi=[v[::nr//n,::nr//n] for v,nr in zip(chain,cfg['reference_intervals'])];refs[n,case]=(hi.copy(),relative(lo,hi));np.savez_compressed(a.out/'references'/f'n{n}_case{case}.npz',coarse=lo,fine=hi);d['references'].append(dict(intervals=n,case=case,reference_delta=relative(lo,hi)))
        print('reference',case,flush=True)
    persist()
    for n in cfg['intervals']:
        operators={};qr={};bankinfo={};engines={}
        for name,(params,codes) in models.items():
            ops=assemble(params,codes,n,cfg['requested_modes'],cfg['lm_budget']);Q,R=jnp.linalg.qr(ops['bank'],mode='reduced');values=np.asarray(jnp.linalg.svd(R,compute_uv=False));rank=int(np.count_nonzero(values>values[0]*max(ops['bank'].shape)*np.finfo(float).eps));bankinfo[name]=dict(rank=rank,rank_valid=rank==R.shape[1],condition_number=float(values[0]/values[-1]),singular_values=values.tolist(),R_sha256=sha(R));operators[name]=ops;qr[name]=(Q,R)
        for method in cfg['trained_model_ids']:
            base=cfg['method_base_model'][method];ops=operators[base];codes=models[base][1];count=cfg['correction_counts'][method];C=directions if count else np.zeros((ops['B'].shape[1],0));engine=prepare_correction(ops,codes,C,count,cfg);engines[method]=engine
            setup={**ops['info'],'method':method,'base_model':base,**engine['info'],'bank_rank':bankinfo[base],'cache':engine['cache']['info']};d['setup'].append(setup);persist()
            np.savez_compressed(a.out/f'cache_n{n}_{method}.npz',codes=np.asarray(engine['cache']['codes']),predictions=np.asarray(engine['cache']['predictions']),B=np.asarray(ops['B']),projected_B=np.asarray(engine['Bp']),C=np.asarray(engine['C']),Q=np.asarray(engine['Q']),R=np.asarray(engine['R']),bank_R=np.asarray(qr[base][1]))
            assert bankinfo[base]['rank_valid'] and engine['info']['linear_rank_valid'],'Failed operator rank retained before evaluation'
        cg=make_cg(n,cfg['cg_maxiter']);lam=jnp.asarray(eigenvalues(n));methods=cfg['trained_model_ids']+[f'cg_{tol:.0e}' for tol in cfg['cg_tolerances']]+['dst']
        def query(method,source):
            if method in engines:
                base=cfg['method_base_model'][method];ops=operators[base];field,row=correction_query(source,ops,engines[method],cfg);row.update(model=method,base_model=base,bank_rank_valid=bankinfo[base]['rank_valid'],preset=cfg['online_preset'],requested_modes=cfg['requested_modes'],retained_modes=ops['info']['retained_modes'],checkpoint_sha256=next(c['sha256'] for c in d['checkpoints'] if c['id']==base),operator_sha256=ops['info']['operator_sha256'],projected_operator_sha256=engines[method]['info']['projected_operator_sha256'],correction_matrix_sha256=engines[method]['info']['correction_matrix_sha256']);row['solver_valid']=row['solver_valid'] and row['bank_rank_valid'];return field,row
            if method=='dst':
                field,row=fom_query(source,lam);row['fused_device_seconds']=row['solver_seconds'];row['solver_valid']=bool(np.isfinite(field).all());return field,row
            return cg_query(source,cg,float(method.removeprefix('cg_')))
        for case,param in enumerate(draws):
            source=full_source(n,param);same=reference(param,n);fine,delta=refs[n,case]
            for base in models:
                Q,R=qr[base];v=jnp.asarray(same[1:-1,1:-1].ravel());field=np.asarray(jnp.pad((Q@(Q.T@v)).reshape(n-1,n-1),1));d['oracles'].append(dict(base_model=base,intervals=n,case=case,group=groups[case],same_grid_relative_error=relative(field,same),physical_error=relative(field,fine),field_sha256=field_store(field),rank_valid=bankinfo[base]['rank_valid'],scope='full fixed-bank projection; unchanged by correction prefix, never online'))
            warm=time.perf_counter()
            for method in methods:
                for _ in range(cfg['warmup']):query(method,source)
            d['warmup'].append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm))
            for rep in range(cfg['repetitions']):
                for ordinal,method in enumerate(methods if rep%2==0 else methods[::-1]):
                    burn(cfg['burn_seconds']);field,row=query(method,source);error=relative(field,fine);stride=n//cfg['observation_intervals'];row.update(intervals=n,nodes_per_axis=n+1,case=case,group=groups[case],repetition=rep,order_index=ordinal,method=method,source_sha256=sha(source),field_sha256=field_store(field),physical_error=error,same_grid_error=relative(field,same),reference_delta=delta,conservative_physical_error=(error+delta)/(1-delta),common_observation_error=relative(field[::stride,::stride],fine[::stride,::stride]),finite=bool(np.isfinite(field).all()));d['rows'].append(row)
            persist();print('evaluation',n,case,groups[case],flush=True)
        del operators,qr,bankinfo,engines,cg;gc.collect();jax.clear_caches()
    assert len(d['rows'])==len(draws)*len(cfg['intervals'])*cfg['repetitions']*(len(cfg['trained_model_ids'])+len(cfg['cg_tolerances'])+1);d['complete']=True;persist();print('CORRECTION ACCURACY COMPLETE',flush=True)


if __name__=='__main__':main()
