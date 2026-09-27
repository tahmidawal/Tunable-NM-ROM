"""Approved fixed-compute coverage x loss-normalization continuation pilot."""
import argparse,json,os,socket,gc
from pathlib import Path
import scipy
from training_core import *
from kernel_solver import make_lm_kernel,specialized_kernel,specialized_query,solution_agreement
from followup import fused_kernel,fused_query,parity,oracle_fit
from pilot import reference,save
from pilot02 import score


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    dev=jax.devices()[0];assert dev.platform=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    params,codes,ckcfg=sc.load_pkl(a.checkpoint);assert (ckcfg['N'],ckcfg['k'],ckcfg['r'])==(256,16,64)
    original_draw=source_params(cfg['original_draw_seed'],cfg['original_draw_count'])
    original=original_draw[:cfg['original_training_prefix_count']]
    additional=source_params(cfg['additional_training_seed'],cfg['additional_training_count'])
    training_draws=np.concatenate((original,additional));assert len(codes)==len(original)
    draws=np.concatenate((source_params(cfg['existing_development_seed'],cfg['existing_development_count']),
        source_params(cfg['additional_development_seed'],cfg['additional_development_count'])))
    groups=['existing_development']*cfg['existing_development_count']+['fresh_development']*cfg['additional_development_count']
    d=dict(config=cfg,checkpoint_config=ckcfg,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        provenance=dict(commit=os.environ['COMMIT'],job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),
            gpu=dev.device_kind,backend='gpu',x64=True,matmul_precision='highest',jax=jax.__version__,scipy=scipy.__version__),
        cohort=dict(parameters=draws.tolist(),parameter_sha256=sha(draws),groups=groups,count=len(draws),descriptors_never_passed_to_head=True),
        contract=dict(input='full host nodal source',output='full host nodal field and solver metadata',
            initialization='mean of each checkpoint training codes; no evaluation answers or descriptors',
            reference='independent SciPy FD-DST nested differences on common observations; empirical, no rigorous bound',
            selection='fixed scheduled endpoint, no best-step checkpoint selection; development cohorts only',
            factorial='coverage crossed with fixed-full-field global versus per-snapshot normalization',
            cost='same-job full queries with all guards and generic fallback charged; offline setup/train/compile separately',
            training='matched updates and batches, not matched per-source passes; truncation retained'),
        training=dict(original_full_draw_parameters=original_draw.tolist(),original_draw_sha256=sha(original_draw),
            union_parameters=training_draws.tolist(),union_sha256=sha(training_draws),arms=[]),
        checkpoints=[],references=[],setup=[],oracles=[],parity=[],rows=[],complete=False)
    path=a.out/'result.json';persist=lambda:save(path,d);persist()
    offline=time.perf_counter()
    U,coords,info=field_dataset(training_draws,cfg['training_nodes_per_axis']);d['training']['data']=info
    norm=float(jnp.mean(U[:len(original)]**2));d['training']['shared_global_normalizer']=norm
    # Tight CG is a manufacturing-correctness check, never the cost baseline.
    source=jnp.asarray(mp.source_interior(256,*original[0]))
    cg=jax.scipy.sparse.linalg.cg(lambda v:mp.neg_lap_interior(v,256),source,tol=1e-13,maxiter=100000)[0]
    cg_error=relative(cg.ravel(),U[0]);assert cg_error<1e-10
    d['training']['cg_check']=dict(tolerance=1e-13,relative_dst_discrepancy=cg_error)
    allcodes,init=initialize_codes(params,codes,U,coords,cfg);d['training']['code_initialization']=init
    np.savez_compressed(a.out/'training_code_initialization.npz',codes=allcodes)
    d['training']['original_checkpoint_training_metrics']=reconstruction_metrics(params,allcodes,U,coords,norm)
    models={'original_frozen':(params,codes)}
    d['checkpoints'].append(dict(model='original_frozen',sha256=d['checkpoint_sha256'],path='in/model.pkl',training_count=len(codes),steps_done=0,valid_endpoint=True))
    persist()
    for tag in cfg['arms'][1:]:
        count=len(original) if tag.startswith('original') else len(training_draws)
        p,z,record=continue_training(params,allcodes[:count],U[:count],coords,norm,cfg,tag.endswith('relative'),tag)
        record['training_metrics']=reconstruction_metrics(p,z,U[:count],coords,norm)
        record['valid_endpoint']=record['finite'] and record['steps_done']==cfg['training_steps_each'] and not record['time_capped']
        checkpoint=a.out/(tag+'.pkl');sc.save_pkl(checkpoint,p,z,{**ckcfg,'continuation':record,'training_draw_sha256':sha(training_draws[:count])})
        d['training']['arms'].append(record)
        d['checkpoints'].append(dict(model=tag,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),path=checkpoint.name,
            training_count=count,steps_done=record['steps_done'],valid_endpoint=record['valid_endpoint']))
        models[tag]=(p,z);persist()
    d['training']['offline_seconds_including_data_codefit_compilation_and_diagnostics']=time.perf_counter()-offline
    del U,coords,allcodes;gc.collect();persist()
    obs=cfg['observation_intervals'];refs=[];t0=time.perf_counter()
    for case,param in enumerate(draws):
        chain=[reference(param,n) for n in cfg['reference_intervals']]
        nested=[observation(u,n,obs) for u,n in zip(chain,cfg['reference_intervals'])]
        gaps=[relative(x,y) for x,y in zip(nested[:-1],nested[1:])];assert gaps[-1]<gaps[0]/2
        refs.append((nested[-1],gaps[-1]));np.savez_compressed(a.out/f'reference_case{case}.npz',observation=nested[-1],coarser_observation=nested[-2])
        d['references'].append(dict(case=case,group=groups[case],intervals=cfg['reference_intervals'],successive_relative_gaps=gaps,
            reference_delta=gaps[-1],uncertainty_kind='empirical_difference',reference_converging=True));persist()
    d['reference_setup_seconds']=time.perf_counter()-t0
    fields=a.out/'fields';fields.mkdir();stored=set()
    def store_field(field):
        digest=sha(field)
        if digest not in stored:np.savez_compressed(fields/(digest+'.npz'),field=field);stored.add(digest)
    for n in cfg['intervals']:
        opss={};kernels={};special={};qrs={};oracles={}
        for tag,(p,z) in models.items():
            ops=assemble(p,z,n,cfg['requested_modes_ladder'][0],cfg['lm_budget']);opss[tag]=ops
            ops['info']['model']=tag;d['setup'].append(ops['info'])
            kernels[tag]=fused_kernel(ops)
            special[tag]=specialized_kernel(ops,make_lm_kernel(ops,cfg['lm_budget'],True,False,cfg['linear_backward_limit']))
            q,r,qrinfo=qr_bank(ops);qrs[tag]=(q,r);ops['info']['qr_bank']=qrinfo
            oracles[tag]=oracle_solvers(p,r,ckcfg['k'],cfg['oracle_budgets'],ops['info']['trust_delta'])
        lam=jnp.asarray(eigenvalues(n));coarselam=jnp.asarray(eigenvalues(128))
        def query(arm,tag,tau,source):
            if arm=='dst':return fom_query(source,lam)
            if arm=='dst_coarse128':return coarse_query(source,coarselam)
            if arm=='rom_modular':return rom_query(source,opss[tag],tau)
            if arm=='rom_fused':return fused_query(source,opss[tag],tau,kernels[tag])
            return specialized_query(source,opss[tag],tau,special[tag])
        subjects=[('dst',None,None),('dst_coarse128',None,None)]+[(arm,tag,tau) for tag in models for tau in cfg['taus'] for arm in ('rom_modular','rom_fused','rom_gj')]
        for case,param in enumerate(draws):
            source=full_source(n,param);same=np.asarray(dst_solve(jnp.asarray(source),lam));gates={};controls={}
            for tag,(p,z) in models.items():
                starts=[z.mean(0)]+[z[index] for index in cfg['oracle_training_code_indices']]
                start=time.perf_counter()
                record,bank,head=oracle_fit(same,opss[tag],*qrs[tag],oracles[tag],starts)
                record.update(model=tag,case=case,group=groups[case],intervals=n,diagnostic_seconds=time.perf_counter()-start,
                    bank_metrics=score(bank,same,*refs[case],n,obs),best_head_metrics=score(head,same,*refs[case],n,obs))
                d['oracles'].append(record);store_field(bank);store_field(head)
                for tau in cfg['taus']:
                    control=query('rom_modular',tag,tau,source);fused=query('rom_fused',tag,tau,source);candidate=query('rom_gj',tag,tau,source)
                    exact=parity(control,fused);agreement=solution_agreement(control,candidate,cfg['agreement_limits'])
                    agreement.update(original_fusion=exact,model=tag,case=case,intervals=n,tau=tau)
                    d['parity'].append(agreement);controls[(tag,tau)]=control;gates[(tag,tau)]=agreement['passed']
            warm=time.perf_counter()
            for arm,tag,tau in subjects:
                for _ in range(cfg['warmup']):query(arm,tag,tau,source)
            d.setdefault('warmup',[]).append(dict(intervals=n,case=case,seconds=time.perf_counter()-warm))
            for rep in range(cfg['repetitions']):
                order=subjects if rep%2==0 else subjects[::-1]
                for ordinal,(arm,tag,tau) in enumerate(order):
                    burn(cfg['burn_seconds']);field,row=query(arm,tag,tau,source)
                    row.update(score(field,same,*refs[case],n,obs));isrom=tag is not None
                    stationary=not isrom or row['stationarity']<=cfg['stationarity_tolerance'];tau_ok=isrom and row['reason']==2
                    gate=True
                    if arm=='rom_gj':
                        agreement=solution_agreement(controls[(tag,tau)],(field,row),cfg['agreement_limits'])
                        gate=gates[(tag,tau)] and agreement['passed'] and row['max_linear_backward_error']<=cfg['linear_backward_limit']
                        row['numerical_agreement']=agreement
                    if arm=='rom_fused':gate=parity(controls[(tag,tau)],(field,row))['passed']
                    row.update(model=tag,arm=arm,requested_modes=cfg['requested_modes_ladder'][0] if isrom else None,
                        retained_modes=opss[tag]['info']['retained_modes'] if isrom else None,intervals=n,case=case,group=groups[case],
                        repetition=rep,order_index=ordinal,tau=tau,source_sha256=sha(source),stationary=stationary,tau_reached=tau_ok,
                        parity_passed=gate,solver_valid=row['finite'] and gate and (not isrom or tau_ok or stationary))
                    d['rows'].append(row)
                    store_field(field)
                persist()
            print('evaluation',n,case,groups[case],flush=True)
    d['complete']=True;persist();print('ALL-DONE',flush=True)

if __name__=='__main__':main()
