"""Bounded fixed-bank Burgers head training with paired complete-query controls."""
import argparse,hashlib,json,os,pickle,time,shutil
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import engines as e
import iterative_paths as ip
import accuracy_paths as ap
import accuracy_coverage_paths as cp


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def host(x):return jax.tree_util.tree_map(np.asarray,x)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--reuse-dir',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
    begin=time.perf_counter();ck=pickle.load(open(a.checkpoint,'rb'));params=host(ck['params']);Zold=ck['Z_tr'];K=Zold.shape[1];R=params['h_lin'].shape[1]
    report=dict(config=cfg,commit=os.environ.get('COMMIT'),job_id=os.environ.get('SLURM_JOB_ID'),backend=jax.default_backend(),gpu=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],jax_version=jax.__version__,checkpoint_sha256=sha(a.checkpoint),
        checkpoint_cfg={k:v for k,v in ck['cfg'].items() if k not in ['hfit_pick','train']},K=K,R=R,M=4*K,m=16*K,
        spatial_bank_frozen=True,network_weights_frozen_during_evaluation=True,final_cohort_unopened=True,
        output_times=[0,.05,.1,.15,.2,.25],timing_contract='supplied full initial field on GPU to six full GPU output fields; same-invocation host transfers also measured; includes all solver diagnostics',
        reference=[],reference_metrics={},mesh_setup=[],invocations=[],declared_subjects=[],verification=ip.verify(),complete=False)
    save=lambda:dump(out/'result.json',report);save()
    reuse=Path(a.reuse_dir);lineage=json.loads((reuse/'INHERITED.json').read_text())
    for item in lineage['files']:assert sha(reuse/item['destination'])==item['sha256']
    parent=json.loads((reuse/'parent-result.json').read_text())
    for key in ['meshes','seed','cases','fresh_seed','fresh_cases','dt','reference_mesh','reference_dt','training','strict']:
        assert cfg[key]==parent['config'][key]
    assert parent['job_id']==lineage['parent_job_id'] and parent['commit']==lineage['parent_source_commit']
    assert sha(a.checkpoint)==parent['checkpoint_sha256']
    for name in ['trained_checkpoint.pkl','training_targets.npz','training.json']:shutil.copy2(reuse/name,out/name)
    trainedck=pickle.loads((out/'trained_checkpoint.pkl').read_bytes());trained=host(trainedck['params']);Ztrained=trainedck['Z_tr'];train_info=json.loads((out/'training.json').read_text())
    assert sha(out/'trained_checkpoint.pkl')==parent['trained_checkpoint_sha256']
    assert all(np.array_equal(x,y) for key in ['B','g','out_scale'] for x,y in zip(jax.tree_util.tree_leaves(params[key]),jax.tree_util.tree_leaves(trained[key])))
    report['spatial_bank_arrays_byte_identical']=True
    report['trained_checkpoint_sha256']=sha(out/'trained_checkpoint.pkl');report['training_seconds']=train_info['elapsed_seconds'];report['training_complete_before_development_opened']=True
    report['training_and_reference_lineage']=lineage;report['training_rerun']=False;report['failed_instrumentation_parent']=lineage['parent_job_id'];report['failures']=[]
    report['additional_development_cohort_note']='All six development cases were opened before this coverage follow-up; no new cases or final cohort are opened, and the sole terminal training endpoint is frozen before this evaluation.'
    report['inherited_576_training_completed_before_original_development_opening']=True
    report.pop('training_complete_before_development_opened',None)
    save();print('REUSED_FROZEN_TRAINING',lineage['parent_job_id'],flush=True)
    coverage_out=out/'coverage_training';coverage_out.mkdir()
    print('COVERAGE_TRAIN_BEGIN',flush=True)
    coverage,Zcoverage,coverage_info=cp.train_head(jax.tree_util.tree_map(jnp.asarray,params),Zold,cfg['coverage_training'],coverage_out,lambda value:dump(coverage_out/'training.json',value))
    coverage=host(coverage);coverage_ck=dict(params=coverage,Z_tr=Zcoverage,cfg=dict(ck['cfg'],accuracy_training=cfg['coverage_training'],accuracy_original_codes=len(Zold)))
    for key in ['B','g','out_scale']:assert all(np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(params[key]),jax.tree_util.tree_leaves(coverage[key])))
    with (coverage_out/'trained_checkpoint.pkl').open('wb') as handle:pickle.dump(coverage_ck,handle)
    report['coverage_trained_checkpoint_sha256']=sha(coverage_out/'trained_checkpoint.pkl');report['coverage_training']=cfg['coverage_training'];report['coverage_training_seconds']=coverage_info['elapsed_seconds'];report['coverage_terminal_frozen_before_query_evaluation']=True
    report['training_rerun']=False;report['coverage_training_performed']=True;report['paired_control_parent']=lineage['parent_job_id'];report.pop('failed_instrumentation_parent',None)
    save();jax.clear_caches();print('COVERAGE_TRAIN_FROZEN',flush=True)

    # Supplied development inputs are regenerated after freezing; verified reference fields are reused.
    physical=np.concatenate((e.params_draw(cfg['seed'],cfg['cases']),e.params_draw(cfg['fresh_seed'],cfg['fresh_cases'])))
    cfg_cases=len(physical);report['physical_cases']=physical.tolist();report['cohort_roles']=['opened development']*cfg['cases']+['fresh development']*cfg['fresh_cases'];save()
    meshes=cfg['meshes'];largest=max(meshes);rf=cfg['reference_mesh'];rt=cfg['reference_dt'];refs={}
    settings=[(rf//2,rt),(rf,2*rt),(rf,rt),(rf//4,2*rt),(rf//2,2*rt),(rf,4*rt)]
    for L,dt in settings:
        print('REUSED_REFERENCE',L,dt,flush=True)
        for case,phys in enumerate(physical):
            item=next(v for v in parent['reference'] if v['intervals']==L and v['dt']==dt and v['case']==case)
            name=item['artifact'];shutil.copy2(reuse/name,out/name);rr=dict(np.load(out/name));f=rr['fields']
            assert np.isfinite(f).all() and np.max(rr['residuals'])<2e-11 and f.shape==(6,largest+1,largest+1)
            refs[L,dt,case]=f;report['reference'].append(dict(item,reused_from_job=lineage['parent_job_id'],reused_from_gpu=parent['gpu'],source_sha256=sha(out/name)));save()
    for L in meshes:
        rows=[]
        for case in range(cfg_cases):
            def diff(L1,t1,L2,t2):return e.errors(refs[L1,t1,case][:,::largest//L,::largest//L],refs[L2,t2,case][:,::largest//L,::largest//L],L)['fixed_initial_max']
            space=diff(rf//2,rt,rf,rt);temporal=diff(rf,2*rt,rf,rt);sc=diff(rf//4,2*rt,rf//2,2*rt);sf=diff(rf//2,2*rt,rf,2*rt);tc=diff(rf,4*rt,rf,2*rt)
            rows.append(dict(case=case,space_difference=space,time_difference=temporal,margin=space+temporal,spatial_coarse=sc,spatial_fine=sf,temporal_coarse=tc,
                observed_space_order=float(np.log2(sc/sf)),observed_time_order=float(np.log2(tc/temporal)),decrease=bool(sf<sc and temporal<tc)))
        report['reference_metrics'][str(L)]=rows
    save();rng=np.random.default_rng(cfg['order_seed']);weights={'frozen':params,'trained576':trained,'trained4608':coverage};code_pools={'frozen':Zold,'trained576':Ztrained,'trained4608':Zcoverage}
    for L in meshes:
        print('ASSEMBLE',L,flush=True);models={};foms={};subjects=[];diagnostics={}
        for model,pnp in weights.items():
            pdev=jax.tree_util.tree_map(jnp.asarray,pnp)
            # Same recorded old-code indices fit EQ for all three heads; no test truth enters.
            data,info=e.build_rom(pdev,Zold,L,4*K,16*K,candidate_cap=cfg['candidate_cap'],fit_states=cfg['fit_states'])
            if model!='frozen':
                data=list(data);data[7]=jnp.concatenate((data[7],jnp.asarray(code_pools[model][len(Zold):])));data=tuple(data)
            cold,cinfo=e.build_gauss_cold(pdev,data[7]);info.update(model=model,cold=cinfo,actual_cold_candidates=len(data[7]),eq_code_pool='unchanged original recorded training codes',trust_radius_source='original recorded codes; identical to frozen control')
            np.savez_compressed(out/f'operators_{model}_L{L}.npz',A=np.asarray(data[1]),lam=np.asarray(data[2]),G5=np.asarray(data[3]),Pq=np.asarray(data[4]),
                candidate_Z=np.asarray(data[7]),cold_xy=np.asarray(cold[0]),cold_w=np.asarray(cold[1]),cold_Q=np.asarray(cold[2]),cold_R=np.asarray(cold[3]))
            solvers={'stationary':ap.make_rom(pdev,L,cfg['dt'],info['trust_radius'],**cfg['strict'])}
            models[model]=(data,cold,solvers,pdev);diagnostics[model]=ap.make_diagnostics(pdev,L,cfg['dt']);report['mesh_setup'].append(info)
            for solver in solvers:subjects.append(dict(name=model+'_'+solver,method='rom',model=model,solver=solver,dt=cfg['dt']))
        for pc in ['fft']:foms[pc]=ip.make_fom(L,cfg['dt'],pc)
        subjects += [dict(method='fom',dt=cfg['dt'],**s) for s in cfg['fom_settings']]
        report['declared_subjects'] += [dict(intervals=L,**s) for s in subjects]
        inputs=[e.initial(L,phys) for phys in physical];gpu_inputs=[jnp.asarray(u) for u in inputs];jax.block_until_ready(gpu_inputs)
        def invoke(s,u,case):
            nu=float(physical[case,4])
            if s['method']=='rom':
                data,cold,solvers,pdev=models[s['model']];return solvers[s['solver']](u,nu,data,cold)
            q,predata=foms[s['preconditioner']];return q(u,nu,s['ntol'],s['ltol'],*predata)
        t=time.perf_counter()
        for s in subjects:jax.block_until_ready(invoke(s,gpu_inputs[0],0))
        report.setdefault('compile_warmup',[]).append(dict(intervals=L,seconds=time.perf_counter()-t));save()
        artifacts={}
        for rep in range(cfg['reps']):
            for case in range(cfg_cases):
                truth=refs[rf,rt,case][:,::largest//L,::largest//L]
                for i in rng.permutation(len(subjects)):
                    s=subjects[i];e.burn(cfg['burn_seconds'])
                    ht=time.perf_counter();u=jax.device_put(np.array(inputs[case],copy=True));jax.block_until_ready(u)
                    gt=time.perf_counter();value=invoke(s,u,case);jax.block_until_ready(value);gs=time.perf_counter()-gt
                    f=np.asarray(value[0]);hs=time.perf_counter()-ht;v=host(value)
                    if not (np.isfinite(f).all() and all(np.isfinite(x).all() for x in v)):
                        failure=f'FAILED_L{L}_{s["name"]}_case{case}_rep{rep}.npz'
                        np.savez_compressed(out/failure,**{f'value{i}':x for i,x in enumerate(v)})
                        report['failures'].append(dict(intervals=L,case=case,rep=rep,name=s['name'],gpu_seconds=gs,host_seconds=hs,artifact=failure,reason='nonfinite output retained before validation'))
                        save();raise RuntimeError('Nonfinite output; complete raw diagnostic arrays retained')
                    h=hashlib.sha256(f.tobytes()).hexdigest();key=(s['name'],case,h)
                    if key not in artifacts:
                        filename=f'L{L}_{s["name"]}_case{case}_rep{rep}.npz'
                        extra=dict(internal_latents=v[7]) if s['method']=='rom' else dict(previous_output_steps=v[4])
                        np.savez_compressed(out/filename,fields=f,**extra);artifacts[key]=filename
                    row=dict(intervals=L,nodes_per_axis=L+1,case=case,cohort=report['cohort_roles'][case],rep=rep,**s,gpu_seconds=gs,host_seconds=hs,output_bytes=f.nbytes,
                        field_sha256=h,artifact=artifacts[key],error=e.errors(f,truth,L),iterations=v[1].tolist(),residuals=v[2].tolist(),finite=True)
                    if s['method']=='rom':
                        row.update(stop_reasons=v[3].tolist(),latent_states=v[4].tolist(),ic_iterations=int(v[5]),ic_reason=int(v[6]),internal_latents=v[7].tolist())
                        data,cold,solvers,pdev=models[s['model']]
                        gn,icgn=host(diagnostics[s['model']](jnp.asarray(v[7]),gpu_inputs[case],float(physical[case,4]),data,cold));icgn=float(icgn)
                        row.update(normalized_stationarity=gn.tolist(),ic_normalized_stationarity=icgn,stationary=bool(max(np.max(gn),icgn)<=cfg['strict']['gtol']*(1+1e-7)))
                        if s['solver']=='stationary':
                            row.update(charged_normalized_stationarity=v[8].tolist(),charged_ic_normalized_stationarity=float(v[9]),
                                charged_posthoc_weak_gradient_max_delta=float(np.max(abs(gn-v[8]))),charged_posthoc_initial_gradient_delta=float(abs(icgn-v[9])),
                                charged_posthoc_threshold_disagreements=int(np.sum((gn<=cfg['strict']['gtol'])!=(v[8]<=cfg['strict']['gtol'])))+int((icgn<=cfg['strict']['gtol'])!=(v[9]<=cfg['strict']['gtol'])))
                            # Roundoff near stationarity is audited from every saved array; no evidence is discarded here.
                    else:row.update(linear_relative_residuals=v[3].tolist(),nonlinear_converged=bool(np.max(v[2])<=s['ntol']*(1+1e-9)),
                        linear_converged=bool(all(np.max(x[:int(n)],initial=0.)<=s['ltol']*(1+1e-7) for x,n in zip(v[3],v[1]))))
                    report['invocations'].append(row);save()
            print('TIMED',L,rep,'elapsed',time.perf_counter()-begin,flush=True)
        del models,diagnostics,data,cold,solvers,pdev,foms,gpu_inputs;jax.clear_caches()
    report['checkpoint_sha256_after']=sha(a.checkpoint);assert report['checkpoint_sha256']==report['checkpoint_sha256_after']
    assert report['trained_checkpoint_sha256']==sha(out/'trained_checkpoint.pkl') and report['coverage_trained_checkpoint_sha256']==sha(coverage_out/'trained_checkpoint.pkl')
    report['elapsed_seconds']=time.perf_counter()-begin;report['complete']=True;save();(out/'COMPLETE').write_text('complete\n')
    print('ACCURACY BURGERS COMPLETE',flush=True)

if __name__=='__main__':main()
