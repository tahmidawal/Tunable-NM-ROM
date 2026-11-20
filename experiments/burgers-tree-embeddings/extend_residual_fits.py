"""Offline continuation of selected unfinished fits; no capacity-floor claims."""
import argparse
from common import *
from neural import residual_tree as R, fields as F
from neural.build_data import resolve
from neural.parent_provenance import load_parent
from evaluate_residual_tree import load_group, owned_job
from training_io import read_checkpoint, checkpoint
import data


def verified_manifest(job, ns, required):
    path=HERE/'manifests'/f'retention-{job}.json';record=json.loads(path.read_text())
    assert record['local_and_remote_verified'] and record['terminal_state']=='COMPLETED'
    rows={r['file']:r for r in record['files']}
    for name,p in required.items():
        assert p.stat().st_size==rows[name]['bytes'] and digest(p)==rows[name]['sha256']
    return digest(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--model-job',default='b8childgn')
    ap.add_argument('--qualification-job',default='b8implprofile');ap.add_argument('--max-seconds',type=int,default=1680)
    a=ap.parse_args();owned_job(a.model_job);owned_job(a.qualification_job);assert 0<a.max_seconds<=1680
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    begun=time.perf_counter();metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/a.model_job/'output/result.json',ns);r=json.loads(rp.read_text())
    assert r['status']=='complete_nonlinear_child_multidepth_subset' and not r['final_test_opened']
    mp=rp.parent/'nonlinear_tree.pkl';assert digest(mp)==r['selected_sha256']
    retention=verified_manifest(a.model_job,ns,{'output/result.json':rp,'output/nonlinear_tree.pkl':mp})
    top=resolve(ns/a.qualification_job/'output/result.json',ns);qp=top.parent/'jacobian/result.json';q=json.loads(qp.read_text())
    assert q['status']=='complete_profile' and len(q['states'])==3
    checks=[v['parity'][k] for v in q['states'] for k in ['root_jacobian','residual_jacobian']]
    assert all(v['passed'] and v['max_absolute']<=1e-9 and v['relative']<=1e-10 for v in checks)
    source_manifest=top.parent.parent/'SOURCE.sha256'
    qretention=verified_manifest(a.qualification_job,ns,{'output/result.json':top,
        'output/jacobian/result.json':qp,'SOURCE.sha256':source_manifest})
    source_entries=[line.split(maxsplit=1) for line in source_manifest.read_text().splitlines() if line.strip()]
    derivative_source_hashes={}
    for name in ['neural/stable.py','neural/residual_tree.py','neural/residual_weak.py']:
        matches=[sha for sha,path in source_entries if path=='experiments/burgers-tree-embeddings/'+name]
        assert len(matches)==1 and digest(HERE/name)==matches[0], 'Derivative source differs from retained GPU profile: '+name
        derivative_source_hashes[name]=matches[0]

    model=read_checkpoint(mp);params=jax.tree.map(jnp.asarray,model['params'])
    pp,parent_record,parent,parent_proof=load_parent(r,ns)
    dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(dp)==r['primary_data_sha256']==parent['data_sha256']
    with np.load(dp) as v:arrays={k:np.array(v[k]) for k in v.files}
    jobs=[v['job'] for v in r['expanded_sources'] if v['cohort']=='dev']
    dv=load_group(ns,'dev',jobs,arrays,parent)
    assert dv['labels']==r['dev_labels']
    assert dv['provenance']==[v for v in r['expanded_sources'] if v['cohort']=='dev']
    xy,w=map(jnp.asarray,[arrays['xy'],arrays['w']]);assert len(xy)==128**2
    xf,wf=map(jnp.asarray,data.grid(256,True));cases=['b8_dev006','b8_dev000','b8_dev001','align004'];times=[0.,.125,.25]
    states=[];fine={};summary=[]
    for case in cases:
        ci=dv['labels'].index(case);source=dv['sources'][ci];sp=resolve(source['file'],ns);assert digest(sp)==source['sha256']
        with np.load(sp) as v:
            assert np.array_equal(v['times'],data.TIMES)
            fine[case]=data.synth_grid(jnp.asarray(v['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
        for t in times:
            for layout in ['uniform','prescribed']:
                rows=[v for v in model['evaluation'] if v['case']==case and abs(v['time']-t)<1e-12 and v['layout']==layout]
                assert len(rows)==1;row=rows[0];i=11*ci+round(t/.025);assert row['index']==i
                z=jnp.asarray(row['codes']);cells=jnp.asarray(row['cells']);st=np.asarray(row['status'])
                assert np.isfinite(np.asarray(z)).all() and np.isfinite(st).all()
                label=f'{case}_{layout}_t{round(t/.025):02d}'
                states.append(dict(z=z,cells=cells,index=i,case=case,time=t,layout=layout,label=label,done=bool(st[-1]),failed=False))
                summary.append(dict(case=case,time=t,layout=layout,index=i,initial_status=st,
                    initial_independent_error=row['independent_error'],initial_fit_error=float(st[0]),
                    initial_stationarity=float(st[1]),initial_physical_update=float(st[2]),initial_code_norm=float(jnp.linalg.norm(z)),
                    chunks=[],converged=bool(st[-1]),failed=False,additional_iterations=0,additional_accepted=0,
                    final=dict(chunk=0,status=st,fit_error=float(st[0]),stationarity=float(st[1]),physical_update=float(st[2]),
                        independent_error=row['independent_error'],code_norm=float(jnp.linalg.norm(z)),converged=bool(st[-1]),
                        note='Selected checkpoint state; replaced after successful continuation')))
    result=dict(meta=metadata,status='running_offline_fit_continuation',parent_job=parent_proof['parent_job'],parent_provenance=parent_proof,model_job=a.model_job,source_result_sha256=digest(rp),model_sha256=digest(mp),
        model_retention_sha256=retention,qualification_sha256=digest(qp),qualification_retention_sha256=qretention,
        derivative_checks=checks,overall_profile_parity_passed=q['parity_passed'],
        profile_source_manifest_sha256=digest(source_manifest),derivative_source_hashes=derivative_source_hashes,
        derivative_scope='Derivative values qualified; overall PDE trace parity is not claimed',
        primary_data_sha256=digest(dp),expanded_sources=dv['provenance'],rows=summary,
        max_chunks=5,iterations_per_chunk=200,max_seconds=a.max_seconds,final_test_opened=False,
        weights_changed=False,representation_admitted=False,pde_admitted=False,
        root_codes_jointly_refitted=True,offline_truth_used=True,
        interpretation='Unfinished fits are optimization evidence, never capacity floors. Independent256 scores use development reference fields offline.',
        lm_restart='Each200-iteration chunk restarts LM damping at1e-3; code persists. Same station/update convergence rules.',
        residual_module_sha256=digest(HERE/'neural/residual_tree.py'),stable_module_sha256=digest(HERE/'neural/stable.py'))
    save(out/'result.json',result);fit=F.make_fit(R.cached,200,jacobian=R.cached_jacobian_reverse)
    # Round-robin chunks give every selected field one continuation before a fifth on any field.
    for chunk in range(1,6):
        for state,row in zip(states,summary):
            if state['done'] or state['failed']:continue
            if time.perf_counter()-begun>=a.max_seconds:
                result.update(status='partial_time_cap',elapsed_seconds=time.perf_counter()-begun);save(out/'result.json',result);return
            start=time.perf_counter();i=state['index'];cells=state['cells'];z=state['z']
            new=None;st=None
            try:
                B=R.feature_matrix(params['feature'],cells,xy);cp=dict(params,B=B)
                new,st=fit(cp,z,xy,dv['y'][i],w,dv['norm'][i]);jax.block_until_ready(st)
                if not np.isfinite(np.asarray(new)).all() or not np.isfinite(np.asarray(st)).all():
                    raise FloatingPointError('Nonfinite continuation code/status')
                Bf=R.feature_matrix(params['feature'],cells,xf)
                pred=R.cached(dict(params,B=Bf),new,xf);truth=fine[state['case']][i%11]
                independent=float(jnp.sqrt(jnp.sum(wf*(pred-truth)**2))/dv['norm'][i])
                if not np.isfinite(independent):raise FloatingPointError('Nonfinite independent score')
                metrics=dict(chunk=chunk,status=np.asarray(st),fit_error=float(st[0]),stationarity=float(st[1]),physical_update=float(st[2]),
                    accepted=int(st[3]),iterations=int(st[4]),converged=bool(st[5]),independent_error=independent,
                    scoring_change=abs(independent-float(st[0])),code_norm=float(jnp.linalg.norm(new)),elapsed_seconds=time.perf_counter()-start)
                filename=f"{state['label']}_chunk{chunk}.pkl"
                metrics['payload']=filename;metrics['payload_sha256']=checkpoint(out/filename,dict(codes=new,cells=cells,metrics=metrics,model_sha256=digest(mp)))
                state['z']=new;state['done']=bool(st[-1]);row['chunks'].append(metrics)
                row.update(converged=state['done'],final=metrics,additional_iterations=row['additional_iterations']+int(st[4]),additional_accepted=row['additional_accepted']+int(st[3]))
            except Exception as exc:
                state['failed']=True;row.update(failed=True,failure=dict(chunk=chunk,type=type(exc).__name__,message=str(exc),elapsed_seconds=time.perf_counter()-start))
                filename=f"{state['label']}_chunk{chunk}_failed.pkl"
                row['failure']['payload']=filename
                row['failure']['payload_sha256']=checkpoint(out/filename,dict(previous_codes=z,candidate_codes=new,status=st,cells=cells))
            result['elapsed_seconds']=time.perf_counter()-begun;save(out/'result.json',result)
            print('long_fit',state['label'],chunk,row.get('final',row.get('failure')),flush=True)
    result.update(status='complete_offline_fit_continuation',elapsed_seconds=time.perf_counter()-begun,
        converged=sum(v['converged'] for v in summary),failed=sum(v['failed'] for v in summary),
        unfinished=sum(not v['converged'] and not v['failed'] for v in summary))
    save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',required=True)
    args,_=parser.parse_known_args();path=Path(args.out)/'result.json'
    result_existed=path.exists()
    try:
        main()
    except Exception as exc:
        # A rejected reused output directory must never mutate its prior evidence.
        if not result_existed:
            record=json.loads(path.read_text()) if path.exists() else dict(meta=meta(),final_test_opened=False,
                representation_admitted=False,pde_admitted=False)
            record.update(status_before_exception=record.get('status'),status='failed_setup_or_driver_exception',
                exception_type=type(exc).__name__,exception=str(exc))
            save(path,record)
        raise
