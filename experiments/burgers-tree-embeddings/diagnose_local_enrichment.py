"""Offline nested local enrichment using fixed weights and initial-only geometry."""
import argparse
from common import *
from neural import residual_tree as R, allocations as A, fields as F
from neural.parent_provenance import load_parent
from neural.build_data import resolve
from evaluate_residual_tree import load_group, owned_job
from extend_residual_fits import verified_manifest
from training_io import read_checkpoint,checkpoint
import data


def added_direction_audit(p,old_code,new_code,B,xy,w):
    """Project new zero-code columns off the current existing-field tangent."""
    J=R.cached_jacobian_reverse(dict(p,B=B),new_code,xy)*jnp.sqrt(w)[:,None]
    old,new=J[:,:len(old_code)],J[:,len(old_code):]
    oldnorm=jnp.linalg.norm(old,axis=0);newnorm=jnp.linalg.norm(new,axis=0)
    U,s,_=jnp.linalg.svd(old/jnp.maximum(oldnorm,1e-30),full_matrices=False)
    keep=s>jnp.maximum(s[0],1e-30)*1e-6;U=U*keep[None,:]
    normalized=new/jnp.maximum(newnorm,1e-30);residual=normalized-U@(U.T@normalized)
    gram=residual.T@residual;sig=jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh((gram+gram.T)/2),0.))
    ratio=sig/jnp.maximum(sig[-1],1e-30);rank=int(jnp.sum(ratio>1e-6))
    return dict(existing_directions=len(old_code),existing_rank=int(jnp.sum(keep)),new_directions=len(new_code)-len(old_code),
        independent_new_directions=rank,old_column_norms=np.asarray(oldnorm),new_column_norms=np.asarray(newnorm),
        projected_column_norms=np.asarray(jnp.linalg.norm(residual,axis=0)),projected_singular_values=np.asarray(sig),
        relative_cutoff=1e-6,useful_independent_directions=rank==len(new_code)-len(old_code) and float(newnorm.min())>1e-12)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--model-job',required=True)
    ap.add_argument('--qualification-job',default='b8implprofile');ap.add_argument('--max-seconds',type=int,default=1740)
    a=ap.parse_args();owned_job(a.model_job);owned_job(a.qualification_job);assert 0<a.max_seconds<=1740
    begun=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    meta_=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/a.model_job/'output/result.json',ns);r=json.loads(rp.read_text());assert r['status']=='complete_nonlinear_child_multidepth_subset' and not r['final_test_opened']
    mp=rp.parent/'nonlinear_tree.pkl';cp=rp.parent/'allocation_comparison.pkl'
    assert digest(mp)==r['selected_sha256'] and digest(cp)==r['comparison_sha256']
    retention=verified_manifest(a.model_job,ns,{'output/result.json':rp,'output/nonlinear_tree.pkl':mp,'output/allocation_comparison.pkl':cp})
    model=read_checkpoint(mp);comparison=read_checkpoint(cp);p=jax.tree.map(jnp.asarray,model['params']);root_k,k=R.dimensions(p);assert root_k==32 and r['nodes']==20
    pp,pr,parent,parent_proof=load_parent(r,ns)
    assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(p['root']),jax.tree.leaves(parent['network'])))
    top=resolve(ns/a.qualification_job/'output/result.json',ns);qp=top.parent/'jacobian/result.json';q=json.loads(qp.read_text())
    qret=verified_manifest(a.qualification_job,ns,{'output/result.json':top,'output/jacobian/result.json':qp})
    assert q['status']=='complete_profile' and len(q['states'])==3
    checks=[v['parity'][name] for v in q['states'] for name in ['root_jacobian','residual_jacobian']]
    assert all(v['passed'] and v['max_absolute']<=1e-9 and v['relative']<=1e-10 for v in checks)
    qm=json.loads((HERE/'manifests'/f'retention-{a.qualification_job}.json').read_text());qh={v['file']:v['sha256'] for v in qm['files']}
    sm=top.parent.parent/'SOURCE.sha256';assert digest(sm)==qh['SOURCE.sha256'];entries=[line.split(maxsplit=1) for line in sm.read_text().splitlines() if line.strip()]
    for module in ['stable.py','residual_tree.py']:
        hashes=[sha for sha,name in entries if name=='experiments/burgers-tree-embeddings/neural/'+module]
        assert len(hashes)==1 and digest(HERE/'neural'/module)==hashes[0]
    dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(dp)==r['primary_data_sha256']==parent['data_sha256']
    with np.load(dp) as v:arrays={key:np.array(v[key]) for key in v.files}
    sources=[v for v in r['expanded_sources'] if v['cohort']=='dev'];dv=load_group(ns,'dev',[v['job'] for v in sources],arrays,parent)
    assert dv['labels']==r['dev_labels'] and dv['provenance']==sources
    xy,w=map(jnp.asarray,[arrays['xy'],arrays['w']]);xf,wf=map(jnp.asarray,data.grid(256,True))
    cases=['align004','b8_dev006','b8_dev000','b8_dev001'];states=[];fine={}
    for case in cases:
        ci=dv['labels'].index(case);source=dv['sources'][ci];path=resolve(source['file'],ns);assert digest(path)==source['sha256']
        with np.load(path) as v:
            assert np.array_equal(v['times'],data.TIMES)
            fine[case]=data.synth_grid(jnp.asarray(v['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
        for ti in [0,5,10]:
            for layout in ['uniform','prescribed']:
                matches=[v for v in comparison if v['case']==case and v['layout']==layout and abs(v['time']-data.TIMES[ti])<1e-12]
                assert len(matches)==1;row=matches[0];assert row['index']==11*ci+ti
                selected_rows=[v for v in model['evaluation'] if v['case']==case and v['layout']==layout and abs(v['time']-data.TIMES[ti])<1e-12]
                assert len(selected_rows)==1
                for key in ['codes','cells','status']:assert np.array_equal(row[key],selected_rows[0][key])
                assert row['independent_error']==selected_rows[0]['independent_error']
                cells=np.asarray(row['cells']);code=jnp.asarray(row['codes'])
                assert cells.shape==(20,3) and code.shape==(32+20*k,) and np.isfinite(np.asarray(code)).all()
                expected=A.allocate(layout,arrays['xy'],np.asarray(dv['y'][11*ci]),r['seed']+ci,nodes=20,max_depth=3)
                assert np.array_equal(cells,expected)
                states.append(dict(case=case,ci=ci,ti=ti,index=11*ci+ti,layout=layout,cells=cells,code=code,
                    control_code=code,control_failed=False,controls=[],failed=False,stages=[],
                    baseline=dict(nodes=20,status=row['status'],independent_error=row['independent_error'],code_norm=float(jnp.linalg.norm(code)))))
    result=dict(meta=meta_,status='running_offline_local_enrichment',model_job=a.model_job,model_sha256=digest(mp),source_result_sha256=digest(rp),
        comparison_sha256=digest(cp),retention_sha256=retention,parent_provenance=parent_proof,qualification_sha256=digest(qp),
        qualification_retention_sha256=qret,qualified_source_manifest_sha256=digest(sm),overall_profile_trace_passed=q['parity_passed'],
        nodes=[20,36,52,84],uniform_stage_labels=['uniform20','breadth-first prefix36','breadth-first prefix52','uniform84'],
        max_seconds=a.max_seconds,weights_changed=False,root_codes_jointly_refitted=True,geometry='Initial-field-only prescribed or breadth-first; fixed max_depth3',
        offline_truth_used_for_fit_score_only=True,final_test_opened=False,actual4096=False,pde_admitted=False,representation_admitted=False,
        interpretation='Local enrichment diagnostic, not deployable adaptation or speed evidence',
        continuation_control='Independent20-node branch continues across all three stages, with the same60+120+200 maximum schedule per stage. Actual stopping counts/times can differ; this controls allowed fitting effort, not equal runtime.',rows=[])
    def persist():
        result['rows']=[{key:value for key,value in v.items() if key not in ['code','cells','control_code']} for v in states]
        result['elapsed_seconds']=time.perf_counter()-begun;save(out/'result.json',result)
    persist();fitters=[F.make_fit(R.cached,n,jacobian=R.cached_jacobian_reverse) for n in [60,120,200]]
    # Round robin across all fields at each budget. Existing20-node fit is retained baseline.
    for nodes in [36,52,84]:
        for state in states:
            if state['failed']:continue
            if time.perf_counter()-begun>=a.max_seconds:result['status']='partial_time_cap';persist();return
            # Independent fixed20 branch: never feed these continued codes into
            # the enriched branch. Each receives the same per-stage iteration cap.
            if not state['control_failed']:
                control_begin=time.perf_counter();control_statuses=[];control_candidate=None
                try:
                    ci,ti,i=state['ci'],state['ti'],state['index'];control_cells=state['cells'][:20]
                    CB=R.feature_matrix(p['feature'],jnp.asarray(control_cells),xy)
                    CBf=R.feature_matrix(p['feature'],jnp.asarray(control_cells),xf)
                    control_candidate=state['control_code']
                    for limit,fit in zip([60,120,200],fitters):
                        control_candidate,control_status=fit(dict(p,B=CB),control_candidate,xy,dv['y'][i],w,dv['norm'][i])
                        jax.block_until_ready(control_status);control_statuses.append(dict(limit=limit,status=np.asarray(control_status)))
                        assert np.isfinite(np.asarray(control_candidate)).all() and np.isfinite(np.asarray(control_status)).all()
                        if bool(control_status[-1]):break
                    control_pred=R.cached(dict(p,B=CBf),control_candidate,xf)
                    control_error=float(jnp.sqrt(jnp.sum(wf*(control_pred-fine[state['case']][ti])**2))/dv['norm'][i]);assert np.isfinite(control_error)
                    control_row=dict(matched_enrichment_nodes=nodes,nodes=20,statuses=control_statuses,
                        fit_error=float(control_status[0]),independent_error=control_error,converged=bool(control_status[-1]),
                        iterations=sum(int(v['status'][4]) for v in control_statuses),accepted=sum(int(v['status'][3]) for v in control_statuses),
                        code_norm=float(jnp.linalg.norm(control_candidate)),elapsed_seconds=time.perf_counter()-control_begin)
                    filename=f"{state['case']}_{state['layout']}_t{ti:02d}_control20_stage{nodes}.pkl"
                    control_row['payload']=filename;control_row['payload_sha256']=checkpoint(out/filename,
                        dict(code=control_candidate,cells=control_cells,row=control_row,model_sha256=digest(mp)))
                    state['controls'].append(control_row);state['control_code']=control_candidate
                except Exception as exc:
                    state['control_failed']=True;filename=f"{state['case']}_{state['layout']}_t{state['ti']:02d}_control20_stage{nodes}_failure.pkl"
                    sha=checkpoint(out/filename,dict(candidate=control_candidate,statuses=control_statuses))
                    state['control_failure']=dict(matched_enrichment_nodes=nodes,error=repr(exc),payload=filename,sha256=sha)
                persist()
            if time.perf_counter()-begun>=a.max_seconds:result['status']='partial_time_cap';persist();return
            begin=time.perf_counter();ci,ti,i=state['ci'],state['ti'],state['index'];oldcells=state['cells'];old=state['code']
            new=None;statuses=[]
            try:
                cells=A.allocate(state['layout'],arrays['xy'],np.asarray(dv['y'][11*ci]),r['seed']+ci,nodes=nodes,max_depth=3)
                A.validate(cells);assert np.array_equal(cells[:len(oldcells)],oldcells)
                new=jnp.pad(old,(0,(nodes-len(oldcells))*k));B=R.feature_matrix(p['feature'],jnp.asarray(cells),xy);Bf=R.feature_matrix(p['feature'],jnp.asarray(cells),xf)
                fit_old=R.cached(dict(p,B=B[:,:len(oldcells)]),old,xy);fit_new=R.cached(dict(p,B=B),new,xy)
                score_old=R.cached(dict(p,B=Bf[:,:len(oldcells)]),old,xf);score_new=R.cached(dict(p,B=Bf),new,xf)
                if len(oldcells)==20:
                    original_fit=float(jnp.sqrt(jnp.sum(w*(fit_old-dv['y'][i])**2))/dv['norm'][i])
                    original_score=float(jnp.sqrt(jnp.sum(wf*(score_old-fine[state['case']][ti])**2))/dv['norm'][i])
                    assert abs(original_fit-float(state['baseline']['status'][0]))<=1e-10
                    assert abs(original_score-state['baseline']['independent_error'])<=1e-10
                    state['baseline'].update(recomputed_fit_error=original_fit,recomputed_independent_error=original_score,verified_current_decoder=True)
                preservation=dict(fit_max_abs=float(jnp.max(jnp.abs(fit_old-fit_new))),independent_max_abs=float(jnp.max(jnp.abs(score_old-score_new))))
                assert max(preservation.values())<=1e-12
                directions=added_direction_audit(p,old,new,B,xy,w)
                assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(directions))
                for iterations,fit in zip([60,120,200],fitters):
                    new,st=fit(dict(p,B=B),new,xy,dv['y'][i],w,dv['norm'][i]);jax.block_until_ready(st)
                    statuses.append(dict(limit=iterations,status=np.asarray(st)))
                    assert np.isfinite(np.asarray(new)).all() and np.isfinite(np.asarray(st)).all()
                    if bool(st[-1]):break
                pred=R.cached(dict(p,B=Bf),new,xf);err=float(jnp.sqrt(jnp.sum(wf*(pred-fine[state['case']][ti])**2))/dv['norm'][i]);assert np.isfinite(err)
                row=dict(nodes=nodes,layout_label=('breadth-first prefix'+str(nodes)) if state['layout']=='uniform' and nodes in [36,52] else state['layout']+str(nodes),
                    preservation=preservation,added_directions=directions,statuses=statuses,independent_error=err,fit_error=float(st[0]),converged=bool(st[-1]),
                    additional_iterations=sum(int(v['status'][4]) for v in statuses),accepted=sum(int(v['status'][3]) for v in statuses),
                    variables=len(new),code_norm=float(jnp.linalg.norm(new)),elapsed_seconds=time.perf_counter()-begin)
                matched=[v for v in state['controls'] if v['matched_enrichment_nodes']==nodes]
                row['matched_control_available']=bool(matched)
                if matched:
                    row['continued20_independent_error']=matched[0]['independent_error']
                    row['error_minus_continued20']=err-matched[0]['independent_error']
                filename=f"{state['case']}_{state['layout']}_t{ti:02d}_n{nodes}.pkl";row['payload']=filename
                row['payload_sha256']=checkpoint(out/filename,dict(code=new,cells=cells,row=row,model_sha256=digest(mp)))
                state['stages'].append(row);state['code']=new;state['cells']=cells
            except Exception as exc:
                state['failed']=True;filename=f"{state['case']}_{state['layout']}_t{ti:02d}_n{nodes}_failure.pkl"
                failure_hash=checkpoint(out/filename,dict(previous_code=old,candidate_code=new,statuses=statuses))
                state['failure']=dict(nodes=nodes,error=repr(exc),payload=filename,sha256=failure_hash,elapsed_seconds=time.perf_counter()-begin)
            persist();print('local_enrichment',state['case'],state['layout'],ti,nodes,state.get('failure') or state['stages'][-1]['independent_error'],flush=True)
    result.update(status='complete_offline_local_enrichment',failed_fields=sum(v['failed'] for v in states),
        failed_controls=sum(v['control_failed'] for v in states));persist()


if __name__=='__main__':main()
