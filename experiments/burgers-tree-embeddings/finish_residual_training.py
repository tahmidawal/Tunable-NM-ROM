"""Finish retained trained weights with qualified reverse derivatives; no SGD."""
import argparse, shutil, traceback
from common import *
from neural import residual_tree as R, fields as F, allocations as A
from neural.build_data import resolve
from neural.parent_provenance import load_parent
from evaluate_residual_tree import load_group, make_initializer, equal_tree, owned_job, zero_child_audit
from extend_residual_fits import verified_manifest
from training_io import checkpoint, read_checkpoint
import data


def run():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--source-job',default='b8childgn')
    ap.add_argument('--qualification-job',default='b8implprofile');ap.add_argument('--max-seconds',type=int,default=5280)
    a=ap.parse_args();owned_job(a.source_job);owned_job(a.qualification_job);assert 0<a.max_seconds<=5280
    started=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/a.source_job/'output/result.json',ns);source=json.loads(rp.read_text())
    manifest=HERE/'manifests'/f'retention-{a.source_job}.json';retained=json.loads(manifest.read_text())
    terminal=retained['terminal_state'];assert terminal=='COMPLETED' or terminal.startswith('CANCELLED')
    if terminal=='COMPLETED':assert source['status']=='complete_nonlinear_child_multidepth_subset'
    assert retained['local_and_remote_verified'] and str(retained['job'])==str(source['meta']['job'])
    assert retained['source_commit']==source['meta']['commit'] and not source['final_test_opened']
    names=['result.json','nonlinear_tree.pkl','latest_training.pkl','initialization.pkl','root_initialization.pkl']
    entries={v['file']:v for v in retained['files']}
    for name in names:
        path=rp.parent/name;entry=entries['output/'+name]
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
    assert digest(rp.parent/'nonlinear_tree.pkl')==source['selected_sha256']
    assert digest(rp.parent/'initialization.pkl')==source['initialization_sha256']
    assert digest(rp.parent/'root_initialization.pkl')==source['root_initialization_sha256']
    old=read_checkpoint(rp.parent/'nonlinear_tree.pkl');latest=read_checkpoint(rp.parent/'latest_training.pkl')
    assert latest['update']==30000
    init=read_checkpoint(rp.parent/'initialization.pkl');rootinit=read_checkpoint(rp.parent/'root_initialization.pkl')
    top=resolve(ns/a.qualification_job/'output/result.json',ns);qp=top.parent/'jacobian/result.json';q=json.loads(qp.read_text())
    qret=verified_manifest(a.qualification_job,ns,{'output/result.json':top,'output/jacobian/result.json':qp})
    qmanifest=json.loads((HERE/'manifests'/f'retention-{a.qualification_job}.json').read_text())
    qhashes={v['file']:v['sha256'] for v in qmanifest['files']}
    source_manifest=top.parent.parent/'SOURCE.sha256';assert digest(source_manifest)==qhashes['SOURCE.sha256']
    source_entries=[line.split(maxsplit=1) for line in source_manifest.read_text().splitlines() if line.strip()]
    qualified_modules={}
    for module in ['stable.py','residual_tree.py']:
        hashes=[sha for sha,name in source_entries if name=='experiments/burgers-tree-embeddings/neural/'+module]
        assert len(hashes)==1 and digest(HERE/'neural'/module)==hashes[0], 'Derivative implementation differs from qualified source'
        qualified_modules[module]=hashes[0]
    assert q['status']=='complete_profile' and len(q['states'])==3
    checks=[row['parity'][key] for row in q['states'] for key in ['root_jacobian','residual_jacobian']]
    assert all(v['passed'] and v['max_absolute']<=1e-9 and v['relative']<=1e-10 for v in checks)
    pp,parent_record,parent,parent_proof=load_parent(source,ns)
    dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(dp)==source['primary_data_sha256']==parent['data_sha256']
    with np.load(dp) as v:arrays={k:np.array(v[k]) for k in v.files}
    groups={}
    for group in ['train','dev']:
        provenance=[v for v in source['expanded_sources'] if v['cohort']==group]
        groups[group]=load_group(ns,group,[v['job'] for v in provenance],arrays,parent)
        assert groups[group]['provenance']==provenance and groups[group]['labels']==source[group+'_labels']
    tr,dv=groups['train'],groups['dev'];assert len(tr['labels'])==160 and len(dv['labels'])==52
    assert not set(tr['labels'])&set(dv['labels'])
    variants=['coarse4','uniform20','prescribed20'];nodes=source['nodes'];child_k=source['child_k'];dim=32+nodes*child_k
    assert nodes==20 and source['variant_labels']==variants
    cells=np.asarray(init['cells']);assert cells.shape==(3,160,20,3)
    for candidate in [old,latest]:
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(candidate['params']))
        assert candidate['parent_sha256']==source['parent_sha256'] and candidate['expanded_sources']==source['expanded_sources']
        assert candidate['variant_labels']==variants and np.array_equal(candidate['training_cells'],cells)
        assert np.asarray(candidate['training_codes']).shape==(3,1760,dim)
        assert np.isfinite(np.asarray(candidate['training_codes'])).all()
        assert np.count_nonzero(np.asarray(candidate['training_codes'])[0,:,32+4*child_k:])==0
        assert equal_tree(candidate['params']['root'],parent['network'])
    assert np.array_equal(rootinit['dev_root_codes'],init['dev_root_codes'])
    devroot=jnp.asarray(init['dev_root_codes']);assert devroot.shape==(572,32)
    xy,w=map(jnp.asarray,[arrays['xy'],arrays['w']]);xf,wf=map(jnp.asarray,data.grid(256,True))
    selection=[11*c+t for c in [0,3,13,20]+list(range(44,52)) for t in [0,5,10]]
    fine={}
    for ci in sorted({i//11 for i in selection}):
        row=dv['sources'][ci];path=resolve(row['file'],ns);assert digest(path)==row['sha256']
        with np.load(path) as v:
            assert np.array_equal(v['times'],data.TIMES)
            fine[ci]=data.synth_grid(jnp.asarray(v['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
    for name in ['initialization.pkl','root_initialization.pkl']:shutil.copyfile(rp.parent/name,out/name)
    if source.get('representative_zero_audit_required',False):
        initial_audit=source['initial_zero_child_audit'];name=initial_audit['file'];path=rp.parent/name
        assert path.parent.resolve()==rp.parent.resolve()
        entry=entries['output/'+name]
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']==initial_audit['sha256']
        shutil.copyfile(path,out/name)
    result=dict(source);result.update(meta=metadata,status='finishing_trained_candidates',parent_job=parent_proof['parent_job'],parent_provenance=parent_proof,history=[],
        source_job=a.source_job,source_status=source['status'],source_terminal_state=terminal,
        source_retention_sha256=digest(manifest),source_files={name:entries['output/'+name]['sha256'] for name in names},
        weight_updates_performed=0,source_trained_update=30000,all_final_refit=False,
        optimizer_continuation=False,max_seconds=a.max_seconds,
        qualification_sha256=digest(qp),qualification_retention_sha256=qret,derivative_checks=checks,
        qualified_source_manifest_sha256=digest(source_manifest),qualified_module_hashes=qualified_modules,
        overall_profile_parity_passed=q['parity_passed'],jacobian='cached_jacobian_reverse',
        completion_protocol='Reevaluate prior selection and30k weights; refit all3 code banks for SELECTED weights, not necessarily the final30k weights',
        final_test_opened=False,representation_admitted=False,pde_admitted=False,
        initialization_sha256=digest(out/'initialization.pkl'),root_initialization_sha256=digest(out/'root_initialization.pkl'))
    # Do not expose inherited selected hashes as if their files existed in this output.
    for key in ['selected_sha256','selected_update','comparison_sha256','worst_subset_error','unfinished_subset_fits','coarse4_diagnostic',
                'selected_zero_child_audit','selected_zero_child_directions_pass']:
        result.pop(key,None)
    def persist():
        result['elapsed_seconds']=time.perf_counter()-started;save(out/'result.json',result)
    def budget():
        if time.perf_counter()-started>=a.max_seconds:
            result['status']='partial_time_cap';persist();return False
        return True
    persist();fit=F.make_fit(R.cached,60,jacobian=R.cached_jacobian_reverse);more=F.make_fit(R.cached,120,jacobian=R.cached_jacobian_reverse)
    def finish(p,z,target,norm):
        z,st=fit(p,z,xy,target,w,norm);attempts=[np.asarray(st)]
        if not bool(st[-1]):z,st=more(p,z,xy,target,w,norm);attempts.append(np.asarray(st))
        assert np.isfinite(np.asarray(z)).all() and np.isfinite(np.asarray(st)).all()
        return z,st,attempts
    def evaluate(params,label,kinds):
        rows=[]
        for kind in kinds:
            active=4 if kind=='coarse4' else nodes;initializer=make_initializer(child_k,active)
            cache={}
            for i in selection:
                if not budget():return None
                ci,ti=divmod(i,11)
                if ci not in cache:
                    cache.clear();cell=jnp.asarray(A.allocate('uniform' if kind=='coarse4' else kind,np.asarray(xy),np.asarray(dv['y'][11*ci]),source['seed']+ci,nodes=active,max_depth=3))
                    cache[ci]=(cell,R.feature_matrix(params['feature'],cell,xy),R.feature_matrix(params['feature'],cell,xf))
                cell,B,Bf=cache[ci];z,ie=initializer(params,devroot[i:i+1],B,dv['y'][i:i+1],dv['norm'][i:i+1],xy,w)
                z,st,attempts=finish(dict(params,B=B),z[0],dv['y'][i],dv['norm'][i])
                pred=R.cached(dict(params,B=Bf),z,xf);err=float(jnp.sqrt(jnp.sum(wf*(pred-fine[ci][ti])**2))/dv['norm'][i]);assert np.isfinite(err)
                rows.append(dict(index=i,case=dv['labels'][ci],time=float(data.TIMES[ti]),layout=kind,codes=np.asarray(z),cells=np.asarray(cell),
                    status=np.asarray(st),fit_attempts=attempts,independent_error=err,scoring_change=abs(err-float(st[0]))))
                if len(rows)%6==0:
                    checkpoint(out/f'{label}_panel_partial.pkl',dict(rows=rows,complete=len(rows)==len(selection)*len(kinds)))
            print('finishing_panel',label,kind,len(rows),flush=True)
        return rows
    candidates=[]
    for label,candidate,update in [('previous_selected',old,source['selected_update']),('latest30000',latest,30000)]:
        params=jax.tree.map(jnp.asarray,candidate['params']);ev=evaluate(params,label,['uniform','prescribed'])
        if ev is None:return
        metrics=dict(candidate=label,update=update,subset_worst=max(v['independent_error'] for v in ev),
            subset_median=float(np.median([v['independent_error'] for v in ev])),unfinished=sum(not bool(v['status'][-1]) for v in ev))
        payload=dict(candidate,evaluation=ev,metrics=metrics)
        evaluated_hash=checkpoint(out/f'{label}_evaluated.pkl',payload)
        payload['evaluated_checkpoint_sha256']=evaluated_hash
        result['history'].append(dict(metrics,sha256=evaluated_hash));candidates.append((metrics['subset_worst'],label,payload));persist()
    _,label,selected=min(candidates,key=lambda v:v[0]);params=jax.tree.map(jnp.asarray,selected['params']);bank=jnp.asarray(selected['training_codes'])
    result.update(selected_candidate=label,selected_update=selected['metrics']['update'],selection_metric='Worst independent256 error on unchanged72-field panel')
    statuses=[];completed=0
    for variant in range(3):
        active=4 if variant==0 else nodes;k=32+active*child_k
        for ci in range(160):
            B=R.feature_matrix(params['feature'],jnp.asarray(cells[variant,ci,:active]),xy);cp=dict(params,B=B)
            for ti in range(11):
                if not budget():return
                i=11*ci+ti;z,st,attempts=finish(cp,bank[variant,i,:k],tr['y'][i],tr['norm'][i])
                bank=bank.at[variant,i,:k].set(z);completed+=1
                statuses.append(dict(variant=variants[variant],index=i,case=tr['labels'][ci],time=float(data.TIMES[ti]),status=np.asarray(st),attempts=attempts))
                if completed%32==0 or completed==5280:
                    ph=checkpoint(out/'training_refresh_partial.pkl',dict(training_codes=bank,records=statuses,completed=completed,expected=5280,
                        selected_candidate=label,selected_evaluated_sha256=selected['evaluated_checkpoint_sha256'],training_cells=cells,variant_labels=variants))
                    result.update(refresh_completed=completed,refresh_partial_sha256=ph);persist();print('finishing_training_codes',completed,5280,flush=True)
    coarse=evaluate(params,'selected_coarse',['coarse4'])
    if coarse is None:return
    ev=selected['evaluation'];result['comparison_sha256']=checkpoint(out/'allocation_comparison.pkl',ev+coarse)
    selected.update(params=params,training_codes=bank,training_fit_records=statuses,optimizer=None,
        optimizer_note='No weight updates; embedding refits invalidate any optimizer continuation')
    result['selected_sha256']=checkpoint(out/'nonlinear_tree.pkl',selected)
    if source.get('representative_zero_audit_required',False):
        audit_start=time.perf_counter();primary=data.parameters('train',128)
        narrow=np.flatnonzero((primary[:,0]==0)&(primary[:,3]<=.05));assert len(narrow)>0
        representatives=[('narrow_primary',int(narrow[0])),('ridge',tr['family'].index('ridge')),('two_pulse',tr['family'].index('two_pulse'))]
        audit_rows=[]
        for family,ci in representatives:
            for variant,layout in enumerate(variants):
                if not budget():return
                active=4 if variant==0 else nodes;k=32+active*child_k
                cell=jnp.asarray(cells[variant,ci,:active]);B=R.feature_matrix(params['feature'],cell,xy)
                audit=zero_child_audit(params,bank[variant,11*ci,:k],B,xy,w)
                audit_rows.append(dict(family=family,case=tr['labels'][ci],cohort='train',time=0.,layout=layout,cells=np.asarray(cell),audit=audit))
                checkpoint(out/'zero_child_audit_selected_partial.pkl',dict(rows=audit_rows,complete=len(audit_rows)==9,
                    model_sha256=result['selected_sha256'],parent_sha256=digest(pp)))
        passed=all(v['audit']['zero_child_preserved'] and v['audit']['useful_independent_directions'] for v in audit_rows)
        filename='zero_child_audit_selected.pkl'
        audit_hash=checkpoint(out/filename,dict(stage='selected_after_training_code_refresh',rows=audit_rows,
            model_sha256=result['selected_sha256'],parent_sha256=digest(pp),passed=passed,
            scope='Three training trajectories at t0 across three layouts; not global qualification'))
        result['selected_zero_child_audit']=dict(file=filename,sha256=audit_hash,passed=passed,fields=9,
            elapsed_seconds=time.perf_counter()-audit_start,cases=[tr['labels'][ci] for _,ci in representatives],
            regenerated_after_code_refresh=True,scope='Representative directions only; all-case qualification remains required')
        result['selected_zero_child_directions_pass']=passed
    result.update(status='complete_nonlinear_child_multidepth_subset',all_final_refit=True,
        worst_subset_error=max(v['independent_error'] for v in ev),unfinished_subset_fits=sum(not bool(v['status'][-1]) for v in ev),
        coarse4_diagnostic=dict(worst=max(v['independent_error'] for v in coarse),unfinished=sum(not bool(v['status'][-1]) for v in coarse)),
        final_training_refit=dict(count=5280,worst=max(float(v['status'][0]) for v in statuses),unfinished=sum(not bool(v['status'][-1]) for v in statuses)))
    persist()


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',required=True)
    args,_=parser.parse_known_args();path=Path(args.out)/'result.json';already_existed=path.exists()
    try:
        run()
    except Exception as exc:
        # A rejected attempt to reuse an output must never overwrite prior evidence.
        if not already_existed:
            previous=json.loads(path.read_text()) if path.exists() else dict(final_test_opened=False,weight_updates_performed=0)
            previous.update(previous_status=previous.get('status'),status='failed_finishing',
                error=repr(exc),traceback=traceback.format_exc(),all_final_refit=False)
            save(path,previous)
        raise


if __name__=='__main__':main()
