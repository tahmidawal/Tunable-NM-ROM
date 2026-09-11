"""Independent NumPy/SciPy audit of paired Burgers accuracy experiment outputs."""
import argparse,collections,hashlib,itertools,json,pickle,subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit
import audit_iterative as ai


def head_jac(p,z):
    x=z;J=np.eye(len(z))
    for i,(w,b) in enumerate(p['h']):
        x=x@w+b;J=w.T@J
        if i<len(p['h'])-1:
            s=expit(x);J=(s+x*s*(1-s))[:,None]*J;x=x*s
    return x+z@p['h_lin'],J+p['h_lin'].T


def complex_head(p,z):
    x=z
    for i,(w,b) in enumerate(p['h']):
        x=x@w+b
        if i<len(p['h'])-1:x=x/(1+np.exp(-x))
    return x+z@p['h_lin']


def stationarity(r,J):return float(np.linalg.norm(J.T@r)/(np.linalg.norm(J)*np.linalg.norm(r)+1e-300))
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('record',type=Path);a=parser.parse_args();record=a.record;run=record/'archive';out=run/'out';root=Path(__file__).resolve().parents[2]
    d=json.loads((out/'result.json').read_text());cfg=d['config'];old_record=record.parent/'iterative06';old=json.loads((old_record/'archive/out/result.json').read_text())
    assert d['complete'] and d['backend']=='gpu' and d['x64'] and d['matmul_precision']=='highest' and d['spatial_bank_arrays_byte_identical'] and d['network_weights_frozen_during_evaluation'] and d['final_cohort_unopened']
    assert d['checkpoint_sha256']==d['checkpoint_sha256_after']==ai.sha(run/'in/checkpoint.pkl')==old['checkpoint_sha256']
    assert d['trained_checkpoint_sha256']==ai.sha(out/'trained_checkpoint.pkl') and d['training_complete_before_development_opened']
    assert cfg==json.loads((run/'code/config-accuracy.json').read_text())
    assert d['commit']==(run/'COMMIT.txt').read_text().strip() and str(d['job_id'])==json.loads((record/'SUBMISSION.json').read_text())['job_id']
    for p in json.loads((run/'PROVENANCE.json').read_text()):
        data=subprocess.check_output(['git','show',p['commit']+':'+p['source']],cwd=root)
        assert hashlib.sha256(data).hexdigest()==p['sha256']==ai.sha(run/p['staged'])
    collection=json.loads((record/'COLLECTION-CHECK.json').read_text());archive_metadata=json.loads((record/'ARCHIVE.json').read_text());submission=json.loads((record/'SUBMISSION.json').read_text())
    assert collection['archive_verified']
    assert d['commit']==collection['source_commit']==archive_metadata['source_commit']==submission['source_commit']
    assert str(d['job_id'])==collection['job_id']==archive_metadata['job_id']==submission['job_id']
    if 'training_and_reference_lineage' in d:
        lineage=d['training_and_reference_lineage'];assert lineage==json.loads((run/'in/reuse/INHERITED.json').read_text())
        parent_record=record.parent/'accuracy07';parent_archive=json.loads((parent_record/'ARCHIVE.json').read_text());parent_result=json.loads((parent_record/'archive/out/result.json').read_text())
        assert lineage['parent_archive_sha256']==parent_archive['joined_sha256'] and lineage['parent_job_id']==parent_result['job_id'] and lineage['parent_source_commit']==parent_result['commit']
        assert lineage['parent_gpu']==parent_result['gpu'] and lineage['parent_reference_seed']==cfg['seed'] and lineage['parent_reference_fresh_seed']==cfg['fresh_seed']
        assert len(lineage['files'])==40
        for item in lineage['files']:
            assert ai.sha(run/'in/reuse'/item['destination'])==item['sha256']==ai.sha(parent_record/'archive'/item['source'])
        for row in d['reference']:assert ai.sha(out/row['artifact'])==row['source_sha256']==ai.sha(run/'in/reuse'/row['artifact'])
        assert not d['training_rerun'] and not d['failures']

    stdout='\n'.join(p.read_text() for p in (run/'logs').glob('*.out'));stderr='\n'.join(p.read_text() for p in (run/'logs').glob('*.err'))
    assert 'jax_backend=gpu' in stdout and 'ACCURACY BURGERS COMPLETE' in stdout and 'ALL-DONE' in stdout and not stderr.strip(),stderr
    training_audit_path=record/'TRAINING-AUDIT.json'
    if not training_audit_path.exists():training_audit_path=record.parent/'accuracy07/TRAINING-AUDIT.json'
    ta=json.loads(training_audit_path.read_text());assert ta['passed'] and ta['R_rank']==512
    # Direct training audit may precede collection; require the exact same scientific bytes.
    for path in [out/'training_targets.npz',out/'training.json',out/'trained_checkpoint.pkl']:
        expected=[v for k,v in ta['source_sha256'].items() if Path(k).name==path.name];assert len(expected)==1 and ai.sha(path)==expected[0]
    weights={'frozen':pickle.loads((run/'in/checkpoint.pkl').read_bytes()),'trained':pickle.loads((out/'trained_checkpoint.pkl').read_bytes())}
    meshes=cfg['meshes'];nc=cfg['cases']+cfg['fresh_cases'];names=['frozen_accepted','frozen_stationary','trained_accepted','trained_stationary']+[s['name'] for s in cfg['fom_settings']]
    expected=set(itertools.product(meshes,range(cfg['reps']),range(nc),names));actual=[(r['intervals'],r['rep'],r['case'],r['name']) for r in d['invocations']]
    assert len(actual)==len(expected) and set(actual)==expected and len(d['declared_subjects'])==len(meshes)*len(names)
    def draw(seed,count):
        rng=np.random.default_rng(seed)
        return np.stack([rng.uniform(.15,.85,count),rng.uniform(.15,.85,count),rng.uniform(.05,.2,count),rng.uniform(.5,2.,count),np.exp(rng.uniform(np.log(.01),np.log(.1),count))],1)
    physical=np.concatenate((draw(cfg['seed'],cfg['cases']),draw(cfg['fresh_seed'],cfg['fresh_cases'])));assert np.allclose(physical,d['physical_cases'],rtol=2e-15,atol=0)
    physical=np.array(d['physical_cases']);assert d['cohort_roles']==['opened development']*cfg['cases']+['fresh development']*cfg['fresh_cases']
    rng=np.random.default_rng(cfg['order_seed'])
    for L,rep,case in itertools.product(meshes,range(cfg['reps']),range(nc)):
        assert [names[i] for i in rng.permutation(len(names))]==[r['name'] for r in d['invocations'] if (r['intervals'],r['rep'],r['case'])==(L,rep,case)]
    largest=max(meshes);rf=cfg['reference_mesh'];rt=cfg['reference_dt'];refs={};ref_delta=0.
    for row in d['reference']:
        z=np.load(out/row['artifact']);f=z['fields'];assert f.dtype==np.float64 and f.shape==(6,largest+1,largest+1) and np.isfinite(f).all()
        assert max(z['residuals'])<2e-11 and abs(max(z['residuals'])-row['max_relative_residual'])<1e-15
        refs[row['intervals'],row['dt'],row['case']]=f
    assert len(refs)==6*nc
    for L in meshes:
        for row in d['reference_metrics'][str(L)]:
            c=row['case']
            def diff(n1,t1,n2,t2):return ai.errors(refs[n1,t1,c][:,::largest//L,::largest//L],refs[n2,t2,c][:,::largest//L,::largest//L])[0]['fixed_initial_max']
            sd=diff(rf//2,rt,rf,rt);td=diff(rf,2*rt,rf,rt);sc=diff(rf//4,2*rt,rf//2,2*rt);sf=diff(rf//2,2*rt,rf,2*rt);tc=diff(rf,4*rt,rf,2*rt)
            vals=dict(space_difference=sd,time_difference=td,margin=sd+td,spatial_coarse=sc,spatial_fine=sf,temporal_coarse=tc,observed_space_order=np.log2(sc/sf),observed_time_order=np.log2(tc/td))
            ref_delta=max(ref_delta,max(abs(v-row[k]) for k,v in vals.items()));assert row['decrease']==bool(sf<sc and td<tc)
    assert ref_delta<1e-10
    grouped=collections.defaultdict(list)
    for row in d['invocations']:grouped[row['artifact']].append(row)
    maxima=collections.defaultdict(float);state_count=0;fom_count=0;parities=[];operator_checks=[];convergence_audit={};threshold_disagreements=[]
    for L in meshes:
        ops={model:dict(np.load(out/f'operators_{model}_L{L}.npz')) for model in weights}
        for model,op in ops.items():
            p=weights[model]['params'];info=next(x for x in d['mesh_setup'] if x['model']==model and x['intervals']==L)
            gc=ai.features(p,op['cold_xy'])*op['cold_w'][:,None]
            assert np.linalg.norm(gc-op['cold_Q']@op['cold_R'])/np.linalg.norm(gc)<2e-12
            assert np.linalg.norm(op['cold_Q'].T@op['cold_Q']-np.eye(512))<1e-10
            ij=np.stack(np.unravel_index(np.array(info['eq_indices']),(L-1,L-1)),1)+1;offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
            g5=ai.features(p,((ij[:,None,:]+offsets)/L).reshape(-1,2)).reshape(op['G5'].shape)
            assert np.linalg.norm(g5-op['G5'])/np.linalg.norm(g5)<2e-12
            ids=np.array(info['mode_ids']);phi=(2./L)*np.sin(np.pi*ij[:,0,None]/L*ids[:,0])*np.sin(np.pi*ij[:,1,None]/L*ids[:,1])
            assert np.linalg.norm(phi*np.array(info['eq_weights'])[:,None]-op['Pq'])<1e-12
            lam=4*L*L*(np.sin(np.pi*ids[:,0]/(2*L))**2+np.sin(np.pi*ids[:,1]/(2*L))**2);assert np.max(abs(lam-op['lam']))<1e-10
            old_codes=weights['frozen']['Z_tr'];candidate=old_codes[::max(1,len(old_codes)//8192)]
            if model=='trained':candidate=np.concatenate((candidate,weights['trained']['Z_tr'][len(old_codes):]))
            assert np.array_equal(candidate,op['candidate_Z'])
            radius=.01*float(np.max(np.linalg.norm(old_codes-old_codes.mean(0),axis=1)));assert abs(radius-info['trust_radius'])<1e-14
            fit=np.sort(np.random.default_rng(info['eq_seed']).choice((L-1)**2,min(cfg['candidate_cap'],(L-1)**2),replace=False))
            # RNG continuation for fit-state draw is checked separately from candidate locations.
            rr=np.random.default_rng(info['eq_seed']);rr.choice((L-1)**2,min(cfg['candidate_cap'],(L-1)**2),replace=False)
            assert np.array_equal(np.sort(rr.choice(len(old_codes),cfg['fit_states'],replace=False)),info['eq_fit_rows'])
            operator_checks.append(dict(intervals=L,model=model,eq_relative_fit=info['eq_relative_fit'],cold_candidates=len(candidate),trust_radius=radius))
        ns=min(31,L-1);ids=np.unique(np.rint(np.linspace(1,L-1,ns)).astype(int));ix,iy=np.meshgrid(ids,ids,indexing='ij');xy=np.stack((ix.ravel()/L,iy.ravel()/L),1)
        gsample=ai.features(weights['frozen']['params'],xy)
        for artifact,rows in grouped.items():
            row=rows[0]
            if row['intervals']!=L:continue
            case=row['case'];nu=physical[case,4];saved=dict(np.load(out/artifact));f=saved['fields'];truth=refs[rf,rt,case][:,::largest//L,::largest//L]
            assert f.shape==(6,L+1,L+1) and f.dtype==np.float64 and np.isfinite(f).all() and np.max(abs(f[:,[0,-1],:]))==0 and np.max(abs(f[:,:,[0,-1]]))==0
            metrics,_=ai.errors(f,truth);hash_=hashlib.sha256(f.tobytes()).hexdigest()
            xx,yy=np.meshgrid(np.arange(L+1)/L,np.arange(L+1)/L,indexing='ij');cx,cy,w,amp,_=physical[case];u0=amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*w*w));u0[[0,-1]]=0;u0[:,[0,-1]]=0
            assert np.max(abs(u0-truth[0]))<1e-14
            for row in rows:
                assert row['field_sha256']==hash_ and row['output_bytes']==f.nbytes and row['gpu_seconds']>0 and row['host_seconds']>=row['gpu_seconds'] and row['finite']
                assert row['dt']==cfg['dt'] and row['cohort']==d['cohort_roles'][case] and len(row['iterations'])==len(row['residuals'])==50
                maxima['full_error_metric_delta']=max(maxima['full_error_metric_delta'],max(float(np.max(abs(np.asarray(v)-row['error'][k]))) for k,v in metrics.items()))
            row=rows[0]
            if row['method']=='rom':
                p=weights[row['model']]['params'];op=ops[row['model']];states=saved['internal_latents'];assert states.shape==(51,16)
                for rr in rows:assert np.array_equal(states,np.array(rr['internal_latents'])) and np.array_equal(states[::10],np.array(rr['latent_states']))
                pred=(gsample@ai.head(p,states[::10]).T).T.reshape(6,len(ids),len(ids));sampled_actual=f[:,ids[:,None],ids[None,:]]
                maxima['sampled_decoder_relative_delta']=max(maxima['sampled_decoder_relative_delta'],float(np.linalg.norm(pred-sampled_actual)/np.linalg.norm(sampled_actual)))
                h,Jh=head_jac(p,states[0])
                complex_jac=np.column_stack([complex_head(p,states[0].astype(complex)+1e-25j*np.eye(16)[j]).imag/1e-25 for j in range(16)])
                maxima['complex_head_jacobian_relative_delta']=max(maxima['complex_head_jacobian_relative_delta'],float(np.linalg.norm(Jh-complex_jac)/np.linalg.norm(complex_jac)))
                target=op['cold_Q'].T@(ai.sample_field(u0,op['cold_xy'],L)*op['cold_w']);r=op['cold_R']@h-target;icgn=stationarity(r,op['cold_R']@Jh);prev=op['A']@h
                gn=[];rn=[]
                for step_index,z in enumerate(states[1:]):
                    h,Jh=head_jac(p,z);us=np.einsum('msr,r->ms',op['G5'],h);dus=np.einsum('msr,rk->msk',op['G5'],Jh)
                    c,xp,xm,yp,ym=us.T;dc,dxp,dxm,dyp,dym=dus.transpose(1,0,2)
                    diff=np.where(c>0,2*c-xm-ym,xp+yp-2*c);ddiff=np.where((c>0)[:,None],2*dc-dxm-dym,dxp+dyp-2*dc)
                    adv=L*c*diff;dadv=L*(dc*diff[:,None]+c[:,None]*ddiff);ah=op['A']@h;dah=op['A']@Jh;scale=1+cfg['dt']*nu*op['lam']
                    r=(ah-prev+cfg['dt']*(op['Pq'].T@adv+nu*op['lam']*ah))/scale
                    J=(dah+cfg['dt']*(op['Pq'].T@dadv+nu*op['lam'][:,None]*dah))/scale[:,None]
                    if step_index==0:
                        def complex_weak(zz):
                            hh=complex_head(p,zz);uu=np.einsum('msr,r->ms',op['G5'],hh);cc,xpp,xmm,ypp,ymm=uu.T
                            advv=L*cc*np.where(c>0,2*cc-xmm-ymm,xpp+ypp-2*cc);ahh=op['A']@hh
                            return (ahh-prev+cfg['dt']*(op['Pq'].T@advv+nu*op['lam']*ahh))/scale
                        complex_jac=np.column_stack([complex_weak(z.astype(complex)+1e-25j*np.eye(16)[j]).imag/1e-25 for j in range(16)])
                        maxima['complex_weak_jacobian_relative_delta']=max(maxima['complex_weak_jacobian_relative_delta'],float(np.linalg.norm(J-complex_jac)/np.linalg.norm(complex_jac)))
                    rn.append(float(np.linalg.norm(r)));gn.append(stationarity(r,J));prev=ah
                for rr in rows:
                    maxima['weak_residual_delta']=max(maxima['weak_residual_delta'],float(np.max(abs(np.array(rn)-rr['residuals']))))
                    maxima['normalized_gradient_delta']=max(maxima['normalized_gradient_delta'],float(np.max(abs(np.array(gn)-rr['normalized_stationarity']))),abs(icgn-rr['ic_normalized_stationarity']))
                    cpu_values=np.array([icgn]+gn);post_values=np.array([rr['ic_normalized_stationarity']]+rr['normalized_stationarity']);threshold=cfg['strict']['gtol']
                    charged_values=np.array([rr.get('charged_ic_normalized_stationarity',rr['ic_normalized_stationarity'])]+rr.get('charged_normalized_stationarity',rr['normalized_stationarity']))
                    cpu_mask=cpu_values<=threshold;post_mask=post_values<=threshold;charged_mask=charged_values<=threshold
                    disagreements=(cpu_mask!=post_mask)|(cpu_mask!=charged_mask)|(post_mask!=charged_mask)
                    conservative=cpu_mask&post_mask&charged_mask
                    key=(L,rr['name'],case,rr['rep']);convergence_audit[key]=dict(stationary=bool(np.all(conservative)),stationary_states=int(np.sum(conservative)),maximum_cpu_gradient=float(cpu_values.max()))
                    if np.any(disagreements):threshold_disagreements.append(dict(intervals=L,name=rr['name'],case=case,rep=rr['rep'],states=np.flatnonzero(disagreements).tolist(),cpu=cpu_values[disagreements].tolist(),posthoc=post_values[disagreements].tolist(),charged=charged_values[disagreements].tolist()))
                    if 'charged_normalized_stationarity' in rr:
                        maxima['charged_cpu_normalized_gradient_delta']=max(maxima['charged_cpu_normalized_gradient_delta'],float(np.max(abs(charged_values-cpu_values))))
                        maxima['charged_posthoc_normalized_gradient_delta']=max(maxima['charged_posthoc_normalized_gradient_delta'],float(np.max(abs(charged_values-post_values))))
                        assert abs(max(abs(charged_values[1:]-post_values[1:]))-rr['charged_posthoc_weak_gradient_max_delta'])<1e-15
                        assert abs(abs(charged_values[0]-post_values[0])-rr['charged_posthoc_initial_gradient_delta'])<1e-15
                    ib=180 if rr['solver']=='accepted' else cfg['strict']['ic_budget'];sb=30 if rr['solver']=='accepted' else cfg['strict']['step_budget']
                    allowed={0,1,2,3} if rr['solver']=='accepted' else {0,1,2,3,4};assert 0<=rr['ic_iterations']<=ib and rr['ic_reason'] in allowed
                    assert all(0<=v<=sb for v in rr['iterations']) and set(rr['stop_reasons'])<=allowed
                    assert all(v<=cfg['strict']['gtol']*(1+1e-7) for v,reason in zip(rr.get('charged_normalized_stationarity',rr['normalized_stationarity']),rr['stop_reasons']) if reason==4)
                    if rr['ic_reason']==4:assert rr.get('charged_ic_normalized_stationarity',rr['ic_normalized_stationarity'])<=cfg['strict']['gtol']*(1+1e-7)
                    state_count+=51
                if row['name']=='frozen_accepted' and case<cfg['cases']:
                    oldrow=next(v for v in old['invocations'] if v['name']=='nmrom' and v['intervals']==L and v['case']==case and v['rep']==0)
                    oldfield=np.load(old_record/'archive/out'/oldrow['artifact'])['fields'];delta=float(np.linalg.norm(f-oldfield)/np.linalg.norm(oldfield))
                    counters=all(np.array_equal(row[k],oldrow[k]) for k in ['iterations','stop_reasons','ic_iterations','ic_reason'])
                    latent_delta=float(np.max(abs(states-np.array(oldrow['internal_latents']))));parities.append(dict(intervals=L,case=case,relative_field_delta=delta,counters_equal=counters,max_latent_delta=latent_delta))
                    assert delta<1e-10 and counters
            else:
                setting=next(s for s in cfg['fom_settings'] if s['name']==row['name']);assert all(row[k]==v for k,v in setting.items())
                prev=saved['previous_output_steps'];assert prev.shape==(5,L+1,L+1);rnorm=[]
                for i in range(5):
                    adv,lap=ai.spatial(f[i+1],L);r=f[i+1,1:-1,1:-1]-prev[i,1:-1,1:-1]+cfg['dt']*(adv-nu*lap);rnorm.append(np.linalg.norm(r)/np.linalg.norm(prev[i,1:-1,1:-1]))
                for rr in rows:
                    maxima['fom_output_residual_delta']=max(maxima['fom_output_residual_delta'],float(np.max(abs(np.array(rnorm)-np.array(rr['residuals'])[9::10]))))
                    assert rr['nonlinear_converged']==bool(max(rr['residuals'])<=rr['ntol']*(1+1e-9))
                    assert rr['linear_converged']==bool(all(max(v[:int(n)],default=0)<=rr['ltol']*(1+1e-7) for v,n in zip(rr['linear_relative_residuals'],rr['iterations'])))
                    assert np.array_equal(f[0],truth[0]);fom_count+=5
        print('AUDITED_MESH',L,flush=True)
    assert maxima['charged_cpu_normalized_gradient_delta']<2e-9
    assert maxima['complex_head_jacobian_relative_delta']<1e-11 and maxima['complex_weak_jacobian_relative_delta']<1e-10
    assert maxima['full_error_metric_delta']<1e-11 and maxima['sampled_decoder_relative_delta']<2e-11 and maxima['weak_residual_delta']<2e-10 and maxima['normalized_gradient_delta']<2e-9 and maxima['fom_output_residual_delta']<2e-10
    summary=[]
    for L,name,cohort in itertools.product(meshes,names,['all','opened development','fresh development']):
        rows=[r for r in d['invocations'] if r['intervals']==L and r['name']==name and (cohort=='all' or r['cohort']==cohort)];cases=sorted(set(r['case'] for r in rows));first=rows[0]
        outliers=sum(sum(r['gpu_seconds']>2*np.median([v['gpu_seconds'] for v in rows if v['case']==c]) for r in rows if r['case']==c) for c in cases)
        reference_pass=all(d['reference_metrics'][str(L)][c]['margin']<=cfg['target_fixed_initial']*cfg['reference_margin_fraction'] and d['reference_metrics'][str(L)][c]['decrease'] for c in cases)
        error_pass=all(r['error']['fixed_initial_max']+d['reference_metrics'][str(L)][r['case']]['margin']<=cfg['target_fixed_initial'] for r in rows)
        numerical=all(convergence_audit[(L,name,r['case'],r['rep'])]['stationary'] for r in rows) if first['method']=='rom' else all(r['nonlinear_converged'] and r['linear_converged'] for r in rows)
        summary.append(dict(intervals=L,name=name,method=first['method'],cohort=cohort,cases=len(cases),invocations=len(rows),gpu_ms=1000*float(np.median([r['gpu_seconds'] for r in rows])),host_ms=1000*float(np.median([r['host_seconds'] for r in rows])),
            worst_fixed_initial=max(r['error']['fixed_initial_max'] for r in rows),worst_current_relative=max(r['error']['current_relative_max'] for r in rows),median_case_fixed_initial=float(np.median([max(r['error']['fixed_initial_max'] for r in rows if r['case']==c) for c in cases])),
            worst_initial_error=max(r['error']['fixed_initial_per_time'][0] for r in rows),worst_final_error=max(r['error']['fixed_initial_per_time'][-1] for r in rows),
            stationary_invocations=sum(convergence_audit[(L,name,r['case'],r['rep'])]['stationary'] for r in rows) if first['method']=='rom' else None,
            stationary_states=sum(convergence_audit[(L,name,r['case'],r['rep'])]['stationary_states'] for r in rows) if first['method']=='rom' else None,
            maximum_stationarity=max(max(r['normalized_stationarity']+[r['ic_normalized_stationarity']]) for r in rows) if first['method']=='rom' else None,
            numerical_pass=numerical,reference_pass=reference_pass,error_plus_margin_pass=error_pass,physical_pass=reference_pass and error_pass,physical_and_numerical_pass=reference_pass and error_pass and numerical,
            timing_outliers_gt2x_case_median=int(outliers),initial_reasons=dict(collections.Counter(r['ic_reason'] for r in rows)) if first['method']=='rom' else None,
            evolution_reasons=dict(collections.Counter(s for r in rows for s in r['stop_reasons'])) if first['method']=='rom' else None,
            median_total_iterations=float(np.median([sum(r['iterations'])+(r['ic_iterations'] if first['method']=='rom' else 0) for r in rows]))))
    comparisons=[]
    for L,cohort in itertools.product(meshes,['all','opened development','fresh development']):
        rows=[r for r in summary if r['intervals']==L and r['cohort']==cohort]
        for rom,fom in itertools.product([r for r in rows if r['method']=='rom'],[r for r in rows if r['method']=='fom']):
            comparisons.append(dict(intervals=L,cohort=cohort,rom=rom['name'],fom=fom['name'],gpu_ratio=fom['gpu_ms']/rom['gpu_ms'],host_ratio=fom['host_ms']/rom['host_ms'],both_physical_and_numerical_pass=rom['physical_and_numerical_pass'] and fom['physical_and_numerical_pass']))
    audit=dict(passed=True,result_sha256=ai.sha(out/'result.json'),audit_script_sha256=ai.sha(Path(__file__)),audit_dependency_sha256={str(Path(ai.__file__)):ai.sha(Path(ai.__file__))},normalized_gradient_agreement_bound=2e-9,threshold_classification_disagreements=threshold_disagreements,classification_policy='A state is stationary only when charged, posthoc and independent NumPy values all meet the unchanged threshold; disagreements fail convergence conservatively.',job_id=d['job_id'],source_commit=d['commit'],gpu=d['gpu'],invocations=len(actual),distinct_full_fields=len(grouped),audited_rom_states_with_repetitions=state_count,audited_fom_output_steps_with_repetitions=fom_count,reference_metric_max_delta=ref_delta,maxima=dict(maxima),frozen_control_parity=parities,
        scope='Independent full saved-field norms, cold/stencil decoder identities, every weak state residual and analytic Jacobian stationarity; FOM output-step residuals; source/provenance and training bytes. Inner linear correction residuals checked from records, not reconstructed unavailable Newton states.')
    dump(record/'AUDIT.json',audit);dump(record/'PANEL.json',dict(status='audited development',pde='burgers2d',job_id=d['job_id'],source_commit=d['commit'],config=cfg,gpu=d['gpu'],rows=summary,comparisons=comparisons,training=ta,operator_checks=operator_checks,reference_metrics=d['reference_metrics'],audit=audit))
    print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
