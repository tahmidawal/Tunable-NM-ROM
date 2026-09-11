"""Independent NumPy/SciPy full-field, rank, geometry and provenance audit."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
import scipy.linalg
from audit_dynamics import geometry, recompute, compare_metrics, initial_fields, parameter_rows, NAMES, energy2
from audit_iterative_extra import modal_fields, sample_bank


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def jacobian_direction(head,transform,z,w):
    """Independent analytic derivative of the head Jacobian in direction w."""
    x1=z@head['p/l1/w']+head['p/l1/b'];s1=1/(1+np.exp(-x1));b1=s1*(1-s1)
    d1=s1+x1*b1;dd1=b1*(2+x1*(1-2*s1));j1=d1[:,None]*head['p/l1/w'].T
    v1=w@head['p/l1/w'];dj1=(dd1*v1)[:,None]*head['p/l1/w'].T
    x2=(x1*s1)@head['p/l2/w']+head['p/l2/b'];s2=1/(1+np.exp(-x2));b2=s2*(1-s2)
    d2=s2+x2*b2;dd2=b2*(2+x2*(1-2*s2));j2=head['p/l2/w'].T@j1
    v2=(d1*v1)@head['p/l2/w'];dj2=(dd2*v2)[:,None]*j2+d2[:,None]*(head['p/l2/w'].T@dj1)
    return transform@(float(head['frozen/output_scale'])*head['p/out/w'].T@dj2)


def projection_check(field,row,mesh,head,u0,v0,c):
    g,m,k=mesh['g'],mesh['mass'],mesh['stiffness'];method=row['method']
    a0=g.T@(m*u0.ravel());b0=g.T@(m*v0.ravel())
    np.testing.assert_allclose(a0,field['projection_a0'],atol=1e-12)
    np.testing.assert_allclose(b0,field['projection_b0'],atol=1e-12)
    eigen,vectors=np.linalg.eigh(k);omega=c*np.sqrt(eigen)
    times=np.arange(len(field['u']))*.05;co,si=np.cos(times[:,None]*omega),np.sin(times[:,None]*omega)
    targets=(co*(vectors.T@a0)+si*(vectors.T@b0)/omega)@vectors.T
    velocities=(-si*omega*(vectors.T@a0)+co*(vectors.T@b0))@vectors.T
    if method=='linear_bank64':
        aa,bb=targets,velocities;implicit=[];fitdelta=[];hessiandelta=[]
    else:
        np.testing.assert_allclose(targets,field['projection_targets'],atol=1e-10)
        np.testing.assert_allclose(velocities,field['projection_velocity_targets'],atol=1e-9)
        weight=np.eye(64) if method=='projected_l2' else k/(np.trace(k)/64)
        root=np.linalg.cholesky(weight).T
        scale=float(field['projection_scale']);aa=[];bb=[];implicit=[];fitdelta=[];hessiandelta=[]
        for ii,(z,w,target,velocity) in enumerate(zip(field['projection_z'],field['projection_w'],targets,velocities)):
            a,b,jac,_=geometry(head,mesh['transform'],z,w);aa.append(a);bb.append(b)
            residue=a-target;grad=jac.T@weight@residue
            projected=(root@residue)/scale;jscaled=(root@jac)/scale
            gradient=np.max(abs(jscaled.T@projected))/max(1.,np.linalg.norm(jscaled)*np.linalg.norm(projected))
            stat=np.linalg.norm(np.linalg.qr(jscaled,mode='reduced')[0].T@projected)/max(np.linalg.norm(projected),1e-10)
            best=int(field['projection_fit_selected'][ii])
            fitdelta.append(max(abs(gradient-field['projection_fit_gradient'][ii,best]),abs(stat-field['projection_fit_stationarity'][ii,best]),
                                abs(projected@projected-field['projection_fit_objective'][ii,best])))
            derivative=jacobian_direction(head,mesh['transform'],z,w)
            rhs=jac.T@weight@velocity;lhs=jac.T@weight@b+derivative.T@weight@residue
            implicit.append(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(rhs),1e-12))
            if ii in (0,len(times)//2,len(times)-1):
                hessian=np.column_stack([jac.T@weight@jac[:,i]+jacobian_direction(head,mesh['transform'],z,np.eye(32)[i]).T@weight@residue for i in range(32)])
                sv=np.linalg.eigvalsh((hessian+hessian.T)/2)
                hessiandelta.append(max(abs(sv[0]-field['projection_fit_hessian_min_eigenvalue'][ii]),abs(sv[-1]-field['projection_fit_hessian_max_eigenvalue'][ii])))
                assert np.linalg.norm(hessian-hessian.T)<1e-9
        aa,bb=np.asarray(aa),np.asarray(bb)
        assert max(implicit)<1e-7 and max(fitdelta)<1e-8 and max(hessiandelta)<1e-9
    coefficient_delta=max(float(np.max(abs(aa-field['projection_coefficients']))),float(np.max(abs(bb-field['projection_velocity_coefficients']))))
    assert coefficient_delta<1e-8
    decode_delta=0.
    for offset in range(0,len(g),8192):
        part=g[offset:offset+8192]
        decode_delta=max(decode_delta,float(np.max(abs(aa@part.T-field['u'].reshape(len(aa),-1)[:,offset:offset+len(part)]))),
                         float(np.max(abs(bb@part.T-field['v'].reshape(len(aa),-1)[:,offset:offset+len(part)]))))
    assert decode_delta<1e-8
    return dict(coefficient_max_absolute=coefficient_delta,decode_max_absolute=decode_delta,
                maximum_implicit_velocity_equation_relative=float(max(implicit,default=0)),
                maximum_fit_diagnostic_error=float(max(fitdelta,default=0)),maximum_sampled_hessian_eigenvalue_error=float(max(hessiandelta,default=0)))


def audit(record):
    cluster=record/'cluster';native=cluster/'out/pilot'
    result=json.loads((native/'result.json').read_text());cfg=result['config'];meta=result['provenance']
    assert result['complete'] and not result['final_test_opened']
    assert meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    submission=json.loads((record/'submission.json').read_text());root=Path(__file__).resolve().parents[2]
    for path,expected in submission['source_hashes'].items():
        assert sha(cluster/path)==expected
        source='experiments/'+str(Path(path).relative_to('code'))
        payload=subprocess.check_output(['git','show',submission['source_commit']+':'+source],cwd=root)
        assert hashlib.sha256(payload).hexdigest()==expected
    for path,expected in result['input_sha256'].items():assert sha(cluster/'in'/path)==expected
    for path,expected in result['output_sha256'].items():assert sha(native/path)==expected
    cleanup=json.loads((record/'cleanup.json').read_text())
    assert cleanup['remote_deleted_and_absence_checked'] and cleanup['all_three_manifests_verified']
    logs='\n'.join(p.read_text() for p in (cluster/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs
    for bad in ('captured constant','Captured constant','out of memory','RESOURCE_EXHAUSTED','Traceback','No space left','cuInit'):
        assert bad not in logs,bad
    assert (cluster/'EXIT_CODE').read_text().strip()=='0'
    origin=json.loads((cluster/'in/ORIGIN.json').read_text())
    lineages=[dict(commit=origin['checkpoint_commit'],sources=origin['sources']),origin['head32_origin']]
    if 'new_head_origin' in origin:lineages.append(origin['new_head_origin'])
    for parent in origin.get('new_head_origin',{}).get('parent_results',{}).values():
        payload=subprocess.check_output(['git','show',origin['new_head_origin']['commit']+':'+parent['result_path']],cwd=root)
        assert hashlib.sha256(payload).hexdigest()==parent['result_sha256']
    for lineage in lineages:
        for path,expected in lineage['sources'].items():
            if '/training_ladder_' in path:
                # Large original arrays are content-addressed by the committed,
                # audited parent result and still checked against local bytes.
                payload=subprocess.check_output(['git','show',lineage['commit']+':'+lineage['result_path']],cwd=root)
                assert hashlib.sha256(payload).hexdigest()==lineage['result_sha256']
                assert json.loads(payload)['output_sha256'][Path(path).name]==expected
                assert sha(root/path)==expected
            else:
                payload=subprocess.check_output(['git','show',lineage['commit']+':'+path],cwd=root)
                assert hashlib.sha256(payload).hexdigest()==expected
    with np.load(cluster/'in/dirichlet/head32.npz') as f:head={k:f[k] for k in f.files}
    groups=defaultdict(list)
    for row in result['invocations']:groups[row['intervals'],row['case'],row['method'],row['setting']].append(row)
    countcases=sum(len(c['indices']) for c in cfg['cohorts'])
    assert len(result['invocations'])==len(cfg['meshes'])*countcases*(len(cfg['arms'])+len(cfg['cg_tolerances'])+len(cfg.get('cg_timestep_arms',[]))+1)*cfg['repetitions']
    nested_audit=None
    if result.get('nested_architecture'):
        with np.load(native/'head_trained_nested40.npz') as f:enriched={k:f[k] for k in f.files}
        with np.load(cluster/'in/dirichlet'/cfg['frozen_head_inputs']['trained_phase']) as f:old={k:f[k] for k in f.files}
        with np.load(native/'nested_basis.npz') as f:basis=f['basis']
        for key in ('p/l1/b','p/l2/w','p/l2/b','p/out/w','p/out/b','p/bias','frozen/output_scale'):
            np.testing.assert_array_equal(enriched[key],old[key])
        np.testing.assert_array_equal(enriched['p/l1/w'][:32],old['p/l1/w'])
        np.testing.assert_array_equal(enriched['p/l1/w'][32:],0.)
        np.testing.assert_array_equal(enriched['p/linear'][:,:32],old['p/linear'])
        np.testing.assert_array_equal(enriched['p/linear'][:,32:],basis)
        np.testing.assert_array_equal(enriched['codes'][:,:32],old['codes'])
        errors=[];ranks=[];rng=np.random.default_rng(691217)
        for idx in np.linspace(0,len(old['codes'])-1,6,dtype=int):
            z,w=old['codes'][idx],rng.normal(size=32)
            a,b,j,c=geometry(old,np.eye(64),z,w)
            aa,bb,jj,cc=geometry(enriched,np.eye(64),np.r_[z,np.zeros(8)],np.r_[w,np.zeros(8)])
            errors.append(max(np.max(abs(a-aa)),np.max(abs(b-bb)),np.max(abs(j-jj[:,:32])),np.max(abs(c-cc))))
            singular=np.linalg.svd(jj,compute_uv=False);ranks.append(singular[-1]/singular[0])
        assert max(errors)<1e-10 and min(ranks)>1e-8
        nested_audit=dict(original_parameters_exact=True,independent_inclusion_max_absolute=float(max(errors)),independent_minimum_sampled_rank_ratio=float(min(ranks)))
    training_audit=None
    if result.get('head_training'):
        assert result['training_manifest']['data_hashes_match'] and result['training_manifest']['saved_initialization_matched']
        with np.load(native/'training_ladder_dirichlet.npz') as f,np.load(native/'fixed_encoder_training.npz') as encoder:
            a,b=f['a'].reshape(-1,64),f['b'].reshape(-1,64)
            independent_z=(a-encoder['center'])@np.linalg.pinv(encoder['linear']).T
            independent_w=b@np.linalg.pinv(encoder['linear']).T
            code_delta=max(float(np.max(abs(independent_z-encoder['z']))),float(np.max(abs(independent_w-encoder['w']))))
            assert code_delta<1e-8
            path=next(p for p in origin['sources'] if 'reflective01' in p and p.endswith('bank_tables.npz'))
            payload=subprocess.check_output(['git','show',origin['checkpoint_commit']+':'+path],cwd=root)
            with np.load(io.BytesIO(payload)) as table:
                stiffness_delta=float(np.linalg.norm(encoder['stiffness']-table['stiffness_unit'])/np.linalg.norm(table['stiffness_unit']))
            assert stiffness_delta<1e-10
            for rec in result['head_training']:
                assert rec['steps']==cfg['head_training']['steps'] and rec['seed']==cfg['head_training']['seed']
                assert sha(native/rec['checkpoint_path'])==rec['checkpoint_sha256']
                with np.load(native/rec['checkpoint_path']) as hh:np.testing.assert_allclose(hh['codes'],encoder['z'],atol=0,rtol=0)
            training_audit=dict(fixed_encoder_code_and_velocity_max_absolute=code_delta,
                original_stiffness_relative_error=stiffness_delta,endpoint_count=len(result['head_training']),original_truth_hashes_matched=True)
    checked=[];references=[];banks=[];parities=[];refinements=[];kinematics=[]
    for n in cfg['meshes']:
        with np.load(native/f'mesh_dirichlet_{n}.npz') as f:mesh={k:f[k] for k in f.files}
        banks.append(dict(intervals=n,sampled_bank_error=sample_bank(cluster/'in/dirichlet',mesh,n,'dirichlet')))
        for cohort in cfg['cohorts']:
            pars=parameter_rows(cohort['seed'],max(cohort['indices'])+1)
            for ci in cohort['indices']:
                case=f"{cohort['name']}_{ci}";par=pars[ci];c=par[5]
                with np.load(native/f'reference_{n}_{case}.npz') as f:ut,vt,u0,v0=f['u'],f['v'],f['u0'],f['v0']
                iu,iv=initial_fields(par,n,'dirichlet')
                np.testing.assert_allclose(iu,u0,atol=2e-14,rtol=2e-14)
                np.testing.assert_allclose(iv,v0,atol=2e-13,rtol=2e-13)
                times=np.arange(round(cfg['end_time']/cfg['observation_dt'])+1)*cfg['observation_dt']
                error=0.
                for ii,(u,v) in enumerate(modal_fields(u0,v0,c,n,times)):
                    error=max(error,float(np.max(abs(u-ut[ii]))),float(np.max(abs(v-vt[ii]))))
                assert error<1e-10
                references.append(dict(intervals=n,case=case,independent_modal_max_absolute=error))
                for key,rows in groups.items():
                    if key[:2]!=(n,case):continue
                    row=next(r for r in rows if r['repetition']==0)
                    assert sorted(r['repetition'] for r in rows)==list(range(cfg['repetitions']))
                    with np.load(native/row['field_artifact']) as f:field={k:f[k] for k in f.files}
                    u,v=field['u'],field['v'];actual=recompute(u,v,ut,vt,n,'dirichlet',c)
                    hashes={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in [('u',u),('v',v)]}
                    delta=max(compare_metrics(actual,r['same_grid_discrepancy']) for r in rows)
                    assert delta<1e-10
                    for r in rows:assert r['output_sha256']==hashes
                    item=dict(intervals=n,case=case,method=row['method'],setting=row['setting'],full_field_metric_disagreement=delta)
                    row_head=head
                    if row['method'].startswith('trained_'):
                        head_path=(cluster/'in/dirichlet'/cfg['frozen_head_inputs'][row['method']]) if row['method'] in cfg.get('frozen_head_inputs',{}) else native/f"head_{row['method']}.npz"
                        with np.load(head_path) as f:row_head={k:f[k] for k in f.files}
                    if 'projection_coefficients' in field:
                        item['projection_audit']=projection_check(field,row,mesh,head,u0,v0,c)
                    if 'coefficients' in field:
                        aa=[];bb=[];rank=[];acceleration_delta=[];backwards=[];bound_defects=[];normal_force=[]
                        for z,w in zip(field['rollout_z'],field['rollout_w']):
                            a,b,jac,curve=geometry(row_head,mesh['transform'],z,w)
                            aa.append(a);bb.append(b)
                            singular=np.linalg.svd(jac,compute_uv=False);rank.append(singular[-1]/singular[0])
                            force=curve+c*c*mesh['stiffness']@a+c*mesh['damping']@b
                            q,rr=np.linalg.qr(jac,mode='reduced');exact=scipy.linalg.solve_triangular(rr,-q.T@force)
                            normal_force.append(float(np.linalg.norm(jac@exact+force)/max(np.linalg.norm(force),1e-12)))
                            lower=np.linalg.cholesky(jac.T@jac)
                            bound=1/(np.linalg.norm(jac)*np.linalg.norm(scipy.linalg.solve_triangular(lower,np.eye(jac.shape[1]),lower=True)))
                            bound_defects.append(max(0.,bound-rank[-1]))
                            normal=scipy.linalg.cho_solve((lower,True),-jac.T@force)
                            acceleration_delta.append(np.linalg.norm(normal-exact)/max(np.linalg.norm(exact),1e-10))
                            gram=jac.T@jac;rhs=-jac.T@force
                            backwards.append(np.linalg.norm(gram@normal-rhs)/(np.linalg.norm(gram)*np.linalg.norm(normal)+np.linalg.norm(rhs)))
                        aa,bb=np.asarray(aa),np.asarray(bb)
                        coefficient_delta=max(float(np.max(abs(aa-field['coefficients']))),float(np.max(abs(bb-field['velocity_coefficients']))))
                        assert coefficient_delta<1e-9 and min(rank)>1e-8 and max(bound_defects)<1e-12
                        decode_error=0.
                        for offset in range(0,mesh['g'].shape[0],8192):
                            g=mesh['g'][offset:offset+8192]
                            decode_error=max(decode_error,float(np.max(abs(aa@g.T-u.reshape(len(u),-1)[:,offset:offset+len(g)]))),
                                             float(np.max(abs(bb@g.T-v.reshape(len(v),-1)[:,offset:offset+len(g)]))))
                        assert decode_error<1e-8
                        a,b,jac,_=geometry(row_head,mesh['transform'],field['rollout_z'][0],field['rollout_w'][0])
                        target=mesh['g'].T@(mesh['mass']*u0.ravel());scale=np.sqrt(np.sum(mesh['mass']*u0.ravel()**2))
                        residual=(a-target)/scale;jac=jac/scale
                        gradient=np.max(abs(jac.T@residual))/max(1.,np.linalg.norm(jac)*np.linalg.norm(residual))
                        stat=np.linalg.norm(np.linalg.qr(jac,mode='reduced')[0].T@residual)/max(np.linalg.norm(residual),1e-10)
                        fits=row['cold_fit'];best=fits['selected']
                        fitdelta=max(abs(gradient-fits['gradient'][best]),abs(stat-fits['stationarity'][best]),abs(residual@residual-fits['objective'][best]))
                        assert fitdelta<1e-8
                        item.update(head_coefficient_max_absolute=coefficient_delta,full_decode_max_absolute=decode_error,
                            min_observed_exact_rank_ratio=float(min(rank)),max_cholesky_qr_acceleration_relative=float(max(acceleration_delta)),
                            max_observed_normal_backward_error=float(max(backwards)),initial_fit_diagnostic_error=float(fitdelta),
                            observed_weak_normal_force_relative=normal_force,max_observed_weak_normal_force_relative=max(normal_force),
                            weak_equations=jac.shape[0],latent_dimension=jac.shape[1],weak_to_latent_ratio=jac.shape[0]/jac.shape[1],
                            max_relative_energy_balance_defect=float(np.max(abs(energy2(u,v,n,'dirichlet',c)-energy2(u[0],v[0],n,'dirichlet',c)))/energy2(u[0],v[0],n,'dirichlet',c)))
                        if row['method']!='baseline':
                            assert row['maximum_normal_backward_error']<1e-10
                            assert float(field['rollout_normal_backward_error'][-1])==row['maximum_normal_backward_error']
                    checked.append(item)
                for item in result['parity']:
                    if (item['intervals'],item['case'])!=(n,case):continue
                    baseline=next(r for r in result['invocations'] if (r['intervals'],r['case'],r['method'],r['setting'])==(n,case,'baseline',cfg['primary_dt']))
                    selected=next(r for r in result['invocations'] if (r['intervals'],r['case'],r['method'],r['setting'])==(n,case,item['method'],cfg['primary_dt']))
                    with np.load(native/baseline['field_artifact']) as f,np.load(native/selected['field_artifact']) as g:
                        dd=recompute(g['u'],g['v'],f['u'],f['v'],n,'dirichlet',c)
                    maxima={name:float(np.max(dd[name]['absolute'])/baseline['same_grid_discrepancy'][name]['initial_scale']) for name in NAMES}
                    assert max(abs(maxima[name]-item['maxima'][name]) for name in NAMES)<1e-10
                    assert item['passed']==(max(maxima.values())<cfg['parity_target'])
                    parities.append(dict(intervals=n,case=case,method=item['method'],maxima=maxima,passed=item['passed']))
                for row in result['accuracy_controls']:
                    if (row['intervals'],row['case'])!=(n,case):continue
                    with np.load(native/row['field_artifact']) as f:fu,fv=f['u'],f['v']
                    assert compare_metrics(recompute(fu,fv,ut,vt,n,'dirichlet',c),row['same_grid_discrepancy'])<1e-10
                    baseline=next(r for r in result['invocations'] if (r['intervals'],r['case'],r['method'],r['setting'])==(n,case,row['method'],row['setting']*2))
                    with np.load(native/baseline['field_artifact']) as f:dd=recompute(f['u'],f['v'],fu,fv,n,'dirichlet',c)
                    maxima={name:float(np.max(dd[name]['absolute'])/row['same_grid_discrepancy'][name]['initial_scale']) for name in NAMES}
                    saved=next(r for r in result['time_refinement'] if (r['intervals'],r['case'],r['method'],r['dt'])==(n,case,row['method'],row['setting']*2))
                    assert max(abs(maxima[name]-saved['maxima'][name]) for name in NAMES)<1e-10
                    refinements.append(dict(intervals=n,case=case,method=row['method'],dt=row['setting']*2,maxima=maxima,passed=saved['passed']))
                for row in result.get('projection_kinematics',[]):
                    if (row['intervals'],row['case'])!=(n,case):continue
                    timed=next(r for r in result['invocations'] if (r['intervals'],r['case'],r['method'])==(n,case,row['method']))
                    with np.load(native/timed['field_artifact']) as f,np.load(native/row['artifact']) as near:
                        a0,b0=f['projection_a0'],f['projection_b0'];scale=np.sqrt(b0@b0+c*c*a0@mesh['stiffness']@a0)
                        for check in row['checks']:
                            delta=check['delta'];derivative=(near[f'plus_{delta}']-near[f'minus_{delta}'])/(2*delta)
                            norm=np.linalg.norm(derivative-f['projection_velocity_coefficients'],axis=1)
                            actual=float(np.max(norm)/scale)
                            assert abs(actual-check['max_initial_velocity_scaled_difference'])<1e-10
                            kinematics.append(dict(intervals=n,case=case,method=row['method'],delta=delta,max_initial_scaled_difference=actual,
                                neighboring_fit_scope='Saved neighboring coefficients and reported stationarity; full timed projection stationarity/Hessian audited independently.'))
    output=dict(passed=True,source_commit=submission['source_commit'],job_id=meta['job_id'],
        timed_invocations=len(result['invocations']),distinct_timed_fields=len(groups),accuracy_control_fields=len(result['accuracy_controls']),
        banks=banks,references=references,fields=checked,parities=parities,refinements=refinements,kinematics=kinematics,training=training_audit,nested=nested_audit)
    (record/'audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);audit(ap.parse_args().record)
