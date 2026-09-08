"""Independent NumPy audit of larger-head training lineage and complete queries.

Fine accuracy controls are audited separately and cannot enter speed comparison.
Preserved references are reused; fine-grid timed FOM fields at nonreference CFLs
remain outside the full-grid reconstruction scope.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
import scipy.linalg
from audit_dynamics import digest,write,restriction,recompute,compare_metrics,geometry,audit_bank,parameter_rows,initial_fields


def audit(record):
    cluster=record/'cluster';native=cluster/'out/pilot';out=record/'analysis';out.mkdir(exist_ok=True)
    data=json.loads((native/'result.json').read_text());cfg=data['config'];meta=data['provenance']
    assert data['complete'] and not data['final_test_opened'] and meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    cleanup=json.loads((record/'cleanup.json').read_text());assert cleanup['remote_deleted_and_absence_checked'] and cleanup['all_three_manifests_verified']
    assert (cluster/'EXIT_CODE').read_text().strip()=='0'
    logs='\n'.join(p.read_text() for p in (cluster/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs
    for bad in ('captured constant','Captured constant','out of memory','RESOURCE_EXHAUSTED','Traceback','No space left','cuInit'):assert bad not in logs,bad
    tree=Path(__file__).resolve().parents[2]
    submission=json.loads((record/'submission.json').read_text())
    for path,sha in submission['source_hashes'].items():
        assert digest(cluster/path)==sha
        payload=subprocess.check_output(['git','show',submission['source_commit']+':experiments/'+str(Path(path).relative_to('code'))],cwd=tree)
        assert hashlib.sha256(payload).hexdigest()==sha
    origin=json.loads((cluster/'in/ORIGIN.json').read_text());blobs={}
    for path,sha in origin['sources'].items():
        payload=subprocess.check_output(['git','show',origin['checkpoint_commit']+':'+path],cwd=tree)
        assert hashlib.sha256(payload).hexdigest()==sha;blobs[path]=payload
    for name,sha in data['input_sha256'].items():assert digest(cluster/'in'/name)==sha
    for name,sha in data['output_sha256'].items():assert digest(native/name)==sha
    assert len(data['invocations'])==cfg['expected_timed_invocations']
    assert len(data['accuracy_controls'])==cfg['expected_accuracy_only_queries']
    assert len(data['warmups'])==cfg['expected_warmup_queries']
    for row in data['invocations']:
        assert row['measurement_role']=='timed_comparison' and row['comparison_eligible'] and row['warmup_performed']
        assert row['setting']!=cfg['accuracy_only_dt']
    for row in data['accuracy_controls']:
        assert row['measurement_role']=='accuracy_refinement_only' and not row['comparison_eligible']
        assert row['repetition']==0 and row['setting']==cfg['accuracy_only_dt'] and row['method'].startswith('new_mlp32_')
    groups=defaultdict(list)
    for row in data['invocations']+data['accuracy_controls']:groups[row['boundary'],row['intervals'],row['case'],row['method'],row['setting'],row['measurement_role']].append(row)
    pars=parameter_rows(cfg['validation_seed'],max(cfg['validation_indices'])+1)
    report=dict(scope=__doc__,result_sha256=digest(native/'result.json'),training=[],bank_checks=[],query_checks=[],fit_checks=[],geometry_checks=[],linear_checks=[],final_test_opened=False)
    for bc,label in (('dirichlet','reflective01'),('absorbing','absorbing02')):
        prefix=f'experiments/fresh-wave-head/runs/{label}/out/campaign/{bc}/'
        for name,source in (('bank_parameters.npz','bank_parameters.npz'),('head.npz','mlp_691200/head.npz'),('data_manifest.json','data_manifest.json')):
            assert (cluster/'in'/bc/name).read_bytes()==blobs[prefix+source]
        with np.load(io.BytesIO(blobs[prefix+'bank_tables.npz'])) as table,np.load(io.BytesIO(blobs[prefix+'common_initialization.npz'])) as common,np.load(cluster/'in'/bc/'coordinates.npz') as provided:
            for supplied,expected in ((provided['qr_r'],table['qr_r']),(provided['common_linear'],common['linear']),(provided['common_center'],common['center'])):np.testing.assert_array_equal(supplied,expected)
        with np.load(native/f'training_ladder_{bc}.npz') as f:train={k:f[k] for k in f.files}
        np.testing.assert_array_equal(train['parameters'],parameter_rows(cfg['training']['train_seed'],cfg['training']['train_count']))
        a=train['a'].reshape(-1,64);center=a.mean(axis=0)
        np.testing.assert_allclose(center,train['center'],atol=1e-14)
        np.testing.assert_allclose((a-center).T@(a-center)/len(a),train['covariance'],atol=1e-14)
        np.testing.assert_allclose(train['basis'].T@train['basis'],np.eye(64),atol=2e-14)
        np.testing.assert_allclose((a-center).T@(a-center)@train['basis'],train['basis']*train['singular_values']**2,atol=1e-10)
        manifest=next(t for t in data['training'] if t['boundary']==bc)
        assert manifest['data_hashes_match'] and manifest['saved_initialization_matched']
        assert hashlib.sha256(train['a'].tobytes()).hexdigest()==manifest['training_coefficients_sha256']
        expected=json.loads((cluster/'in'/bc/'data_manifest.json').read_text())['splits']['train']
        np.testing.assert_allclose(train['case_scales'],expected['scales'],rtol=1e-12,atol=1e-14)
        codes=(a-center)@train['basis'][:,:32]/train['scales'][:32]
        linear=train['standardized_linear'][:,:32];scale=np.repeat(train['case_scales'][:,0],49)
        with np.load(cluster/'in'/bc/'head.npz') as f:heads={'frozen_mlp16_seed691200':{k:f[k] for k in f.files}}
        endpoints=[t for t in data['head_training'] if t['boundary']==bc]
        assert sorted(t['optimizer_seed'] for t in endpoints)==cfg['training']['optimizer_seeds']
        for endpoint in endpoints:
            assert endpoint['training']==cfg['training'] and endpoint['velocity_weight']==0 and endpoint['configuration_dimension']==32
            assert digest(native/endpoint['checkpoint_path'])==endpoint['checkpoint_sha256']
            with np.load(native/endpoint['checkpoint_path']) as f:head={k:f[k] for k in f.files}
            assert head['p/linear'].shape==(64,32) and head['codes'].shape==codes.shape
            assert all(np.all(np.isfinite(x)) for x in head.values())
            assert float(head['frozen/output_scale'])==endpoint['output_scale']
            expected_scale=np.sqrt(np.mean(np.sum(a*a,axis=1))/64)
            assert abs(expected_scale-endpoint['output_scale'])<1e-15
            for arr,key in ((linear,'initializer_linear_sha256'),(train['center'],'initializer_center_sha256')):
                assert hashlib.sha256(arr.tobytes()).hexdigest()==endpoint[key]
            idx=np.random.default_rng(endpoint['optimizer_seed']).integers(0,len(a),cfg['training']['head_batch'])
            first_reconstruction=float(np.mean(np.sum((center+codes[idx]@linear.T-a[idx])**2,axis=1)/scale[idx]**2))
            first_objective=first_reconstruction+cfg['training']['code_penalty']*float(np.mean(codes[idx]**2))
            history=endpoint['history'];assert history[0][0]==0 and history[-1][0]==cfg['training']['head_steps']-1
            assert max(abs(history[0][1]-first_objective),abs(history[0][2]-first_reconstruction))<1e-12 and all(row[3]==0 for row in history)
            heads[endpoint['name']]=head
            report['training'].append(dict(boundary=bc,endpoint=endpoint['name'],first_objective=first_objective,recorded_first_objective=history[0][1],training_shape=list(head['codes'].shape),reconstructed_initial_codes_hash_matches=hashlib.sha256(codes.tobytes()).hexdigest()==endpoint['initializer_codes_sha256'],initial_codes_note='Initial PCA score formula independently reconstructed; exact rederived hash can differ across CPU BLAS arithmetic. First training objective and retained PCA factors are independently checked.',seed_and_frozen_checkpoint_verified=True))
        for n in cfg['meshes']:
            with np.load(native/f'mesh_{bc}_{n}.npz') as f:mesh={k:f[k] for k in f.files}
            report['bank_checks'].append(dict(boundary=bc,intervals=n,maximum_coordinate_bank_difference=audit_bank(cluster/'in'/bc,mesh,n,bc)))
            g=mesh['g'];np.testing.assert_allclose(g.T@(mesh['mass'][:,None]*g),np.eye(64),atol=2e-10)
            for ci in cfg['validation_indices']:
                c=pars[ci,5]
                with np.load(native/f'samegrid_{bc}_{n}_{ci}.npz') as f:su,sv=f['u'],f['v'];u0,v0=f['u0'],f['v0']
                expected_initial=initial_fields(pars[ci],n,bc)
                np.testing.assert_allclose(u0,expected_initial[0],atol=1e-14);np.testing.assert_allclose(v0,expected_initial[1],atol=1e-13)
                with np.load(native/f'reference_{bc}_{ci}.npz') as f:ut,vt=f['u'],f['v']
                for key,rows in groups.items():
                    if key[:3]!=(bc,n,ci):continue
                    name,setting,role=key[3:];expected_reps=cfg['repetitions'] if role=='timed_comparison' else 1
                    assert sorted(r['repetition'] for r in rows)==list(range(expected_reps))
                    assert len({json.dumps(r['output_sha256'],sort_keys=True) for r in rows})==1
                    first=next(r for r in rows if r['repetition']==0);np.testing.assert_array_equal(first['parameters'],pars[ci])
                    with np.load(native/(first['invocation_id']+'.npz')) as f:field={k:f[k] for k in f.files}
                    physical=recompute(field['u'],field['v'],ut,vt,cfg['comparison_intervals'],bc,c)
                    difference=max(compare_metrics(physical,r['physical_reference_error']) for r in rows)
                    native_difference=None;field_difference=None
                    if 'coefficients' in field:
                        uf=(field['coefficients']@g.T).reshape(su.shape);vf=(field['velocity_coefficients']@g.T).reshape(sv.shape)
                        field_difference=max(float(np.max(abs(restriction(x,n,cfg['comparison_intervals'],bc)-field[k]))) for x,k in ((uf,'u'),(vf,'v')))
                        native_difference=max(compare_metrics(recompute(uf,vf,su,sv,n,bc,c),r['same_grid_discrepancy']) for r in rows)
                    if name in heads:
                        values=[geometry(heads[name],mesh['transform'],z,w)[:2] for z,w in zip(field['rollout_z'],field['rollout_w'])]
                        np.testing.assert_allclose(np.stack([x[0] for x in values]),field['coefficients'],atol=2e-11,rtol=2e-11)
                        np.testing.assert_allclose(np.stack([x[1] for x in values]),field['velocity_coefficients'],atol=2e-10,rtol=2e-11)
                        assert first['completed']==bool(np.all(field['rollout_completed']) and np.all(np.isfinite(field['u'])) and np.all(np.isfinite(field['v'])))
                        assert first['minimum_dynamic_rank_ratio']==float(np.min(field['rollout_rank_ratio']))
                        target=g.T@(mesh['mass']*u0.ravel());scale0=np.sqrt(np.sum(mesh['mass']*u0.ravel()**2))
                        aa,jj=geometry(heads[name],mesh['transform'],field['rollout_z'][0]);residual=(aa-target)/scale0;jj/=scale0
                        q=np.linalg.qr(jj)[0];singular=np.linalg.svd(jj,compute_uv=False)
                        cold_values=dict(objective=float(residual@residual),gradient=float(np.max(abs(jj.T@residual))/max(1.,np.linalg.norm(jj)*np.linalg.norm(residual))),
                            stationarity=float(np.linalg.norm(q.T@residual)/max(np.linalg.norm(residual),1e-10)),rank_ratio=float(singular[-1]/singular[0]))
                        for row in rows:
                            cold=row['cold_fit'];si=cold['selected']
                            assert si==int(np.argmin(np.where(cold['finite'],cold['objective'],np.inf)))
                            assert max(abs(cold_values[k]-cold[k][si]) for k in cold_values)<2e-8
                            stationary=bool(cold['finite'][si] and cold_values['rank_ratio']>1e-8 and cold_values['gradient']<=1e-7 and (cold_values['stationarity']<=1e-6 or cold_values['objective']<=1e-20))
                            assert row['fit_stationary']==stationary
                    elif name=='affine32':
                        basis,offset=mesh['affine32_basis'],mesh['affine32_center'];dim=32
                        generator=np.zeros((65,65));generator[:32,32:64]=np.eye(32)
                        generator[32:64]=np.linalg.solve(basis.T@basis,np.column_stack((-c*c*basis.T@mesh['stiffness']@basis,-c*basis.T@mesh['damping']@basis,-c*c*basis.T@mesh['stiffness']@offset)))
                        actual=np.stack([scipy.linalg.expm(generator*i*cfg['observation_dt'])@field['states'][0] for i in range(49)])
                        gd=float(np.max(abs(generator-field['generator'])));sd=float(np.max(abs(actual-field['states'])))
                        assert gd<1e-7 and sd<1e-7
                        report['linear_checks'].append(dict(invocation_id=first['invocation_id'],generator_difference=gd,state_difference=sd))
                    for row in rows:
                        assert abs(sum(v for k,v in row['seconds'].items() if k!='complete_query')-row['seconds']['complete_query'])<1e-12
                        assert row['output_bytes']==2*8*49*su.shape[-1]**2
                    assert difference<2e-8 and (native_difference is None or native_difference<2e-8) and (field_difference is None or field_difference<2e-8)
                    report['query_checks'].append(dict(invocation_id=first['invocation_id'],role=role,repetitions=len(rows),completed=first['completed'],physical_metric_difference=difference,native_grid_ROM_metric_difference=native_difference,common_field_difference=field_difference))
                for model,head in heads.items():
                    diag=next(x for x in data['diagnostics'] if (x['boundary'],x['intervals'],x['case'],x['model'])==(bc,n,ci,model))
                    with np.load(native/f'diagnostic_{bc}_{n}_{ci}_{model}.npz') as f:fit={k:f[k] for k in f.files}
                    for budget in cfg['diagnostic_fit_budgets']:
                        selected=fit[f'fit_{budget}_selected'];np.testing.assert_array_equal(selected,np.argmin(np.where(fit[f'fit_{budget}_finite'],fit[f'fit_{budget}_objective'],np.inf),axis=1))
                        od=gd=sd=rd=0.
                        for ti in range(len(selected)):
                            for si in range(8):
                                aa,jj=geometry(head,mesh['transform'],fit[f'fit_{budget}_z'][ti,si]);residual=(aa-fit['targets'][ti])/diag['displacement_scale'];jj/=diag['displacement_scale']
                                objective=float(residual@residual);gradient=float(np.max(abs(jj.T@residual))/max(1.,np.linalg.norm(jj)*np.linalg.norm(residual)))
                                od=max(od,abs(objective-fit[f'fit_{budget}_objective'][ti,si]));gd=max(gd,abs(gradient-fit[f'fit_{budget}_gradient'][ti,si]))
                                singular=np.linalg.svd(jj,compute_uv=False);rank_ratio=singular[-1]/singular[0]
                                stationarity=np.linalg.norm(np.linalg.qr(jj)[0].T@residual)/max(np.linalg.norm(residual),1e-10)
                                sd=max(sd,abs(stationarity-fit[f'fit_{budget}_stationarity'][ti,si]));rd=max(rd,abs(rank_ratio-fit[f'fit_{budget}_rank_ratio'][ti,si]))
                        assert od<1e-8 and gd<1e-8 and sd<2e-8 and rd<2e-8
                        selected_index=(np.arange(len(selected)),selected)
                        expected_stationary=(fit[f'fit_{budget}_gradient'][selected_index]<=1e-7)&((fit[f'fit_{budget}_stationarity'][selected_index]<=1e-6)|(fit[f'fit_{budget}_objective'][selected_index]<=1e-20))&(fit[f'fit_{budget}_rank_ratio'][selected_index]>1e-8)&fit[f'fit_{budget}_finite'][selected_index]
                        np.testing.assert_array_equal(expected_stationary,next(f for f in diag['fitting'] if f['budget']==budget)['selected_stationary'])
                        report['fit_checks'].append(dict(boundary=bc,intervals=n,case=ci,model=model,budget=budget,endpoints=len(selected)*8,objective_difference=od,gradient_difference=gd,projected_stationarity_difference=sd,rank_ratio_difference=rd))
                    budget=cfg['diagnostic_fit_budgets'][-1];selected=fit[f'fit_{budget}_selected'];z=fit[f'fit_{budget}_z'][np.arange(len(selected)),selected]
                    curvature_difference=normal_difference=0.
                    for i,zz in enumerate(z):
                        aa,jj=geometry(head,mesh['transform'],zz);w=np.linalg.lstsq(jj,fit['velocities'][i],rcond=None)[0]
                        aa,bb,jj,curve=geometry(head,mesh['transform'],zz,w);q=np.linalg.qr(jj)[0]
                        force=-c*c*mesh['stiffness']@aa-c*mesh['damping']@bb-curve;normal=force-q@(q.T@force)
                        curvature_difference=max(curvature_difference,float(np.max(abs(curve-fit[model+'_curvature'][i]))))
                        normal_difference=max(normal_difference,float(np.max(abs(normal-fit[model+'_normal_force'][i]))))
                    assert curvature_difference<2e-6 and normal_difference<2e-6
                    for arm in diag['arms']:
                        aa=fit[arm['arm']+'_coefficients'];bb=fit[arm['arm']+'_velocity_coefficients']
                        ur=(aa[:-1]@g.T).reshape(fit['truth_u'].shape);vr=(bb[:-1]@g.T).reshape(fit['truth_v'].shape)
                        assert compare_metrics(recompute(ur,vr,fit['truth_u'],fit['truth_v'],n,bc,c),arm['snapshot_metrics'])<2e-8
                        assert abs(np.linalg.norm(aa[-1])-arm['zero_field_mass_norm'])<1e-12
                    report['geometry_checks'].append(dict(boundary=bc,intervals=n,case=ci,model=model,curvature_difference=curvature_difference,normal_force_difference=normal_difference))
    report.update(passed=True,audited_comparison_invocations=len(data['invocations']),audited_accuracy_only_invocations=len(data['accuracy_controls']),
        failed_comparison_invocations=sum(not r['completed'] for r in data['invocations']),failed_accuracy_only_invocations=sum(not r['completed'] for r in data['accuracy_controls']),
        maximum_physical_metric_difference=max(r['physical_metric_difference'] for r in report['query_checks']),
        maximum_ROM_native_grid_metric_difference=max(r['native_grid_ROM_metric_difference'] or 0 for r in report['query_checks']),
        file_hashes=len(data['output_sha256']))
    write(out/'audit.json',report)
    print(json.dumps({k:report[k] for k in ('passed','audited_comparison_invocations','audited_accuracy_only_invocations','failed_comparison_invocations','maximum_physical_metric_difference','maximum_ROM_native_grid_metric_difference')},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);audit(ap.parse_args().record)
