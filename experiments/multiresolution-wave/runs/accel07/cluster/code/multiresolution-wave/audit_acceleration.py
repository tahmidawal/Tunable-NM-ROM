"""Independent NumPy/SciPy full-field, rank, geometry and provenance audit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import scipy.linalg
from audit_dynamics import geometry, recompute, compare_metrics, initial_fields, parameter_rows, NAMES, energy2
from audit_iterative_extra import modal_fields, sample_bank


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


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
    for lineage in [dict(commit=origin['checkpoint_commit'],sources=origin['sources']),origin['head32_origin']]:
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
    assert len(result['invocations'])==len(cfg['meshes'])*countcases*(len(cfg['arms'])+len(cfg['cg_tolerances'])+1)*cfg['repetitions']
    checked=[];references=[];banks=[];parities=[];refinements=[]
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
                    if 'coefficients' in field:
                        aa=[];bb=[];rank=[];acceleration_delta=[];backwards=[];bound_defects=[]
                        for z,w in zip(field['rollout_z'],field['rollout_w']):
                            a,b,jac,curve=geometry(head,mesh['transform'],z,w)
                            aa.append(a);bb.append(b)
                            singular=np.linalg.svd(jac,compute_uv=False);rank.append(singular[-1]/singular[0])
                            force=curve+c*c*mesh['stiffness']@a+c*mesh['damping']@b
                            q,rr=np.linalg.qr(jac,mode='reduced');exact=scipy.linalg.solve_triangular(rr,-q.T@force)
                            lower=np.linalg.cholesky(jac.T@jac)
                            bound=1/(np.linalg.norm(jac)*np.linalg.norm(scipy.linalg.solve_triangular(lower,np.eye(32),lower=True)))
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
                        a,b,jac,_=geometry(head,mesh['transform'],field['rollout_z'][0],field['rollout_w'][0])
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
    output=dict(passed=True,source_commit=submission['source_commit'],job_id=meta['job_id'],
        timed_invocations=len(result['invocations']),distinct_timed_fields=len(groups),accuracy_control_fields=len(result['accuracy_controls']),
        banks=banks,references=references,fields=checked,parities=parities,refinements=refinements)
    (record/'audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);audit(ap.parse_args().record)
