"""Prospective diagnostic exception: continuous reverse-Jacobian implementation replay."""
import argparse,subprocess,sys
from common import *
from neural import residual_tree as R,residual_weak as W
from neural.build_data import resolve
from training_io import read_checkpoint
from evaluate_residual_tree import owned_job


def retained(ns, job, relative):
    owned_job(job)
    manifest_path=HERE/'manifests'/f'retention-{job}.json'
    manifest=json.loads(manifest_path.read_text())
    assert manifest['name']==job and manifest['terminal_state']=='COMPLETED' and manifest['local_and_remote_verified']
    entries=[r for r in manifest['files'] if r['file']==relative];assert len(entries)==1
    path=resolve(ns/job/relative,ns)
    assert path.stat().st_size==entries[0]['bytes'] and digest(path)==entries[0]['sha256']
    return path,dict(job=job,file=relative,sha256=digest(path),retention_manifest_sha256=digest(manifest_path),
        retained_source_commit=manifest['source_commit'],replay_source_commit=os.environ['SOURCE_COMMIT'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--profile-job',default='b8implprofile');ap.add_argument('--forward-job',default='b8treepde')
    ap.add_argument('--output-mode',choices=['baseline16','cached64'],default='cached64');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ns=Path(os.environ['STUDY_NAMESPACE'])
    jp,jproof=retained(ns,a.profile_job,'output/jacobian/result.json');j=json.loads(jp.read_text())
    fp,fproof=retained(ns,a.forward_job,'output/result.json');forward=json.loads(fp.read_text())
    assert j['status']=='complete_profile' and not j['final_test_opened']
    assert j['trajectory_record_sha256']==digest(fp) and j['model_sha256']==forward['model_sha256']
    assert j['all_three_step_medians_faster'] and len(j['states'])==3
    for row in j['states']:
        assert row['parity']['root_jacobian']['passed'] and row['parity']['residual_jacobian']['passed']
        p=row['parity']['step'];assert p['relative_field_change']<=1e-10 and p['residual_vector_change']<=1e-10
        for mode in ['forward_status','reverse_status']:
            st=p[mode];assert st[-1]==1 and st[1]<=1e-8 and (st[2]<=1e-10 or st[1]<=1e-10)
    outputproof=None
    if a.output_mode=='cached64':
        op,outputproof=retained(ns,a.profile_job,'output/output/result.json');profile=json.loads(op.read_text())
        assert profile['status']=='complete_equivalent_output_profile' and not profile['final_test_opened']
        assert profile['model_sha256']==forward['model_sha256'] and profile['source_result_sha256']==digest(fp)
        blocks=[r for r in profile['blocks'] if r['variant']=='cached64'];assert len(blocks)==1 and len(blocks[0]['outputs'])==5
        for row in blocks[0]['outputs']:assert row['maximum_absolute_difference']<1e-11 and row['relative_l2_difference']<1e-10
    assert forward['status']=='complete_fixed_tree_diagnostic' and not forward['final_test_opened']
    assert (forward['case'],forward['layout'],forward['nodes'],forward['L'],forward['Q'],forward['M'])==('dev003','uniform',4,16384,128,512)
    assert forward['dt']==.0005 and forward['theta']==.5 and forward['iterations']==60
    assert forward['station_tolerance']==1e-8 and forward['update_tolerance']==1e-10
    result=dict(meta=meta(),status='running_continuous_replay',diagnostic=True,method_admitted=False,
        final_test_opened=False,profile_proof=jproof,forward_proof=fproof,output_profile_proof=outputproof,
        prior_strict_trace_gate_passed=j['parity_passed'],prior_replay_eligibility=j['eligible_for_separately_registered_full_trajectory'],
        prospective_exception='Preserve strict trace gate failure. Test continuous trajectory physical equivalence at1e-8 with unchanged PDE stopping rules; acceptance/iteration differences are diagnostics.',
        parity_threshold=1e-8,online_forward_codes_used=False,paired_runtime_comparison=False,
        runtime_note='New complete cold query recorded separately; no cross-job speedup claim. Post-completion parity excluded from online solve time.')
    save(out/'result.json',result)
    solveout=out/'solve'
    command=[sys.executable,str(HERE/'solve_residual_tree.py'),'--out',str(solveout),'--model-job',forward['model_job'],
        '--diagnostic','--case-label','dev003','--layout','uniform','--nodes','4','--L','16384','--Q','128','--M','512',
        '--dt','.0005','--theta','.5','--iterations','60','--station-tolerance','1e-8','--update-tolerance','1e-10',
        '--root-jacobian','reverse','--output-mode',a.output_mode]
    # No forward trajectory code or solution field is loaded before this returns.
    try:
        run=subprocess.run(command,timeout=1500,check=False)
    except subprocess.TimeoutExpired:
        result.update(status='failed_solve_timeout',passed=False,timeout_seconds=1500)
        save(out/'result.json',result);return
    result['solve_returncode']=run.returncode
    sr=json.loads((solveout/'result.json').read_text());result['solve_result_sha256']=digest(solveout/'result.json')
    result['solve_timing']=sr.get('timing')
    if run.returncode or sr['status']!='complete_fixed_tree_diagnostic':
        result.update(status='failed_continuous_solve',passed=False);save(out/'result.json',result);return
    assert sr['all_steps_converged'] and sr['steps_completed']==500 and sr['requested_outputs_complete']
    assert sr['model_sha256']==forward['model_sha256'] and sr['reference']==forward['reference']
    begin=time.perf_counter()
    cp,cproof=retained(ns,a.forward_job,'output/'+forward['trajectory_codes_file']);result['forward_codes_proof']=cproof
    f=np.load(cp);newpath=solveout/sr['trajectory_codes_file'];assert digest(newpath)==sr['trajectory_codes_sha256'];n=np.load(newpath)
    assert f['codes'].shape==n['codes'].shape==(501,96) and np.array_equal(f['times'],n['times']) and np.array_equal(f['cells'],n['cells'])
    assert np.isfinite(n['codes']).all() and np.isfinite(n['statuses']).all() and np.all(n['statuses'][:,-1]==1)
    status=n['statuses'][1:]
    assert np.all((status[:,1]<=1e-8)&((status[:,2]<=1e-10)|(status[:,1]<=1e-10)))
    assert np.array_equal(n['codes'][0],f['codes'][0]),'Initialization must remain identical'
    mp=resolve(ns/sr['model_job']/'output/nonlinear_tree.pkl',ns);assert digest(mp)==sr['model_sha256']
    params=jax.tree.map(jnp.asarray,read_checkpoint(mp)['params']);cache=W.prepare(params,jnp.asarray(n['cells']),16384,128,512)
    field=jax.jit(W.field);errors=[]
    for i in range(501):
        uf=field(params,jnp.asarray(f['codes'][i]),cache);un=field(params,jnp.asarray(n['codes'][i]),cache)
        errors.append(float(jnp.linalg.norm(un-uf)/jnp.linalg.norm(uf)))
    status_difference=n['statuses'][:,3:5]-f['statuses'][:,3:5]
    result['integration_parity']=dict(normalization='Current forward field discrete L2; implementation parity, not reference accuracy',
        all501_relative_field_differences=errors,worst=max(errors),passed=bool(np.isfinite(errors).all() and max(errors)<=1e-8),
        iteration_difference=status_difference[:,0],accepted_difference=status_difference[:,1],
        differing_iteration_count=int(np.count_nonzero(status_difference[:,0])),differing_acceptance_count=int(np.count_nonzero(status_difference[:,1])))
    output_rows=[]
    for old,new in zip(forward['outputs'],sr['outputs']):
        assert old['time']==new['time']
        oldpath,proof=retained(ns,a.forward_job,'output/'+old['fields']['file'])
        newpath=solveout/new['fields']['file'];assert digest(newpath)==new['fields']['sha256']
        with np.load(oldpath) as ov,np.load(newpath) as nv:
            of=ov['prediction'];nf=nv['prediction'];assert of.shape==nf.shape==(4096,4096)
            norm=float(new['reference_l2_discrete']);assert np.isfinite(norm) and norm>0
            error=float(np.linalg.norm(nf-of)/norm)
        output_rows.append(dict(time=new['time'],relative_field_difference=error,normalization='Stored current-time reference discrete L2',
            reference_l2_discrete=norm,forward_field_proof=proof,new_field_sha256=digest(newpath),passed=bool(np.isfinite(error) and error<=1e-8)))
    result.update(status='complete_implementation_replay',outputs=output_rows,postsolve_parity_seconds=time.perf_counter()-begin,
        passed=bool(result['integration_parity']['passed'] and all(r['passed'] for r in output_rows) and sr['physical_accuracy_pass']),
        physical_accuracy_pass=sr['physical_accuracy_pass'],complete_cold_query_seconds=sr['timing']['complete_cold_query_seconds'])
    save(out/'result.json',result)

if __name__=='__main__':main()
