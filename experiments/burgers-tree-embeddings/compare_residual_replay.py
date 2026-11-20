"""Postmortem component diagnostics; never retroactively passes the failed replay."""
import argparse
from common import *
from neural import residual_weak as W
from neural.build_data import resolve
from training_io import read_checkpoint
from evaluate_residual_tree import owned_job


def retained(ns,job,relative,terminal):
    owned_job(job);mp=HERE/'manifests'/f'retention-{job}.json';m=json.loads(mp.read_text())
    assert m['name']==job and m['terminal_state']==terminal and m['local_and_remote_verified']
    rows=[r for r in m['files'] if r['file']==relative];assert len(rows)==1
    path=resolve(ns/job/relative,ns);assert path.stat().st_size==rows[0]['bytes'] and digest(path)==rows[0]['sha256']
    return path,dict(job=job,file=relative,sha256=rows[0]['sha256'],terminal_state=terminal,
        retention_manifest_sha256=digest(mp),source_commit=m['source_commit'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--replay-job',default='b8treereplay');ap.add_argument('--forward-job',default='b8treepde');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ns=Path(os.environ['STUDY_NAMESPACE'])
    result=dict(meta=preflight(),status='running_component_diagnostics',diagnostic=True,final_test_opened=False,
        original_overall_gate_passed=False,exact_replay_passed=False,passed=False,method_admitted=False,
        contract='Read-only postmortem component diagnostics after bitwise initialization failure; no solve rerun, no retroactive gate pass, no paired runtime claim',
        parity_threshold=1e-8)
    save(out/'result.json',result)
    fp,fproof=retained(ns,a.forward_job,'output/result.json','COMPLETED');f=json.loads(fp.read_text())
    rp,rproof=retained(ns,a.replay_job,'output/result.json','FAILED');replay=json.loads(rp.read_text())
    sp,sproof=retained(ns,a.replay_job,'output/solve/result.json','FAILED');s=json.loads(sp.read_text())
    assert s['status']==f['status']=='complete_fixed_tree_diagnostic'
    assert s['all_steps_converged'] and s['steps_completed']==500 and s['requested_outputs_complete']
    assert not s['final_test_opened'] and not f['final_test_opened'] and not replay['final_test_opened']
    keys=['model_sha256','case','layout','nodes','root_k','child_k','L','Q','M','dt','theta','iterations',
          'station_tolerance','update_tolerance','reference']
    assert all(s[k]==f[k] for k in keys)
    assert (s['L'],s['Q'],s['M'],s['nodes'],s['root_k'],s['child_k'])==(16384,128,512,4,32,16)
    assert s['station_tolerance']==1e-8 and s['update_tolerance']==1e-10
    fc,fcp=retained(ns,a.forward_job,'output/'+f['trajectory_codes_file'],'COMPLETED')
    nc,ncp=retained(ns,a.replay_job,'output/solve/'+s['trajectory_codes_file'],'FAILED')
    assert digest(fc)==f['trajectory_codes_sha256'] and digest(nc)==s['trajectory_codes_sha256']
    old=np.load(fc);new=np.load(nc)
    assert old['codes'].shape==new['codes'].shape==(501,96)
    assert np.array_equal(old['times'],new['times']) and np.array_equal(old['cells'],new['cells'])
    assert np.isfinite(new['codes']).all() and np.isfinite(new['statuses']).all()
    statuses=new['statuses'];assert np.all(statuses[:,-1]==1)
    assert np.all((statuses[1:,1]<=1e-8)&((statuses[1:,2]<=1e-10)|(statuses[1:,1]<=1e-10)))
    delta=new['codes'][0]-old['codes'][0];bitwise=bool(np.array_equal(new['codes'][0],old['codes'][0]))
    assert not bitwise,'This postmortem is scoped to the retained bitwise-initialization failure'
    result.update(forward_proof=fproof,replay_proof=rproof,solve_proof=sproof,forward_codes_proof=fcp,new_codes_proof=ncp,
        original_recorded_status=replay['status'],original_failure='Initial codes were not bitwise equal after otherwise complete solve',
        initial_codes=dict(bitwise_equal=bitwise,maximum_absolute_difference=float(np.max(abs(delta))),
            l2_difference=float(np.linalg.norm(delta)),relative_code_l2_difference=float(np.linalg.norm(delta)/np.linalg.norm(old['codes'][0])),
            changed_coordinates=int(np.count_nonzero(delta)),forward_status=old['statuses'][0],new_status=new['statuses'][0]),
        physical_accuracy_pass=s['physical_accuracy_pass'],solve_complete_cold_query_seconds=s['timing']['complete_cold_query_seconds'])
    save(out/'result.json',result)
    mp=resolve(ns/s['model_job']/'output/nonlinear_tree.pkl',ns);assert digest(mp)==s['model_sha256']
    params=jax.tree.map(jnp.asarray,read_checkpoint(mp)['params']);cache=W.prepare(params,jnp.asarray(new['cells']),16384,128,512)
    field=jax.jit(W.field);errors=[]
    for i in range(501):
        uf=field(params,jnp.asarray(old['codes'][i]),cache);un=field(params,jnp.asarray(new['codes'][i]),cache)
        errors.append(float(jnp.linalg.norm(un-uf)/jnp.linalg.norm(uf)))
    differences=new['statuses'][:,3:5]-old['statuses'][:,3:5]
    result['integration_parity']=dict(normalization='Current forward field discrete L2; implementation comparison, not reference accuracy',
        all501_relative_field_differences=errors,worst=max(errors),component_passed=bool(np.isfinite(errors).all() and max(errors)<=1e-8),
        iteration_difference=differences[:,0],accepted_difference=differences[:,1],
        differing_iteration_count=int(np.count_nonzero(differences[:,0])),differing_acceptance_count=int(np.count_nonzero(differences[:,1])))
    save(out/'result.json',result)
    assert len(f['outputs'])==len(s['outputs'])==5
    rows=[]
    for fr,sr in zip(f['outputs'],s['outputs']):
        assert fr['time']==sr['time']
        op,opproof=retained(ns,a.forward_job,'output/'+fr['fields']['file'],'COMPLETED')
        npth,nproof=retained(ns,a.replay_job,'output/solve/'+sr['fields']['file'],'FAILED')
        assert digest(op)==fr['fields']['sha256'] and digest(npth)==sr['fields']['sha256']
        with np.load(op) as ov,np.load(npth) as nv:
            of=ov['prediction'];nf=nv['prediction'];assert of.shape==nf.shape==(4096,4096)
            norm=float(sr['reference_l2_discrete']);assert np.isfinite(norm) and norm>0
            diff=float(np.linalg.norm(nf-of)/norm)
        rows.append(dict(time=sr['time'],relative_field_difference=diff,reference_l2_discrete=norm,
            normalization='Stored current-time reference discrete L2',forward_proof=opproof,new_proof=nproof,
            component_passed=bool(np.isfinite(diff) and diff<=1e-8)))
        result['outputs']=rows;save(out/'result.json',result)
    result.update(status='complete_postmortem_component_diagnostics',
        physical_field_components_passed=bool(result['integration_parity']['component_passed'] and all(r['component_passed'] for r in rows)),
        original_overall_gate_passed=False,exact_replay_passed=False,passed=False)
    save(out/'result.json',result)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',required=True)
        args,_=parser.parse_known_args();p=Path(args.out)/'result.json'
        r=json.loads(p.read_text()) if p.exists() else dict(diagnostic=True)
        r.update(status='failed_postmortem_exception',passed=False,exception_type=type(exc).__name__,exception=str(exc))
        save(p,r);raise
