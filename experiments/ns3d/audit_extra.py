"""Independent audit of extra03 without rewriting any collected raw bytes.

Frozen numerical-reference/operator checks are inherited only by verified content
hash. Newly generated fields, metrics, checkpoints and capacity fits are audited
here; per-step Jacobian probes use audit_histories.py separately.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.signal import resample
from audit_comparison import file_sha,array_sha,leaves
from audit_pilot import independently_evaluate_bank,head,rel,stats


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--previous',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.collected);out=root/'output';previous=Path(a.previous);old=previous/'output';report=json.loads((out/'result.json').read_text());cfg=report['config']
    result=dict(passed=True,source_commit=report['source_commit'],job_id=report['job_id'],scope='independent checksums, same-invocation physical field errors and capacity reconstruction; separate bounded Jacobian audit required',checks={})
    def gate(name,passed,**values):
        result['checks'][name]=dict(passed=bool(passed),**values);result['passed'] &= bool(passed);print(name,json.dumps(result['checks'][name]),flush=True);Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    failures=[];count=0
    for line in (root/'OUTPUTS.sha256').read_text().splitlines():
        want,path=line.split(None,1);count+=1
        if file_sha(root/path.strip())!=want:failures.append(path)
    gate('all_raw_output_hashes',not failures,count=count,failures=failures)
    stdout=(root/'logs'/f"{report['job_id']}.out").read_text();stderr=(root/'logs'/f"{report['job_id']}.err").read_text()
    gate('backend_complete',report['complete'] and report['x64'] and report['precision']=='highest' and report['device_resident_parameters'] and 'jax_backend=gpu' in stdout and 'PILOT_EXIT=0' in stdout and not any(t in (stdout+stderr).lower() for t in ['captured-large-constant','out of memory','resource_exhausted','no space left']),stderr_preserved_bytes=len(stderr))
    inherited=json.loads((previous.parent/'audit.json').read_text());reuse=report['reuse'];reused=[]
    for entry in reuse['files']:reused.append(file_sha(old/entry['path'])==entry['sha256'])
    gate('frozen_artifact_lineage',inherited['passed'] and all(reused),count=len(reused),inherited_job=inherited['job_id'])
    for kind in ('train_data','dev_data'):
        with np.load(out/(kind+'.npz')) as f:states=f['states'];params=f['parameters'];actual=array_sha(states);parameter_hash=array_sha(params)
        gate(kind+'_regenerated_membership',actual==report[kind]['states_sha256'] and parameter_hash==report[kind]['parameter_sha256'],states_sha256=actual,parameter_sha256=parameter_hash)
        del states
    bad=[]
    for spec in cfg['extra_operators']:
        with (out/spec['kind']/'best.pkl').open('rb') as f:model=pickle.load(f)
        bad += [(spec['kind'],str(v.dtype)) for v in leaves(model['params']) if v.dtype.kind in 'fc' and v.dtype not in (np.float64,np.complex128)]
    gate('new_operator_parameter_precision',not bad,bad=bad)
    with np.load(out/'timing_references.npz') as f:same=f['same_grid'];fine=f['fine_grid']
    with np.load(old/'timing_references.npz') as f:old_same=f['same_grid'];old_fine=f['fine_grid']
    gate('independently_audited_reference_replay',rel(same,old_same)<1e-12 and rel(fine,old_fine)<1e-12,same_relative=rel(same,old_same),fine_relative=rel(fine,old_fine))
    rows=json.loads((out/'timing_rows.json').read_text());cache={};metricmax=0.;bad=[];count=0;replay=[]
    with np.load(out/'timed_fields.npz') as fields,np.load(old/'timed_fields.npz') as frozen:
        for key in fields:
            value=fields[key];parts=key.split('__');kind=parts[0];case=int(parts[1][4:]);count+=1
            error=np.linalg.norm((value-same[case]).reshape(6,-1),axis=1)/np.linalg.norm(same[case,0]);lift=value
            for axis in (-3,-2,-1):lift=resample(lift,cfg['fine_reference_n'],axis=axis)
            physical=np.linalg.norm((lift-fine[case]).reshape(6,-1),axis=1)/np.linalg.norm(fine[case,0]);cache[(kind,case,array_sha(value))]=(error,physical)
            if key in frozen:replay.append(dict(field=key,relative=rel(value,frozen[key])))
        for row in rows:
            key=(row['method'],row['case'],row['field_sha256'])
            if key not in cache:bad.append('unretained invocation');continue
            e,ph=cache[key];difference=max(np.max(abs(e-row['same_grid_errors'])),np.max(abs(ph-row['fine_grid_errors'])));metricmax=max(metricmax,float(difference))
            if difference>1e-10:bad.append('metric discrepancy')
            if not 0<row['gpu_seconds']<=row['with_host_seconds']:bad.append('timing inequality')
    gate('timed_fields',not bad,fields=count,invocations=len(rows),maximum_metric_disagreement=metricmax,failures=bad)
    # Retaining state history can change fusion/roundoff. The threshold checks a
    # bounded replay, not exact bit identity; failures remain visible.
    gate('frozen_method_replay',all(r['relative']<1e-8 for r in replay),worst_relative=max((r['relative'] for r in replay),default=0.),count=len(replay))
    result['derived_summary']={}
    for method in sorted({r['method'] for r in rows}):
        arm=[r for r in rows if r['method']==method];t=np.array([r['gpu_seconds'] for r in arm]);med=np.median(t)
        result['derived_summary'][method]=dict(invocations=len(arm),device_ms_median=float(1000*med),timing_outliers_above_three_median=int(np.sum(t>3*med)),worst_evolved_fine_grid=max(max(r['fine_grid_errors'][1:]) for r in arm),worst_evolved_same_grid=max(max(r['same_grid_errors'][1:]) for r in arm))
    cap=out/'capacity';screen=json.loads((cap/'screen.json').read_text())
    if screen.get('complete'):
        with (cap/'frozen_bank.pkl').open('rb') as f:checkpoint=pickle.load(f)
        G=checkpoint['extra']['bank'];Q=checkpoint['extra']['qr_Q'];Rb=checkpoint['extra']['qr_R'];computed=independently_evaluate_bank(checkpoint['params'],cfg['n'])
        gate('larger_coordinate_bank',rel(computed,G)<1e-11,relative=rel(computed,G));del computed
        with np.load(out/'dev_data.npz') as f:states=f['states']
        X=states.reshape(-1,G.shape[0]);den=np.repeat(np.linalg.norm(states[:,0].reshape(len(states),-1),axis=1),6);norm=np.linalg.norm(X,axis=1)
        floor=np.linalg.norm(X-(X@Q)@Q.T,axis=1)/den
        gate('larger_bank_initial_normalization',all(abs(screen['development_bank_initial_normalized'][k]-v)<1e-10 for k,v in stats(floor).items()),floor=stats(floor))
        for k,record in screen['heads'].items():
            with (cap/f'head_K{k}.pkl').open('rb') as f:checkpoint=pickle.load(f)
            pred=head(checkpoint['params'],np.asarray(record['fits']['z']))@G.T
            errors=np.linalg.norm(pred-X,axis=1)/den;current=np.linalg.norm(pred-X,axis=1)/norm
            gate('head_K'+k+'_reconstruction',np.max(abs(errors-np.asarray(record['error_by_case_time']).ravel()))<1e-10 and np.max(abs(current-record['fits']['head_error']))<1e-10,initial_normalized=stats(errors))
    else:result['capacity_screen_failure_preserved']=report.get('capacity_screen_failure',screen.get('stage'))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
