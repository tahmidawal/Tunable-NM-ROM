"""Independent NumPy field/stopping audit and sampled-state EQ fidelity analysis."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import data as d


def errors(field,truth):
    f=field[:,1:-1,1:-1];t=truth[:,1:-1,1:-1]
    denominator=np.sqrt(np.sum(t[0]*t[0]))
    return np.sqrt(np.sum((f-t)**2,axis=(1,2)))/denominator


def validate_fields(field):
    assert field.shape==(6,257,257) and field.dtype==np.float64
    assert np.isfinite(field).all()
    assert not np.any(field[:,[0,-1],:]) and not np.any(field[:,:,[0,-1]])


def advect(field,L):
    u=field[:,1:-1,1:-1]
    xm,xp=field[:,:-2,1:-1],field[:,2:,1:-1]
    ym,yp=field[:,1:-1,:-2],field[:,1:-1,2:]
    return (u*L*(np.where(u>0,u-xm,xp-u)+np.where(u>0,u-ym,yp-u))).reshape(len(field),-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    x=json.loads(a.diagnosis.read_text());ref=json.loads(a.reference.read_text())
    assert x['complete'] and x['intervals']==256 and len(x['cases'])==8 and len(x['invocations'])==168
    assert x['provenance']['backend']=='gpu' and x['provenance']['f64'] and x['provenance']['matmul_precision']=='highest'
    assert ref['complete'] and ref['gate']['by_output']['256']['passing']
    assert d.sha(a.reference)==x['reference_index_sha256']
    assert d.sha(d.HERE/'diagnose.py')==x['own_source_sha256']
    assert d.sha(d.ROOT/'experiments/mr-burgers2d/accuracy_paths.py')==x['accuracy_paths_sha256']
    for name,sha in x['provenance']['source_sha256'].items():assert d.sha(d.ROOT/name)==sha
    L=256;mode_ids=np.asarray(x['setup']['mode_ids']);axis=np.arange(1,L)/L
    coordinates=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    phi=(2./L)*np.sin(np.pi*coordinates[:,0,None]*mode_ids[:,0])*np.sin(np.pi*coordinates[:,1,None]*mode_ids[:,1])
    lam=4*L*L*(np.sin(np.pi*mode_ids[:,0]/(2*L))**2+np.sin(np.pi*mode_ids[:,1]/(2*L))**2)
    pos=np.asarray(x['setup']['eq_indices']);weights=np.asarray(x['setup']['eq_weights']);assert np.all(weights>=0)
    weighted=phi[pos]*weights[:,None]
    truths={};online={};eq=[];maximum_metric_difference=0.
    for row in x['cases']:
        path=a.diagnosis.parent/row['path'];assert d.sha(path)==row['sha256']
        anchor=next(r for r in ref['solves'] if r['case_id']==row['case_id'] and r['intervals']==4096 and r['dt']==.00015625)
        target_path=a.reference.parent/anchor['path'];assert d.sha(target_path)==anchor['sha256']==row['anchor_sha256']
        target=np.load(target_path)['fields'][:,::4,::4]
        with np.load(path) as arrays:
            assert np.array_equal(arrays['reference'],target)
            assert np.array_equal(arrays['input'],target[0])
            for key,metric in [('bank','bank'),('nonlinear','nonlinear_best_found'),('online','online')]:
                fields=arrays[key];validate_fields(fields);actual=errors(fields,target)
                expected=np.asarray(row['errors'][metric]['per_time'])
                maximum_metric_difference=max(maximum_metric_difference,float(abs(actual-expected).max()))
                assert np.allclose(actual,expected,rtol=1e-10,atol=1e-14)
                assert np.isclose(actual.max(),row['errors'][metric]['maximum'],rtol=1e-10,atol=1e-14)
            assert np.all(np.asarray(row['errors']['bank']['per_time'])<=np.asarray(row['errors']['nonlinear_best_found']['per_time'])+1e-9)
            assert np.all(np.asarray(row['errors']['nonlinear_best_found']['per_time'])<=np.asarray(row['errors']['online']['per_time'])+1e-9)
            station=bool(np.max(arrays['step_gradients'])<=1e-6 and arrays['initial_gradient']<=1e-6)
            assert station==row['online_stationary']
            online[row['case_id']]=arrays['online'].copy()
        truths[row['case_id']]=target
        descriptors=next(r['generation_descriptors'] for r in ref['records'] if r['case_id']==row['case_id'])
        full=advect(online[row['case_id']],L)
        projected=full@phi;sampled=full[:,pos]@weighted;delta=sampled-projected
        relative=np.linalg.norm(delta,axis=1)/np.linalg.norm(projected,axis=1)
        scaled=.005*delta/(1+.005*descriptors['nu']*lam)
        normalized=np.linalg.norm(scaled,axis=1)/np.linalg.norm(target[0,1:-1,1:-1])
        eq.append(dict(case_id=row['case_id'],relative_projected_advection_difference=relative.tolist(),
            preconditioned_step_component_over_initial_norm=normalized.tolist()))
    seen=set()
    for row in x['invocations']:
        key=(row['case_id'],row['method'],row['rep']);assert key not in seen;seen.add(key)
        path=a.diagnosis.parent/row['path'];assert d.sha(path)==row['sha256']
        assert row['gpu_seconds']>0 and row['host_to_host_seconds']>=row['gpu_seconds']
        assert row['host_to_host_seconds']+1e-12>=row['gpu_seconds']+row['input_transfer_seconds']+row['output_transfer_seconds']
        with np.load(path) as arrays:
            fields=arrays['fields'];validate_fields(fields);actual=errors(fields,truths[row['case_id']])
            maximum_metric_difference=max(maximum_metric_difference,float(abs(actual-np.asarray(row['error']['per_time'])).max()))
            assert np.allclose(actual,row['error']['per_time'],rtol=1e-10,atol=1e-14)
            assert int(arrays['iterations'].sum())==row['newton_or_lm_iterations']
            if row['method']=='rom':
                assert np.array_equal(fields,online[row['case_id']])
                assert int(arrays['initial_iterations'])==row['initial_iterations']
                assert bool(arrays['step_gradients'].max()<=1e-6 and arrays['initial_gradient']<=1e-6)==row['stationary']
            else:
                preset=next(p for p in x['presets'] if p['name']==row['method'])
                assert np.array_equal(fields[0],truths[row['case_id']][0])
                assert bool(np.isfinite(arrays['residuals']).all() and arrays['residuals'].max()<=preset['ntol'])==row['converged']
    methods=['rom']+[p['name'] for p in x['presets']]
    assert seen=={(r['case_id'],m,rep) for r in x['cases'] for m in methods for rep in range(3)}
    summary={}
    for method in methods:
        rows=[r for r in x['invocations'] if r['method']==method];times=np.asarray([r['gpu_seconds'] for r in rows])
        q1,q3=np.quantile(times,[.25,.75]);fence=q3+1.5*(q3-q1)
        summary[method]=dict(gpu_median_ms=float(1000*np.median(times)),host_to_host_median_ms=1000*statistics.median(r['host_to_host_seconds'] for r in rows),
            worst_fixed_initial_error=max(r['error']['maximum'] for r in rows),worst_evolved_error=max(max(r['error']['per_time'][1:]) for r in rows),
            failed_stopping_invocations=sum(not r.get('stationary',r.get('converged',False)) for r in rows),
            latency_upper_tukey_outliers=int(np.sum(times>fence)),latency_upper_tukey_fence_seconds=float(fence),gpu_repetition_seconds=times.tolist())
    result=dict(passed=True,scope='NumPy postprocessing of archived GPU fields; no new PDE solve or training',
        diagnosis_index_sha256=d.sha(a.diagnosis),reference_index_sha256=d.sha(a.reference),cases_checked=8,invocations_checked=168,
        every_prediction_hash_and_error_checked=True,all_stopping_records_rechecked=True,complete_case_method_repetition_cartesian_product=True,
        maximum_absolute_metric_discrepancy=maximum_metric_difference,summary=summary,
        eq_fidelity=dict(case_results=eq,nominal_points=len(pos),positive_weights=int((weights>0).sum()),
            interpretation='Advection-component quadrature discrepancies on six saved native output states only; not a trajectory error bound, stationarity proof, or causal error decomposition'))
    d.write_json(a.out,result);print(json.dumps({k:result[k] for k in ['passed','cases_checked','invocations_checked','maximum_absolute_metric_discrepancy']}))


if __name__=='__main__':main()
