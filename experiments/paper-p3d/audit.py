"""Independent NumPy/SciPy saved-field, reference and summary audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.fft import dstn


def sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def error(a,b):return float(np.linalg.norm(np.asarray(a).reshape(-1)-np.asarray(b).reshape(-1))/np.linalg.norm(b))


def audit(out,output=None):
    record=json.loads((out/'result.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert record['complete'] and record['backend']=='gpu' and record['x64'] and record['matmul_precision']=='highest'
    if record['final_cohort_opened']:
        from freeze import digest, configuration, checkpoint_names
        assert record['evaluation_cohort']=='final'
        assert record['freeze']['verified_before_final_parameter_generation']
        assert all(entry.get('reuse') for entry in record['config']['operators'])
        freeze_path=out.parent/record['config']['final_freeze_path']
        freeze=json.loads(freeze_path.read_text())
        assert digest(freeze_path)==record['freeze']['sha256']
        assert freeze['configuration']==configuration(record['config'])
        assert freeze['checkpoint_sha256']=={name:digest(out/name) for name in checkpoint_names(record['config'])}
        assert freeze['selection_result_sha256']==record['reused_checkpoints']['files']['result.json']
    checked=0;references={};max_reference_defect=0.;max_metric_defect=0.
    for row in record['invocations']:
        n=row['intervals'];case=row['case'];key=(n,case)
        if key not in references:
            z=np.load(out/'fields'/f'N{n}_case{case}_reference.npz');references[key]={k:z[k] for k in z.files}
            k=np.arange(1,n);one=4*n*n*np.sin(np.pi*k/(2*n))**2
            lam=one[:,None,None]+one[None,:,None]+one[None,None,:]
            exact=dstn(dstn(z['forcing'],type=1,norm='ortho')/lam,type=1,norm='ortho')
            defect=error(exact,z['same_grid']);max_reference_defect=max(max_reference_defect,defect)
            assert defect<1e-11,defect
        if row['method'].startswith('nmrom_'):
            stats=np.asarray(row['selected_stats']);starts=np.asarray(row['all_starts_stats'])
            assert row['stationary']==bool(int(stats[2])==1 and stats[5]<=record['config']['lm_tolerance'])
            assert row['iterations']==int(np.sum(starts[:,0]))
        if 'field_file' not in row:continue
        pred=np.load(out/row['field_file'])['prediction'];assert sha(pred)==row['field_sha256']
        assert pred.dtype==np.dtype('float64')
        assert row['finite']==bool(np.isfinite(pred).all())
        if row['finite']:
            for metric,truth in [('same_grid_error','same_grid'),('physical_error','physical')]:
                observed=error(pred,references[key][truth]);defect=abs(observed-row[metric]);max_metric_defect=max(max_metric_defect,defect)
                assert defect<1e-12,(metric,defect)
        checked+=1
    for row in summary['rows']:
        records=[r for r in record['invocations'] if r['intervals']==row['intervals'] and r['method']==row['method']]
        assert len(records)==row['invocations']
        assert np.array_equal([r['device_ms'] for r in records],row['device_ms_repetitions'])
        assert abs(np.median(row['device_ms_repetitions'])-row['device_ms_median'])<1e-12
        cases={r['case']:r for r in records};finite=[r for r in cases.values() if r['finite']]
        assert row['cases']==len(cases)
        assert row['nonfinite_cases']==len(cases)-len(finite)
        assert row['nonstationary_cases']==sum(not r['stationary'] for r in cases.values())
        assert row['cases_above_same_grid_target']==sum(r['same_grid_error']>record['config']['same_grid_target'] for r in finite)
        times=np.asarray([r['device_ms'] for r in records]);assert np.all(times>0)
        assert row['timing_outliers_above_1p5_median']==int(np.count_nonzero(times>1.5*np.median(times)))
        assert abs(np.median([r['total_ms'] for r in records])-row['total_ms_median'])<1e-12
        if finite:
            errors=np.asarray([r['same_grid_error'] for r in finite])
            for key,value in [('same_grid_error_worst',np.max(errors)),('same_grid_error_mean',np.mean(errors)),
                              ('same_grid_error_median',np.median(errors)),
                              ('physical_error_worst',max(r['physical_error'] for r in finite))]:
                assert abs(value-row[key])<1e-12,(row['method'],key,value,row[key])
            for case in cases:
                repeated=[r['same_grid_error'] for r in records if r['case']==case and r['finite']]
                if repeated:assert max(repeated)-min(repeated)<1e-12
    result=dict(passed=True,checked_fields=checked,checked_references=len(references),checked_summary_rows=len(summary['rows']),
        maximum_reference_relative_defect=max_reference_defect,maximum_metric_absolute_defect=max_metric_defect,
        limitation='independent field/reference/aggregation audit; no independent retraining or global-optimality proof')
    (output or out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--output');args=p.parse_args();audit(Path(args.out),Path(args.output) if args.output else None)
