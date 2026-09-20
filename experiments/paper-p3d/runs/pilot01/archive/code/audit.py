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


def audit(out):
    record=json.loads((out/'result.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert record['complete'] and record['backend']=='gpu' and record['x64'] and record['matmul_precision']=='highest'
    assert not record['final_cohort_opened'];checked=0;references={};max_reference_defect=0.;max_metric_defect=0.
    for row in record['invocations']:
        n=row['intervals'];case=row['case'];key=(n,case)
        if key not in references:
            z=np.load(out/'fields'/f'N{n}_case{case}_reference.npz');references[key]={k:z[k] for k in z.files}
            k=np.arange(1,n);one=4*n*n*np.sin(np.pi*k/(2*n))**2
            lam=one[:,None,None]+one[None,:,None]+one[None,None,:]
            exact=dstn(dstn(z['forcing'],type=1,norm='ortho')/lam,type=1,norm='ortho')
            defect=error(exact,z['same_grid']);max_reference_defect=max(max_reference_defect,defect)
            assert defect<1e-11,defect
        if 'field_file' not in row:continue
        pred=np.load(out/row['field_file'])['prediction'];assert sha(pred)==row['field_sha256']
        assert pred.dtype==np.dtype('float64')
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
        if finite:assert abs(max(r['same_grid_error'] for r in finite)-row['same_grid_error_worst'])<1e-12
    result=dict(passed=True,checked_fields=checked,checked_references=len(references),checked_summary_rows=len(summary['rows']),
        maximum_reference_relative_defect=max_reference_defect,maximum_metric_absolute_defect=max_metric_defect,
        limitation='independent field/reference/aggregation audit; no independent retraining or global-optimality proof')
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');args=p.parse_args();audit(Path(args.out))
