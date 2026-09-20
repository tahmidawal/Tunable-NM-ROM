"""Independent NumPy saved-field metric and repeat-integrity audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(out, destination=None):
    out=Path(out);record=json.loads((out/'result.json').read_text());failures=[];checked=0
    assert record['backend']=='gpu' and record['x64'] and record['matmul_precision']=='highest'
    def metric(a,b):
        a=a.reshape(len(a),-1);b=b.reshape(len(b),-1)
        diff=np.sqrt(np.sum((a-b)**2,axis=1));norm=np.sqrt(np.sum(b*b,axis=1))
        return diff/np.maximum(norm,1e-300),diff/max(norm[0],1e-300)
    for row in record['invocations']:
        if 'field_file' not in row:continue
        path=out/row['field_file'];pred=np.load(path)['prediction']
        refs=np.load(out/'fields'/f'N{row["intervals"]}_case{row["case"]}_reference.npz')
        for key,refkey in [('same_grid','same_grid'),('physical','physical')]:
            if not row['finite']:continue
            current,initial=metric(pred,refs[refkey])
            for name,actual in [('current_by_time',current),('initial_by_time',initial)]:
                if not np.allclose(actual,row[key][name],rtol=2e-11,atol=2e-13):failures.append(f'{path.name}: {key} {name}')
        blob=np.ascontiguousarray(pred)
        sha=hashlib.sha256(str((blob.shape,blob.dtype.str)).encode()+blob.tobytes()).hexdigest()
        if sha!=row['field_sha256']:failures.append(f'{path.name}: checksum')
        if row['device_ms']<=0 or row['total_ms']<row['device_ms']:failures.append(f'{path.name}: timing')
        checked+=1
    groups={}
    for row in record['invocations']:
        groups.setdefault((row['intervals'],row['case'],row['method']),[]).append(row)
    for key,rows in groups.items():
        if record['complete'] and sorted(r['repetition'] for r in rows)!=list(range(record['config']['repetitions'])):
            failures.append(f'{key}: incomplete repetitions')
    for mesh in record['meshes']:
        n=mesh['intervals']
        for row in mesh['representation']:
            if 'head_best_found' not in row:continue
            data=np.load(out/'fields'/f'N{n}_K{row["k"]}_best_found.npz')
            prediction=data['prediction'][row['case']]
            reference=np.load(out/'fields'/f'N{n}_case{row["case"]}_reference.npz')['same_grid']
            current,initial=metric(prediction,reference)
            if not np.allclose(current,row['head_best_found']['current_by_time'],rtol=2e-11,atol=2e-13):
                failures.append(f'N{n} K{row["k"]} case{row["case"]}: best-found metric')
            checked+=1
    result=dict(passed=not failures,checked_fields=checked,failures=failures,complete=record['complete'],
                final_cohort_opened=record['final_cohort_opened'],
                scope='independent NumPy field metrics, hashes and invocation integrity; not a global solver-optimum proof')
    (Path(destination) if destination else out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--destination');a=p.parse_args();audit(a.out,a.destination)
