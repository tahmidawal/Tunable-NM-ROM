"""Independent NumPy archive checks and source-derived pilot summary."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),median=float(np.median(a)),maximum=float(a.max()),
                count_above_005=int(np.sum(a>.05)))


def audit(root, diagnosis="diagnosis", dataset_root=None):
    root=Path(root)
    data=Path(dataset_root) if dataset_root is not None else root
    calibration=json.loads((data/'calibration/calibration.json').read_text())
    assert calibration['complete'] and calibration['empirical_reference_budget_pass']
    arrays={}; ids=set(); array_hashes=set()
    for split,count in [('train',128),('validation',32)]:
        folder=data/split; index=json.loads((folder/'index.json').read_text())
        assert index['pde']=='poisson' and index['count']==count and index['complete']
        for r in index['records']:
            assert r['case_id'] not in ids; ids.add(r['case_id'])
            expected_seed=int(np.random.SeedSequence([20260914,11,1 if split=='train' else 2,int(r['case_id'].split('-')[-1])]).generate_state(1,dtype=np.uint32)[0])
            assert r['seed']==expected_seed
            path=folder/r['path']; assert digest(path)==r['sha256']
            with np.load(path,allow_pickle=False) as a:
                assert set(a.files)=={'input','target','parameters','times'}
                assert a['input'].shape==(1,257,257) and a['target'].shape==(1,1,257,257)
                assert a['parameters'].shape==(0,) and a['times'].tolist()==[0.]
                assert all(a[k].dtype==np.float64 and np.isfinite(a[k]).all() for k in a.files)
                ih=hashlib.sha256(a['input'].tobytes()).hexdigest()
                assert ih not in array_hashes; array_hashes.add(ih)
                source=a['input'][0]; field=a['target'][0,0]
                rng=np.random.default_rng(expected_seed)
                cx=float(rng.uniform(.15,.85,1)[0]); cy=float(rng.uniform(.15,.85,1)[0])
                width=float(np.exp(rng.uniform(np.log(.02),np.log(.1),1))[0])
                amplitude=float(rng.uniform(.5,2.,1)[0])
                x=np.arange(1,256,dtype=np.float64)/256
                expected_source=amplitude*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width**2))
                np.testing.assert_allclose(source[1:-1,1:-1],expected_source,rtol=2e-12,atol=1e-14)
                applied=256**2*(4*field[1:-1,1:-1]-field[2:,1:-1]-field[:-2,1:-1]-field[1:-1,2:]-field[1:-1,:-2])
                assert rel(applied,source[1:-1,1:-1])<1e-8
                if split=='validation':
                    reference=r['reference']; p=folder/reference['path']
                    assert digest(p)==reference['sha256']
                    with np.load(p) as f: arrays[r['case_id']]=(field.copy(),f['target'][0,0].copy(),f['refinement'][0,0].copy())
    result=json.loads((root/diagnosis/'result.json').read_text()); assert result['complete']
    rows=result['rows']; assert len(rows)==32*7*3
    seen=set()
    def field(entry):
        p=root/diagnosis/entry['field_path']; assert digest(p)==entry['field_sha256']
        with np.load(p) as a: f=a['field'].copy()
        assert np.isfinite(f).all() and f.dtype==np.float64
        assert not (np.any(f[0]) or np.any(f[-1]) or np.any(f[:,0]) or np.any(f[:,-1]))
        assert hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()==entry['array_sha256']
        return f
    for row in rows:
        key=(row['case_id'],row['method'],row['repetition']); assert key not in seen; seen.add(key)
        f=field(row)
        for target,name in zip(arrays[row['case_id']],['discrete_relative_error','physical_candidate_relative_error','refinement_relative_error']):
            np.testing.assert_allclose(rel(f,target),row[name],rtol=2e-12,atol=2e-14)
    oracle_summary={}
    for reference in ['discrete','physical_candidate']:
        entries=[r for r in result['oracles'] if r['reference']==reference]; assert len(entries)==32
        for r in entries:
            target=arrays[r['case_id']][0 if reference=='discrete' else 1]
            for item in [r['bank_projection'],r['augmented_bank_projection'],*r['fit_starts']]:
                np.testing.assert_allclose(rel(field(item),target),item['relative_error'],rtol=2e-12,atol=2e-14)
            assert abs(r['bank_projection']['relative_error']-r['augmented_bank_projection']['relative_error'])<1e-10
            assert r['best_found_nonlinear_linear_fit']['relative_error']+1e-10>=r['bank_projection']['relative_error']
        oracle_summary[reference]={name:stats([r[name]['relative_error'] for r in entries]) for name in ['bank_projection','augmented_bank_projection','best_found_nonlinear_linear_fit']}
        oracle_summary[reference]['best_fit_unsuccessful_stopping_cases']=sum(not r['best_found_nonlinear_linear_fit']['success'] for r in entries)
        oracle_summary[reference]['cases']=[dict(case_id=r['case_id'],bank=r['bank_projection']['relative_error'],
            augmented_bank=r['augmented_bank_projection']['relative_error'],best_found_fit=r['best_found_nonlinear_linear_fit']['relative_error'],
            online=r['online_relative_error'],online_valid=r['online_solver_valid']) for r in entries]
    methods={}
    for method in sorted(set(r['method'] for r in rows)):
        chosen=[r for r in rows if r['method']==method]
        cases=[next(r for r in chosen if r['case_id']==case) for case in sorted(arrays)]
        methods[method]=dict(physical_candidate_relative_error=stats([r['physical_candidate_relative_error'] for r in cases]),
            discrete_relative_error=stats([r['discrete_relative_error'] for r in cases]),
            invalid_invocations=sum(not r['solver_valid'] for r in chosen),
            gpu_seconds_median=float(np.median([r['fused_device_seconds'] for r in chosen])),
            total_seconds_median=float(np.median([r['total_seconds'] for r in chosen])),
            gpu_seconds_repetitions=[r['fused_device_seconds'] for r in chosen],
            total_seconds_repetitions=[r['total_seconds'] for r in chosen])
    return dict(archive_numpy_checks_pass=True,case_count=len(ids),timed_invocations=len(rows),
        reference_budget_failures=sum(not r['empirical_reference_budget_pass'] for r in rows[::21]),
        runtime=result['runtime'],retained_modes=result['setup']['retained_modes'],
        calibration_maximum_change=max(r['reference_relative_change'] for r in calibration['records']),
        oracles=oracle_summary,methods=methods,
        limits='Independent field/hash/metric checks; stationarity flags not independently recomputed here. Inherited ROM unmatched training history. Timings compare only within this job.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(audit(a.root),indent=2,allow_nan=False)+'\n')
