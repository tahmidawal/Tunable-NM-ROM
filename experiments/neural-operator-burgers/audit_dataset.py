"""Audit generated split checkpoints with NumPy; never run a new PDE solve."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import data as d
from audit_calibration import initial
from refine import configure


def audit(index, calibration, *, allow_partial=False):
    x=json.loads(index.read_text());cal=json.loads(calibration.read_text())
    assert x['pde']=='burgers' and x['kind']=='matched-neural-operator-dataset'
    split=x['split'];assert split in ('train','validation')
    assert x['count']==d.PROTOCOL['counts'][split] and x['mesh']==256
    assert x['final_cohort']=='sealed' and not x['descriptors_are_model_inputs']
    if not allow_partial:assert x['complete'] and len(x['records'])==x['count']
    if x['complete']:assert len(x['records'])==x['count']
    assert len(x['records'])<=x['count']
    assert x['protocol_sha256']==d.sha(d.PROTOCOL_PATH)==cal['protocol_sha256']
    assert x['calibration']['sha256']==d.sha(calibration)
    selected=cal['gate']['by_output']['256']['selected']
    assert cal['complete'] and cal['count']==8 and selected['passing']
    assert x['reference_setting']==selected
    provenance=x['provenance']
    assert provenance['backend']=='gpu' and provenance['f64'] and provenance['matmul_precision']=='highest'
    assert provenance['source_sha256']==cal['provenance']['source_sha256']
    for name,sha in provenance['source_sha256'].items():
        assert d.sha(d.ROOT/name)==sha
        blob=subprocess.check_output(['git','-C',str(d.ROOT),'show',provenance['source_commit']+':'+name])
        assert hashlib.sha256(blob).hexdigest()==sha
    rows=[]
    def checked_artifact(record):
        path=(index.parent/record['path']).resolve()
        assert path.parent==index.parent.resolve() and d.sha(path)==record['sha256']
        return path
    for i,row in enumerate(x['records']):
        for key,value in d.case_record(split,i).items():assert row[key]==value
        assert row['mesh']==256
        with np.load(checked_artifact(row),allow_pickle=False) as z:
            assert set(z.files)=={'input','target','parameters','times'}
            for key in z.files:assert z[key].dtype==np.float64 and np.isfinite(z[key]).all()
            supplied,target=z['input'],z['target']
            assert supplied.shape==(1,257,257) and target.shape==(6,1,257,257)
            assert z['parameters'].shape==(1,) and np.array_equal(z['times'],d.TIMES)
            assert np.array_equal(supplied,target[0])
            assert not target[:,:,[0,-1],:].any() and not target[:,:,:,[0,-1]].any()
            expected,nu=initial(256,row['seed'])
            initial_error=float(np.linalg.norm(supplied[0]-expected)/np.linalg.norm(expected))
            assert initial_error<=1e-12 and np.isclose(z['parameters'][0],nu,rtol=1e-13,atol=0)
            input_hash=hashlib.sha256(supplied.tobytes()).hexdigest()
        reference=row['reference']
        assert reference['calibration_sha256']==d.sha(calibration)
        assert reference['intervals']==selected['intervals'] and reference['dt']==selected['dt']
        with np.load(checked_artifact(row['solver_audit']),allow_pickle=False) as z:
            assert set(z.files)=={'iterations','residuals'}
            iterations,residuals=z['iterations'],z['residuals']
            assert iterations.shape==residuals.shape==(round(.25/selected['dt']),)
            assert np.issubdtype(iterations.dtype,np.integer) and residuals.dtype==np.float64
            assert np.isfinite(residuals).all() and (residuals>=0).all() and residuals.max()<=2e-11
            assert (iterations>=0).all() and iterations.max()<=20
            assert int(iterations.sum())==reference['total_newton_iterations']
            assert float(residuals.max())==reference['max_relative_residual']
        rows.append(dict(case_id=row['case_id'],seed=row['seed'],input_sha256=input_hash,
                         initial_regeneration_relative_error=initial_error))
    return dict(passed=True,index_sha256=d.sha(index),split=split,complete=x['complete'],
                expected_count=x['count'],records_checked=len(rows),cases=rows,
                maximum_initial_regeneration_relative_error=max((r['initial_regeneration_relative_error'] for r in rows),default=None),
                calibration_index_sha256=d.sha(calibration),original_calibration_path=x['calibration']['path'],
                audited_calibration_path=str(calibration.resolve()),source_provenance=provenance)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args();configure()
    calibration=a.output/'refinement/index.json'
    results=[audit(a.output/split/'index.json',calibration,allow_partial=a.allow_partial)
             for split in ('train','validation') if (a.output/split/'index.json').exists()]
    if not a.allow_partial:assert len(results)==2
    allrows=[row for result in results for row in result['cases']]
    cal=json.loads(calibration.read_text())
    seeds=[r['seed'] for r in allrows]+[r['seed'] for r in cal['records']]
    assert len(set(seeds))==len(seeds)
    inputs=[r['input_sha256'] for r in allrows];assert len(set(inputs))==len(inputs)
    result=dict(passed=True,complete_dataset=len(results)==2 and all(r['complete'] for r in results),
                partial_mode=a.allow_partial,splits=results,total_records_checked=len(allrows),
                exact_four_key_schema=True,finite_float64_zero_boundary=True,
                input_equals_target_initial=True,independent_seed_regeneration=True,
                no_duplicate_seeds_across_calibration_train_validation=True,no_duplicate_supplied_fields=True,
                every_saved_solver_step_residual_checked=True,
                interpretation='Gaussian continuum-family pilot with analytic fine-reference initial access; empirical independent calibration, not a per-case continuum certificate')
    d.write_json(a.out,result)
    print(json.dumps({key:result[key] for key in ('passed','complete_dataset','total_records_checked')}))


if __name__=='__main__':main()
