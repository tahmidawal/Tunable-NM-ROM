"""Independent seed/cohort, learned-bank, selected head and operator-field audit."""
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.special import expit
from audit_head import audit as audit_head,head


def bank_at(saved,n):
    x=np.arange(1,n,dtype=np.float64)/n
    xyz=np.stack(np.meshgrid(x,x,x,indexing='ij'),-1).reshape(-1,3)
    p=saved['params'];angle=2*np.pi*(xyz@p['freq']);v=np.concatenate((np.sin(angle),np.cos(angle)),axis=1)
    for w,b in p['net'][:-1]:v=v@w+b;v=v*expit(v)
    w,b=p['net'][-1];v=v@w+b
    return ((p['scale']*64*np.prod(xyz*(1-xyz),axis=1))[:,None]*v)@saved['rotation']


def audit(out,destination):
    out=Path(out);record=json.loads((out/'result.json').read_text());cfg=record['config']
    assert record['complete'] and not record['final_cohort_opened']
    assert len(record['seeds'])==len(cfg['seeds']);results=[]
    for setting,row in zip(cfg['seeds'],record['seeds']):
        assert row['directory']==setting['name'] and row['role']==setting['role']
        root=out/setting['name'];d=json.loads((root/'result.json').read_text());c=d['config']
        target=Path(destination).with_name(f'audit-{setting["name"]}.json')
        audit_head(root,target);a=json.loads(target.read_text())
        coordinates=np.load(root/'coordinates.npz');saved=pickle.loads((root/'bank.pkl').read_bytes())
        basis=bank_at(saved,c['train_intervals'])
        np.testing.assert_allclose(basis,coordinates['bank'],rtol=5e-10,atol=5e-12)
        bank_difference=float(np.max(np.abs(basis-coordinates['bank'])))
        for candidate in d['candidates']:
            model=pickle.loads((root/candidate['checkpoint_relative_path']).read_bytes())
            prediction=np.stack([head(model['params'],z)[0] for z in model['codes']])
            residual=coordinates['training_target']-prediction
            errors=np.sqrt((np.sum(residual**2,axis=1)+coordinates['training_perpendicular2'])/coordinates['training_norm2'])
            for key,value in [('mean',np.mean(errors)),('median',np.median(errors)),('worst',np.max(errors))]:
                np.testing.assert_allclose(value,candidate['info']['training_error_'+key],rtol=2e-10,atol=1e-12)
            directions=model['directions'];np.testing.assert_allclose(directions.T@directions,np.eye(len(directions)),rtol=1e-10,atol=1e-10)
            # Singular axes may change sign, but the selected ordered eigenspaces must agree.
            covariance=residual.T@residual;diagonal=directions.T@covariance@directions
            np.testing.assert_allclose(diagonal,np.diag(np.diag(diagonal)),rtol=1e-8,atol=1e-9)
            assert np.all(np.diff(np.diag(diagonal))<=1e-9)
        operators=[]
        for operator in d['operators']:
            directory=root/'operators'/operator['name'];fields=np.load(directory/'development.npz')
            prediction=fields['prediction'];truth=fields['target'];shape=prediction.shape
            errors=np.linalg.norm((prediction-truth).reshape(shape[0],-1,shape[-1]),axis=1)/np.linalg.norm(truth.reshape(shape[0],-1,shape[-1]),axis=1)
            np.testing.assert_allclose(errors,operator['development_error_by_case_time'],rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(np.max(errors),operator['best_validation_worst'],rtol=1e-10,atol=1e-12)
            curve=json.loads((directory/'curve.json').read_text());selected=min(r['validation_worst'] for r in curve if 'validation_worst' in r)
            np.testing.assert_allclose(np.max(errors),selected,rtol=1e-10,atol=1e-12)
            cohort=json.loads((root/'cohorts.json').read_text())
            assert operator['training_parameters_sha256']==cohort['train_sha256']
            assert operator['validation_parameters_sha256']==cohort['validation_sha256']
            if (directory/'pretraining/pretraining.json').exists():
                p=json.loads((directory/'pretraining/pretraining.json').read_text())
                assert not p['development_data_used'] and not p['final_data_used'] and not p['online_architecture_changed']
            operators.append(operator['name'])
        frozen=json.loads((root/'FROZEN.json').read_text())
        assert hashlib.sha256((root/'FROZEN.json').read_bytes()).hexdigest()==row['frozen_manifest_sha256']
        for path,want in frozen['checkpoints'].items():
            with (root/path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==want
        from audit_pretraining import audit as audit_pretraining
        pretraining=audit_pretraining(root,d)
        results.append(dict(seed=setting['name'],head_audit=a,pretraining=pretraining,bank_reconstruction_max_absolute_difference=bank_difference,
            operator_development_fields_checked=operators,all_checkpoint_hashes_verified=True))
    result=dict(passed=True,source_commit=record['source_commit'],job_id=record['job_id'],complete=True,
        final_cohort_opened=False,seeds=results,scope='independent source/cohort/bank/fields/analytic fitting gradients; training evidence only, no trajectory timing claim')
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print('COVERAGE_AUDIT_PASS',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--destination',required=True);a=p.parse_args();audit(a.out,a.destination)
