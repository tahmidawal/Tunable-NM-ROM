"""Independent NumPy/SciPy reconstruction, field-error and fitting-gradient audit."""
import argparse,hashlib,json,pickle,re,subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit

def head(p,z):
    value=np.asarray(z);jac=np.eye(len(z))
    for w,b in p['net'][:-1]:
        pre=value@w+b;sigmoid=expit(pre);derivative=sigmoid+pre*sigmoid*(1-sigmoid)
        value=pre*sigmoid;jac=derivative[:,None]*(w.T@jac)
    w,b=p['net'][-1]
    return value@w+b+z@p['skip'],w.T@jac+p['skip'].T

def audit(out,destination):
    out=Path(out);record=json.loads((out/'result.json').read_text())
    assert record['complete'] and not record['final_cohort_opened']
    assert record['backend']=='gpu' and record['x64'] and record['matmul_precision']=='highest'
    archive=next(p for p in out.parents if (p/'PROVENANCE.json').exists())
    log=(archive/'job.out').read_text();err=(archive/'job.err').read_text()
    assert 'jax_backend=gpu' in log and 'run_exit=0' in log
    assert not re.search(r'captur\w*.{0,30}large.{0,30}constant|RESOURCE_EXHAUSTED|out.of.memory|No space left',log+'\n'+err,re.I)
    provenance=json.loads((archive/'PROVENANCE.json').read_text());assert provenance['source_commit']==record['source_commit']
    for row in provenance['files']:
        blob=subprocess.check_output(['git','show',f'{record["source_commit"]}:{row["path"]}'])
        assert hashlib.sha256(blob).hexdigest()==row['sha256'] and (archive/row['staged']).read_bytes()==blob
    a=np.load(out/'coordinates.npz');truth=np.load(out/'reference.npz')['validation'];bank=a['bank'];matrix=a['validation_matrix']
    from audit_panel import reference
    cfg=record['config'];cohorts=json.loads((out/'cohorts.json').read_text());n=cfg['train_intervals']
    params={}
    for label in ('train','validation'):
        params[label]=np.random.default_rng(cfg[label+'_seed']).random((cfg[label+'_count'],5))*[.3,.3,.3,.05,.4]+[.35,.35,.35,.10,.8]
        key='training_parameters' if label=='train' else 'validation_parameters'
        np.testing.assert_array_equal(params[label],cohorts[key])
    reference_difference=0.;training_coordinate_difference=0.
    for case,parameters in enumerate(params['validation']):
        expected=reference(n,parameters,cfg,False);reference_difference=max(reference_difference,float(np.max(np.abs(expected-truth[case]))))
        np.testing.assert_allclose(expected,truth[case],rtol=2e-11,atol=2e-13)
    for case,parameters in enumerate(params['train']):
        values=reference(n,parameters,cfg,False).reshape(len(cfg['times']),-1);target=values@bank
        selected=a['training_target'][case*len(values):(case+1)*len(values)]
        training_coordinate_difference=max(training_coordinate_difference,float(np.max(np.abs(target-selected))))
        np.testing.assert_allclose(target,selected,rtol=2e-10,atol=2e-11)
    failures=[];fields=0;states=0;gradient_max=0.;reconstruction_max=0.
    for candidate in record['candidates']:
        k=candidate['info']['k'];directory=(out/candidate['checkpoint_relative_path']).parent
        saved=pickle.loads((out/candidate['checkpoint_relative_path']).read_bytes());p=saved['params']
        if candidate['info'].get('initialization',{}).get('kind')=='weighted_pca_linear_skip':
            initial=pickle.loads((directory/f'head_K{k}_initial.pkl').read_bytes());ini=initial['initialization']
            target=a['training_target'];mean=np.average(target,axis=0,weights=1/a['training_norm2'])
            np.testing.assert_allclose(ini['mean'],mean,rtol=2e-12,atol=2e-12)
            axes=np.asarray(ini['axes']);scale=np.asarray(ini['code_scale'])
            np.testing.assert_allclose(initial['codes'],((target-mean)@axes)/scale,rtol=2e-11,atol=2e-11)
            np.testing.assert_allclose(initial['params']['skip'],scale[:,None]*axes.T,rtol=1e-12,atol=1e-12)
        if candidate['newly_trained']:
            curve=json.loads((directory/f'head_K{k}_curve.json').read_text());assert curve[0]['step']==0
            expected=min(row['validation_evolved_worst'] for row in curve if 'validation_evolved_worst' in row)
            np.testing.assert_allclose(candidate['info']['validation_evolved_worst'],expected,rtol=1e-12)
        for fit in candidate['fits']:
            path=out/fit['field_file']
            data=np.load(path);prediction=data['prediction'];stats=data['stats'];latents=data['latents']
            for case in range(len(truth)):
                pflat=prediction[case].reshape(len(truth[case]),-1);tflat=truth[case].reshape(len(truth[case]),-1)
                errors=np.linalg.norm(pflat-tflat,axis=1)/np.linalg.norm(tflat,axis=1)
                np.testing.assert_allclose(errors,fit['metrics'][case]['current_by_time'],rtol=2e-11,atol=2e-13);fields+=1
                for t,z in enumerate(latents[case]):
                    coefficient,jacobian=head(p,z);reconstructed=bank@coefficient
                    reconstruction_max=max(reconstruction_max,float(np.max(np.abs(reconstructed-pflat[t]))))
                    np.testing.assert_allclose(reconstructed,pflat[t],rtol=2e-10,atol=2e-12)
                    target=a['validation_target'][case*len(truth[case])+t];scale=max(np.linalg.norm(target),1e-14)
                    residual=(matrix@coefficient-target)/scale;jac=(matrix@jacobian)/scale
                    gradient=float(np.linalg.norm(jac.T@residual)/max(np.linalg.norm(jac),1e-30))
                    gradient_max=max(gradient_max,abs(gradient-stats[case,t,4]));states+=1
                    np.testing.assert_allclose(gradient,stats[case,t,4],rtol=2e-6,atol=2e-11)
                    if stats[case,t,2]==1:assert gradient<=record['config']['representation_fit_tolerance']+2e-11
    result=dict(passed=True,complete=record['complete'],final_cohort_opened=False,checked_fields=fields,checked_fit_states=states,
        reconstruction_max_absolute_difference=reconstruction_max,gradient_max_absolute_difference=gradient_max,
        source_files_verified=len(provenance['files']),independent_reference_max_absolute_difference=reference_difference,
        independent_training_coordinate_max_absolute_difference=training_coordinate_difference,
        scope='independent NumPy/SciPy PCA initialization, selected-checkpoint field errors, decoded fields and analytic normalized fitting gradients; operator diagnostics await a full paired panel')
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--destination',required=True);a=p.parse_args();audit(a.out,a.destination)
