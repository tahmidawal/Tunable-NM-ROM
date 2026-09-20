"""Independent NumPy/SciPy reconstruction, field-error and fitting-gradient audit."""
import argparse,json,pickle
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
    a=np.load(out/'coordinates.npz');truth=np.load(out/'reference.npz')['validation'];bank=a['bank'];matrix=a['validation_matrix']
    failures=[];fields=0;states=0;gradient_max=0.;reconstruction_max=0.
    for candidate in record['candidates']:
        k=candidate['info']['k'];new=candidate['name'].startswith('pca_')
        directory=out/f'pca_K{k}' if new else out
        saved=pickle.loads((directory/f'head_K{k}.pkl').read_bytes());p=saved['params']
        if new:
            initial=pickle.loads((directory/f'head_K{k}_initial.pkl').read_bytes());ini=initial['initialization']
            target=a['training_target'];mean=np.average(target,axis=0,weights=1/a['training_norm2'])
            np.testing.assert_allclose(ini['mean'],mean,rtol=2e-12,atol=2e-12)
            axes=np.asarray(ini['axes']);scale=np.asarray(ini['code_scale'])
            np.testing.assert_allclose(initial['codes'],((target-mean)@axes)/scale,rtol=2e-11,atol=2e-11)
            np.testing.assert_allclose(initial['params']['skip'],scale[:,None]*axes.T,rtol=1e-12,atol=1e-12)
            curve=json.loads((directory/f'head_K{k}_curve.json').read_text());assert curve[0]['step']==0
            expected=min(row['validation_evolved_worst'] for row in curve if 'validation_evolved_worst' in row)
            np.testing.assert_allclose(candidate['info']['validation_evolved_worst'],expected,rtol=1e-12)
        for fit in candidate['fits']:
            path=directory/f'validation_starts{fit["starts"]}.npz' if new else out/f'old_K{k}_starts{fit["starts"]}.npz'
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
        scope='independent NumPy/SciPy PCA initialization, selected-checkpoint field errors, decoded fields and analytic normalized fitting gradients; operator diagnostics await a full paired panel')
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--destination',required=True);a=p.parse_args();audit(a.out,a.destination)
