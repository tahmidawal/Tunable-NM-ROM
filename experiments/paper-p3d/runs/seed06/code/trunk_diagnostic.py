"""Independent NumPy/SciPy projection into one frozen DeepONet trunk.

This is a representation diagnostic, not an online predictor: coefficients use
the true solution. It never generates final-cohort parameters. No JAX is used.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.fft import dstn
from scipy.linalg import qr
from audit import family, forcing


def trunk(params, coordinates, spec):
    features=[coordinates]
    for frequency in spec.get('trunk_frequencies',[1.,2.,4.]):
        features.extend([np.sin(np.pi*frequency*coordinates),np.cos(np.pi*frequency*coordinates)])
    value=np.concatenate(features,axis=-1)
    for layer in params['trunk'][:-1]:value=np.tanh(value@layer['w']+layer['b'])
    layer=params['trunk'][-1]
    return value@layer['w']+layer['b']


def diagnose(checkpoint,cfg):
    saved=pickle.loads(checkpoint.read_bytes());spec=saved['spec'];params=saved['params']
    assert spec['kind']=='deeponet3d' and not cfg['final_cohort_opened']
    n=cfg['train_intervals'];a=np.arange(1,n,dtype=np.float64)/n*2-1
    coordinates=np.stack(np.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)
    # The fixed learned bias adds an affine offset. Its scale is recovered only
    # from training solutions, identically to the PDE adapter's normalization.
    k=np.arange(1,n);one=4*n*n*np.sin(np.pi*k/(2*n))**2
    lam=one[:,None,None]+one[None,:,None]+one[None,None,:]
    def solutions(role):
        return np.stack([dstn(dstn(forcing(n,p),type=1,norm='ortho')/lam,type=1,norm='ortho').reshape(-1)
            for p in family(cfg[role+'_seed'],cfg[role+'_count'])])
    training=solutions('train');development=solutions('validation')
    output_scale=float(np.sqrt(np.mean(training**2)))
    matrix=trunk(params,coordinates,spec)/np.sqrt(spec['rank'])
    q,r,pivot=qr(matrix,mode='economic',pivoting=True)
    threshold=np.finfo(np.float64).eps*max(matrix.shape)*abs(r[0,0])
    rank=int(np.count_nonzero(abs(np.diag(r))>threshold));q=q[:,:rank]
    bias=float(np.asarray(params['bias']).reshape(-1)[0])*output_scale
    result={}
    for role,truth in [('training',training),('development',development)]:
        target=truth-bias;coefficients=target@q
        prediction=coefficients@q.T+bias
        errors=np.linalg.norm(prediction-truth,axis=1)/np.linalg.norm(truth,axis=1)
        result[role]=dict(cases=len(errors),errors=errors.tolist(),worst=float(errors.max()),
            median=float(np.median(errors)),mean=float(errors.mean()))
    result.update(schema='poisson3d-frozen-deeponet-trunk-projection-v1',intervals=n,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),selected_step=int(saved['step']),
        nominal_rank=spec['rank'],numerical_rank=rank,rank_threshold=float(threshold),
        training_output_scale=output_scale,final_cohort_generated=False,
        interpretation='Exact least-squares free coefficients in the frozen learned trunk with fixed trained bias; representation diagnostic, not online prediction or a global architecture floor.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    result=diagnose(Path(a.checkpoint),json.loads(Path(a.config).read_text()))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['training','development']}))
    print(json.dumps({role:{k:v for k,v in result[role].items() if k!='errors'} for role in ['training','development']}))
