"""Independent NumPy reconstruction of validation-only diagnostic inputs/bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def draw(seed,n):
    rng=np.random.default_rng(seed)
    return np.stack([rng.uniform(.15,.85,n),rng.uniform(.15,.85,n),rng.uniform(.05,.20,n),
                     rng.uniform(.5,2.,n),np.exp(rng.uniform(np.log(.01),np.log(.1),n))],axis=1)


def initial(parameters,L):
    x=np.arange(L+1)/L;cx,cy,width,amplitude,_=parameters
    u=amplitude*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*width**2))
    u[[0,-1],:]=0;u[:,[0,-1]]=0
    return u


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic',required=True);ap.add_argument('--fields',required=True)
    args=ap.parse_args();out=Path(args.diagnostic);fields=Path(args.fields)
    report=json.loads((out/'diagnosis.json').read_text());config=json.loads((out/'config.json').read_text())
    training=draw(config['seeds']['train'],64);validation=draw(config['seeds']['validation'],16)
    results=[]
    for row in report['rows']:
        name=f'validation_{row["method"]}_cap10_dt0.00125_q4_tol0.0001_L256_case{row["case"]}.npz'
        artifact=fields/name;assert sha(artifact)==row['input_artifact_sha256']
        data=np.load(artifact);truth=data['truth_u'];L=truth.shape[1]-1
        np.testing.assert_allclose(initial(validation[row['case']],L),truth[0],rtol=0,atol=1e-15)
        norm=np.linalg.norm(truth[0]);dist=np.asarray([np.linalg.norm(initial(p,L)-truth[0])/norm for p in training])
        centers=np.linalg.norm(training[:,:2]-validation[row['case'],:2],axis=1)
        result=dict(method=row['method'],case=row['case'],input_hash_verified=True,seeded_initial_field_verified=True,
                    physical_parameters=validation[row['case']].tolist(),parameter_columns=['center_x','center_y','width','amplitude','viscosity'],
                    nearest_training_initial_fields=[dict(training_case=int(i),relative_field_distance=float(dist[i]),parameters=training[i].tolist()) for i in np.argsort(dist)[:5]],
                    nearest_center_distance=float(centers.min()),nearest_center_training_case=int(centers.argmin()),
                    training_width_range=[float(training[:,2].min()),float(training[:,2].max())])
        if row['method']=='cp':
            ckpath=out/'checkpoints/cp.pkl';assert sha(ckpath)==row['checkpoint_sha256']
            with ckpath.open('rb') as f:ck=pickle.load(f)
            factors=np.asarray(ck['params']['factors']);G=np.einsum('ri,rj->ijr',factors[0,0],factors[1,0])
            G[[0,-1],:]=0.;G[:,[0,-1]]=0.;G=G.reshape(-1,64)
            mask=np.ones((L+1,L+1));mask[[0,-1],:]=0.;mask[:,[0,-1]]=0.
            offset=float(ck['params']['bias'][0])*mask.ravel()
            coefficients,_,rank,sv=np.linalg.lstsq(G,truth.reshape(6,-1).T-offset[:,None],rcond=1e-12)
            residual=G@coefficients+offset[:,None]-truth.reshape(6,-1).T
            err=np.linalg.norm(residual,axis=0)/norm
            np.testing.assert_allclose(err,row['unconstrained_CP_spatial_span']['errors'],rtol=1e-12,atol=1e-12)
            np.savez_compressed(out/'cp_case3_free_span_projection.npz',coefficients=coefficients,singular_values=sv,
                                residual=residual.T.reshape(truth.shape),errors=err,rank=rank,rcond=1e-12)
            result['independent_cp_span']=dict(errors=err.tolist(),rank=int(rank),rcond=1e-12,
                normalized_orthogonality=float(np.linalg.norm(G.T@residual)/(np.linalg.norm(G)*np.linalg.norm(residual))),
                coefficients_and_full_residual='cp_case3_free_span_projection.npz')
        results.append(result)
    summary=dict(passed=True,kind='independent NumPy input/hash/CP-span audit; validation only; no query changes',
                 diagnostic_sha256=sha(out/'diagnosis.json'),cases=results)
    (out/'independent_diagnostic_audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
