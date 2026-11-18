"""Bounded richer local-span diagnostics; no neural/PDE performance claim."""
import argparse
from common import *
import data,local_polynomial as LP
from training_io import *
from evaluate import finer_dataset

CONFIGS=[(4,28,3,'initial'),(6,28,4,'initial'),(6,28,4,'swept'),
         (8,28,4,'initial'),(8,28,4,'swept'),(6,52,4,'swept')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--dev',required=True);ap.add_argument('--span',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    D=finer_dataset(a.dev,load_reference(a.dev,'dev'),128);x,w,Y=D['xy'],D['w'],D['Y']
    base=np.load(Path(a.span)/'pod_basis.npz')['coefficients'][:128];axis=jnp.unique(x[:,0],size=128)
    G0=data.synth_grid(jnp.asarray(base),axis).reshape(128,-1).T
    result=dict(meta=meta,status='screening',classification='analytical shared-feature span control',screen_cases=[5,16,20,21],
                root_variables=128,scoring=128,configurations=[],final_test_opened=False)
    def fit_case(i,config):
        side,nodes,depth,policy=config;cs=LP.tree(x,w,D['parameters'][i],nodes,depth,policy)
        B,tr=LP.local_blocks(x,w,cs,side);G=jnp.concatenate([G0,B],axis=1);sl=slice(i*11,(i+1)*11)
        c,p,p8=LP.ridge_projection(G,Y[sl],w)
        err=errors(p,Y[sl],w);err8=errors(p8,Y[sl],w)
        assert bool(jnp.all(jnp.isfinite(err)))
        return dict(case=D['labels'][i],errors=np.asarray(err),ridge_sensitivity=float(jnp.max(jnp.abs(err-err8))),
                    variables=G.shape[1],cells=cs,transforms=np.asarray(tr),coefficients=np.asarray(c),prediction=p)
    for config in CONFIGS:
        rows=[fit_case(i,config) for i in result['screen_cases']]
        score=max(np.max(r['errors']) for r in rows)
        row=dict(config=config,worst=score,cases=[{k:v for k,v in r.items() if k not in ['transforms','coefficients','prediction']} for r in rows])
        result['configurations'].append(row);save(out/'result.json',result);print('screen',config,score,flush=True)
    chosen=min(result['configurations'],key=lambda r:r['worst'])['config'];result['chosen_config']=chosen
    full=[];saved=[]
    for i in range(len(D['parameters'])):
        row=fit_case(i,chosen);saved.append({k:v for k,v in row.items() if k!='prediction'})
        full.append(row['errors']);print('full',i,float(np.max(row['errors'])),flush=True)
    result['full_projection']=summary(np.asarray(full));result['max_ridge_sensitivity']=max(r['ridge_sensitivity'] for r in saved)
    result['full_fit_sha256']=checkpoint(out/'full_fits.pkl',saved)
    result['status']='complete_richer_local_span';save(out/'result.json',result)

if __name__=='__main__':main()
