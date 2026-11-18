"""Frozen-checkpoint nonlinear projection audit with an exact dense LM system."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint,summary
from train_shared_child import load_cache
import shared_child_head as H

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--cache',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    model=Path(a.model);rec=json.loads((model/'result.json').read_text());assert rec['status']=='complete_shared_child_representation'
    assert digest(model/'shared_child.pkl')==rec['model_sha256'];m=read_checkpoint(model/'shared_child.pkl')
    cache=Path(a.cache);assert digest(cache/'result.json')==m['cache_sha256'];d,labels=load_cache(cache,'dev',out)
    p=m['params'];scale=jnp.asarray(m['scale']);project=jax.jit(jax.vmap(H.make_project(250,linear_solver='dense'),in_axes=(None,0,0,0,0,None,None,None,None,None)))
    rows=[];errs=[];full=[];codes=[]
    result=dict(meta=meta,status='auditing_frozen_child_projection',model_sha256=rec['model_sha256'],latent=m['latent'],
                active_variables=128+52*m['latent'],dev_labels=labels,linear_solver='dense',max_iterations_per_start=250,
                model_weights_changed=False,final_test_opened=False,cases=rows)
    for i,label in enumerate(labels):
        ze=H.encode(p['encoder'],d['teacher'][i,2],d['geometry'][i],d['h'][i],scale)
        def fit(z):return project(d['S'][i],d['b'][i],d['constant'][i],d['norm2'][i],z,p['head'],d['geometry'][i],d['h'][i],scale,d['mask'][i,2])
        candidates=[fit(jnp.asarray(m['dev']['codes'][i])),fit(ze),fit(jnp.zeros_like(ze))]
        choice=jnp.argmin(jnp.stack([v[1] for v in candidates]),axis=0)
        z=jnp.stack([v[0] for v in candidates])[choice,jnp.arange(11)]
        candidates.append(fit(jnp.concatenate([z[:1],z[:-1]],axis=0)))
        choice=jnp.argmin(jnp.stack([v[1] for v in candidates]),axis=0)
        z,e,s,n=[jnp.stack([v[k] for v in candidates])[choice,jnp.arange(11)] for k in range(4)]
        c=H.head(p['head'],z,d['geometry'][i],d['h'][i],scale).reshape(11,-1)
        root=d['Ainvb'][i]-c@d['AinvB'][i].T
        codes.append(np.asarray(z));full.append(np.asarray(jnp.concatenate([root,c],axis=1)));errs.append(np.asarray(e))
        rows.append(dict(case=label,error=np.asarray(e),stationarity=np.asarray(s),accepted=np.asarray(n),start_errors=[np.asarray(v[1]) for v in candidates]))
        save(out/'result.json',result);print('dense_child_refit',label,float(e.max()),float(s.max()),flush=True)
    result['projection']=summary(np.stack(errs));result['worst_projection']=result['projection']['worst']
    result['max_stationarity']=max(float(np.max(r['stationarity'])) for r in rows)
    result['fit_sha256']=checkpoint(out/'refined_fits.pkl',dict(codes=np.stack(codes),coefficients=np.stack(full),labels=labels))
    result['status']='complete_frozen_child_projection_audit';result['representation_gate']=result['worst_projection']<=.005
    save(out/'result.json',result)

if __name__=='__main__':main()
