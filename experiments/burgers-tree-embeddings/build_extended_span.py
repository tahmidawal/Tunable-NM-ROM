"""Training-only spectral span for the expanded cohort, with explicit original panels."""
import argparse
from common import *
from build_local_cache import cohort
from training_io import summary

def read(dirs):
    coeff=[];norm=[];labels=[];params=[];sources=[]
    for folder,r in cohort(dirs):
        path=folder/r['levels'][-1]['file'];assert digest(path)==r['levels'][-1]['sha256']
        b=np.load(path)['coefficients'];coeff.append(b[:,:128,:128].reshape(11,-1))
        norm.append(np.linalg.norm(b.reshape(11,-1),axis=1)*.5);labels.append(r['case']);params.append(r['parameters'])
        sources.append(dict(case=r['case'],file=str(path),sha256=digest(path)))
    return jnp.asarray(np.concatenate(coeff)),jnp.asarray(np.concatenate(norm)),labels,params,sources

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--train',required=True);ap.add_argument('--dev',required=True);ap.add_argument('--crosscheck',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    cross=json.loads((Path(a.crosscheck)/'result.json').read_text());assert cross['passed'] and len(cross['cases'])==4
    A,tn,tl,tp,ts=read(a.train);B,vn,vl,vp,vs=read(a.dev);assert len(tl)==128 and len(vl)==44
    _,s,vh=jax.jit(lambda z:jnp.linalg.svd(z,full_matrices=False))(A/(2*tn[:,None]));basis=vh[:768]
    Ct=A@basis.T*.5;Cd=B@basis.T*.5;controls={}
    for rank in (16,32,64,128,256,384,512,768):
        residual=B-2*Cd[:,:rank]@basis[:rank]
        tail=jnp.maximum(vn**2-jnp.sum(B*B,axis=1)*.25,0)
        error=jnp.sqrt(jnp.sum(residual*residual,axis=1)*.25+tail)/vn;controls[str(rank)]=summary(error.reshape(-1,11))
    np.savez_compressed(out/'pod_basis.npz',coefficients=2*np.asarray(basis).reshape(768,128,128),singular_values=np.asarray(s))
    np.savez_compressed(out/'reduced_data.npz',train_coefficients=np.asarray(Ct),dev_coefficients=np.asarray(Cd),train_norm=np.asarray(tn),dev_norm=np.asarray(vn),
        train_missing=np.asarray(jnp.maximum(tn**2-jnp.sum(Ct*Ct,axis=1),0)),dev_missing=np.asarray(jnp.maximum(vn**2-jnp.sum(Cd*Cd,axis=1),0)),
        train_parameters=np.asarray(tp),dev_parameters=np.asarray(vp))
    result=dict(meta=meta,status='complete_extended_POD_span',basis_sha256=digest(out/'pod_basis.npz'),reduced_data_sha256=digest(out/'reduced_data.npz'),
        controls=controls,train_labels=tl,dev_labels=vl,training_sources=ts,dev_sources=vs,reference_crosscheck_sha256=digest(Path(a.crosscheck)/'result.json'),final_test_opened=False)
    save(out/'result.json',result);print('expanded_POD',{k:v['worst'] for k,v in controls.items()},flush=True)
if __name__=='__main__':main()
