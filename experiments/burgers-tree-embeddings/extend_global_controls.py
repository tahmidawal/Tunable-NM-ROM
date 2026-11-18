"""Global variable counts comparable to the qualified tree, plus full Fourier spans."""
import argparse
from common import *
from build_extended_span import read
from training_io import summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--train',required=True)
    ap.add_argument('--dev',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    A,tn,tl,tp,ts=read(a.train);B,vn,vl,vp,vs=read(a.dev);assert len(tl)==128 and len(vl)==44
    _,s,vh=jax.jit(lambda z:jnp.linalg.svd(z,full_matrices=False))(A/(2*tn[:,None]));C=B@vh.T*.5
    tail=jnp.maximum(vn**2-jnp.sum(B*B,axis=1)*.25,0);controls={};fourier={}
    for k in (768,960,1024,1280,1376,1408):
        residual=B-2*C[:,:k]@vh[:k]
        err=jnp.sqrt(.25*jnp.sum(residual*residual,axis=1)+tail)/vn
        controls[str(k)]=dict(projection=summary(err.reshape(-1,11)),variables=k,
                             last_training_singular_value=float(s[k-1]),relative_training_singular_value=float(s[k-1]/s[0]))
    for side in (24,32,36,40,48,64):
        b=B.reshape(-1,128,128);keep=b[:,:side,:side];rest=b.at[:,:side,:side].set(0.)
        err=jnp.sqrt(.25*jnp.sum(rest*rest,axis=(1,2))+tail)/vn
        fourier[str(side)]=dict(projection=summary(err.reshape(-1,11)),variables=side*side,
                               classification='complete tensor sine span, no learned weights or nonlinear compression')
    np.savez_compressed(out/'global_POD_full.npz',coefficients=2*np.asarray(vh).reshape(len(vh),128,128),singular_values=np.asarray(s))
    result=dict(meta=meta,status='complete_extended_global_controls',POD=controls,complete_sine=fourier,
                dev_labels=vl,train_labels=tl,training_sources=ts,dev_sources=vs,basis_sha256=digest(out/'global_POD_full.npz'),
                normalization='current-time reference physical L2',final_test_opened=False,reference_assisted=True,PDE_solve=False)
    save(out/'result.json',result)
    print('global_POD',{k:v['projection']['worst'] for k,v in controls.items()},flush=True)
    print('complete_sine',{k:v['projection']['worst'] for k,v in fourier.items()},flush=True)

if __name__=='__main__':main()
