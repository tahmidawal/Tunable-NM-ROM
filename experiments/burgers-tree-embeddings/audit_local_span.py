"""Independent dense scoring of the promising analytical span, no code refitting."""
import argparse
from common import *
import data,local_polynomial as LP
from training_io import read_checkpoint,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    for p in ['dev','span','fits']:ap.add_argument('--'+p,required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    record=json.loads((Path(a.fits)/'result.json').read_text());assert digest(Path(a.fits)/'full_fits.pkl')==record['full_fit_sha256']
    fits=read_checkpoint(Path(a.fits)/'full_fits.pkl');basis=np.load(Path(a.span)/'pod_basis.npz')['coefficients'][:128]
    raw=json.loads((Path(a.dev)/'result.json').read_text());refs={r['case']:r for r in raw['cases']}
    side=record['chosen_config'][0];result=dict(meta=preflight(),status='running',reference_assisted=True,PDE_solve=False,
         active_variables=len(fits[0]['coefficients'][0]),same_codes_at_all_resolutions=True,scoring={})
    errors_by_grid={}
    for n in (128,256,512):
        axis,weights=np.polynomial.legendre.leggauss(n);axis=(axis+1)/2;weights=weights/2;w=np.outer(weights,weights)
        errs=[]
        for row in fits:
            p=Path(a.dev)/refs[row['case']]['levels'][-1]['file'];assert digest(p)==refs[row['case']]['levels'][-1]['sha256']
            b=jnp.asarray(np.load(p)['coefficients']);ref=data.synth_grid(b,axis)
            pred=LP.field_grid(basis,jnp.asarray(row['coefficients']),row['cells'],row['transforms'],side,axis)
            err=jnp.sqrt(jnp.sum((pred-ref)**2*w,axis=(1,2))/jnp.sum(ref**2*w,axis=(1,2)))
            errs.append(np.asarray(err))
        errs=np.asarray(errs);errors_by_grid[n]=errs;result['scoring'][str(n)]=summary(errs)
        save(out/'result.json',result);print('grid',n,float(errs.max()),flush=True)
    result['max_256_512_error_change']=float(np.max(np.abs(errors_by_grid[512]-errors_by_grid[256])))
    result['reference_scoring_qualification']=result['max_256_512_error_change']<.0002
    ii,tt=np.unravel_index(np.argmax(errors_by_grid[512]),errors_by_grid[512].shape);row=fits[ii]
    dense={};refrow=refs[row['case']];bs=np.load(Path(a.dev)/refrow['levels'][-1]['file'])['coefficients']
    for n in (2048,4096):
        axis=np.linspace(0,1,n);pred=LP.field_grid(basis,jnp.asarray(row['coefficients'][tt]),row['cells'],row['transforms'],side,axis)[0]
        ref=data.synth_grid(jnp.asarray(bs[tt]),axis);delta=pred-ref
        rel=float(jnp.linalg.norm(delta)/jnp.linalg.norm(ref));dense[str(n)]=dict(relative_l2=rel,shape=list(pred.shape),
            relative_linf=float(jnp.max(jnp.abs(delta))/jnp.max(jnp.abs(ref))),peak_index=np.unravel_index(int(jnp.argmax(jnp.abs(delta))),pred.shape))
        if n==4096:np.savez_compressed(out/'actual4096.npz',prediction=np.asarray(pred),reference=np.asarray(ref),case=row['case'],time=data.TIMES[tt])
    result['dense_witness']=dict(case=row['case'],time=data.TIMES[tt],grids=dense,change=abs(dense['4096']['relative_l2']-dense['2048']['relative_l2']))
    result['representation_gate']=errors_by_grid[512].max()<=.005 and result['reference_scoring_qualification'] and dense['4096']['relative_l2']<=.005
    result['status']='complete_local_span_scoring';save(out/'result.json',result)
    print('representation_gate',result['representation_gate'],flush=True)
if __name__=='__main__':main()
