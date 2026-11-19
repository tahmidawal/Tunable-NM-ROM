"""Bounded deterministic optimization of the unchanged training-only pilot."""
import argparse
from common import *
from neural import stable as S
from neural.build_data import resolve
from training_io import read_checkpoint,checkpoint
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/'b7parentpilot/output/result.json',ns);r=json.loads(rp.read_text());source=rp.parent/'stable_parent.pkl'
    assert digest(source)==r['selected_sha256'];warm=read_checkpoint(source);ids=np.asarray(warm['pilot_indices'])
    parent=resolve(ns/'b6parent/output/neural_parent.pkl',ns);dp=parent.parent/'coordinate_data.npz';assert digest(dp)==warm['data_sha256']
    d=np.load(dp);xy,w,y,norm=[jnp.asarray(v) for v in [d['xy'],d['w'],d['train_y'][ids],d['train_norm'][ids]]]
    z=jnp.asarray(warm['training_codes'])[ids];x0,unravel=ravel_pytree(dict(network=warm['network'],codes=z))
    @jax.jit
    def objective(x,xy,w,y,norm):
        def loss(x):
            p=unravel(x);pred=S.apply(p['network'],p['codes'][:,None,:],xy[None])
            return jnp.mean(jnp.sum(w[None]*((pred-y)/norm[:,None])**2,axis=1))
        return jax.value_and_grad(loss)(x)
    @jax.jit
    def errors(x,xy,w,y,norm):
        p=unravel(x);pred=S.apply(p['network'],p['codes'][:,None,:],xy[None])
        return jnp.sqrt(jnp.sum(w[None]*(pred-y)**2,axis=1))/norm
    xf,wf=data.grid(256,True);xf,wf=jnp.asarray(xf),jnp.asarray(wf);truth=[]
    for i in ids:
        src=warm['data_record']['groups']['train']['sources'][i//11];p=resolve(src['file'],ns);assert digest(p)==src['sha256']
        b=jnp.asarray(np.load(p)['coefficients'][i%11]);truth.append(data.synth_grid(b,jnp.unique(xf[:,0])).reshape(-1))
    yf=jnp.stack(truth);start=time.monotonic();counter=0;best=np.inf
    result=dict(meta=meta,status='deterministic_pilot_polish',source_sha256=digest(source),final_test_opened=False,
        representation_admitted=False,pde_admitted=False,development_evaluated=False,history=[],pilot_indices=ids,
        optimizer='L-BFGS-B, no bounds, full128² physical quadrature, unchanged weights/code architecture and objective',
        max_iterations=3000,max_seconds=1500)
    def fun(x):
        value,g=objective(jnp.asarray(x),xy,w,y,norm)
        return float(value),np.asarray(g,dtype=np.float64)
    def record(x,it):
        nonlocal best
        fine=np.asarray(errors(jnp.asarray(x),xf,wf,yf,norm));coarse=np.asarray(errors(jnp.asarray(x),xy,w,y,norm))
        value,g=fun(x);row=dict(update=it,loss=value,pilot_worst=float(fine.max()),pilot_median=float(np.median(fine)),
            pilot_scoring_change=float(np.max(abs(fine-coarse))),max_gradient=float(np.max(abs(g))),elapsed_seconds=time.monotonic()-start)
        result['history'].append(row)
        if row['pilot_worst']<best:
            best=row['pilot_worst'];p=unravel(jnp.asarray(x));codes=jnp.asarray(warm['training_codes']).at[ids].set(p['codes'])
            payload=dict(warm,network=p['network'],training_codes=codes,metrics=row)
            result['selected_sha256']=checkpoint(out/'stable_parent.pkl',payload);result['selected_update']=it
            result['pilot_pass']=best<=.005 and row['pilot_scoring_change']<=.0002
        save(out/'result.json',result);print('deterministic_parent_polish',row,flush=True)
    record(np.asarray(x0),0)
    def callback(x):
        nonlocal counter
        counter+=1
        if counter%100==0:
            record(x,counter)
            if result.get('pilot_pass') or time.monotonic()-start>1500:raise StopIteration
    opt=minimize(fun,np.asarray(x0),jac=True,method='L-BFGS-B',callback=callback,
        options=dict(maxiter=3000,maxls=30,maxcor=30,ftol=1e-15,gtol=1e-10))
    record(opt.x,counter)
    result.update(status='complete_deterministic_pilot_polish',optimizer_status=int(opt.status),optimizer_message=str(opt.message),
        iterations=int(opt.nit),function_evaluations=int(opt.nfev),worst_training_error=best)
    save(out/'result.json',result)


if __name__=='__main__':main()
