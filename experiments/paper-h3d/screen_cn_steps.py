"""Development-only SciPy modal screen for useful full-grid CN time steps.

This measures time-discretization error only, with no GPU or timing claim.
"""
import json
from pathlib import Path
import numpy as np
from scipy.fft import dstn,idstn


def main():
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/'developmentA.json').read_text())
    assert cfg['evaluation_cohort']=='development'
    params=np.random.default_rng(cfg['validation_seed']).random((cfg['validation_count'],5))*[.3,.3,.3,.05,.4]+[.35,.35,.35,.10,.8]
    checks=[]
    for n in cfg['evaluation_intervals']:
        x=np.arange(1,n)/n;k=np.arange(1,n)
        eig=4*n*n*np.sin(np.pi*k/(2*n))**2
        lam=eig[:,None,None]+eig[None,:,None]+eig[None,None,:]
        for dt in [cfg['dt'],2*cfg['dt'],4*cfg['dt']]:
            ticks=np.rint(np.asarray(cfg['times'])/dt).astype(int)
            assert np.max(np.abs(ticks*dt-cfg['times']))<1e-12
            errors=[]
            for p in params:
                factors=[x*(1-x)*np.exp(-((x-p[j])/p[3])**2/2) for j in range(3)]
                initial=64*p[4]*factors[0][:,None,None]*factors[1][None,:,None]*factors[2][None,None,:]
                coeff=dstn(initial,type=1,norm='ortho');alpha=cfg['diffusivity']*dt/2
                factor=(1-alpha*lam)/(1+alpha*lam)
                truth=np.stack([idstn(coeff*np.exp(-cfg['diffusivity']*t*lam),type=1,norm='ortho') for t in cfg['times'][1:]])
                pred=np.stack([idstn(coeff*factor**tick,type=1,norm='ortho') for tick in ticks[1:]])
                count=len(truth)
                errors.append(float(np.max(np.linalg.norm((pred-truth).reshape(count,-1),axis=1)/np.linalg.norm(truth.reshape(count,-1),axis=1))))
            checks.append(dict(intervals=n,dt=dt,worst_relative_error=max(errors),median_relative_error=float(np.median(errors)),errors=errors))
    result=dict(scope='development-only independent SciPy modal CN time-discretization screen; no CG iterations or timing measured; final parameters unopened',validation_seed=cfg['validation_seed'],cases=len(params),checks=checks)
    (root/'smokes/cn-development-step-screen.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in checks:print(row['intervals'],row['dt'],row['worst_relative_error'])


if __name__=='__main__':main()
