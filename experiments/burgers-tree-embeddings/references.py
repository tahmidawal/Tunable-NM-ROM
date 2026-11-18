"""Fresh spectral ladders. This reference integrator is NOT the online ROM operator."""
import argparse, time
from pathlib import Path
from functools import lru_cache
from common import *
import data
from vendor import pscore as P

SETTINGS=[(128,.00015625),(256,.00015625),(512,.000625),(512,.0003125),(512,.00015625)]

@lru_cache(None)
def trajectory(N,dt):
    # Adapted from verified continuous.full_trajectory; explicit large coefficients.
    co=P.etdrk4_coefs(P.lin_eig(N,2,data.NU),dt,32)
    @jax.jit
    def run(b0,co):
        def step(v):
            nv,_=P.nonlinear(v,N,2); a=co['E2']*v+co['Q']*nv
            na,_=P.nonlinear(a,N,2); b=co['E2']*v+co['Q']*na
            nb,_=P.nonlinear(b,N,2); c=co['E2']*a+co['Q']*(2*nb-nv)
            nc,_=P.nonlinear(c,N,2)
            return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
        def block(v,_):
            v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v)
            return v,v
        _,v=jax.lax.scan(block,b0,None,length=10)
        return jnp.concatenate([b0[None],v])
    return lambda b:run(b,co)

def relative_difference(lo,hi):
    q=np.zeros_like(hi); q[:,:lo.shape[1],:lo.shape[2]]=lo
    return np.linalg.norm((q-hi).reshape(11,-1),axis=1)/np.linalg.norm(hi.reshape(11,-1),axis=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--group',choices=['train','dev'],required=True)
    ap.add_argument('--limit',type=int); a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    params=data.parameters(a.group)
    labels=[f'{a.group}{i:03d}' for i in range(len(params))]
    if a.group=='dev':
        params=np.concatenate([params,data.alignment(),data.fixtures()])
        labels += [f'align{i:03d}' for i in range(12)]+['fixture000','fixture001']
    if a.limit:params=params[:a.limit];labels=labels[:a.limit]
    np.savez(out/'parameters.npz',parameters=params,labels=labels,times=data.TIMES)
    result=dict(meta=preflight(),group=a.group,normalization='current-time reference L2',
                settings=SETTINGS,cases=[],status='running',final_test_opened=False)
    save(out/'result.json',result)
    for label,p in zip(labels,params):
        levels=[]; records=[]
        for N,dt in SETTINGS:
            path=out/f'{label}_N{N}_dt{dt:.8f}.npz'
            b0=data.initial_coefficients(p,N)
            v,secs=timer(lambda:trajectory(N,dt)(jnp.asarray(b0)))
            v=np.asarray(v);assert np.isfinite(v).all()
            np.savez_compressed(path,coefficients=v,parameters=p,times=data.TIMES)
            levels.append(v);records.append(dict(N=N,dt=dt,seconds=secs,file=path.name,sha256=digest(path),bytes=path.stat().st_size))
            print(label,N,dt,round(secs,3),flush=True)
        s0=relative_difference(levels[0],levels[1]);s1=relative_difference(levels[1],levels[4])
        t0=relative_difference(levels[2],levels[3]);t1=relative_difference(levels[3],levels[4])
        # Independent physical grids, compare field differences, current-time norms.
        egrids=[]
        for n in [257,513]:
            x=np.linspace(0,1,n)
            hi=data.synth_grid(jnp.asarray(levels[4]),x)
            lo=data.synth_grid(jnp.asarray(levels[1]),x)
            egrids.append(np.asarray(jnp.linalg.norm((hi-lo).reshape(11,-1),axis=1)/jnp.linalg.norm(hi.reshape(11,-1),axis=1)))
        extension=np.abs(egrids[1]-egrids[0]);allowance=s1+t1+extension
        # Compact verified training cache; discard no original reference coefficients.
        xy,w=data.grid(64,True); x=np.unique(xy[:,0])
        field=data.synth_grid(jnp.asarray(levels[4]),x)
        np.savez_compressed(out/f'{label}_training.npz',field=np.asarray(field),x=x,w=w,
                            parameters=p,norm=np.linalg.norm(levels[4].reshape(11,-1),axis=1)/2,
                            reference_file=records[-1]['file'],reference_sha256=records[-1]['sha256'])
        row=dict(case=label,parameters=p,levels=records,spatial_previous=s0,spatial_last=s1,
                 temporal_previous=t0,temporal_last=t1,scoring_change=extension,allowance=allowance,
                 provisional_ladder_pass=bool(np.max(allowance)<=.001 and np.max(extension)<.0002),
                 independent_crosscheck='pending',actual4096='pending finalist qualification')
        result['cases'].append(row);save(out/'result.json',result)
    result['status']='spectral_ladders_complete_pending_independent_checks';save(out/'result.json',result)

if __name__=='__main__': main()
