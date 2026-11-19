"""The user's named Burgers extensions, separate from primary and sealed cohorts."""
from common import *
import data
from vendor import pscore as P
from references import relative_difference,SETTINGS
from functools import lru_cache


def initial(xy,case):
    x,y=xy[...,0],xy[...,1];u=jnp.full_like(x,case['background'])
    for g in case['gaussians']:
        angle=np.deg2rad(g['angle_degrees']);dx=x-g['center'][0];dy=y-g['center'][1]
        tangent=np.cos(angle)*dx+np.sin(angle)*dy;normal=-np.sin(angle)*dx+np.cos(angle)*dy
        u=u+g['amplitude']*jnp.exp(-.5*((tangent/g['sigma_t'])**2+(normal/g['sigma_n'])**2))
    return jnp.sin(jnp.pi*x)**2*jnp.sin(jnp.pi*y)**2*u


def initial_coefficients(case,N,order=32):
    x,w=P.composite_gl(64,order);xy=jnp.stack(jnp.meshgrid(jnp.asarray(x),jnp.asarray(x),indexing='ij'),axis=-1)
    u=initial(xy,case);S=jnp.sin(jnp.pi*jnp.asarray(x)[:,None]*jnp.arange(1,N)[None])
    return 4*S.T@(jnp.asarray(w)[:,None]*u*jnp.asarray(w)[None,:])@S


@lru_cache(None)
def trajectory(N,dt,nu):
    coefficients=P.etdrk4_coefs(P.lin_eig(N,2,nu),dt,32)
    @jax.jit
    def run(b0,co):
        def step(v):
            nv,_=P.nonlinear(v,N,2);a=co['E2']*v+co['Q']*nv
            na,_=P.nonlinear(a,N,2);b=co['E2']*v+co['Q']*na
            nb,_=P.nonlinear(b,N,2);c=co['E2']*a+co['Q']*(2*nb-nv)
            nc,_=P.nonlinear(c,N,2)
            return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
        def block(v,_):
            v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v)
            return v,v
        _,v=jax.lax.scan(block,b0,None,length=10)
        return jnp.concatenate([b0[None],v])
    return lambda b:run(b,coefficients)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);spec=HERE/'neural/named_cases.json'
    definition=json.loads(spec.read_text())['burgers']
    result=dict(meta=preflight(),status='named_reference_ladders_running',spec_sha256=digest(spec),
        settings=SETTINGS,cases=[],final_test_opened=False,role='named development extension, not training')
    for ci,case in enumerate(definition['cases']):
        b512=initial_coefficients(case,512);fine=initial_coefficients(case,512,64)
        initial_change=float(jnp.linalg.norm(b512-fine)/jnp.linalg.norm(fine));assert initial_change<1e-10
        for nu in definition['proposed_viscosities']:
            label=f'named{ci}_nu{nu:g}';levels=[];records=[]
            for N,dt in SETTINGS:
                b0=b512[:N-1,:N-1]
                v,seconds=timer(lambda:trajectory(N,dt,nu)(b0));v=np.asarray(v);assert np.isfinite(v).all()
                path=out/f'{label}_N{N}_dt{dt:.8f}.npz'
                np.savez_compressed(path,coefficients=v,times=data.TIMES,nu=nu)
                records.append(dict(N=N,dt=dt,file=path.name,sha256=digest(path),seconds=seconds));levels.append(v)
                print('named_reference',label,N,dt,seconds,flush=True)
            s0=relative_difference(levels[0],levels[1]);s1=relative_difference(levels[1],levels[4])
            t0=relative_difference(levels[2],levels[3]);t1=relative_difference(levels[3],levels[4])
            changes=[]
            for n in [257,513]:
                axis=np.linspace(0,1,n);hi=data.synth_grid(jnp.asarray(levels[4]),axis)
                lo=data.synth_grid(jnp.asarray(levels[1]),axis)
                changes.append(np.asarray(jnp.linalg.norm((hi-lo).reshape(11,-1),axis=1)/jnp.linalg.norm(hi.reshape(11,-1),axis=1)))
            scoring=abs(changes[1]-changes[0]);allowance=s1+t1+scoring
            result['cases'].append(dict(case=label,definition=case,nu=nu,levels=records,
                spatial_previous=s0,spatial_last=s1,temporal_previous=t0,temporal_last=t1,
                scoring_change=scoring,allowance=allowance,initial_quadrature_change=initial_change,
                provisional_ladder_pass=bool(max(allowance)<=.001 and max(scoring)<=.0002),
                independent_discretization_crosscheck='pending',admitted_reference=False))
            save(out/'result.json',result)
    result['status']='complete_named_ladders_pending_independent_crosscheck';save(out/'result.json',result)


if __name__=='__main__':main()
