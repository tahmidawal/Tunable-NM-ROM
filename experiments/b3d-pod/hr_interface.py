"""Archived ECSW equations with explicit schedule-sized buffers and data arguments."""
import numpy as np
import interface as I

jax,jnp,H,R,B=I.jax,I.jnp,I.H,I.R,I.P.BD


def functions_for(kind,trust,steps,solver='common',L=64):
    assert kind in ('none','ecsw') and solver in ('common','native')
    if kind=='none':return I.functions_for(trust,steps,solver,L=L)
    assert steps in (10,15,20,25,40) and L in (5,64)
    ev=R.make_evolve('ecsw',3,L,solver,steps,trust=trust)
    def query(u0,nu,data,sch):
        c0=data['Phi'].T@u0;scale=jnp.linalg.norm(c0)
        W,C,stats=ev(c0,nu,scale,data,sch)
        return dict(fields=W@data['Phi'].T,W=W,C=C,c0=c0,scale=scale,stats=stats)
    def schedule(scheme,dt,level='prod'):
        g,t=R.LEVELS['prod_native' if solver=='native' and level=='prod' else level]
        s=H.sched(scheme,dt,nmax=steps)
        return dict({k:v for k,v in s.items() if k!='nsave'},gtol=jnp.float64(g),tolf=jnp.float64(t))
    return (jax.jit(query),jax.jit(lambda u,d:(d['Phi'].T@u,jnp.linalg.norm(d['Phi'].T@u))),
            jax.jit(ev),jax.jit(lambda W,d:W@d['Phi'].T),R.make_certifier('ecsw',3,L,steps),schedule)


def archived_data(phi_path,support_path):
    _,_,_,_,full,_,meta=I.setup(phi_path)
    z=np.load(support_path);nodes,weights=z['nodes'],z['weights']
    assert len(nodes)==8192 and len(set(nodes.tolist()))==8192
    assert np.isfinite(weights).all() and (weights>0).all()
    data,_=B.ecsw_data(full['Phi'],nodes,weights,3,64)
    sampled=jax.block_until_ready(dict(data,Phi=full['Phi']))
    return full,sampled,meta
