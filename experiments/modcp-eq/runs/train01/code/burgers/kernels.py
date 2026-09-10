"""CP-family weak Burgers residuals and complete device-resident queries."""
import time
import jax
import jax.numpy as jnp
import numpy as np
from common.decoders import decode_grid,decode_points,prepare_points,decode_cached
from common.lm import make_lm
from common.quadrature import fit_quadrature
from . import fom


def test_modes(L,M):
    P,lam,ids=fom.modes(L,M)
    return P*L,lam,ids


def sampled_advection(values,L):
    c,xp,xm,yp,ym=[values[:,i] for i in range(5)]
    return c*L*(jnp.where(c>0,c-xm,xp-c)+jnp.where(c>0,c-ym,yp-c))


def build_rule(params,Z,cfg,L,M,m,*,fit_states=32,candidate_cap=8192):
    """Fit output mass and FOM-exact advection, never FOM residual snapshots."""
    assert M>=4*cfg.k and m>=4*M
    Phi,lam,mode_ids=test_modes(L,M);P=(L-1)**2
    candidates=np.unique(np.rint(np.linspace(0,P-1,min(candidate_cap,P))).astype(int))
    pick=np.unique(np.rint(np.linspace(0,len(Z)-1,min(fit_states,len(Z)))).astype(int))
    targets=[np.array([P/L**2])];rows=[np.ones((1,len(candidates)))]
    decode=jax.jit(lambda p,z:decode_grid(p,z,L,cfg)[1:-1,1:-1,0].reshape(-1))
    spatial=jax.jit(lambda u:fom.spatial(u,L)[0])
    negatives=[]
    for index in pick:
        u=np.asarray(decode(params,jnp.asarray(Z[index])));adv=np.asarray(spatial(jnp.asarray(u)))
        negatives.append(float(np.mean(u<0)))
        for values in (u,adv):
            full=Phi*values[:,None]
            scale=np.maximum(np.sqrt(np.mean(full**2,axis=0)),1e-12)
            targets.append(np.sum(full,axis=0)/L**2/scale)
            rows.append((full[candidates]/scale).T)
    design=np.concatenate(rows);target=np.concatenate(targets)
    pos,w,info=fit_quadrature(design,target,m,candidate_ids=candidates)
    ij=np.stack(np.unravel_index(pos,(L-1,L-1)),axis=1)+1
    offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    xy=((ij[:,None,:]+offsets[None,:,:])/L).reshape(-1,2)
    cache=prepare_points(params,jnp.asarray(xy),cfg)
    info.update(intervals=L,M=M,m=m,fit_training_codes=pick.tolist(),mode_ids=mode_ids.tolist(),
                indices=pos.tolist(),weights=w.tolist(),decoded_negative_fractions=negatives,
                constant_integral=P/L**2,constant_fit_error=float(abs(w.sum()-P/L**2)),
                candidate_selection='deterministic evenly spaced flattened interior grid pool')
    return dict(cache=cache,Pq=jnp.asarray(Phi[pos]*w[:,None]),lam=jnp.asarray(lam)),info


def moments(params,z,data,cfg):
    us=decode_cached(params,z,data['cache'],cfg)[:,0].reshape((-1,5))
    return data['Pq'].T@us[:,0],us


def weak(z,prev,nu,params,data,cfg,L,dt,scale=1.):
    a,us=moments(params,z,data,cfg)
    adv=data['Pq'].T@sampled_advection(us,L)
    return (a-prev+dt*(adv+nu*data['lam']*a))/(1+dt*nu*data['lam'])/scale


def full_weak(z,prevz,nu,params,cfg,L,M,dt,scale=1.):
    Phi,lam,_=test_modes(L,M);P=jnp.asarray(Phi)/L**2
    u=decode_grid(params,z,L,cfg)[1:-1,1:-1,0].reshape(-1)
    prev=decode_grid(params,prevz,L,cfg)[1:-1,1:-1,0].reshape(-1)
    adv,lap=fom.spatial(u,L)
    return P.T@(u-prev+dt*(adv-nu*lap))/(1+dt*nu*jnp.asarray(lam))/scale


def build_initial_data(params,Z,cfg,L,points_axis=32):
    ids=np.unique(np.rint(np.linspace(1,L-1,min(points_axis,L-1))).astype(int))
    ii,jj=np.meshgrid(ids,ids,indexing='ij')
    xy=np.stack((ii.ravel(),jj.ravel()),axis=1)/L
    cache=prepare_points(params,jnp.asarray(xy),cfg)
    candidates=jnp.asarray(Z[::6]) # training initial states, six snapshots/case
    pred=jax.jit(lambda p,c,codes:jax.vmap(lambda z:decode_cached(p,z,c,cfg)[:,0])(codes))(params,cache,candidates)
    return dict(cache=cache,ii=jnp.asarray(ii.ravel()),jj=jnp.asarray(jj.ravel()),codes=candidates,pred=pred)


def make_rom(cfg,L,dt,cap=30,*,horizon=.25,output_spacing=.05):
    substeps=round(output_spacing/dt);nout=round(horizon/output_spacing)
    assert abs(substeps*dt-output_spacing)<1e-12
    def icfun(z,target,scale,params,ic):
        return (decode_cached(params,z,ic['cache'],cfg)[:,0]-target)/scale/jnp.sqrt(target.size)
    initial_lm=make_lm(icfun,cap=60)
    def rfun(z,prev,nu,params,data,scale):
        return weak(z,prev,nu,params,data,cfg,L,dt,scale)
    step_lm=make_lm(rfun,cap=cap)
    def initialize(field,params,ic):
        target=field[ic['ii'],ic['jj']]
        scale=jnp.maximum(jnp.sqrt(jnp.mean(target**2)),1e-12)
        loss=jnp.sum((ic['pred']-target[None,:])**2,axis=1)
        z0=ic['codes'][jnp.argmin(loss)]
        z,it,reason,stat,rn=initial_lm(z0,(target,scale,params,ic),1e-6)
        return z,scale,it,reason,stat,rn
    def evolve(z,nu,scale,params,data,tol):
        def step(z,_):
            prev,_=moments(params,z,data,cfg)
            zn,it,reason,stat,rn=step_lm(z,(prev,nu,params,data,scale),tol)
            return zn,(zn,it,reason,stat,rn)
        def block(z,_):
            zn,detail=jax.lax.scan(step,z,None,length=substeps)
            return zn,(zn,detail)
        _,(zo,detail)=jax.lax.scan(block,z,None,length=nout)
        return jnp.concatenate((z[None],zo)),jax.tree_util.tree_map(lambda x:x.reshape((-1,)+x.shape[2:]),detail)
    def reconstruct(Z,params):return jax.lax.map(lambda z:decode_grid(params,z,L,cfg)[...,0],Z)
    def query(field,nu,params,data,ic,tol):
        z,scale,it0,reason0,stat0,rn0=initialize(field,params,ic)
        Zo,detail=evolve(z,nu,scale,params,data,tol)
        return reconstruct(Zo,params),Zo,detail,(it0,reason0,stat0,rn0,scale)
    return jax.jit(query),dict(initialize=jax.jit(initialize),evolve=jax.jit(evolve),reconstruct=jax.jit(reconstruct),residual=rfun)


def errors(fields,truth):
    f=np.asarray(fields);t=np.asarray(truth)
    norm=max(float(np.linalg.norm(t[0])),1e-30)
    per=np.linalg.norm((f-t).reshape((len(f),-1)),axis=1)/norm
    return dict(displacement=float(per.max()),per_time=per.tolist(),initial=float(per[0]))


def burn(seconds=.75):
    a=jnp.ones((1024,1024),dtype=jnp.float64)
    f=jax.jit(lambda x:x@x/1024.)
    t=time.perf_counter()
    while time.perf_counter()-t<seconds:a=f(a);jax.block_until_ready(a)
