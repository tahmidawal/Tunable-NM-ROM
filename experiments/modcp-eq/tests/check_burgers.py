"""Small-grid correctness checks; runs in under one local GPU minute."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from common.decoders import *
from common.lm import make_lm
from burgers import fom,kernels as k


def main():
    assert jax.config.jax_enable_x64
    L=12;cfg=DecoderConfig(k=3,rank=6,intervals=L,head_width=12,width=8,inr_width=10,frequencies=3)
    key=jax.random.PRNGKey(10);p=init_decoder(key,cfg);z=jnp.asarray([.1,-.2,.3])
    xy=jnp.asarray([[.1,.2],[.8,.2],[.25,.75],[0.,.5]])
    mod=replace(cfg,architecture='modcp');pm=add_modulation(p,mod,key)
    np.testing.assert_allclose(decode_points(p,z,xy,cfg),decode_points(pm,z,xy,mod),atol=1e-14)
    np.testing.assert_allclose(jax.jacfwd(lambda z:decode_points(p,z,xy,cfg))(z),jax.jacfwd(lambda z:decode_points(pm,z,xy,mod))(z),atol=1e-14)
    for arch in ('cp','modcp','film'):
        c=replace(cfg,architecture=arch);q=init_decoder(key,c)
        cache=prepare_points(q,xy,c)
        np.testing.assert_allclose(decode_points(q,z,xy,c),decode_cached(q,z,cache,c),atol=1e-13)
        direction=jnp.array([.2,.3,-.1]);eps=1e-5
        ad=jax.jvp(lambda z:decode_cached(q,z,cache,c),(z,),(direction,))[1]
        fd=(decode_cached(q,z+eps*direction,cache,c)-decode_cached(q,z-eps*direction,cache,c))/(2*eps)
        np.testing.assert_allclose(ad,fd,rtol=1e-6,atol=1e-10)
        target=2*L;gridxy=jnp.asarray(fom.coords(target,False))
        np.testing.assert_allclose(decode_grid(q,z,target,c).reshape(-1,1),decode_points(q,z,gridxy,c),atol=1e-13)
        near=jnp.asarray([[.5/L,.4],[1./L,.4]])
        values=decode_points(q,z,near,c)
        np.testing.assert_allclose(values[0],.5*values[1],atol=1e-13)
        empty=jnp.zeros((0,2),dtype=jnp.float64)
        assert decode_cached(q,z,prepare_points(q,empty,c),c).shape==(0,1)
    # Complete-grid positive quadrature must recover adjoint diffusion and the
    # exact sign-dependent FOM advection, including negative decoded values.
    M=12;Phi,lam,_=k.test_modes(L,M)
    pos=np.arange((L-1)**2);ij=np.stack(np.unravel_index(pos,(L-1,L-1)),axis=1)+1
    offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    stxy=jnp.asarray(((ij[:,None]+offsets[None])/L).reshape(-1,2))
    data=dict(cache=prepare_points(p,stxy,cfg),Pq=jnp.asarray(Phi/L**2),lam=jnp.asarray(lam))
    prevz=z*.7;prev,_=k.moments(p,prevz,data,cfg)
    sampled=k.weak(z,prev,.02,p,data,cfg,L,.005)
    full=k.full_weak(z,prevz,.02,p,cfg,L,M,.005)
    np.testing.assert_allclose(sampled,full,atol=1e-13)
    contracted=k.precontract_mass(data,cfg)
    np.testing.assert_allclose(k.weak(z,prev,.02,p,contracted,cfg,L,.005),sampled,atol=1e-13)
    uncontractedJ=jax.jacfwd(lambda z:k.weak(z,prev,.02,p,data,cfg,L,.005))(z)
    contractedJ=jax.jacfwd(lambda z:k.weak(z,prev,.02,p,contracted,cfg,L,.005))(z)
    np.testing.assert_allclose(uncontractedJ,contractedJ,atol=1e-13)
    training_codes=jnp.stack([z,z*.9,z*.8,z*.7,z*.6,z*.5])
    initial_data=k.build_initial_data(p,training_codes,cfg,L)
    query,_=k.make_rom(cfg,L,.005,10,horizon=.05)
    initial_field=decode_grid(p,z,L,cfg)[...,0]
    original=query(initial_field,.02,p,data,initial_data,1e-6)
    optimized=query(initial_field,.02,p,contracted,initial_data,1e-6)
    np.testing.assert_allclose(original[0],optimized[0],atol=1e-10,rtol=1e-8)
    u=decode_grid(p,z,L,cfg)[1:-1,1:-1,0].reshape(-1)
    _,values=k.moments(p,z,data,cfg)
    np.testing.assert_allclose(k.sampled_advection(values,L),fom.spatial(u,L)[0],atol=1e-12)
    linear=lambda x,A,b:A@x-b
    A=jnp.array([[1.,0.],[0.,2.],[1.,1.]])
    b=A@jnp.array([.3,-.8]);fit=make_lm(linear,30)(jnp.zeros(2),(A,b),1e-8)
    np.testing.assert_allclose(fit[0],[.3,-.8],atol=1e-8)
    constant=make_lm(lambda z:jnp.ones(4),3)(jnp.zeros(2),(),1e-6)
    assert int(constant[2])==5
    # FFT preconditioner inverts the discrete Dirichlet Helmholtz operator.
    v=jnp.arange((L-1)**2,dtype=jnp.float64);nu=.025;dt=.005
    inverse=fom.helmholtz(v,nu,dt,L)
    np.testing.assert_allclose(inverse-dt*nu*fom.spatial(inverse,L)[1],v,atol=1e-10)
    q,_=fom.make_fom(L,.005,horizon=.05)
    fields,it,r=q(jnp.asarray(fom.initial(L,fom.params_draw(37,1)[0])),.025,1e-10,1e-8)
    assert np.isfinite(fields).all() and float(jnp.max(r))<2e-10
    print('PASS decoder parity, transfer, autodiff, exact weak/upwind, LM reasons, FFT/FOM',flush=True)


if __name__=='__main__':main()
