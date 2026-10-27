"""Phase-only derivative of frozen psjax: same ETDRK4 operations, coefficient outputs before synthesis.

Production timing always calls the byte-identical original fused psjax kernel. These
separately timed components are indicative and cannot substitute for its full query.
"""
import jax
import jax.numpy as jnp
import pscore as PS
from psjax import lin_eig_j

def make(N,dt,Mc=32):
    keep=int(round(.05/dt));assert abs(keep*dt-.05)<1e-12
    def step(v,co):
        Nv,_=PS.nonlinear(v,N,3)
        a=co['E2']*v+co['Q']*Nv
        Na,_=PS.nonlinear(a,N,3)
        b=co['E2']*v+co['Q']*Na
        Nb,_=PS.nonlinear(b,N,3)
        c=co['E2']*a+co['Q']*(2*Nb-Nv)
        Nc,_=PS.nonlinear(c,N,3)
        return co['E']*v+Nv*co['f1']+2*(Na+Nb)*co['f2']+Nc*co['f3']
    @jax.jit
    def evolve(b0,nu):
        co=PS.etdrk4_coefs(lin_eig_j(N,nu),dt,Mc)
        def interval(v,_):
            v=jax.lax.fori_loop(0,keep,lambda i,x:step(x,co),v)
            return v,v
        _,B=jax.lax.scan(interval,b0,None,length=5)
        return jnp.concatenate((b0[None],B))
    @jax.jit
    def decode(B,S):return jax.vmap(lambda b:PS.synth_lattice(b,S,3))(B)
    return evolve,decode,PS.lattice_S(N,64)
