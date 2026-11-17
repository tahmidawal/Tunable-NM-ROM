"""ETDRK4 source arithmetic, with 11 retained states for matched training."""
from campaign import *
def make(N=128,dt=.000625,M=64):
 k=jnp.arange(1,N,dtype=jnp.float64);eig=-jnp.pi**2*(k[:,None]**2+k[None,:]**2)
 S=P.lattice_S(N,M)
 def run(b0,nu):
  co=P.etdrk4_coefs(nu*eig,dt,32)
  def step(v):
   nv,_=P.nonlinear(v,N,2);a=co['E2']*v+co['Q']*nv;na,_=P.nonlinear(a,N,2)
   b=co['E2']*v+co['Q']*na;nb,_=P.nonlinear(b,N,2)
   c=co['E2']*a+co['Q']*(2*nb-nv);nc,_=P.nonlinear(c,N,2)
   return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v)
   return v,P.synth_lattice(v,S,2)
  _,f=jax.lax.scan(block,b0,None,length=10)
  return jnp.concatenate([P.synth_lattice(b0,S,2)[None],f])
 return jax.jit(run)
