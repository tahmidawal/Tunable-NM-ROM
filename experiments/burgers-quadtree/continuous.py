"""C1 continuous sine weak Galerkin, physical L2 scaling, ETDRK4.
Not the inherited upwind operator. V has orthonormal columns in b/2 coordinates,
where b is the continuous sine coefficient vector and ||u||_L2=||b||/2.
"""
from campaign import *
def eigenvalues(N):
 k=jnp.arange(1,N,dtype=jnp.float64)
 return (jnp.pi**2*(k[:,None]**2+k[None,:]**2)).reshape(-1)
def diffusion_basis(V,N):
 lam=eigenvalues(N);D=V.T@(lam[:,None]*V)
 eig,R=jnp.linalg.eigh((D+D.T)/2)
 return V@R,eig,R

def full_trajectory(N,dt=.000625):
 lam=eigenvalues(N).reshape(N-1,N-1)
 @jax.jit
 def run(b0,nu):
  co=P.etdrk4_coefs(-nu*lam,dt,32)
  def step(v):
   nv,_=P.nonlinear(v,N,2);a=co['E2']*v+co['Q']*nv;na,_=P.nonlinear(a,N,2)
   b=co['E2']*v+co['Q']*na;nb,_=P.nonlinear(b,N,2);c=co['E2']*a+co['Q']*(2*nb-nv);nc,_=P.nonlinear(c,N,2)
   return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v);return v,v
  _,v=jax.lax.scan(block,b0,None,length=10)
  return jnp.concatenate([b0[None],v])
 return run

def reduced_trajectory(N,dt=.0025):
 @jax.jit
 def run(a0,nu,W,eigs):
  co=P.etdrk4_coefs(-nu*eigs,dt,32)
  def nonlin(a):
   b=(2*(W@a)).reshape(N-1,N-1)
   f,_=P.nonlinear(b,N,2)
   return W.T@f.reshape(-1)/2
  def step(v):
   nv=nonlin(v);a=co['E2']*v+co['Q']*nv;na=nonlin(a)
   b=co['E2']*v+co['Q']*na;nb=nonlin(b);c=co['E2']*a+co['Q']*(2*nb-nv);nc=nonlin(c)
   return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v);return v,v
  _,v=jax.lax.scan(block,a0,None,length=10)
  return jnp.concatenate([a0[None],v])
 return run

def flux_tables(W,N,q):
 from numpy.polynomial.legendre import leggauss
 x,w=leggauss(q);x=(x+1)/2;w=w/2;k=jnp.arange(1,N,dtype=jnp.float64)
 S=jnp.sin(jnp.pi*jnp.asarray(x)[:,None]*k);D=jnp.pi*k*jnp.cos(jnp.pi*jnp.asarray(x)[:,None]*k)
 V=W.reshape(N-1,N-1,-1)
 def tensor(A,B):return jnp.einsum('ai,ijr,bj->abr',A,V,B).reshape(q*q,-1)
 G=2*tensor(S,S);grad=2*(tensor(D,S)+tensor(S,D));weights=jnp.asarray(np.outer(w,w).ravel())
 return G,.5*weights[:,None]*grad

def flux_trajectory(dt=.005):
 @jax.jit
 def run(a0,nu,G,H,eigs):
  co=P.etdrk4_coefs(-nu*eigs,dt,32)
  def nonlin(a):return H.T@(G@a)**2
  def step(v):
   nv=nonlin(v);a=co['E2']*v+co['Q']*nv;na=nonlin(a);b=co['E2']*v+co['Q']*na;nb=nonlin(b)
   c=co['E2']*a+co['Q']*(2*nb-nv);nc=nonlin(c)
   return co['E']*v+co['f1']*nv+2*co['f2']*(na+nb)+co['f3']*nc
  def block(v,_):
   v=jax.lax.fori_loop(0,round(.025/dt),lambda i,x:step(x),v);return v,v
  _,v=jax.lax.scan(block,a0,None,length=10)
  return jnp.concatenate([a0[None],v])
 return run
