"""Smooth canonical neural features for input-conditioned frames."""
from campaign import jax,jnp,np
import local_bank as B
def init(key,k=128):
 dims=[22,512,512,k]
 return [(jax.random.normal(q,(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b)),jnp.zeros(b)) for q,a,b in zip(jax.random.split(key,3),dims[:-1],dims[1:])]
def apply(p,z):
 f=jnp.array([1.,2.,4.,8.,16.]);x=jnp.concatenate([z,jnp.sin(jnp.pi*z[:,:,None]*f).reshape(len(z),-1),jnp.cos(jnp.pi*z[:,:,None]*f).reshape(len(z),-1)],axis=1)
 for w,b in p[:-1]:x=jnp.tanh(x@w+b)
 return x@p[-1][0]+p[-1][1]
def features(p,xy,patch,k):return apply(p,(xy-jnp.asarray(patch[:2]))/patch[2])[:,:k]*B.window(xy,patch)[:,None]
def weighted_teacher(Y,W,k=128,iterations=10):
 """Y[c,p,t], W[c,p]; alternating weighted low rank with per-case geometry."""
 mean=jnp.maximum(jnp.mean(W,axis=0),1e-10);X=(jnp.sqrt(W)[:,:,None]*Y).transpose(1,0,2).reshape(Y.shape[1],-1);U,_,_=jnp.linalg.svd(X,full_matrices=False);F=U[:,:k]/jnp.sqrt(mean)[:,None]
 @jax.jit
 def coefficients(F,Y,W):
  gram=jnp.einsum('pi,cp,pj->cij',F,W,F);rhs=jnp.einsum('pi,cp,cpt->cit',F,W,Y);ridge=1e-8*jnp.trace(gram,axis1=1,axis2=2)/k+1e-12
  return jnp.linalg.solve(gram+ridge[:,None,None]*jnp.eye(k),rhs)
 @jax.jit
 def update(F,Y,W):
  C=coefficients(F,Y,W);gram=jnp.einsum('cp,cit,cjt->pij',W,C,C);rhs=jnp.einsum('cp,cpt,cit->pi',W,Y,C);ridge=1e-8*jnp.trace(gram,axis1=1,axis2=2)/k+1e-12
  F=jnp.linalg.solve(gram+ridge[:,None,None]*jnp.eye(k),rhs[:,:,None])[:,:,0]
  Q,_=jnp.linalg.qr(F*jnp.sqrt(mean)[:,None]);F=Q/jnp.sqrt(mean)[:,None]
  C=coefficients(F,Y,W);loss=jnp.sum(W[:,:,None]*(jnp.einsum('pi,cit->cpt',F,C)-Y)**2)/Y.shape[0]
  return F,C,loss
 history=[]
 for it in range(iterations):F,C,loss=update(F,Y,W);history.append(float(loss))
 cov=jnp.einsum('cit,cjt->ij',C,C);s,R=jnp.linalg.eigh(cov);return F@R[:,::-1],s[::-1],history
