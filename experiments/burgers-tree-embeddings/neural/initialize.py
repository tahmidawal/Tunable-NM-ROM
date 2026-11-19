"""Offline missing-field initialization; never supplied to an online PDE solve."""
from common import *
from neural import fields as F


@jax.jit
def missing_field(params,root_code,cells,xy,target,weight,norm):
    """Sequential zero-code tangent fits with measured nonlinear backtracking.

    This only initializes independent child variables. Subsequent training and
    representation fits jointly adjust the assembled root and all child codes.
    """
    prediction=F.root(params['root'],root_code,xy)
    scale=jnp.sqrt(weight)/norm
    def one(prediction,cell):
        zero=jnp.zeros(8)
        fun=lambda z:F.child(params['child'],z,cell,xy)
        J=jax.jacfwd(fun)(zero)*scale[:,None]
        rhs=(target-prediction)*scale
        diag=jnp.maximum(jnp.sum(J**2,axis=0),1e-14);inv=jax.lax.rsqrt(diag)
        scaled=J*inv[None,:]
        code=inv*jnp.linalg.solve(scaled.T@scaled+1e-3*jnp.eye(8),scaled.T@rhs)
        # Limit the first proposal in code space, then measure true field loss.
        code=code*jnp.minimum(1.,1./jnp.maximum(jnp.linalg.norm(code),1e-30))
        candidates=jnp.array([0.,1.,.5,.25,.125,.0625])[:,None]*code[None,:]
        corrections=jax.vmap(fun)(candidates)
        losses=jnp.sum(((prediction[None,:]+corrections-target)*scale)**2,axis=1)
        best=jnp.argmin(losses)
        return prediction+corrections[best],(candidates[best],losses[best])
    prediction,(codes,loss)=jax.lax.scan(one,prediction,cells)
    return jnp.concatenate([root_code,codes.ravel()]),loss
