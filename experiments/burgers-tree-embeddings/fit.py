"""Physical-metric linear fits and bounded damped nonlinear projection."""
from common import *
import decoder as dec

@jax.jit
def linear_fit(G,Y,w):
    A=G*jnp.sqrt(w[:,None]);B=Y*jnp.sqrt(w[None,:])
    U,s,Vh=jnp.linalg.svd(A,full_matrices=False)
    keep=s>jnp.maximum(s[0]*1e-11,1e-14)
    C=(B@U)*jnp.where(keep,1/jnp.maximum(s,1e-300),0)
    C=C@Vh
    pred=C@G.T
    return C,pred,s

def make_project(coefficients,maxiter=25):
    """Returns best feasible iterate; tolerances use the current physical target norm."""
    @jax.jit
    def project(G,y,w,z0):
        sw=jnp.sqrt(w);norm=jnp.sqrt(jnp.sum(w*y*y))
        A=G*sw[:,None]/norm;target=y*sw/norm
        residual=lambda z:A@coefficients(z)-target
        jac=jax.jacfwd(coefficients)
        def body(_,state):
            z,loss,damping,accepted,done=state
            r=residual(z);J=A@jac(z);grad=J.T@r;H=J.T@J
            diag=jnp.maximum(jnp.diag(H),1e-10)
            delta=jnp.linalg.solve(H+damping*jnp.diag(diag)+1e-14*jnp.eye(len(z)),-grad)
            trial=z+delta;newloss=jnp.sum(residual(trial)**2)
            good=(newloss<loss)&jnp.isfinite(newloss)&(~done)
            stationary=jnp.max(jnp.abs(grad)/jnp.sqrt(diag))<1e-9
            return (jnp.where(good,trial,z),jnp.where(good,newloss,loss),
                    jnp.clip(jnp.where(good,damping/3,damping*5),1e-12,1e12),
                    accepted+good.astype(jnp.int32),done|stationary)
        initial=(z0,jnp.sum(residual(z0)**2),jnp.float64(1e-3),jnp.int32(0),jnp.bool_(False))
        z,loss,damp,accepted,done=jax.lax.fori_loop(0,maxiter,body,initial)
        r=residual(z);J=A@jac(z)
        station=jnp.max(jnp.abs(J.T@r)/jnp.maximum(jnp.linalg.norm(J,axis=0),1e-14))
        return z,jnp.sqrt(loss),station,accepted
    return project

def multi_fit(project,G,Y,w,starts):
    fn=jax.jit(jax.vmap(project,in_axes=(None,0,None,0)))
    results=[fn(G,Y,w,z) for z in starts]
    errors=jnp.stack([r[1] for r in results]);which=jnp.argmin(errors,axis=0)
    codes=jnp.stack([r[0] for r in results])[which,jnp.arange(len(Y))]
    stat=jnp.stack([r[2] for r in results])[which,jnp.arange(len(Y))]
    return codes,jnp.min(errors,axis=0),stat,errors
