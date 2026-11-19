"""Coordinate neural fields with independent, exactly zero-preserving child codes."""
from common import *


def init_mlp(key,widths,sine=False):
    keys=jax.random.split(key,len(widths)-1);layers=[]
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        bound=(1/a if i==0 else np.sqrt(6/a)/30) if sine else np.sqrt(6/(a+b))
        w=jax.random.uniform(keys[i],(a,b),minval=-bound,maxval=bound,dtype=jnp.float64)
        if sine and i==0:
            w=w.at[:2].multiply(a/2)
        layers.append(dict(w=w,b=jnp.zeros(b)))
    return layers


def mlp(layers,x,sine=False):
    for layer in layers[:-1]:
        x=x@layer['w']+layer['b'];x=jnp.sin(30*x) if sine else jnp.tanh(x)
    return x@layers[-1]['w']+layers[-1]['b']


def boundary(x):return 16*x[...,0]*(1-x[...,0])*x[...,1]*(1-x[...,1])


def root(params,z,xy):
    z=jnp.broadcast_to(z,xy.shape[:-1]+(z.shape[-1],))
    return boundary(xy)*mlp(params,jnp.concatenate([2*xy-1,z],axis=-1),True)[...,0]


def child(params,z,cell,xy):
    h=2.**(-cell[0]);center=(cell[1:]+.5)*h;q=(xy-center)/h
    t=jnp.clip((jnp.abs(q)-.5)/.25,0,1)
    window=jnp.prod(1-(10*t**3-15*t**4+6*t**5),axis=-1)
    geom=jnp.concatenate([2*center-1,-cell[:1]/4])
    geom=jnp.broadcast_to(geom,xy.shape[:-1]+(3,))
    z=jnp.broadcast_to(z,xy.shape[:-1]+(z.shape[-1],))
    coord=jnp.clip(q/.75,-1,1)
    def value(code):return mlp(params,jnp.concatenate([coord,code,geom],axis=-1),True)[...,0]
    return boundary(xy)*window*(value(z)-value(jnp.zeros_like(z)))


def tree(params,z,xy,cells,root_k=8,child_k=8):
    u=root(params['root'],z[:root_k],xy)
    local=z[root_k:].reshape(len(cells),child_k)
    return u+jnp.sum(jax.vmap(child,in_axes=(None,0,0,None))(params['child'],local,cells,xy),axis=0)


def intrinsic(params,features,k=8):
    skip=jnp.pad(features,[(0,0)]*(features.ndim-1)+[(0,k-features.shape[-1])])
    return skip+.1*mlp(params,features)


def make_fit(apply,maxiter=60,station_tolerance=1e-8,update_tolerance=1e-10):
    """Residual-based LM in physical tangent-scaled coordinates; no target subtraction loss."""
    @jax.jit
    def fit(params,z0,xy,target,weight,norm):
        scale=jnp.sqrt(weight)/norm
        fun=lambda z:(apply(params,z,xy)-target)*scale
        def body(_,state):
            z,loss,damping,accepted,done,lastupdate,used=state
            def advance(state):
                z,loss,damping,accepted,done,lastupdate,used=state
                r=fun(z);J=jax.jacfwd(fun)(z);g=J.T@r
                H=J.T@J;diag=jnp.maximum(jnp.diag(H),1e-14);inv=jax.lax.rsqrt(diag)
                scaled=H*inv[:,None]*inv[None,:]+jnp.eye(len(z))*damping
                delta=inv*jnp.linalg.solve(scaled,-inv*g)
                update=jnp.linalg.norm(J@delta)
                rr=fun(z+delta);newloss=rr@rr
                predicted=-2*g@delta-delta@H@delta
                rho=(loss-newloss)/jnp.maximum(predicted,1e-30)
                good=jnp.isfinite(newloss)&(newloss<loss)&(predicted>0)
                station=jnp.max(jnp.abs(g)*inv)
                done=(station<=station_tolerance)&((update<=update_tolerance)|(station<=station_tolerance*.01))
                damp=jnp.where(good,jnp.where(rho>.75,damping/3,jnp.where(rho<.25,damping*2,damping)),damping*5)
                return (jnp.where(good,z+delta,z),jnp.where(good,newloss,loss),
                        jnp.clip(damp,1e-10,1e12),accepted+good.astype(jnp.int32),done,update,used+1)
            return jax.lax.cond(done,lambda s:s,advance,state)
        r=fun(z0);initial=(z0,r@r,jnp.float64(1e-3),jnp.int32(0),jnp.bool_(False),jnp.float64(0),jnp.int32(0))
        z,loss,damping,accepted,done,update,used=jax.lax.fori_loop(0,maxiter,body,initial)
        r=fun(z);J=jax.jacfwd(fun)(z);diag=jnp.maximum(jnp.sum(J*J,axis=0),1e-14)
        station=jnp.max(jnp.abs(J.T@r)/jnp.sqrt(diag))
        converged=(station<=station_tolerance)&((update<=update_tolerance)|(station<=station_tolerance*.01))
        return z,jnp.array([jnp.linalg.norm(r),station,update,accepted,used,converged],dtype=jnp.float64)
    return fit
