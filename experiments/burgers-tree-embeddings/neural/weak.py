"""Nonlinear coordinate decoder in the verified, fine-stencil weak equations.

Q changes integration only; L fixes the upwind stencil and diffusion spectrum.
All decoder weights, coordinates and test indices are runtime arguments.
"""
from common import *
from discrete_weak import dst2,selected_modes


def prepare(L,Q,M):
    assert L%Q==0 and Q<=L
    idx,lam=selected_modes(L,M)
    assert idx.max()<Q-1,'Integration grid cannot resolve these test modes'
    axis=np.arange(1,Q)/Q
    xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])/L
    return dict(xy=jnp.asarray(xy),stencil=jnp.asarray(xy[None]+offsets[:,None]),
                indices=jnp.asarray(idx),lam=jnp.asarray(lam))


def make_operators(apply,L,Q,theta=.5):
    assert theta in (.5,1.)
    def project(u,cache):
        b=dst2(u.reshape(Q-1,Q-1))/Q
        return b[cache['indices'][0],cache['indices'][1]]
    def mass(params,z,cache):return project(apply(params,z,cache['xy']),cache)
    def advection(params,z,cache):
        c,xp,xm,yp,ym=jax.vmap(lambda xy:apply(params,z,xy))(cache['stencil'])
        a=c*L*(jnp.where(c>0,c-xm,xp-c)+jnp.where(c>0,c-ym,yp-c))
        return project(a,cache)
    def previous(params,z,cache,dt,nu):
        m=mass(params,z,cache)
        return m*(1-(1-theta)*dt*nu*cache['lam'])-(1-theta)*dt*advection(params,z,cache)
    def residual(params,z,old,cache,dt,nu,norm):
        denom=1+theta*dt*nu*cache['lam']
        return (mass(params,z,cache)+(theta*dt*advection(params,z,cache)-old)/denom)/norm
    return mass,advection,previous,residual


def make_step(apply,L,Q,iterations=60,theta=.5):
    """Full assembled Jacobian LM; stopping and update size use physical fields."""
    mass,adv,previous,residual=make_operators(apply,L,Q,theta)
    @jax.jit
    def step(params,z0,cache,dt,nu,station_tolerance=1e-8,update_tolerance=1e-10):
        u0=apply(params,z0,cache['xy'])
        norm=jnp.maximum(jnp.linalg.norm(u0)/Q,1e-15)
        old=previous(params,z0,cache,dt,nu)
        fun=lambda z:residual(params,z,old,cache,dt,nu,norm)
        field=lambda z:apply(params,z,cache['xy'])/(Q*norm)
        def body(_,state):
            def advance(state):
                z,loss,damp,accepted,done,update,used=state
                r=fun(z);J=jax.jacfwd(fun)(z);g=J.T@r;H=J.T@J
                physical=jax.jacfwd(field)(z)
                diag=jnp.maximum(jnp.sum(physical**2,axis=0),1e-14)
                scale=jax.lax.rsqrt(diag)
                delta=scale*jnp.linalg.solve(H*scale[:,None]*scale[None,:]+damp*jnp.eye(len(z)),-scale*g)
                update=jnp.linalg.norm(field(z+delta)-field(z))
                rr=fun(z+delta);newloss=rr@rr;predicted=-2*g@delta-delta@H@delta
                good=jnp.isfinite(newloss)&(newloss<loss)&(predicted>0)
                ratio=(loss-newloss)/jnp.maximum(predicted,1e-30)
                station=jnp.max(jnp.abs(g)*scale)
                done=(station<=station_tolerance)&((update<=update_tolerance)|(station<=station_tolerance*.01))
                damp=jnp.where(good,jnp.where(ratio>.75,damp/3,jnp.where(ratio<.25,damp*2,damp)),damp*5)
                return (jnp.where(good,z+delta,z),jnp.where(good,newloss,loss),jnp.clip(damp,1e-10,1e12),
                        accepted+good.astype(jnp.int32),done,update,used+1)
            return jax.lax.cond(state[4],lambda s:s,advance,state)
        r=fun(z0)
        initial=(z0,r@r,jnp.float64(1e-3),jnp.int32(0),jnp.bool_(False),jnp.float64(0),jnp.int32(0))
        z,loss,damp,accepted,done,update,used=jax.lax.fori_loop(0,iterations,body,initial)
        r=fun(z);J=jax.jacfwd(fun)(z);P=jax.jacfwd(field)(z)
        station=jnp.max(jnp.abs(J.T@r)/jnp.sqrt(jnp.maximum(jnp.sum(P**2,axis=0),1e-14)))
        converged=(station<=station_tolerance)&((update<=update_tolerance)|(station<=station_tolerance*.01))
        return z,jnp.array([jnp.linalg.norm(r),station,update,used,accepted,converged])
    return step
