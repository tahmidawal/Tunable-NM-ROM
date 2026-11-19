"""Safeguarded inverse-BFGS diagnostic for the unchanged tested weak objective.

This is a separately selected optimizer, never a silent replacement of LM.
Every rejected Armijo trial is counted. No reference field enters a step.
"""
from common import *
import discrete_weak as W
import operators


def minimize(fun,x0,H0,station,update_norm,iterations,station_tolerance,update_tolerance):
    """All coordinates are dimensionless; H0 is a fixed SPD inverse metric."""
    value_gradient=jax.value_and_grad(fun)
    loss,g=value_gradient(x0)
    initial=(x0,loss,g,H0,jnp.int32(0),jnp.int32(0),jnp.int32(1),jnp.float64(0),jnp.bool_(False))
    def body(_,state):
        def advance(state):
            x,loss,g,H,used,accepted,evaluations,lastupdate,done=state
            p=-H@g
            descent=jnp.isfinite(g@p)&(g@p<0)
            p=jnp.where(descent,p,-H0@g)
            slope=g@p
            def condition(v):
                alpha,trial_loss,count,good=v
                return (~good)&(count<20)
            def backtrack(v):
                alpha,trial_loss,count,good=v
                trial_loss=fun(x+alpha*p)
                good=jnp.isfinite(trial_loss)&(trial_loss<=loss+1e-4*alpha*slope)&(slope<0)
                return jnp.where(good,alpha,alpha*.5),trial_loss,count+1,good
            alpha,trial_loss,trials,good=jax.lax.while_loop(condition,backtrack,
                (jnp.float64(1),loss,jnp.int32(0),jnp.bool_(False)))
            newx=jnp.where(good,x+alpha*p,x)
            newloss,newg=value_gradient(newx)
            s=newx-x;y=newg-g;ys=y@s;Hy=H@y
            valid=good&(ys>1e-12*jnp.linalg.norm(s)*jnp.linalg.norm(y))&(ys>1e-30)
            safe_ys=jnp.where(valid,ys,1.)
            B=H+((1+(y@Hy)/safe_ys)/safe_ys)*jnp.outer(s,s)-(jnp.outer(s,Hy)+jnp.outer(Hy,s))/safe_ys
            H=jnp.where(valid,(B+B.T)*.5,H)
            H=jnp.where(good,H,H0)
            relative_update=update_norm(newx,x)
            st=station(newg,newx)
            done=(st<station_tolerance)&((relative_update<update_tolerance)|(st<station_tolerance*.01))
            return newx,newloss,newg,H,used+1,accepted+good.astype(jnp.int32),evaluations+trials+1,relative_update,done
        return jax.lax.cond(state[-1],lambda v:v,advance,state)
    return jax.lax.fori_loop(0,iterations,body,initial)


def make_solver(field,L,iterations=120,theta=.5,advection_tests=None,
                coefficient_map=None,tested_mass_jacobian=None):
    assert (coefficient_map is None)==(tested_mass_jacobian is None)
    decode=(lambda z,ca:z) if coefficient_map is None else coefficient_map
    _,residual=W.make_solver(field,L,theta=theta,advection_tests=advection_tests,
        coefficient_map=coefficient_map,tested_mass_jacobian=tested_mass_jacobian)
    def adv(c,ca):
        if advection_tests is not None:return advection_tests(c,ca)
        a,_=operators.spatial(field(c,ca),L)
        return W.dst2(a.reshape(L-1,L-1))[ca['indices'][0],ca['indices'][1]]/L
    @jax.jit
    def step(z0,ca,dt,nu,station_tolerance,update_tolerance):
        c0=decode(z0,ca);norm=jnp.sqrt(jnp.maximum(c0@ca['gram']@c0,1e-30))
        previous=ca['A']@c0
        if theta!=1.:
            previous=previous*(1-(1-theta)*dt*nu*ca['lam'])-(1-theta)*dt*adv(c0,ca)
        mass=ca['A'] if tested_mass_jacobian is None else tested_mass_jacobian(z0,ca)
        metric=(mass.T@mass)/norm**2;diag=jnp.maximum(jnp.diag(metric),1e-12)
        scale=jax.lax.rsqrt(diag)
        scaled_metric=metric*scale[:,None]*scale[None,:]+1e-4*jnp.eye(len(z0))
        H0=jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(scaled_metric),True),jnp.eye(len(z0)))
        def fun(x):
            r=residual(z0+scale*x,previous,ca,dt,nu,norm)
            return .5*(r@r)
        def station(g,x):
            d=diag
            if tested_mass_jacobian is not None:
                A=tested_mass_jacobian(z0+scale*x,ca)
                d=jnp.maximum(jnp.sum(A*A,axis=0)/norm**2,1e-12)
            return jnp.max(jnp.abs(g/scale)/jnp.sqrt(d))
        def update(new,old):
            dc=decode(z0+scale*new,ca)-decode(z0+scale*old,ca)
            return jnp.sqrt(jnp.maximum(dc@ca['gram']@dc,0))/norm
        x,loss,g,H,used,accepted,evaluations,lastupdate,done=minimize(fun,jnp.zeros_like(z0),H0,
            station,update,iterations,station_tolerance,update_tolerance)
        return z0+scale*x,jnp.array([jnp.sqrt(2*loss),station(g,x),lastupdate,evaluations,used,accepted],dtype=jnp.float64)
    return step,residual
