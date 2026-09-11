"""Compiled mass-symmetric implicit midpoint wave FOM with counted CG.

This uses only the frozen fresh-wave stiffness/mass/boundary damping. Unknowns
are sqrt(M) v, so the matrix is SPD also at absorbing trapezoid endpoints.
"""
from functools import partial
import numpy as np
import pilot as base
from pilot import jax, jnp
from fresh_fom import positive_laplacian, damping_ratio


def cg_solve(apply, rhs, x, tolerance, max_iterations):
    r = rhs-apply(x)
    rr = jnp.vdot(r,r).real
    norm = jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)
    threshold = (tolerance*norm)**2
    def condition(state):
        k, x, r, p, rr = state
        return (k<max_iterations)&(rr>threshold)&jnp.isfinite(rr)
    def step(state):
        k, x, r, p, rr = state
        ap = apply(p)
        alpha = rr/jnp.vdot(p,ap).real
        xn, rn = x+alpha*p, r-alpha*ap
        rrn = jnp.vdot(rn,rn).real
        pn = rn+(rrn/rr)*p
        return k+1,xn,rn,pn,rrn
    k,x,r,p,rr = jax.lax.while_loop(condition,step,(jnp.int32(0),x,r,r,rr))
    true_relative = jnp.linalg.norm(rhs-apply(x))/norm
    return x,jnp.array([k,jnp.sqrt(rr)/norm,true_relative])


@partial(jax.jit,static_argnames=('grid','steps','stride','max_iterations'))
def implicit_wave(u0,v0,c,dt,tolerance,*,grid,steps,stride,max_iterations=1000):
    root_mass=jnp.sqrt(jnp.asarray(grid.mass()))
    damping=damping_ratio(grid,c)
    def stiffness(x):
        return c*c*root_mass*positive_laplacian(x/root_mass,grid)
    def apply(x):
        return (1+dt/2*damping)*x+dt*dt/4*stiffness(x)
    def advance(carry,_):
        u,v=carry
        rhs=(1-dt/2*damping)*v-dt*dt/4*stiffness(v)-dt*stiffness(u)
        vn,info=cg_solve(apply,rhs,v,tolerance,max_iterations)
        un=u+dt/2*(v+vn)
        return (un,vn),info
    def block(carry,_):
        state,infos=jax.lax.scan(advance,carry,None,length=stride)
        return state,(state[0]/root_mass,state[1]/root_mass,infos)
    _,(u,v,infos)=jax.lax.scan(block,(root_mass*u0,root_mass*v0),None,length=steps//stride)
    return (jnp.concatenate((u0[None],u)),jnp.concatenate((v0[None],v)),infos.reshape(steps,3))


def verify():
    """Independent edge-matrix assembly + dense first-order trapezoid solve."""
    import scipy.linalg
    from scipy.fft import dstn,idstn
    result={}
    for bc in ('dirichlet','absorbing'):
        grid=base.Grid(6,bc,bc); side=grid.shape[0]; size=side**2
        edge=np.diff(np.eye(side+2),axis=0)[:,1:-1] if bc=='dirichlet' else np.diff(np.eye(side),axis=0)
        w=np.ones(side)/grid.n
        if bc=='absorbing':w[[0,-1]]*=.5
        mass=np.outer(w,w).reshape(-1)
        s=edge.T@edge/grid.h
        stiffness=np.kron(s,np.diag(w))+np.kron(np.diag(w),s)
        c=1.07; boundary=np.zeros(side)
        if bc=='absorbing':boundary[[0,-1]]=2/grid.h
        damp=c*(boundary[:,None]+boundary[None,:]).reshape(-1)
        operator=np.block([[np.zeros((size,size)),np.eye(size)],[-c*c*stiffness/mass[:,None],-np.diag(damp)]])
        rng=np.random.default_rng(790711); u0=rng.normal(size=grid.shape);v0=rng.normal(size=grid.shape)
        dt=.003;steps=10
        propagation=scipy.linalg.solve(np.eye(2*size)-dt/2*operator,np.eye(2*size)+dt/2*operator)
        expected=np.concatenate((u0.reshape(-1),v0.reshape(-1)))
        for _ in range(steps):expected=propagation@expected
        u,v,info=implicit_wave(jnp.asarray(u0),jnp.asarray(v0),c,dt,1e-12,grid=grid,steps=steps,stride=steps)
        actual=np.concatenate((np.asarray(u[-1]).reshape(-1),np.asarray(v[-1]).reshape(-1)))
        error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        symmetry=float(np.linalg.norm(stiffness-stiffness.T))
        eigmin=float(np.linalg.eigvalsh(np.eye(size)+dt/2*np.diag(damp)+dt*dt/4*c*c*stiffness/np.sqrt(mass[:,None]*mass[None,:])).min())
        assert error<1e-10 and symmetry<1e-13 and eigmin>0
        assert np.max(np.asarray(info)[:,2])<1.01e-12
        exact=scipy.linalg.expm(steps*dt*operator)@np.concatenate((u0.reshape(-1),v0.reshape(-1)))
        uf,vf,_=implicit_wave(jnp.asarray(u0),jnp.asarray(v0),c,dt/2,1e-12,grid=grid,steps=2*steps,stride=2*steps)
        finer=np.concatenate((np.asarray(uf[-1]).reshape(-1),np.asarray(vf[-1]).reshape(-1)))
        convergence=float(np.linalg.norm(actual-exact)/np.linalg.norm(finer-exact))
        assert 3.9<convergence<4.1
        result[bc]=dict(dense_relative_error=error,minimum_system_eigenvalue=eigmin,temporal_refinement_ratio=convergence,maximum_true_relative_residual=float(np.max(np.asarray(info)[:,2])))
    return result


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
