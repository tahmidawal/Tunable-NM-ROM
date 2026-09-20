"""Counted full-grid cold-start CG for the SPD Dirichlet Poisson stencil."""
import jax
import jax.numpy as jnp
import common as C


def settings(cfg, n):
    spec=cfg.get('iterative_cg')
    if spec is None:return []
    assert spec['preconditioner']=='identity'
    tolerances=spec['relative_tolerances']
    assert tolerances==[1e-2,1e-4,1e-6]
    cap=int(spec['max_iterations_per_interval'])*n
    assert cap>0
    return [(f'cg_identity_rtol{tol:.0e}',float(tol),cap) for tol in tolerances]


def engine(n, rtol, maxiter, retain_history=True):
    @jax.jit
    def solve(f):
        f=jnp.asarray(f,dtype=jnp.float64)
        norm2=jnp.vdot(f,f);threshold=rtol**2*norm2
        def condition(s):
            x,r,p,rho,k,healthy,history=s
            return (k<maxiter)&(rho>threshold)&healthy
        def step(s):
            x,r,p,rho,k,healthy,history=s
            ap=C.negative_laplacian(p,n);pap=jnp.vdot(p,ap)
            healthy=healthy&jnp.isfinite(pap)&(pap>0)
            alpha=jnp.where(healthy,rho/pap,0.)
            x=x+alpha*p;r=r-alpha*ap;newrho=jnp.vdot(r,r)
            beta=jnp.where(rho>0,newrho/rho,0.)
            if retain_history:history=history.at[k].set(jnp.array([alpha,beta,rho,pap,newrho]))
            return x,r,r+beta*p,newrho,k+1,healthy&jnp.isfinite(newrho),history
        x,r,p,rho,k,healthy,history=jax.lax.while_loop(condition,step,
            (jnp.zeros_like(f),f,f,norm2,jnp.int32(0),jnp.bool_(True),jnp.zeros((maxiter if retain_history else 0,5),dtype=jnp.float64)))
        # Timed true-residual certification, independent of recurrence stopping.
        residual=f-C.negative_laplacian(x,n)
        denom=jnp.maximum(jnp.sqrt(norm2),jnp.finfo(jnp.float64).tiny)
        true=jnp.linalg.norm(residual)/denom
        recursive=jnp.sqrt(rho)/denom
        converged=healthy&jnp.isfinite(true)&(true<=rtol)
        stats=jnp.array([k,true,recursive,converged,k>=maxiter,healthy,k+1],dtype=jnp.float64)
        return (x,stats,history) if retain_history else (x,stats)
    return solve


def counters(stats,rtol,maxiter):
    k,true,recursive,converged,cap,healthy,matvecs=map(float,stats)
    return dict(stationary=bool(converged),iterations=int(k),cg_converged=bool(converged),
        cg_true_relative_residual=true,cg_recursive_relative_residual=recursive,
        cg_hit_iteration_cap=bool(cap),cg_healthy=bool(healthy),cg_matvecs=int(matvecs),
        cg_relative_tolerance=rtol,cg_max_iterations=maxiter,cg_preconditioner='identity',
        cg_stopping_reason='true_residual_pass' if converged else ('breakdown' if not healthy else
            ('iteration_cap' if cap else 'true_residual_failed')))
