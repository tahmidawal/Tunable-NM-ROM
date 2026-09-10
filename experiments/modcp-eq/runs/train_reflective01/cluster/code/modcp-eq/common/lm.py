"""Damped least squares with explicit normalized-gradient stopping semantics."""
import jax
import jax.numpy as jnp


def stationarity(r,J):
    # Cosine of residual with each tangent direction; scale invariant. A zero
    # residual is stationary. A zero tangent is recorded in diagnostics by users.
    rn=jnp.linalg.norm(r)
    den=jnp.linalg.norm(J,axis=0)*jnp.maximum(rn,1e-30)
    return jnp.max(jnp.abs(J.T@r)/jnp.maximum(den,1e-30))


def make_lm(fun,cap=30):
    """Return z, attempts, reason, stationarity, residual norm.

    reason: 0 budget; 1 normalized-gradient convergence; 2 tiny accepted step;
    3 nonfinite value/derivative; 4 damping ceiling; 5 zero tangent at nonzero
    residual. Only reason 1 is stationary; full-rank audits are separate.
    """
    derivative=jax.jacfwd(fun)
    def solve(z0,args,tol):
        r=fun(z0,*args);J=derivative(z0,*args)
        rn=jnp.linalg.norm(r);stat=stationarity(r,J)
        finite=jnp.all(jnp.isfinite(J))&jnp.isfinite(rn)&jnp.isfinite(stat)
        degenerate=(jnp.linalg.norm(J)<=1e-24)&(rn>1e-14)
        reason=jnp.where(~finite,3,jnp.where(degenerate,5,jnp.where((stat<=tol)|(rn<=1e-14),1,0))).astype(jnp.int32)
        def body(s):
            z,r,J,rn,stat,lam,it,reason=s
            H=J.T@J;g=J.T@r
            diag=jnp.maximum(jnp.diag(H),jnp.maximum(jnp.max(jnp.diag(H))*1e-12,1e-24))
            dz=jnp.linalg.solve(H+lam*jnp.diag(diag),-g)
            zn=z+dz;r2=fun(zn,*args);rn2=jnp.linalg.norm(r2)
            accept=jnp.all(jnp.isfinite(dz))&jnp.isfinite(rn2)&(rn2<rn)
            z2=jnp.where(accept,zn,z);r2=jnp.where(accept,r2,r)
            J2=jax.lax.cond(accept,lambda:derivative(z2,*args),lambda:J)
            rn2=jnp.where(accept,rn2,rn);st2=stationarity(r2,J2)
            tiny=accept&(jnp.linalg.norm(dz)<=1e-12*(1+jnp.linalg.norm(z)))
            finite2=jnp.all(jnp.isfinite(J2))&jnp.isfinite(st2)
            degenerate2=(jnp.linalg.norm(J2)<=1e-24)&(rn2>1e-14)
            reason=jnp.where(~finite2,3,jnp.where(degenerate2,5,jnp.where((st2<=tol)|(rn2<=1e-14),1,jnp.where(tiny,2,jnp.where((~accept)&(lam>=1e12),4,0))))).astype(jnp.int32)
            return z2,r2,J2,rn2,st2,jnp.where(accept,jnp.maximum(lam/3,1e-12),jnp.minimum(lam*10,1e12)),it+1,reason
        s=jax.lax.while_loop(lambda s:(s[6]<cap)&(s[7]==0),body,
             (z0,r,J,rn,stat,jnp.asarray(1e-3),jnp.int32(0),reason))
        return s[0],s[6],s[7],s[4],s[3]
    return solve
