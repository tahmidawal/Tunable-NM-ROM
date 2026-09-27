"""Instrumented original unpreconditioned CG: supplied field, zero start, f64."""
from core import *


def make_cg(intervals, maxiter):
    def operator(u):
        padded=jnp.pad(u,1)
        return intervals**2*(4*u-padded[:-2,1:-1]-padded[2:,1:-1]
                             -padded[1:-1,:-2]-padded[1:-1,2:])
    @jax.jit
    def solve(source,tolerance):
        b=source[1:-1,1:-1];rr=jnp.vdot(b,b).real;threshold=tolerance**2*rr
        init=(jnp.zeros_like(b),b,b,rr,jnp.int32(0))
        def cond(state):return (state[3]>threshold)&(state[4]<maxiter)&jnp.isfinite(state[3])
        def body(state):
            x,r,p,r2,count=state;ap=operator(p);alpha=r2/jnp.vdot(p,ap).real
            x=x+alpha*p;r=r-alpha*ap;new2=jnp.vdot(r,r).real
            return x,r,r+(new2/r2)*p,new2,count+1
        x,r,p,r2,count=jax.lax.while_loop(cond,body,init)
        true_relative=jnp.linalg.norm(b-operator(x))/(jnp.sqrt(rr)+1e-300)
        recursive_relative=jnp.sqrt(r2/(rr+1e-300))
        converged=jnp.isfinite(true_relative)&(true_relative<=tolerance*(1+1e-6)+1e-12)
        return jnp.pad(x,1),(count,true_relative,recursive_relative,converged)
    return solve


def cg_query(host_source,kernel,tolerance):
    start=time.perf_counter();src=jax.device_put(host_source);src.block_until_ready();input_end=time.perf_counter()
    field,info=kernel(src,jnp.asarray(tolerance));jax.block_until_ready((field,info));device_end=time.perf_counter()
    field,info=jax.device_get((field,info));end=time.perf_counter()
    count,true,recursive,converged=info
    return np.asarray(field),dict(total_seconds=end-start,input_seconds=input_end-start,
        fused_device_seconds=device_end-input_end,output_seconds=end-device_end,
        iterations=int(count),true_relative_residual=float(true),recursive_relative_residual=float(recursive),
        cg_converged=bool(converged),cg_tolerance=tolerance,reason='converged' if converged else 'budget_or_breakdown',
        solver_valid=bool(converged))


def verify_cg():
    from scipy.fft import dstn
    n=16;source=np.pad(np.random.default_rng(7090741).normal(size=(n-1,n-1)),1)
    kernel=make_cg(n,1000);out,info=kernel(jnp.asarray(source),1e-12)
    lam=eigenvalues(n);exact=np.pad(dstn(dstn(source[1:-1,1:-1],type=1,norm='ortho')/lam,type=1,norm='ortho'),1)
    installed=jax.scipy.sparse.linalg.cg(lambda u:mp.neg_lap_interior(u,n+1),jnp.asarray(source[1:-1,1:-1]),tol=1e-12,maxiter=1000)[0]
    err=relative(out,exact);parity=relative(out[1:-1,1:-1],installed)
    assert err<1e-11 and parity<1e-11 and bool(info[3])
    return dict(scipy_dst_field_relative=err,installed_jax_cg_field_relative=parity,iterations=int(info[0]))
