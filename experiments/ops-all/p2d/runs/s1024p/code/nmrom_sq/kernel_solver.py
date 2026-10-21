"""Controlled small SPD solve ablation of the unchanged incumbent Poisson LM.

The unrolled elimination follows the existing b1d_fast_common/Burgers algebra.
This version adds symmetric diagonal scaling and a backward-error gate, with a
charged generic solve fallback. The incumbent ctol_tol.py is never edited.
"""
from core import *


def unrolled_gj(A,b):
    n=A.shape[0];mat=jnp.concatenate((A,b[:,None]),axis=1);rows=jnp.arange(n)
    for k in range(n):
        row=mat[k]/mat[k,k]
        mat=mat-jnp.where(rows==k,0.,mat[:,k])[:,None]*row[None,:]
        mat=jnp.where((rows==k)[:,None],row[None,:],mat)
    return mat[:,n]


def backward_error(A,x,b):
    return jnp.linalg.norm(A@x-b)/(jnp.linalg.norm(A)*jnp.linalg.norm(x)+jnp.linalg.norm(b)+1e-300)


def guarded_gj(A,b,residual_limit=1e-12):
    diagonal=jnp.diag(A)
    scale=jax.lax.rsqrt(jnp.where(diagonal>0,diagonal,1.))
    x=scale*unrolled_gj(scale[:,None]*A*scale[None,:],scale*b)
    eta=backward_error(A,x,b)
    good=jnp.all(diagonal>0)&jnp.all(jnp.isfinite(x))&jnp.isfinite(eta)&(eta<=residual_limit)
    result=jax.lax.cond(good,lambda:x,lambda:jnp.linalg.solve(A,b))
    return result,backward_error(A,result,b),(~good).astype(jnp.int32),eta


def make_lm_kernel(ops,budget,specialized=True,trace=False,residual_limit=1e-12,stationarity_tol=None):
    """Incumbent LM schedules/stops, changed linear solve plus diagnostics.

    Returns (same original seven outputs, [max backward error, fallbacks,
    max proposed backward error], optional trajectory arrays). Trace is a
    validation replay, not included in production timing. Optional stationarity_tol
    activates a normalized-gradient stop (reason 6); None is the unchanged
    incumbent schedule. The post-query audit threshold remains separate.
    """
    B,params=ops['B'],ops['params'];K=ops['z0'].shape[0];trust=ops['info']['trust_delta']
    def r_of(z,fm):
        return jnp.ones(B.shape[0])*(B@(jnp.ones(B.shape[1])*sc.head(params,z)))-fm
    rJ=lambda z,fm:(r_of(z,fm),jax.jacfwd(r_of)(z,fm))
    rn_fn=lambda z,fm:jnp.linalg.norm(r_of(z,fm))
    def lm(z0,fm,tau):
        r0,J0=rJ(z0,fm);v0=jnp.linalg.norm(r0);tol=tau*v0
        reason=jnp.where(~jnp.isfinite(v0),jnp.int32(5),jnp.where((tau>0)&(v0<=tol),jnp.int32(2),jnp.int32(0)))
        if stationarity_tol is not None:
            gradient=jnp.linalg.norm(J0.T@r0)/(jnp.linalg.norm(J0)*v0+1e-300)
            reason=jnp.where((reason==0)&(gradient<=stationarity_tol),jnp.int32(6),reason)
        init=(z0,J0,r0,v0,jnp.asarray(1e-6,jnp.float64),jnp.int32(0),jnp.int32(0),jnp.int32(1),reason,
              jnp.asarray(0.,jnp.float64),jnp.int32(0),jnp.asarray(0.,jnp.float64))
        if trace:
            init += (jnp.zeros((budget,K,K)),jnp.zeros((budget,K)),jnp.zeros((budget,K)),
                     jnp.zeros(budget),jnp.zeros(budget),jnp.zeros(budget,jnp.int32),
                     jnp.zeros((budget,K)),jnp.zeros(budget,jnp.bool_))
        def cond(s):return (s[8]==0)&(s[5]<budget)
        def body(s):
            z,J,r,val,lam,att,acc,nJ,_=s[:9]
            H=J.T@J;g=J.T@r
            D=jnp.diag(jnp.diag(H))+1e-30*jnp.eye(K,dtype=jnp.float64)
            A=H+lam*D
            if specialized:dz,eta,fallback,proposed_eta=guarded_gj(A,-g,residual_limit)
            else:
                dz=jnp.linalg.solve(A,-g);eta=backward_error(A,dz,-g)
                fallback=jnp.int32(0);proposed_eta=eta
            finite=jnp.all(jnp.isfinite(dz));within=jnp.linalg.norm(dz)<=trust;admissible=finite&within
            z_new=z+jnp.where(admissible,dz,0.)
            v_new=jnp.where(admissible,rn_fn(z_new,fm),jnp.inf)
            accept=admissible&jnp.isfinite(v_new)&(v_new<val)
            rel_dec=jnp.where(accept,(val-v_new)/(jnp.abs(val)+1e-300),1.)
            step=jnp.linalg.norm(dz)/(1.+jnp.linalg.norm(z))
            r2,J2=jax.lax.cond(accept,lambda:rJ(z_new,fm),lambda:(r,J))
            z_final=jnp.where(accept,z_new,z);val=jnp.where(accept,v_new,val)
            lam_new=jnp.where(accept,jnp.maximum(lam/3.,1e-12),jnp.minimum(lam*10.,1e12))
            acc=acc+accept.astype(jnp.int32);nJ=nJ+accept.astype(jnp.int32)
            reason=jnp.where(accept&(tau>0)&(val<=tol),jnp.int32(2),
                jnp.where(accept&((rel_dec<1e-12)|(step<1e-13)),jnp.int32(1),
                    jnp.where((~accept)&(lam_new>=1e12),jnp.int32(3),jnp.int32(0))))
            if stationarity_tol is not None:
                gradient=jnp.linalg.norm(J2.T@r2)/(jnp.linalg.norm(J2)*val+1e-300)
                reason=jnp.where((reason==0)&(gradient<=stationarity_tol),jnp.int32(6),reason)
            result=(z_final,J2,r2,val,lam_new,att+1,acc,nJ,reason,
                jnp.maximum(s[9],eta),s[10]+fallback,jnp.maximum(s[11],jnp.nan_to_num(proposed_eta,nan=jnp.inf)))
            if trace:
                result += (s[12].at[att].set(A),s[13].at[att].set(-g),s[14].at[att].set(dz),
                    s[15].at[att].set(lam),s[16].at[att].set(eta),s[17].at[att].set(fallback),
                    s[18].at[att].set(z),s[19].at[att].set(accept))
            return result
        done=jax.lax.while_loop(cond,body,init)
        z,J,r,val,lam,att,acc,nJ,reason=done[:9]
        return (z,val,v0,nJ,acc,att,reason),jnp.stack((done[9],done[10],done[11])),done[12:]
    return jax.jit(lm)


def specialized_kernel(ops,solve):
    @jax.jit
    def complete(source,bank,S,I,J,W,params,z0,tau):
        fm=ops['project'](source,S,I,J,W)
        ans,diagnostics,_=solve(z0,fm,tau)
        field=ops['decode'](ans[0],bank,params)
        return field,ans,fm,diagnostics
    return complete


def specialized_query(host_source,ops,tau,kernel):
    start=time.perf_counter();src=jax.device_put(host_source);src.block_until_ready();input_end=time.perf_counter()
    field,ans,fm,diagnostics=kernel(src,ops['bank'],ops['S'],ops['I'],ops['J'],ops['W'],ops['params'],ops['z0'],jnp.asarray(tau))
    jax.block_until_ready((field,ans,fm,diagnostics));kernel_end=time.perf_counter()
    field,host_ans,diagnostics=jax.device_get((field,ans,diagnostics));end=time.perf_counter()
    stationary=float(ops['stationary'](ans[0],fm,ops['B'],ops['params']))
    z,residual,initial,njac,accepted,attempts,reason=host_ans
    return np.asarray(field),dict(total_seconds=end-start,input_seconds=input_end-start,
        fused_device_seconds=kernel_end-input_end,output_seconds=end-kernel_end,
        projection_init_seconds=None,solver_seconds=None,reason=int(reason),attempts=int(attempts),
        accepted=int(accepted),jacobians=int(njac),residual=float(residual),initial_residual=float(initial),
        stationarity=stationary,latent=z.tolist(),max_linear_backward_error=float(diagnostics[0]),
        fallback_count=int(diagnostics[1]),max_proposed_backward_error=float(diagnostics[2]))


def solution_agreement(control,candidate,limits):
    uc,rc=control;uk,rk=candidate
    field=relative(uk,uc);latent=relative(rk['latent'],rc['latent'])
    # Scale by the initial residual: terminal gradients/residual differences
    # near cancellation must not acquire meaningless relative blow-ups.
    objective=abs(rk['residual']-rc['residual'])/(rc['initial_residual']+1e-300)
    out=dict(field_relative=field,latent_relative=latent,objective_scaled_difference=objective,
        candidate_stationarity=rk['stationarity'],control_stationarity=rc['stationarity'],
        candidate_reason=rk['reason'],control_reason=rc['reason'],
        attempt_difference=rk['attempts']-rc['attempts'],accepted_difference=rk['accepted']-rc['accepted'],
        jacobian_difference=rk['jacobians']-rc['jacobians'])
    out['passed']=bool(field<=limits['field_relative'] and latent<=limits['latent_relative'] and objective<=limits['objective_initial_scaled'])
    return out
