"""Exact QR diagnostics and fused invocation of the unchanged weak solver."""
from core import *


def fused_kernel(ops):
    @jax.jit
    def complete(source, bank, S, I, J, W, params, z0, tau):
        fm=ops['project'](source,S,I,J,W)
        ans=ops['solve'](z0,fm,tau)
        field=ops['decode'](ans[0],bank,params)
        return field,ans,fm
    return complete


def fused_query(host_source,ops,tau,kernel):
    start=time.perf_counter()
    src=jax.device_put(host_source)
    src.block_until_ready()
    input_end=time.perf_counter()
    field,ans,fm=kernel(src,ops['bank'],ops['S'],ops['I'],ops['J'],ops['W'],ops['params'],ops['z0'],jnp.asarray(tau))
    jax.block_until_ready((field,ans,fm))
    kernel_end=time.perf_counter()
    field,host_ans=jax.device_get((field,ans))
    end=time.perf_counter()
    stationary=float(ops['stationary'](ans[0],fm,ops['B'],ops['params']))
    z,residual,initial,njac,accepted,attempts,reason=host_ans
    return np.asarray(field),dict(total_seconds=end-start,input_seconds=input_end-start,
        fused_device_seconds=kernel_end-input_end,output_seconds=end-kernel_end,
        projection_init_seconds=None,solver_seconds=None,
        component_note='Projection/solve/decode are fused; separate stage costs are unavailable in this same invocation.',
        reason=int(reason),attempts=int(attempts),accepted=int(accepted),jacobians=int(njac),
        residual=float(residual),initial_residual=float(initial),stationarity=stationary,latent=np.asarray(z).tolist())


def parity(modular, fused):
    uf,rf=fused; um,rm=modular
    counters=['reason','attempts','accepted','jacobians']
    metrics=dict(field_relative=relative(uf,um),latent_relative=relative(rf['latent'],rm['latent']),
        residual_absolute=abs(rf['residual']-rm['residual']),
        initial_absolute=abs(rf['initial_residual']-rm['initial_residual']),
        counters_match=all(rf[k]==rm[k] for k in counters),
        modular_counters={k:rm[k] for k in counters},fused_counters={k:rf[k] for k in counters})
    metrics['passed']=bool(metrics['field_relative']<1e-10 and metrics['latent_relative']<1e-8 and metrics['counters_match'])
    return metrics


def qr_bank(ops):
    start=time.perf_counter()
    q,r=jnp.linalg.qr(ops['bank'],mode='reduced')
    jax.block_until_ready((q,r))
    error=float(jnp.linalg.norm(q@r-ops['bank'])/jnp.linalg.norm(ops['bank']))
    orth=float(jnp.linalg.norm(q.T@q-jnp.eye(r.shape[0])))
    return q,r,dict(seconds=time.perf_counter()-start,relative_reconstruction=error,
        orthogonality_frobenius=orth,qr_r_sha256=sha(r),q_bytes=int(q.size*q.dtype.itemsize))


def oracle_solvers(params,r,k,budgets,trust):
    return {budget:ctol_tol.lm_tau_poisson(lambda z,xy:sc.head(params,z),k,
        np.zeros((r.shape[1],2)),np.ones(r.shape[1]),r,np.ones(r.shape[0]),budget,trust)[0]
        for budget in budgets}


def oracle_fit(field,ops,q,r,solvers,zstarts):
    """Reference-only fit, never returned as a deployed initial guess."""
    u=jnp.asarray(field[1:-1,1:-1].ravel())
    target=q.T@u
    projected=q@target
    perpendicular=float(jnp.linalg.norm(u-projected))
    norm=float(jnp.linalg.norm(u))
    @jax.jit
    def diag(z,params,r,target):
        fun=lambda zz:r@sc.head(params,zz)-target
        residual=fun(z)
        jac=jax.jacfwd(fun)(z)
        grad=jnp.linalg.norm(jac.T@residual)/(jnp.linalg.norm(jac)*jnp.linalg.norm(residual)+1e-300)
        return jnp.linalg.norm(residual),grad
    rows=[];outputs={}
    for budget,solver in solvers.items():
        for start_index,z0 in enumerate(zstarts):
            start=time.perf_counter()
            ans=solver(jnp.asarray(z0),target,jnp.asarray(0.0))
            jax.block_until_ready(ans)
            elapsed=time.perf_counter()-start
            z,residual,initial,njac,accepted,attempts,reason=jax.device_get(ans)
            reduced_norm,stationary=map(float,diag(ans[0],ops['params'],r,target))
            output=np.asarray(ops['decode'](ans[0],ops['bank'],ops['params']))
            actual=relative(output,field)
            qr_error=np.sqrt(reduced_norm**2+perpendicular**2)/norm
            identity=abs(qr_error-actual)
            rows.append(dict(budget=budget,start_index=start_index,latent=z.tolist(),
                same_grid_error=actual,qr_total_error=qr_error,qr_identity_absolute=identity,
                stationarity=stationary,reason=int(reason),attempts=int(attempts),
                accepted=int(accepted),jacobians=int(njac),fit_seconds=elapsed,
                field_sha256=sha(output)))
            outputs[(budget,start_index)]=output
    # Preserve all starts/budgets; best selection is explicitly a diagnostic oracle.
    best=min(rows,key=lambda row:row['same_grid_error'])
    bank_field=np.pad(np.asarray(projected).reshape(field.shape[0]-2,field.shape[1]-2),1)
    return dict(full_bank_same_grid_error=perpendicular/norm,rows=rows,
        best_budget=best['budget'],best_start_index=best['start_index'],
        best_same_grid_error=best['same_grid_error'],best_stationarity=best['stationarity']),bank_field,outputs[(best['budget'],best['start_index'])]
