"""Frozen nonlinear head plus nested linear corrections with exact elimination."""
from speed_core import *


def prepare_correction(ops,codes,directions,count,cfg):
    begin=time.perf_counter();B=ops['B'];C=jnp.asarray(directions[:,:count]);L=B@C
    if count:
        Q,R=jnp.linalg.qr(L,mode='reduced');values=np.asarray(jnp.linalg.svd(R,compute_uv=False));rank=int(np.count_nonzero(values>values[0]*max(L.shape)*np.finfo(float).eps));Bp=B-Q@(Q.T@B)
    else:
        Q=jnp.zeros((B.shape[0],0));R=jnp.zeros((0,0));values=np.zeros(0);rank=0;Bp=B
    reduced={**ops,'B':Bp,'info':{**ops['info'],'operator_sha256':sha(Bp)}};cache=weak_code_cache(reduced,codes)
    info=dict(correction_count=count,nominal_latent_dimension=len(ops['z0'])+count,nonlinear_optimizer_dimension=len(ops['z0']),linear_rank=rank,linear_rank_valid=rank==count,linear_singular_values=values.tolist(),linear_condition_number=float(values[0]/values[-1]) if count else None,
        projected_operator_sha256=sha(Bp),correction_matrix_sha256=sha(C),weak_correction_sha256=sha(L),Q_sha256=sha(Q),R_sha256=sha(R),setup_seconds_including_compile=time.perf_counter()-begin,
        mechanism='added linear correction capacity plus exact analytic elimination; not a pure width-only solver comparison')
    solve=make_lm_kernel(reduced,cfg['online_preset']['budget'],True,False,cfg['linear_backward_error_limit'],cfg['online_preset']['stationarity_stop']);n=ops['info']['intervals'];k=len(ops['z0'])
    @jax.jit
    def kernel(source,bank,S,I,J,W,params,predictions,cached_codes,Q,R,C,B,Bp):
        f=ops['project'](source,S,I,J,W);fp=f-Q@(Q.T@f) if count else f
        squared=jnp.sum((predictions-fp[None,:])**2,axis=1);_,indices=jax.lax.top_k(-squared,1);index=indices[0];z0=cached_codes[index]
        answer,stats,_=solve(z0,fp,jnp.asarray(cfg['online_preset']['tau']));z=answer[0];h=sc.head(params,z);h0=sc.head(params,z0)
        if count:
            rhs=Q.T@(f-B@h);y=jax.scipy.linalg.solve_triangular(R,rhs,lower=False);initial_y=jax.scipy.linalg.solve_triangular(R,Q.T@(f-B@h0),lower=False)
            linear_error=jnp.linalg.norm(R@y-rhs)/(jnp.linalg.norm(R)*jnp.linalg.norm(y)+jnp.linalg.norm(rhs)+1e-300)
        else:y=jnp.zeros((0,));initial_y=y;linear_error=jnp.asarray(0.)
        coefficients=h+C@y;field=jnp.pad((bank@coefficients).reshape(n-1,n-1),1)
        residual=B@coefficients-f;reduced_residual=Bp@h-fp;D=jax.jacfwd(lambda zz:sc.head(params,zz))(z);Jfull=B@D;Jreduced=Bp@D;linear_columns=B@C
        full_gradient=jnp.concatenate((Jfull.T@residual,linear_columns.T@residual));full_stationarity=jnp.linalg.norm(full_gradient)/(jnp.sqrt(jnp.sum(Jfull*Jfull)+jnp.sum(linear_columns*linear_columns))*jnp.linalg.norm(residual)+1e-300)
        reduced_stationarity=jnp.linalg.norm(Jreduced.T@reduced_residual)/(jnp.linalg.norm(Jreduced)*jnp.linalg.norm(reduced_residual)+1e-300)
        singular=jnp.linalg.svd(Jreduced,compute_uv=False);threshold=singular[0]*max(Jreduced.shape)*jnp.finfo(jnp.float64).eps;projected_rank=jnp.sum(singular>threshold)
        reconstruct=jnp.linalg.norm(residual-reduced_residual)/(jnp.linalg.norm(f)+jnp.linalg.norm(B@h)+1e-300)
        initial_full=jnp.linalg.norm(B@(h0+C@initial_y)-f)
        diagnostics=(full_stationarity,reduced_stationarity,linear_error,reconstruct,jnp.linalg.norm(residual),initial_full,projected_rank,singular)
        return field,answer,stats,index,z0,y,initial_y,diagnostics
    return dict(kernel=kernel,cache=cache,Q=Q,R=R,C=C,Bp=Bp,info=info)


def correction_query(host_source,ops,engine,cfg):
    start=time.perf_counter();source=jax.device_put(host_source);source.block_until_ready();input_end=time.perf_counter();cache=engine['cache']
    output=engine['kernel'](source,ops['bank'],ops['S'],ops['I'],ops['J'],ops['W'],ops['params'],cache['predictions'],cache['codes'],engine['Q'],engine['R'],engine['C'],ops['B'],engine['Bp']);jax.block_until_ready(output);device_end=time.perf_counter()
    field,answer,stats,index,z0,y,y0,diag=jax.device_get(output);end=time.perf_counter();z,res,initial,njac,accepted,attempts,reason=answer;full_gradient,reduced_gradient,recovery_error,reconstruction_error,full_residual,initial_full,rank,singular=diag
    full_ok=bool(full_gradient<=cfg['stationarity_tolerance']);reduced_ok=bool(reduced_gradient<=cfg['stationarity_tolerance']);linear_ok=bool(stats[0]<=cfg['linear_backward_error_limit'] and recovery_error<=cfg['linear_backward_error_limit']);rank_ok=bool(rank==len(z) and engine['info']['linear_rank_valid']);finite=bool(np.isfinite(field).all())
    item=dict(latent=z.tolist(),initial_latent=z0.tolist(),correction_coefficients=y.tolist(),initial_correction_coefficients=y0.tolist(),augmented_latent=np.concatenate((z,y)).tolist(),selected_training_code_index=int(index),residual=float(res),initial_residual=float(initial),full_residual=float(full_residual),initial_full_residual=float(initial_full),absolute_tau_threshold=float(cfg['online_preset']['tau']*initial),reason=int(reason),attempts=int(attempts),accepted=int(accepted),jacobians=int(njac),
        stationarity=float(full_gradient),stationary=full_ok,reduced_stationarity=float(reduced_gradient),reduced_stationary=reduced_ok,max_linear_backward_error=float(stats[0]),fallback_count=int(stats[1]),max_proposed_backward_error=float(stats[2]),linear_recovery_backward_error=float(recovery_error),residual_reconstruction_scaled=float(reconstruction_error),projected_jacobian_rank=int(rank),projected_jacobian_singular_values=np.asarray(singular).tolist(),projected_jacobian_rank_valid=rank_ok)
    row=dict(starts=[item],selected_start=0,**item,total_seconds=end-start,input_seconds=input_end-start,fused_device_seconds=device_end-input_end,output_seconds=end-device_end,total_attempts=int(attempts),total_jacobians=int(njac),all_start_linear_valid=linear_ok,
        correction_count=engine['info']['correction_count'],nominal_latent_dimension=engine['info']['nominal_latent_dimension'],nonlinear_optimizer_dimension=len(z),projection='skinny_sine_products',initialization='nearest_cached_projected_weak_training_prediction',
        solver_valid=bool(finite and full_ok and reduced_ok and linear_ok and rank_ok and reconstruction_error<=cfg['residual_reconstruction_limit']),
        timing_scope='input, source contraction, projected nearest-code lookup, nonlinear solve, exact y recovery, full decode, full/reduced stationarity and projected-Jacobian rank diagnostics, output copy')
    return np.asarray(field),row
