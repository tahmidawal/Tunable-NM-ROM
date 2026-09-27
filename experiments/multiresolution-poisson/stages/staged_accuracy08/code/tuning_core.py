"""Fixed-checkpoint online work controls; source-based starts, exact weak algebra."""
from speed_core import *


def make_tuning_kernel(ops,preset,linear_limit):
    solve=make_lm_kernel(ops,preset['budget'],True,False,linear_limit,preset['stationarity_stop'])
    count=preset['starts']
    @jax.jit
    def complete(source,bank,S,I,J,W,params,cache_predictions,cache_codes,tau):
        fm=ops['project'](source,S,I,J,W)
        squared=jnp.sum((cache_predictions-fm[None,:])**2,axis=1)
        _,indices=jax.lax.top_k(-squared,count)
        guesses=cache_codes[indices]
        # Sequential map preserves the single-start LM arithmetic; all starts
        # are charged. Selection uses weak residual only, never a truth field.
        answers,stats=jax.lax.map(lambda z:solve(z,fm,tau)[:2],guesses)
        chosen=jnp.argmin(jnp.where(jnp.isfinite(answers[1]),answers[1],jnp.inf))
        field=ops['decode'](answers[0][chosen],bank,params)
        return field,answers,stats,chosen,indices,guesses,fm
    return complete


def tuning_query(host_source,ops,cache,preset,kernel,limits):
    start=time.perf_counter();source=jax.device_put(host_source);source.block_until_ready();input_end=time.perf_counter()
    field,answers,stats,chosen,indices,guesses,fm=kernel(source,ops['bank'],ops['S'],ops['I'],ops['J'],ops['W'],ops['params'],cache['predictions'],cache['codes'],jnp.asarray(preset['tau']))
    jax.block_until_ready((field,answers,stats,chosen,indices,guesses,fm));device_end=time.perf_counter()
    field,host_answers,stats,chosen,indices,guesses=jax.device_get((field,answers,stats,chosen,indices,guesses));end=time.perf_counter()
    selected=int(chosen)
    stationarity=[float(ops['stationary'](answers[0][j],fm,ops['B'],ops['params'])) for j in range(preset['starts'])]
    starts=[]
    for j in range(preset['starts']):
        z,residual,initial,njac,accepted,attempts,reason=[a[j] for a in host_answers]
        starts.append(dict(latent=z.tolist(),initial_latent=guesses[j].tolist(),selected_training_code_index=int(indices[j]),residual=float(residual),initial_residual=float(initial),absolute_tau_threshold=float(preset['tau']*initial),reason=int(reason),attempts=int(attempts),accepted=int(accepted),jacobians=int(njac),stationarity=stationarity[j],stationary=stationarity[j]<=limits['stationarity_tolerance'],max_linear_backward_error=float(stats[j,0]),fallback_count=int(stats[j,1]),max_proposed_backward_error=float(stats[j,2])))
    row=dict(starts=starts,selected_start=selected,**starts[selected])
    row.update(total_seconds=end-start,input_seconds=input_end-start,fused_device_seconds=device_end-input_end,output_seconds=end-device_end,
        total_attempts=sum(x['attempts'] for x in starts),total_jacobians=sum(x['jacobians'] for x in starts),
        all_start_linear_valid=all(x['max_linear_backward_error']<=limits['linear_backward_error_limit'] for x in starts),
        projection='skinny_sine_products',initialization='nearest_cached_scaled_weak_prediction',
        start_selection='minimum terminal exact weak residual, deterministic nearest-training-code starts',
        solver_valid=bool(np.isfinite(field).all() and all(x['max_linear_backward_error']<=limits['linear_backward_error_limit'] for x in starts) and (starts[selected]['stationary'] or starts[selected]['reason']==2)))
    return np.asarray(field),row
