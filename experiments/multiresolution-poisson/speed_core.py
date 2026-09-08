"""Generic, isolated source projection and training-code initialization utilities.

No selected continuation or evaluation answer is embedded. Cache arrays remain
explicit compiled-function arguments, and online lookup/guards are charged.
"""
from core import *
from kernel_solver import make_lm_kernel


@jax.jit
def project_source_dst(source,I,J,W):
    return dst2(source[1:-1,1:-1])[I,J]*W


@jax.jit
def nearest_code(fm,predictions,codes):
    squared=jnp.sum((predictions-fm[None,:])**2,axis=1)
    index=jnp.argmin(squared)
    first=squared[index]
    second=jnp.min(jnp.where(jnp.arange(len(squared))==index,jnp.inf,squared))
    return codes[index],index,jnp.sqrt(first),second-first


def weak_code_cache(ops,codes):
    start=time.perf_counter();zc=jnp.asarray(codes)
    build=jax.jit(lambda p,z,B:sc.head(p,z)@B.T)
    predictions=build(ops['params'],zc,ops['B']);predictions.block_until_ready()
    return dict(codes=zc,predictions=predictions,info=dict(count=len(codes),
        prediction_shape=list(predictions.shape),prediction_bytes=int(predictions.size*predictions.dtype.itemsize),
        code_bytes=int(zc.size*zc.dtype.itemsize),setup_seconds_including_compile=time.perf_counter()-start,
        training_codes_sha256=sha(codes),prediction_sha256=sha(predictions),
        operator_sha256=ops['info']['operator_sha256'],
        scope='Cached scaled weak decoder predictions from training codes only; no reference fields or descriptors'))


def make_speed_kernel(ops,budget,projection,initialization,linear_limit=1e-12):
    assert projection in ('skinny_sine_products','forward_dst_and_gather')
    assert initialization in ('mean_training_code','nearest_cached_scaled_weak_prediction')
    solve=make_lm_kernel(ops,budget,True,False,linear_limit)
    @jax.jit
    def complete(source,bank,S,I,J,W,params,mean_code,cache_predictions,cache_codes,tau):
        if projection=='forward_dst_and_gather':fm=project_source_dst(source,I,J,W)
        else:fm=ops['project'](source,S,I,J,W)
        if initialization=='nearest_cached_scaled_weak_prediction':
            z0,index,distance,gap=nearest_code(fm,cache_predictions,cache_codes)
        else:
            z0=mean_code;index=jnp.int64(-1);distance=jnp.asarray(0.);gap=jnp.asarray(0.)
        ans,stats,_=solve(z0,fm,tau)
        field=ops['decode'](ans[0],bank,params)
        return field,ans,fm,stats,(index,distance,gap,z0)
    return complete


def speed_query(host_source,ops,cache,tau,kernel,projection,initialization):
    start=time.perf_counter();source=jax.device_put(host_source);source.block_until_ready();input_end=time.perf_counter()
    field,ans,fm,stats,init=kernel(source,ops['bank'],ops['S'],ops['I'],ops['J'],ops['W'],
        ops['params'],ops['z0'],cache['predictions'],cache['codes'],jnp.asarray(tau))
    jax.block_until_ready((field,ans,fm,stats,init));device_end=time.perf_counter()
    field,host_ans,stats,init=jax.device_get((field,ans,stats,init));end=time.perf_counter()
    stationarity=float(ops['stationary'](ans[0],fm,ops['B'],ops['params']))
    z,residual,initial,njac,accepted,attempts,reason=host_ans
    index,distance,gap,z0=init;nearest=initialization=='nearest_cached_scaled_weak_prediction'
    return np.asarray(field),dict(total_seconds=end-start,input_seconds=input_end-start,
        fused_device_seconds=device_end-input_end,output_seconds=end-device_end,
        projection_init_seconds=None,solver_seconds=None,projection=projection,initialization=initialization,
        reason=int(reason),attempts=int(attempts),accepted=int(accepted),jacobians=int(njac),
        residual=float(residual),initial_residual=float(initial),absolute_tau_threshold=float(tau*initial),
        stationarity=stationarity,latent=z.tolist(),initial_latent=np.asarray(z0).tolist(),
        selected_training_code_index=int(index),cache_distance=float(distance) if nearest else None,
        cache_squared_distance_gap=float(gap) if nearest else None,
        cache_relative_squared_gap=float(gap/(distance*distance+1e-300)) if nearest else None,
        cache_near_tie=bool(gap<=1e-12*(distance*distance+1e-300)) if nearest else None,
        cache_near_tie_relative_threshold=1e-12 if nearest else None,
        max_linear_backward_error=float(stats[0]),fallback_count=int(stats[1]),max_proposed_backward_error=float(stats[2]),
        component_note='Source projection, optional nearest lookup, guarded LM and decoding share the charged device interval; initialization metadata is copied to host before the timer ends.')


def projection_agreement(source,ops,cache):
    """Untimed coefficients/lookup validation, not an online timing component."""
    source=jnp.asarray(source)
    thin=ops['project'](source,ops['S'],ops['I'],ops['J'],ops['W'])
    fft=project_source_dst(source,ops['I'],ops['J'],ops['W'])
    left=nearest_code(thin,cache['predictions'],cache['codes'])
    right=nearest_code(fft,cache['predictions'],cache['codes'])
    return dict(coefficient_relative=relative(fft,thin),
        thin_selected_code_index=int(left[1]),dst_selected_code_index=int(right[1]),
        lookup_index_matches=bool(int(left[1])==int(right[1])),
        thin_squared_distance_gap=float(left[3]),dst_squared_distance_gap=float(right[3]),
        coefficient_scope='same scaled weak sine coefficients; floating-point FFT and matrix-product roundoff may differ')
