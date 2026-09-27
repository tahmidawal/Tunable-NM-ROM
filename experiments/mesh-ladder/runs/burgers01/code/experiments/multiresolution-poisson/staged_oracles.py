"""Untimed, full-field stationary head fits; truth never enters online queries."""
from staged_training import *
from kernel_solver import make_lm_kernel


def prepare_head_oracle(ops,Q,R,codes,cfg,linear_limit):
    augmented=jnp.vstack((R,jnp.zeros((1,R.shape[1]))));local=dict(B=augmented,params=ops['params'],z0=ops['z0'],info=ops['info'])
    solve=make_lm_kernel(local,cfg['budget'],True,False,linear_limit,cfg['stationarity_tolerance'])
    run=jax.jit(lambda guesses,target:jax.lax.map(lambda z:solve(z,target,jnp.asarray(0.))[:2],guesses))
    @jax.jit
    def stationarity(z,target):
        fun=lambda zz:augmented@sc.head(ops['params'],zz)-target
        r=fun(z);J=jax.jacfwd(fun)(z);return jnp.linalg.norm(J.T@r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300)
    predictions=sc.head(ops['params'],jnp.asarray(codes))@R.T;predictions.block_until_ready()
    return dict(run=run,stationarity=stationarity,predictions=predictions,codes=jnp.asarray(codes),mean=jnp.asarray(codes.mean(0)))


def head_oracle(same,ops,Q,R,engine,cfg,linear_limit):
    start=time.perf_counter();truth=jnp.asarray(same[1:-1,1:-1].ravel());target=Q.T@truth;perpendicular=jnp.linalg.norm(truth-Q@target)
    distances=jnp.sum((engine['predictions']-target)**2,axis=1);index=int(jnp.argmin(distances));guesses=jnp.stack((engine['mean'],engine['codes'][index]));augmented_target=jnp.concatenate((target,-perpendicular[None]))
    answers,stats=engine['run'](guesses,augmented_target);jax.block_until_ready((answers,stats));host,hs=jax.device_get((answers,stats));outputs=[]
    for i in range(len(guesses)):
        z,residual,initial,njac,accepted,attempts,reason=[x[i] for x in host];gradient=float(engine['stationarity'](answers[0][i],augmented_target));field=np.asarray(ops['decode'](answers[0][i],ops['bank'],ops['params']))
        row=dict(start_id=cfg['starts'][i],initial_latent=np.asarray(guesses[i]).tolist(),latent=z.tolist(),training_code_index=-1 if i==0 else index,
            residual=float(residual),initial_residual=float(initial),stationarity=gradient,stationary=gradient<=cfg['stationarity_tolerance'],attempts=int(attempts),accepted=int(accepted),jacobians=int(njac),reason=int(reason),
            max_linear_backward_error=float(hs[i,0]),fallback_count=int(hs[i,1]),same_grid_relative_error=relative(field,same),perpendicular_norm=float(perpendicular),
            solver_valid=bool(np.isfinite(field).all() and gradient<=cfg['stationarity_tolerance'] and hs[i,0]<=linear_limit))
        outputs.append((field,row))
    return outputs,dict(seconds_including_compile_and_diagnostics=time.perf_counter()-start,objective='full field Euclidean norm, QR residual plus perpendicular constant; reference-only, never online',nearest_full_field_training_code_index=index)
