"""Free-coefficient neural bank training, field-metric head fitting and continuation."""
from training_core import *


def bank_qr(params,U,coords):
    start=time.perf_counter();G=jax.jit(sc.features)(params,coords);Q,R=jnp.linalg.qr(G,mode='reduced')
    left,singular,right=jnp.linalg.svd(R,full_matrices=False);values=np.asarray(singular);threshold=float(values[0]*max(G.shape)*np.finfo(float).eps);rank=int((values>threshold).sum())
    valid=rank==R.shape[1]
    if valid:
        T=U@Q;C=jax.scipy.linalg.solve_triangular(R,T.T,lower=False).T
    else:
        Q=Q@left[:,:rank];R=singular[:rank,None]*right[:rank];T=U@Q;C=T@(right[:rank]/singular[:rank,None])
    perpendicular=jnp.sum((U-T@Q.T)**2,axis=1);jax.block_until_ready((G,Q,R,T,C,perpendicular))
    return G,Q,R,T,C,perpendicular,dict(seconds_including_compile=time.perf_counter()-start,bank_sha256=sha(G),R_sha256=sha(R),
        singular_values=values.tolist(),rank=rank,rank_valid=valid,projection_kind='full_rank_QR' if valid else 'truncated_SVD_diagnostic_failed_rank',rank_threshold=threshold,condition_number=float(values[0]/values[-1]),
        worst_projection_error=float(jnp.sqrt(jnp.max(perpendicular/jnp.sum(U*U,axis=1)))))


def make_step(params,phase,cfg):
    # Only these parameter subtrees enter optimization; frozen trees are not
    # members of the optimizer state, rather than merely receiving zero steps.
    keys=('B','g') if phase=='bank' else ('h','h_lin') if phase=='head' else tuple(x for x in params if x!='out_scale')
    train={k:params[k] for k in keys};frozen={k:params[k] for k in params if k not in keys}
    optimizer=optax.scale_by_adam()
    def loss(v,U,coords,denom,R,T,perp,si,pi):
        weights,latent=v;p={**frozen,**weights}
        if phase=='head':
            residual=sc.head(p,latent[si])@R.T-T[si]
            reconstruction=jnp.mean((jnp.sum(residual*residual,axis=1)+perp[si])/denom[si])
            return reconstruction,(reconstruction,jnp.asarray(0.,jnp.float64))
        G=sc.features(p,coords[pi]);H=latent[si] if phase=='bank' else sc.head(p,latent[si])
        residual=H@G.T-U[si[:,None],pi[None,:]]
        reconstruction=jnp.mean(jnp.mean(residual*residual,axis=1)/(denom[si]/U.shape[1]))
        gram=G.T@G/(G.shape[0]*p['out_scale']**2);orth=jnp.mean((gram-jnp.eye(gram.shape[0]))**2)
        return reconstruction+cfg['orthogonality_weight']*orth,(reconstruction,orth)
    @jax.jit
    def step(v,state,key,U,coords,denom,R,T,perp,rate):
        key,ks,kp=jax.random.split(key,3);si=jax.random.randint(ks,(cfg['source_batch'],),0,U.shape[0]);pi=jax.random.randint(kp,(cfg['point_batch'],),0,U.shape[1])
        (value,parts),grad=jax.value_and_grad(loss,has_aux=True)(v,U,coords,denom,R,T,perp,si,pi)
        update,state=optimizer.update(grad,state,v);v=optax.apply_updates(v,jax.tree_util.tree_map(lambda x:-rate*x,update))
        return v,state,key,value,parts
    return train,frozen,optimizer,step


def learning_rate(progress,cfg):
    warm=cfg['warmup_fraction'];factor=progress/warm if progress<warm else cfg['final_learning_rate_fraction']+(1-cfg['final_learning_rate_fraction'])*.5*(1+np.cos(np.pi*(progress-warm)/(1-warm)))
    return float(cfg['learning_rate']*max(cfg['final_learning_rate_fraction'],factor))


def train_phase(params,latent,U,coords,cfg,phase,tag,*,R=None,T=None,perpendicular=None,matched_seconds=None):
    start=time.perf_counter();denom=jnp.sum(U*U,axis=1)
    if R is None:R=jnp.zeros((params['h_lin'].shape[1],)*2);T=jnp.zeros((len(U),R.shape[0]));perpendicular=jnp.zeros(len(U))
    train,frozen,opt,step=make_step(params,phase,cfg);v=(train,jnp.asarray(latent));state=opt.init(v);key=jax.random.PRNGKey(cfg['rng_seed'])
    initial_hash=weights_sha(params);compile_start=time.perf_counter()
    compiled=step.lower(v,state,key,U,coords,denom,R,T,perpendicular,jnp.asarray(cfg['learning_rate'])).compile()
    compile_seconds=time.perf_counter()-compile_start;burn(cfg['training_burn_seconds'])
    elapsed=0.;done=0;records=[];blocks=[];finite=True;budget=cfg['steps'] if matched_seconds is None else cfg['max_matched_steps'];block=cfg['timing_block_updates']
    while done<budget and (matched_seconds is None or elapsed<matched_seconds):
        count=min(block,budget-done);progress=(done/cfg['steps']) if matched_seconds is None else min(elapsed/matched_seconds,1.)
        rate=jnp.asarray(learning_rate(progress,cfg));t0=time.perf_counter()
        for _ in range(count):v,state,key,value,parts=compiled(v,state,key,U,coords,denom,R,T,perpendicular,rate)
        jax.block_until_ready((v,state,value,parts));block_seconds=time.perf_counter()-t0;elapsed+=block_seconds;done+=count
        blocks.append(dict(first_update=done-count+1,last_update=done,seconds=block_seconds,learning_rate=float(rate)))
        scalar=float(value);finite=bool(np.isfinite(scalar))
        if done==block or done%cfg['log_every_updates']==0 or not finite or done==budget or (matched_seconds is not None and elapsed>=matched_seconds):
            record=dict(updates=done,loss=scalar,reconstruction=float(parts[0]),orth=float(parts[1]),optimizer_seconds=elapsed,block_seconds=block_seconds,learning_rate=float(rate));records.append(record);print('training',tag,record,flush=True)
        if not finite:break
    trained,z=v;final={**frozen,**trained};jax.block_until_ready(final)
    return final,np.asarray(z),dict(tag=tag,phase=phase,updates=done,source_batch=cfg['source_batch'],point_batch=cfg['point_batch'] if phase!='head' else None,
        sampled_source_exposures=done*cfg['source_batch'],sampled_field_entry_exposures=done*cfg['source_batch']*cfg['point_batch'] if phase!='head' else None,
        head_metric='full-field QR coefficient metric plus directly computed perpendicular constant' if phase=='head' else None,
        finite=finite,scheduled_updates=cfg['steps'] if matched_seconds is None else None,matched_optimizer_seconds=matched_seconds,
        budget_reached=done==budget if matched_seconds is None else elapsed>=matched_seconds,optimizer_seconds=elapsed,
        compile_seconds=compile_seconds,total_seconds=time.perf_counter()-start,
        timing_boundary='sum of synchronized optimizer-block elapsed intervals; compilation, burn-in, QR, diagnostics and JSON writes excluded',
        matching_round_rule='first complete block reaching target; at most one timing block overshoot',
        optimizer_blocks=blocks,initial_weights_sha256=initial_hash,final_weights_sha256=weights_sha(final),initial_latent_sha256=sha(latent),final_latent_sha256=sha(z),progress=records)


def normalized_pod(U,rank):
    start=time.perf_counter();norm=jnp.linalg.norm(U,axis=1);normalized=U/norm[:,None];eigen,vectors=jnp.linalg.eigh(normalized@normalized.T)
    eigen=eigen[::-1];vectors=vectors[:,::-1];assert float(eigen[rank-1])>1e-12*float(eigen[0])
    raw=normalized.T@(vectors[:,:rank]/jnp.sqrt(eigen[:rank]));Q,_=jnp.linalg.qr(raw,mode='reduced');Q.block_until_ready()
    error=jnp.linalg.norm(normalized-(normalized@Q)@Q.T,axis=1)
    return Q,dict(rank=rank,normalized_snapshot_sha256=sha(normalized),basis_sha256=sha(Q),eigenvalues=np.asarray(eigen).tolist(),
        training_relative_errors=np.asarray(error).tolist(),training_worst_relative_error=float(jnp.max(error)),orthogonality_error=float(jnp.linalg.norm(Q.T@Q-jnp.eye(rank))),
        seconds_including_compile=time.perf_counter()-start,scope='normalized training snapshots only; diagnostic span, not a deployed neural decoder')
