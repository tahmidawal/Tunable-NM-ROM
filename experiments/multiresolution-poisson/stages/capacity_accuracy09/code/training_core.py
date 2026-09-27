"""Bounded continuation of inherited separable weights; training answers only."""
from core import *
from followup import qr_bank,oracle_solvers
import optax


def field_dataset(draws,nodes,chunk=16):
    """GPU FD-DST training truth, no saved/synchronized training dataset."""
    start=time.perf_counter();n=nodes-1
    x=jnp.arange(1,n,dtype=jnp.float64)/n;xx,yy=jnp.meshgrid(x,x,indexing='ij')
    coords=jnp.column_stack((xx.ravel(),yy.ravel()));lam=jnp.asarray(eigenvalues(n))
    def one(p,xx,yy,lam):
        cx,cy,width,amp=p
        f=amp*jnp.exp(-((xx-cx)**2+(yy-cy)**2)/(2*width**2))
        return dst2(dst2(f)/lam).ravel()
    batch=jax.jit(jax.vmap(one,in_axes=(0,None,None,None)))
    groups=[]
    for i in range(0,len(draws),chunk):groups.append(batch(jnp.asarray(draws[i:i+chunk]),xx,yy,lam))
    U=jnp.concatenate(groups);U.block_until_ready()
    return U,coords,dict(seconds_including_compile=time.perf_counter()-start,
        nodes=nodes,intervals=n,count=len(draws),field_shape=list(U.shape),field_sha256=sha(U),
        mean_squares=np.asarray(jnp.mean(U*U,axis=1)).tolist())


def initialize_codes(params,codes,U,coords,cfg):
    """One fixed original-head reference fit for new training fields only."""
    start=time.perf_counter();original=len(codes)
    G=jax.jit(sc.features)(params,coords);q,r=jnp.linalg.qr(G,mode='reduced')
    target=U[original:]@q;target.block_until_ready()
    trust=float(np.linalg.norm(codes-codes.mean(0),axis=1).max())
    solve=oracle_solvers(params,r,codes.shape[1],[cfg['new_code_fit_budget']],trust)[cfg['new_code_fit_budget']]
    batched=jax.jit(jax.vmap(lambda t:solve(jnp.asarray(codes.mean(0)),t,jnp.asarray(0.))))
    def stationarity(z,t):
        fun=lambda z:r@sc.head(params,z)-t
        residual=fun(z);jac=jax.jacfwd(fun)(z)
        return jnp.linalg.norm(jac.T@residual)/(jnp.linalg.norm(jac)*jnp.linalg.norm(residual)+1e-300)
    diagnostics=jax.jit(jax.vmap(stationarity))
    zs=[];records=[]
    for i in range(0,len(target),64):
        ans=batched(target[i:i+64]);jax.block_until_ready(ans)
        z,res,initial,nj,acc,att,reason=jax.device_get(ans)
        grad=np.asarray(diagnostics(ans[0],target[i:i+64]));finite=np.isfinite(z).all(axis=1)
        z=np.where(finite[:,None],z,codes.mean(0));zs.append(z)
        for j in range(len(z)):records.append(dict(index=original+i+j,residual=float(res[j]),initial=float(initial[j]),
            jacobians=int(nj[j]),accepted=int(acc[j]),attempts=int(att[j]),reason=int(reason[j]),
            stationarity=float(grad[j]),stationary=bool(np.isfinite(grad[j]) and grad[j]<=cfg['stationarity_tolerance']),
            finite=bool(finite[j]),nonfinite_mean_code_fallback=bool(not finite[j])))
    allcodes=np.concatenate((codes,*zs))
    return allcodes,dict(seconds_including_compile=time.perf_counter()-start,
        original_code_sha256=sha(codes),new_code_sha256=sha(allcodes[original:]),all_code_sha256=sha(allcodes),
        records=records,scope='Offline reference-only fit on new training fields; never online initialization')


def loss_components(params,z,truth,coords,denominators,normalizer,relative_loss,orth_weight):
    G=sc.features(params,coords);H=sc.head(params,z);error=H@G.T-truth
    per=jnp.mean(error*error,axis=1)
    reconstruction=jnp.mean(per/denominators) if relative_loss else jnp.mean(per)/normalizer
    gram=G.T@G/(G.shape[0]*params['out_scale']**2)
    orth=jnp.mean((gram-jnp.eye(gram.shape[0]))**2)
    return reconstruction+orth_weight*orth,(reconstruction,orth)


def training_step(config,relative_loss):
    schedule=optax.warmup_cosine_decay_schedule(0.,config['learning_rate'],config['warmup_steps'],
        config['training_steps_each'],config['learning_rate']*config['final_learning_rate_fraction'])
    optimizer=optax.adam(schedule)
    def loss(pz,U,coords,denom,normalizer,source_indices,point_indices):
        p,z=pz
        return loss_components(p,z[source_indices],U[source_indices[:,None],point_indices[None,:]],
            coords[point_indices],denom[source_indices],normalizer,relative_loss,config['orthogonality_weight'])
    @jax.jit
    def step(pz,state,key,U,coords,denom,normalizer):
        key,ks,kp=jax.random.split(key,3)
        si=jax.random.randint(ks,(config['source_batch'],),0,U.shape[0])
        pi=jax.random.randint(kp,(config['point_batch'],),0,U.shape[1])
        (value,parts),grad=jax.value_and_grad(loss,has_aux=True)(pz,U,coords,denom,normalizer,si,pi)
        grad[0]['out_scale']=jnp.zeros_like(grad[0]['out_scale'])
        update,state=optimizer.update(grad,state,pz)
        return optax.apply_updates(pz,update),state,key,value,parts
    return optimizer,step


def reconstruction_metrics(params,codes,U,coords,normalizer):
    G=jax.jit(sc.features)(params,coords)
    evaluate=jax.jit(lambda z,u,g,p:jnp.mean((sc.head(p,z)@g.T-u)**2,axis=1))
    per=[]
    for i in range(0,len(codes),64):per.append(np.asarray(evaluate(jnp.asarray(codes[i:i+64]),U[i:i+64],G,params)))
    mse=np.concatenate(per);denom=np.asarray(jnp.mean(U*U,axis=1));rel=np.sqrt(mse/denom)
    return dict(global_relative_mse=float(mse.mean()/normalizer),mean_per_snapshot_relative_mse=float(np.mean(mse/denom)),
        per_snapshot_relative_l2=rel.tolist(),median_relative_l2=float(np.median(rel)),worst_relative_l2=float(rel.max()))


def continue_training(params,codes,U,coords,normalizer,config,relative_loss,tag):
    start=time.perf_counter();denom=jnp.mean(U*U,axis=1)
    pz=(params,jnp.asarray(codes));optimizer,step=training_step(config,relative_loss)
    state=optimizer.init(pz);key=jax.random.PRNGKey(config['training_rng_seed'])
    compile_start=time.perf_counter()
    compiled=step.lower(pz,state,key,U,coords,denom,jnp.asarray(normalizer)).compile()
    compile_seconds=time.perf_counter()-compile_start
    loop_start=time.perf_counter();records=[];done=0;capped=False;finite=True
    for i in range(config['training_steps_each']):
        pz,state,key,value,parts=compiled(pz,state,key,U,coords,denom,jnp.asarray(normalizer));done=i+1
        if done==1 or done%250==0:
            jax.block_until_ready((pz,value));v=float(value)
            records.append(dict(step=done,loss=v,reconstruction=float(parts[0]),orth=float(parts[1]),elapsed=time.perf_counter()-start))
            print('train',tag,records[-1],flush=True)
            finite=bool(np.isfinite(v))
            if not finite:break
        if done%25==0:
            jax.block_until_ready(value)
            if time.perf_counter()-start>config['training_wall_cap_seconds_each']:
                capped=done<config['training_steps_each'];break
    jax.block_until_ready(pz);loop_seconds=time.perf_counter()-loop_start
    finalp,finalz=pz
    return finalp,np.asarray(finalz),dict(tag=tag,coverage_count=len(U),relative_loss=relative_loss,
        steps_done=done,steps_requested=config['training_steps_each'],time_capped=capped,finite=finite,
        compile_seconds=compile_seconds,optimizer_loop_seconds=loop_seconds,
        elapsed_seconds_including_compile=time.perf_counter()-start,progress=records,
        initial_codes_sha256=sha(codes),initial_weights_sha256=weights_sha(params),final_weights_sha256=weights_sha(finalp))


def weights_sha(params):
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(params):h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()
