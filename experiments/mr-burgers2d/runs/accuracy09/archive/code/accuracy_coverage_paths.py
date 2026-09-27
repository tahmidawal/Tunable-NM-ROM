"""One bounded initial-family coverage expansion; same head optimizer and capacity."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import engines as e
from accuracy_paths import make_stationary_lm,whiten_head,unwhiten_head

def train_head(params,Zold,cfg,out,save_progress):
    """New truth is only regenerated initial fields; replay targets are teacher outputs."""
    begin=time.perf_counter();L=cfg['mesh'];K=Zold.shape[1];rng=np.random.default_rng(cfg['seed'])
    physical=np.concatenate([e.params_draw(item['seed'],item['cases']) for item in cfg['physical_draws']]);assert len(physical)==cfg['cases']
    G=e.sc.SeparableDecoder(params,K,params['h_lin'].shape[1]).feat_at(e.coords(L),chunk=8192)/L
    Q,R=jnp.linalg.qr(G,mode='reduced');jax.block_until_ready(R);qr_seconds=time.perf_counter()-begin;projection_begin=time.perf_counter()
    # Target coefficients and orthogonal loss preserve the full-grid field metric.
    norm_parts=[];y_parts=[]
    for start in range(0,len(physical),8):
        U=np.stack([e.initial(L,p)[1:-1,1:-1].ravel()/L for p in physical[start:start+8]])
        norm_parts.append(np.sum(U*U,axis=1));y_parts.append(np.asarray(jnp.asarray(U)@Q))
    norm2=jnp.asarray(np.concatenate(norm_parts));Y=jnp.asarray(np.concatenate(y_parts));del U,G,Q;projection_seconds=time.perf_counter()-projection_begin;codefit_begin=time.perf_counter()
    floor2=jnp.maximum(norm2-jnp.sum(Y*Y,1),0.)
    cand_ids=np.arange(0,len(Zold),max(1,len(Zold)//8192));Zc=jnp.asarray(Zold[cand_ids]);hp=whiten_head(params,R)
    H=e.sc.head(hp,Zc);idx=jnp.argmin(jnp.sum(H*H,1)[None,:]-2*Y@H.T,axis=1);Z=Zc[idx]
    lm=make_stationary_lm(lambda z,y:e.sc.head(hp,z)-y,K,cfg['code_fit_budget'],gtol=1e-6)
    fit=jax.jit(jax.vmap(lambda z,y:lm(z,(y,),0.)))
    zparts=[];fit_info=[]
    for start in range(0,len(physical),16):
        val=jax.tree_util.tree_map(np.asarray,fit(Z[start:start+16],Y[start:start+16]));zparts.append(val[0]);fit_info.append([a.tolist() for a in val[1:]])
    Z=jnp.asarray(np.concatenate(zparts));codefit_seconds=time.perf_counter()-codefit_begin;replay_ids=np.sort(rng.choice(len(Zold),cfg['replay_cases'],replace=False));Zrep=jnp.asarray(Zold[replay_ids]);Yrep=e.sc.head(hp,Zrep);repnorm=jnp.sum(Yrep*Yrep,1)
    initial_relative=np.sqrt(np.asarray((jnp.sum((e.sc.head(hp,Z)-Y)**2,axis=1)+floor2)/norm2))
    info=dict(feature_qr_seconds=qr_seconds,field_projection_seconds=projection_seconds,code_fit_seconds=codefit_seconds,per_update_head_initial_batch=cfg['batch'],per_update_head_replay_batch=cfg['batch'],optimized_code_scalars=int(Z.size),initial_examples_per_epoch=cfg['cases'],initial_relative_before_training=initial_relative.tolist(),replay_relative_before_training=np.zeros(len(replay_ids)).tolist(),config=cfg,physical_cases=physical.tolist(),replay_ids=replay_ids.tolist(),candidate_ids=cand_ids.tolist(),code_fit=fit_info,
        bank_relative_floor=np.sqrt(np.asarray(floor2/norm2)).tolist(),loss_trace=[],replay_note='frozen decoder outputs on original recorded training codes; regularization targets, not PDE truth')
    np.savez_compressed(out/'training_targets.npz',R=np.asarray(R),Y=np.asarray(Y),norm2=np.asarray(norm2),floor2=np.asarray(floor2),physical=physical,
        Z_init=np.asarray(Z),Z_replay=np.asarray(Zrep),Y_replay=np.asarray(Yrep),replay_ids=replay_ids)
    schedule=optax.cosine_decay_schedule(cfg['lr'],cfg['steps'],alpha=.05)
    optimizer=optax.chain(optax.clip_by_global_norm(10.),optax.adam(schedule));state=(hp,Z);ost=optimizer.init(state)
    def objective(state,i,j,Y,norm2,floor2,Yrep,Zrep,repnorm):
        h,z=state;err=(jnp.sum((e.sc.head(h,z[i])-Y[i])**2,axis=1)+floor2[i])/norm2[i]
        rep=jnp.sum((e.sc.head(h,Zrep[j])-Yrep[j])**2,axis=1)/repnorm[j]
        return jnp.mean(err)+cfg['replay_weight']*jnp.mean(rep)
    @jax.jit
    def step(state,ost,i,j,Y,norm2,floor2,Yrep,Zrep,repnorm):
        loss,grad=jax.value_and_grad(objective)(state,i,j,Y,norm2,floor2,Yrep,Zrep,repnorm)
        upd,ost=optimizer.update(grad,ost,state);return optax.apply_updates(state,upd),ost,loss
    optimizer_begin=time.perf_counter()
    for it in range(cfg['steps']):
        i=jnp.asarray(rng.integers(0,len(physical),cfg['batch']));j=jnp.asarray(rng.integers(0,len(replay_ids),cfg['batch']))
        state,ost,loss=step(state,ost,i,j,Y,norm2,floor2,Yrep,Zrep,repnorm)
        if it==0 or (it+1)%500==0:
            loss=float(loss);assert np.isfinite(loss)
            curh,curz=state
            initial_loss=float(jnp.mean((jnp.sum((e.sc.head(curh,curz)-Y)**2,axis=1)+floor2)/norm2))
            replay_loss=float(jnp.mean(jnp.sum((e.sc.head(curh,Zrep)-Yrep)**2,axis=1)/repnorm))
            info['loss_trace'].append(dict(step=it+1,batch_loss=loss,initial_relative_squared_loss=initial_loss,replay_relative_squared_loss=replay_loss,seconds=time.perf_counter()-begin));save_progress(info)
            print('HEAD_TRAIN',it+1,loss,flush=True)
    info['optimizer_loop_seconds']=time.perf_counter()-optimizer_begin
    info['optimizer_time_note']='One actual training allocation, including update compilation and periodic training diagnostics; no cross-job training-speed comparison.'
    hp,Z=state;final=dict(params);final.update(unwhiten_head(hp,R));jax.block_until_ready(final)
    fitpred=e.sc.head(hp,Z);repred=e.sc.head(hp,Zrep)
    info['final_initial_relative']=np.sqrt(np.asarray((jnp.sum((fitpred-Y)**2,axis=1)+floor2)/norm2)).tolist()
    info['final_replay_relative']=np.sqrt(np.asarray(jnp.sum((repred-Yrep)**2,axis=1)/repnorm)).tolist()
    back=e.sc.head(final,Z)@R.T;info['unwhitening_relative_parity']=float(jnp.linalg.norm(back-fitpred)/jnp.linalg.norm(fitpred));assert info['unwhitening_relative_parity']<1e-10
    # Preserve all old codes for dynamics and append newly optimized initial codes.
    Zout=np.concatenate((np.asarray(Zold),np.asarray(Z)))
    info['elapsed_seconds']=time.perf_counter()-begin;save_progress(info)
    return final,Zout,info

