"""Stationarity-aware controls and fixed-bank initial-field head refinement."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import engines as e


def make_stationary_lm(fun,K,budget,trust=np.inf,gtol=1e-6):
    def lm(z0,args,tol):
        def evaluate(z):
            r=fun(z,*args);J=jax.jacfwd(fun)(z,*args)
            return r,J,jnp.linalg.norm(r)
        def grad(r,J):return jnp.linalg.norm(J.T@r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300)
        r,J,rn=evaluate(z0)
        reason=jnp.where(jnp.isfinite(rn),jnp.where(grad(r,J)<=gtol,4,jnp.where(rn<=tol,1,0)),3).astype(jnp.int32)
        def body(s):
            z,r,J,rn,lam,it,reason=s;H=J.T@J;g=J.T@r
            dz=e.gj_solve(H+lam*jnp.diag(jnp.diag(H)+1e-30),-g)
            ok=jnp.all(jnp.isfinite(dz))&(jnp.linalg.norm(dz)<=trust)
            zn=z+jnp.where(ok,dz,0.);rn2=jnp.linalg.norm(fun(zn,*args));accept=ok&jnp.isfinite(rn2)&(rn2<rn)
            r2,J2,rn2=jax.lax.cond(accept,lambda:evaluate(zn),lambda:(r,J,rn))
            gn=grad(r2,J2)
            tiny=ok&(jnp.linalg.norm(dz)<=1e-14*(1+jnp.linalg.norm(z)))
            reason=jnp.where(gn<=gtol,4,jnp.where(rn2<=tol,1,jnp.where(tiny,2,jnp.where((~accept)&(lam>=1e14),3,0)))).astype(jnp.int32)
            return jnp.where(accept,zn,z),r2,J2,rn2,jnp.where(accept,jnp.maximum(lam/3,1e-12),jnp.minimum(lam*10,1e14)),it+1,reason
        z,r,J,rn,lam,it,reason=jax.lax.while_loop(lambda s:(s[5]<budget)&(s[6]==0),body,
            (z0,r,J,rn,jnp.asarray(1e-6),jnp.int32(0),reason))
        return z,rn,it,reason,grad(r,J)
    return lm


def make_rom(params,L,dt,trust,ic_budget=400,step_budget=180,gtol=1e-6):
    K=params['h_lin'].shape[0]
    ic=make_stationary_lm(lambda z,y,R:R@e.sc.head(params,z)-y,K,ic_budget,gtol=gtol)
    lm=make_stationary_lm(lambda z,p,nu,data:e.weak(z,p,nu,data,params,L,dt),K,step_budget,trust,gtol)
    def query(u0,nu,data,cold):
        xy,w,Q,R,Hrot,Hnorm=cold;ui=e.sample_field(u0,xy,L)*w;y=Q.T@ui
        idx=jnp.argmin(Hnorm-2*Hrot@y);z,icrn,icit,icreason,icgn=ic(data[7][idx],(y,R),0.)
        scale=jnp.linalg.norm(ui)*jnp.sqrt(len(w))
        def step(carry,_):
            z,zprev=carry;p=data[1]@e.sc.head(params,z);ze=z+(z-zprev)
            r0=jnp.linalg.norm(e.weak(z,p,nu,data,params,L,dt));re=jnp.linalg.norm(e.weak(ze,p,nu,data,params,L,dt))
            zi=jnp.where(jnp.isfinite(re)&(re<r0),ze,z)
            z2,rn,it,reason,gn=lm(zi,(p,nu,data),1e-9*scale)
            return (z2,z),(z2,rn,it,reason,gn)
        _,(zs,rn,it,reason,gn)=jax.lax.scan(step,(z,z),None,length=int(round(.25/dt)))
        internal=jnp.concatenate((z[None],zs));Z=internal[::int(round(.05/dt))]
        fields=jax.vmap(lambda z:e.output_field(data[0]@e.sc.head(params,z),L,L))(Z)
        return fields,it,rn,reason,Z,icit,icreason,internal,gn,icgn
    return jax.jit(query)


def whiten_head(params,R):
    hp={k:params[k] for k in ['h','h_lin']}
    hp['h']=list(hp['h']);w,b=hp['h'][-1];hp['h'][-1]=(w@R.T,b@R.T);hp['h_lin']=hp['h_lin']@R.T
    return hp


def unwhiten_head(hp,R):
    def invert(x):return jax.scipy.linalg.solve_triangular(R,x.T,lower=False).T
    out=dict(hp);out['h']=list(hp['h']);w,b=out['h'][-1];out['h'][-1]=(invert(w),invert(b[None])[0]);out['h_lin']=invert(out['h_lin'])
    return out


def train_head(params,Zold,cfg,out,save_progress):
    """New truth is only regenerated initial fields; replay targets are teacher outputs."""
    begin=time.perf_counter();L=cfg['mesh'];K=Zold.shape[1];rng=np.random.default_rng(cfg['seed'])
    physical=e.params_draw(cfg['physical_seed'],cfg['cases'])
    G=e.sc.SeparableDecoder(params,K,params['h_lin'].shape[1]).feat_at(e.coords(L),chunk=8192)/L
    Q,R=jnp.linalg.qr(G,mode='reduced');jax.block_until_ready(R)
    # Target coefficients and orthogonal loss preserve the full-grid field metric.
    norm_parts=[];y_parts=[]
    for start in range(0,len(physical),8):
        U=np.stack([e.initial(L,p)[1:-1,1:-1].ravel()/L for p in physical[start:start+8]])
        norm_parts.append(np.sum(U*U,axis=1));y_parts.append(np.asarray(jnp.asarray(U)@Q))
    norm2=jnp.asarray(np.concatenate(norm_parts));Y=jnp.asarray(np.concatenate(y_parts));del U,G,Q
    floor2=jnp.maximum(norm2-jnp.sum(Y*Y,1),0.)
    cand_ids=np.arange(0,len(Zold),max(1,len(Zold)//8192));Zc=jnp.asarray(Zold[cand_ids]);hp=whiten_head(params,R)
    H=e.sc.head(hp,Zc);idx=jnp.argmin(jnp.sum(H*H,1)[None,:]-2*Y@H.T,axis=1);Z=Zc[idx]
    lm=make_stationary_lm(lambda z,y:e.sc.head(hp,z)-y,K,cfg['code_fit_budget'],gtol=1e-6)
    fit=jax.jit(jax.vmap(lambda z,y:lm(z,(y,),0.)))
    zparts=[];fit_info=[]
    for start in range(0,len(physical),16):
        val=jax.tree_util.tree_map(np.asarray,fit(Z[start:start+16],Y[start:start+16]));zparts.append(val[0]);fit_info.append([a.tolist() for a in val[1:]])
    Z=jnp.asarray(np.concatenate(zparts));replay_ids=np.sort(rng.choice(len(Zold),cfg['replay_cases'],replace=False));Zrep=jnp.asarray(Zold[replay_ids]);Yrep=e.sc.head(hp,Zrep);repnorm=jnp.sum(Yrep*Yrep,1)
    initial_relative=np.sqrt(np.asarray((jnp.sum((e.sc.head(hp,Z)-Y)**2,axis=1)+floor2)/norm2))
    info=dict(initial_relative_before_training=initial_relative.tolist(),replay_relative_before_training=np.zeros(len(replay_ids)).tolist(),config=cfg,physical_cases=physical.tolist(),replay_ids=replay_ids.tolist(),candidate_ids=cand_ids.tolist(),code_fit=fit_info,
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
    hp,Z=state;final=dict(params);final.update(unwhiten_head(hp,R));jax.block_until_ready(final)
    fitpred=e.sc.head(hp,Z);repred=e.sc.head(hp,Zrep)
    info['final_initial_relative']=np.sqrt(np.asarray((jnp.sum((fitpred-Y)**2,axis=1)+floor2)/norm2)).tolist()
    info['final_replay_relative']=np.sqrt(np.asarray(jnp.sum((repred-Yrep)**2,axis=1)/repnorm)).tolist()
    back=e.sc.head(final,Z)@R.T;info['unwhitening_relative_parity']=float(jnp.linalg.norm(back-fitpred)/jnp.linalg.norm(fitpred));assert info['unwhitening_relative_parity']<1e-10
    # Preserve all old codes for dynamics and append newly optimized initial codes.
    Zout=np.concatenate((np.asarray(Zold),np.asarray(Z)))
    info['elapsed_seconds']=time.perf_counter()-begin;save_progress(info)
    return final,Zout,info


def make_diagnostics(params,L,dt):
    """Compile once per model/mesh; scoring is outside complete-query timers."""
    def diagnostics(internal,u0,nu,data,cold):
        def station(z,zprev):
            p=data[1]@e.sc.head(params,zprev)
            fun=lambda z:e.weak(z,p,nu,data,params,L,dt);r=fun(z);J=jax.jacfwd(fun)(z)
            return jnp.linalg.norm(J.T@r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300)
        gn=jax.vmap(station)(internal[1:],internal[:-1])
        y=cold[2].T@(e.sample_field(u0,cold[0],L)*cold[1]);fun=lambda z:cold[3]@e.sc.head(params,z)-y
        rr=fun(internal[0]);JJ=jax.jacfwd(fun)(internal[0])
        return gn,jnp.linalg.norm(JJ.T@rr)/(jnp.linalg.norm(JJ)*jnp.linalg.norm(rr)+1e-300)
    return jax.jit(diagnostics)
