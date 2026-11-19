"""B7 repair2: alternating code fits and full-quadrature network training."""
import argparse,optax
from common import *
from neural import stable as S,fields as F,code_optimizer as C
from neural.build_data import resolve
from training_io import checkpoint,read_checkpoint
import data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    a.phase='full';a.updates=200000
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    original=resolve(ns/'b6parent/output/neural_parent.pkl',ns);old=read_checkpoint(original)
    dp=original.parent/'coordinate_data.npz';assert digest(dp)==old['data_sha256'];dr=old['data_record']
    arrays=np.load(dp);xy,w=jnp.asarray(arrays['xy']),jnp.asarray(arrays['w'])
    train=jnp.asarray(arrays['train_y']);enc=jnp.asarray(arrays['train_encoder']);norm=jnp.asarray(arrays['train_norm'])
    codes,pca=S.initialize_codes(enc,32);network=S.initialize(jax.random.PRNGKey(2026111961),32,128,4)
    params=data.parameters('train',128);pulses=np.where(params[:,0]==0)[0]
    cases=sorted(set([3,int(pulses[np.argmin(params[pulses,3])]),int(pulses[np.argmax(params[pulses,3])]),int(pulses[np.argmax(params[pulses,4])])]))
    pilot_indices=np.array([11*c+t for c in cases for t in [0,5,10]])
    source=resolve(ns/'b7parentfull/output/result.json',ns);source_result=json.loads(source.read_text())
    assert source_result['status']=='complete_stable_parent' and not source_result['parent_quality_pass']
    diagnostic=resolve(ns/'b7parentdiagnosis/output/result.json',ns);diagnosis=json.loads(diagnostic.read_text());assert diagnosis['status']=='complete'
    path=source.parent/'stable_parent.pkl';assert digest(path)==source_result['selected_sha256'];warm=read_checkpoint(path)
    network=jax.tree.map(jnp.asarray,warm['network']);codes=jnp.asarray(warm['training_codes']);pca=warm['pca']
    netopt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(optax.cosine_decay_schedule(1e-4,a.updates,alpha=.05)))
    code_rate=optax.cosine_decay_schedule(3e-3,a.updates,alpha=.1)
    nsopt=netopt.init(network);csopt=C.initialize(codes);rng=np.random.default_rng(2026111964)
    @jax.jit
    def step(p,c,no,co,si,xy,w,y,norm,it):
        def loss(p,selected):
            pred=S.apply(p,selected[:,None,:],xy[None]);target=y[si]
            return jnp.mean(jnp.sum(w[None]*((pred-target)/norm[si,None])**2,axis=1))
        value,(gp,gc)=jax.value_and_grad(loss,argnums=(0,1))(p,c[si])
        up,no=netopt.update(gp,no,p);c,co=C.update(c,co,si,gc,code_rate(it))
        return optax.apply_updates(p,up),c,no,co,value
    @jax.jit
    def score(p,z,xy,w,y,norm):
        prediction=S.apply(p,z[:,None,:],xy[None])
        return jnp.sqrt(jnp.sum(w[None]*(prediction-y)**2,axis=1))/norm
    decode=jax.jit(S.apply);fit=F.make_fit(S.apply,60);curvature=F.make_fit(S.apply,120,newton=True)
    xf,wf=data.grid(256,True);xf,wf=jnp.asarray(xf),jnp.asarray(wf)
    def fine_truth(group,ci):
        row=dr['groups'][group]['sources'][ci];path=resolve(row['file'],ns);assert digest(path)==row['sha256']
        return data.synth_grid(jnp.asarray(np.load(path)['coefficients']),jnp.unique(xf[:,0])).reshape(11,-1)
    pilot_truth=jnp.concatenate([fine_truth('train',ci)[jnp.array([0,5,10])] for ci in cases])
    if a.phase=='full':
        dy=jnp.asarray(arrays['dev_y']);de=jnp.asarray(arrays['dev_encoder']);dn=jnp.asarray(arrays['dev_norm'])
    def refit(p,c,indices):
        zz=[];stats=[]
        for i in indices:
            near,interp=S.nearest_starts(de[i],enc,c)
            candidates=[fit(p,z,xy,dy[i],w,dn[i]) for z in [near,interp,jnp.zeros(32)]]
            z,st=min(candidates,key=lambda v:float(v[1][0]))
            if float(st[-1])==0:z,st=curvature(p,z,xy,dy[i],w,dn[i])
            zz.append(np.asarray(z));stats.append(np.asarray(st))
        return dict(indices=np.asarray(indices),codes=np.stack(zz),status=np.stack(stats))
    result=dict(meta=metadata,status='training',phase=a.phase,seed=2026111961,shuffle_seed=2026111964,latent=32,width=128,depth=4,
        decoder='Tanh coordinate network with fixed spatial Fourier inputs and smooth latent shifts; no periodic latent inputs or online POD lift',
        training='Repair2: per-snapshot Adam clocks, full spatial quadrature, explicit training-code refits every50000 updates; unchanged physical reconstruction objective',
        repair=2,source_full_sha256=digest(source),diagnosis_sha256=digest(diagnostic),max_updates=a.updates,
        initialization='Training-only field nearest-neighbor and eight-neighbor interpolated latent codes, plus zero restart',
        data_sha256=digest(dp),source_parent_sha256=digest(original),pilot_indices=pilot_indices,
        final_test_opened=False,history=[],representation_admitted=False,pde_admitted=False)
    result['initialization_sha256']=checkpoint(out/'initialization.pkl',dict(pca=pca,training_codes=codes,pilot_indices=pilot_indices))
    best=np.inf;stale=0;selection=np.array([11*c+t for c in [0,3,7,13,31,32,34,43] for t in [0,5,10]])
    def refresh(p,c,it):
        after=[];zs=[]
        for i in range(len(train)):
            z,st=fit(p,c[i],xy,train[i],w,norm[i]);zs.append(z);after.append(np.asarray(st))
            if (i+1)%200==0:print('training_code_refresh',it,i+1,flush=True)
        updated=jnp.stack(zs);stats=np.stack(after)
        record=dict(update=it,loss=float(np.mean(stats[:,0]**2)),worst=float(stats[:,0].max()),median=float(np.median(stats[:,0])),unfinished=int(np.sum(stats[:,-1]==0)))
        result.setdefault('training_code_refreshes',[]).append(record)
        checkpoint(out/f'code_refresh_{it:06d}.pkl',dict(codes=updated,status=stats))
        save(out/'result.json',result);print('training_code_refresh_complete',record,flush=True)
        return updated,C.initialize(updated)
    codes,csopt=refresh(network,codes,0)
    loss=result['training_code_refreshes'][-1]['loss']
    for it in range(a.updates+1):
        if it>0:
            si=rng.choice(len(train),size=8,replace=False)
            network,codes,nsopt,csopt,loss=step(network,codes,nsopt,csopt,jnp.asarray(si),xy,w,train,norm,it-1)
            if it%50000==0:codes,csopt=refresh(network,codes,it)
        if it%10000:continue
        coarse=np.asarray(score(network,codes[pilot_indices],xy,w,train[pilot_indices],norm[pilot_indices]))
        independent=np.asarray(score(network,codes[pilot_indices],xf,wf,pilot_truth,norm[pilot_indices]))
        row=dict(update=it,loss=float(loss),full_spatial_quadrature=True,pilot_worst=float(independent.max()),pilot_median=float(np.median(independent)),
            pilot_scoring_change=float(np.max(abs(coarse-independent))))
        ev=None
        if a.phase=='full':
            ev=refit(network,codes,selection);row.update(subset_worst=float(ev['status'][:,0].max()),
                subset_median=float(np.median(ev['status'][:,0])),unfinished_fits=int(np.sum(ev['status'][:,-1]==0)))
        metric=row.get('subset_worst',row['pilot_worst']);result['history'].append(row)
        payload=dict(network=network,training_codes=codes,training_encoder=enc,pca=pca,metrics=row,
            data_sha256=digest(dp),data_record=dr,evaluation=ev,pilot_indices=pilot_indices)
        if metric<best:
            significant=metric<best*.99;stale=0 if significant else stale+1;best=metric
            result['selected_sha256']=checkpoint(out/'stable_parent.pkl',payload);result['selected_update']=it
        else:stale+=1
        row['checkpoint_sha256']=checkpoint(out/f'parent_{it:05d}.pkl',payload)
        save(out/'result.json',result);print('stable_parent_training',row,flush=True)
        if a.phase=='pilot' and independent.max()<=.005 and row['pilot_scoring_change']<=.0002:break
        if it>=100000 and stale>=5:break
    selected=read_checkpoint(out/'stable_parent.pkl');network=selected['network'];codes=jnp.asarray(selected['training_codes'])
    if a.phase=='pilot':
        result.update(status='complete_training_only_pilot',pilot_pass=selected['metrics']['pilot_worst']<=.005 and selected['metrics']['pilot_scoring_change']<=.0002,
            worst_training_error=selected['metrics']['pilot_worst'],development_evaluated=False)
        save(out/'result.json',result);return
    full=refit(network,codes,np.arange(len(dy)));result['all_fits_sha256']=checkpoint(out/'development_fits.pkl',full)
    allfine=[]
    for ci in range(44):
        truth=fine_truth('dev',ci);lo=11*ci
        allfine.extend(np.asarray(score(network,jnp.asarray(full['codes'][lo:lo+11]),xf,wf,truth,dn[lo:lo+11])).tolist())
    trainerrors=[]
    for lo in range(0,len(train),16):trainerrors.extend(np.asarray(score(network,codes[lo:lo+16],xy,w,train[lo:lo+16],norm[lo:lo+16])).tolist())
    worst=int(np.argmax(allfine));ci,it=divmod(worst,11);source=dr['groups']['dev']['sources'][ci]
    path=resolve(source['file'],ns);assert digest(path)==source['sha256'];b=jnp.asarray(np.load(path)['coefficients'][it])
    axis=np.linspace(0,1,4096);prediction=np.empty((4096,4096),np.float64)
    for lo in range(0,4096,16):
        points=np.stack(np.meshgrid(axis[lo:lo+16],axis,indexing='ij'),-1).reshape(-1,2)
        prediction[lo:lo+16]=np.asarray(decode(network,jnp.asarray(full['codes'][worst]),jnp.asarray(points))).reshape(16,4096)
    truth=np.asarray(data.synth_grid(b,jnp.asarray(axis)));e4096=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth))
    np.savez_compressed(out/'worst_actual4096.npz',reference=truth,prediction=prediction,case=source['case'],time=data.TIMES[it])
    unfinished=int(np.sum(full['status'][:,-1]==0));change=float(np.max(abs(np.asarray(allfine)-full['status'][:,0])))
    result.update(status='complete_stable_parent',independent_errors=allfine,training_errors=trainerrors,
        worst_independent_error=max(allfine),worst_training_error=max(trainerrors),unfinished_fits=unfinished,max_scoring_change=change,
        actual4096=dict(case=source['case'],time=float(data.TIMES[it]),relative_error=e4096,shape=[4096,4096]),
        parent_quality_pass=max(allfine)<=.02 and e4096<=.02 and change<=.0002 and unfinished==0,
        parent_representation_pass=max(allfine)<=.005 and e4096<=.005 and change<=.0002 and unfinished==0)
    save(out/'result.json',result)


if __name__=='__main__':main()
