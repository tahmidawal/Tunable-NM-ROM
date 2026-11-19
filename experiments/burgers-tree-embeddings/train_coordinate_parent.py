"""Train the actual coordinate neural parent; descriptors are offline-only."""
import argparse,optax
from common import *
from training_io import checkpoint
from neural import fields as F
from neural.build_data import build,resolve
import data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--updates',type=int,default=20000);ap.add_argument('--seed',type=int,default=2026111951)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight()
    arrays,dr=build(out);d={k:jnp.asarray(v) for k,v in arrays.items()}
    keys=jax.random.split(jax.random.PRNGKey(a.seed),3);k=8
    params=dict(decoder=F.init_mlp(keys[0],[k+2,128,128,128,1],True),
                intrinsic=F.init_mlp(keys[1],[7,32,32,k]),encoder=F.init_mlp(keys[2],[256,128,64,k]))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(optax.cosine_decay_schedule(2e-4,a.updates,alpha=.05)))
    state=opt.init(params);rng=np.random.default_rng(a.seed)
    @jax.jit
    def step(p,state,si,ci,xy,w,y,enc,features,norm):
        def loss(p):
            z=F.intrinsic(p['intrinsic'],features[si]);ze=F.mlp(p['encoder'],enc[si])
            points=xy[ci];target=y[si[:,None],ci];weight=w[ci]*len(w)
            direct=(F.root(p['decoder'],z[:,None,:],points)-target)/norm[si,None]
            encoded=(F.root(p['decoder'],ze[:,None,:],points)-target)/norm[si,None]
            align=jnp.mean((ze-jax.lax.stop_gradient(z))**2)
            shift=jnp.zeros(7).at[-1].set(.1)
            curvature=(F.intrinsic(p['intrinsic'],features[si]+shift)-2*z+
                       F.intrinsic(p['intrinsic'],features[si]-shift))/.01
            regular=sum(jnp.mean(layer['w']**2) for layer in p['decoder'])
            reconstruction=.5*jnp.mean(weight*(direct**2+encoded**2))
            return reconstruction+.01*align+1e-4*jnp.mean(curvature**2)+1e-6*regular
        value,grad=jax.value_and_grad(loss)(p);update,state=opt.update(grad,state,p)
        return optax.apply_updates(p,update),state,value
    fit=F.make_fit(F.root,60)
    # Fixed subset is a checkpoint-selection diagnostic, never full admission.
    case_ids=[0,3,7,13,31,32,34,43];indices=np.array([11*i+j for i in case_ids for j in [0,5,10]])
    def evaluate(p,ids):
        fits=[];stats=[];initial=[]
        for i in ids:
            ze=F.mlp(p['encoder'],d['dev_encoder'][i])
            candidates=[fit(p['decoder'],z,d['xy'],d['dev_y'][i],d['w'],d['dev_norm'][i])
                        for z in [ze,jnp.zeros(k)]]
            chosen=min(candidates,key=lambda v:float(v[1][0]))
            fits.append(np.asarray(chosen[0]));stats.append(np.asarray(chosen[1]));initial.append(np.asarray(ze))
        return dict(indices=np.asarray(ids),codes=np.stack(fits),status=np.stack(stats),encoded=np.stack(initial))
    result=dict(meta=metadata,status='training_coordinate_neural_parent',seed=a.seed,latent=k,
        decoder='coordinate plus embedding sine network, three hidden layers of128; no POD/Fourier decoder expansion',
        training_only_intrinsic_inputs='normalized parameters and time; absent from online initial fitting and PDE solve',
        input_encoder='known sampled field on16x16 Gauss grid',data_sha256=dr['sha256'],
        selection_subset=indices,selection_not_full_admission=True,history=[],final_test_opened=False)
    best=(float('inf'),float('inf'));stale=0;selected=None
    for it in range(1,a.updates+1):
        si=jnp.asarray(rng.integers(len(d['train_y']),size=4));ci=jnp.asarray(rng.integers(len(d['w']),size=(4,1024)))
        params,state,value=step(params,state,si,ci,d['xy'],d['w'],d['train_y'],d['train_encoder'],d['train_features'],d['train_norm'])
        if it%1000:continue
        ev=evaluate(params,indices);e=ev['status'][:,0];metric=(float(e.max()),float(np.median(e)))
        row=dict(update=it,loss=float(value),subset_worst=metric[0],subset_median=metric[1],
                 unfinished_fits=int(np.sum(ev['status'][:,-1]==0)),max_stationarity=float(ev['status'][:,1].max()))
        payload=dict(params=jax.device_get(params),evaluation=ev,metrics=row,latent=k,
                     data_sha256=dr['sha256'],data_record=dr,feature_mean=arrays['feature_mean'],feature_scale=arrays['feature_scale'])
        row['sha256']=checkpoint(out/f'parent_{it:05d}.pkl',payload);result['history'].append(row)
        print('coordinate_parent',row,flush=True)
        if metric<best:
            significant=metric[0]<best[0]*.99;stale=0 if significant else stale+1;best=metric;selected=payload
            result['selected_sha256']=checkpoint(out/'neural_parent.pkl',selected);result['selected_update']=it
        else:stale+=1
        save(out/'result.json',result)
        if it>=6000 and stale>=5:break
    assert selected is not None
    params=selected['params'];full=evaluate(params,np.arange(len(d['dev_y'])))
    result['full_fits_sha256']=checkpoint(out/'all_development_fits.pkl',full)
    result.update(worst_128_projection=float(full['status'][:,0].max()),
        unfinished_full_fits=int(np.sum(full['status'][:,-1]==0)),max_stationarity=float(full['status'][:,1].max()))
    # Fixed-code independent physical scoring of every development snapshot.
    xy256,w256=data.grid(256,True);axis256=np.unique(xy256[:,0]);decode=jax.jit(F.root);synth=jax.jit(data.synth_grid)
    independent=[];ns=Path(os.environ['STUDY_NAMESPACE'])
    for ci,row in enumerate(dr['groups']['dev']['sources']):
        path=resolve(row['file'],ns);assert digest(path)==row['sha256']
        b=np.load(path)['coefficients'];truth=np.asarray(synth(jnp.asarray(b),jnp.asarray(axis256))).reshape(11,-1)
        for j in range(11):
            i=11*ci+j;u=np.asarray(decode(params['decoder'],jnp.asarray(full['codes'][i]),jnp.asarray(xy256)))
            independent.append(float(np.sqrt(np.sum(w256*(u-truth[j])**2))/arrays['dev_norm'][i]))
    result['independent_256_errors']=independent;result['worst_256_projection']=max(independent)
    result['max_128_to_256_error_change']=float(np.max(abs(np.asarray(independent)-full['status'][:,0])))
    worst=int(np.argmax(independent));case,snapshot=divmod(worst,11)
    row=dr['groups']['dev']['sources'][case];path=resolve(row['file'],ns);b=np.load(path)['coefficients'][snapshot]
    axis=np.linspace(0,1,4096);prediction=np.empty((4096,4096),dtype=np.float64)
    for lo in range(0,4096,16):
        xy=np.stack(np.meshgrid(axis[lo:lo+16],axis,indexing='ij'),axis=-1).reshape(-1,2)
        prediction[lo:lo+16]=np.asarray(decode(params['decoder'],jnp.asarray(full['codes'][worst]),jnp.asarray(xy))).reshape(16,4096)
    truth=np.asarray(synth(jnp.asarray(b),jnp.asarray(axis)))
    error=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth))
    np.savez_compressed(out/'worst_actual4096.npz',prediction=prediction,reference=truth,case=row['case'],time=data.TIMES[snapshot])
    result.update(actual4096=dict(case=row['case'],time=float(data.TIMES[snapshot]),relative_error=error,shape=[4096,4096]),
        parent_representation_gate=bool(max(independent)<=.005 and error<=.005 and result['max_128_to_256_error_change']<=.0002),
        status='complete_coordinate_parent_diagnostic',child_model_trained=False,
        admission='Parent alone is diagnostic; no neural-tree/PDE admission implied')
    save(out/'result.json',result)


if __name__=='__main__':main()
