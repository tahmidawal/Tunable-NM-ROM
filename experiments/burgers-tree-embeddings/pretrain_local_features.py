"""Train actual local coordinate-network features from missing training fields.

An offline patch SVD supplies supervised targets only. Deployment evaluates the
coordinate network, never this SVD or a global POD/Fourier coefficient expansion.
"""
import argparse,optax
from common import *
from neural import fields as F,factorized as S
from neural.build_data import resolve
from training_io import checkpoint,read_checkpoint


def quadrature(n):
    x,w=np.polynomial.legendre.leggauss(n);x=x*.75;w=w*.75
    xy=np.stack(np.meshgrid(x,x,indexing='ij'),-1).reshape(-1,2)
    t=np.clip((abs(xy)-.5)/.25,0,1);window=np.prod(1-(10*t**3-15*t**4+6*t**5),-1)
    return dict(axis=jnp.asarray(x),xy=jnp.asarray(xy),w=jnp.asarray(np.outer(w,w).ravel()),window=jnp.asarray(window))


@jax.jit
def patches(root,codes,b,cells,norm,q):
    h=2.**(-cells[:,0]);centers=(cells[:,1:]+.5)*h[:,None]
    xs=centers[:,0,None]+h[:,None]*q['axis'];ys=centers[:,1,None]+h[:,None]*q['axis']
    k=jnp.arange(1,b.shape[-1]+1)
    sx=jnp.sin(jnp.pi*xs[:,:,None]*k);sy=jnp.sin(jnp.pi*ys[:,:,None]*k)
    truth=jnp.einsum('nai,tij,nbj->ntab',sx,b,sy).reshape(len(cells),len(codes),-1)
    xy=centers[:,None,:]+h[:,None,None]*q['xy'][None]
    flat=xy.reshape(-1,2)
    parent=jax.vmap(lambda z:F.root(root,z,flat))(codes).reshape(len(codes),len(cells),-1).transpose(1,0,2)
    env=F.boundary(xy);denom=jax.vmap(lambda c:S.window(c,flat)[1])(cells).sum(0).reshape(len(cells),-1)
    valid=jnp.all((xy>=0)&(xy<=1),axis=(1,2))
    # Exclude patches crossing the physical wall from the shared shape teacher;
    # setting an arbitrary outside-domain field to zero would create false jumps.
    residual=(truth-parent)/jnp.where(abs(env[:,None])>1e-8,env[:,None],1.)/norm[None,:,None]
    target=residual*q['window'][None,None,:]/jnp.maximum(denom[:,None,:],1e-12)
    target=target*valid[:,None,None]
    return target.reshape(-1,len(q['xy'])),valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=50000)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);metadata=preflight()
    ns=Path(os.environ['STUDY_NAMESPACE']);parent=resolve(ns/'b6parent/output/neural_parent.pkl',ns)
    pp=read_checkpoint(parent);root=pp['params']['decoder'];dr=pp['data_record']
    previous=resolve(ns/'b6treecurv/output/initialization.pkl',ns);ini=read_checkpoint(previous)
    codes=jnp.asarray(ini['codes'][:,:8]);cells=jnp.asarray(ini['cells'][::11]);q=quadrature(64);fine=quadrature(128)
    source=[];samples=[];valid_nodes=[]
    for ci,row in enumerate(dr['groups']['train']['sources']):
        assert row['case'].startswith('train');path=resolve(row['file'],ns);assert digest(path)==row['sha256']
        b=jnp.asarray(np.load(path)['coefficients']);norm=jnp.linalg.norm(b.reshape(11,-1),axis=1)/2
        y,valid=patches(root,codes[11*ci:11*(ci+1)],b,cells[ci],norm,q)
        samples.append(np.asarray(y));valid_nodes.append(np.asarray(valid));source.append(dict(row,file=str(path)))
        if ci%16==0:print('local_neural_teacher',ci,flush=True)
    raw=jnp.asarray(np.concatenate(samples));del samples
    norms=jnp.sqrt(jnp.sum(raw**2*q['w'][None],axis=1));normalization=jnp.maximum(norms,1e-3)
    X=raw*jnp.sqrt(q['w'])[None]/normalization[:,None]
    covariance=X.T@X
    values,vectors=jnp.linalg.eigh(covariance);values=values[-32:][::-1];V=vectors[:,-32:][:,::-1]
    sign=jnp.sign(V[jnp.argmax(abs(V),axis=0),jnp.arange(32)]);V=V*sign[None]
    scores=X@V;weighted_fine=jnp.zeros((len(fine['xy']),32))
    # Evaluate the same data-derived teacher independently, without interpolation.
    for ci,row in enumerate(source):
        b=jnp.asarray(np.load(row['file'])['coefficients']);norm=jnp.linalg.norm(b.reshape(11,-1),axis=1)/2
        y,_=patches(root,codes[11*ci:11*(ci+1)],b,cells[ci],norm,fine)
        lo=ci*220;hi=lo+220
        weighted_fine+=(y/normalization[lo:hi,None]).T@(scores[lo:hi]/values[None])
    weighted_fine*=jnp.sqrt(fine['w'])[:,None]
    assert bool(jnp.all(jnp.isfinite(weighted_fine))) and float(values[-1])>0
    teacher=dict(xy=q['xy'],w=q['w'],window=q['window'],weighted_modes=V,
        fine_xy=fine['xy'],fine_w=fine['w'],fine_window=fine['window'],fine_weighted_modes=weighted_fine,
        eigenvalues=values,normalization=normalization,source=source,valid_nodes=np.stack(valid_nodes),
        parent_sha256=digest(parent),initialization_sha256=digest(previous))
    result=dict(meta=metadata,status='local_neural_feature_training',final_test_opened=False,
        teacher='Offline training-patch SVD used only for neural-feature supervision',
        online_representation='Coordinate sine network only; no online SVD reconstruction',
        valid_patch_rows=int(np.sum(valid_nodes))*11,total_patch_rows=int(len(raw)),
        captured_teacher_variance=float(jnp.sum(values)/jnp.trace(covariance)),
        teacher_physical_gram_change=float(jnp.max(abs(weighted_fine.T@weighted_fine-jnp.eye(32)))),history=[])
    result['teacher_sha256']=checkpoint(out/'feature_teacher.pkl',teacher)
    del raw,X,covariance,vectors,scores
    feature=F.init_mlp(jax.random.PRNGKey(2026111954),[2,128,128,128,32],True)
    opt=optax.chain(optax.clip_by_global_norm(10.),optax.adam(optax.cosine_decay_schedule(1e-3,a.updates,alpha=.02)))
    state=opt.init(feature)
    @jax.jit
    def step(feature,state,xy,w,window,target):
        def loss(p):
            prediction=F.mlp(p,xy/.75,True)*(jnp.sqrt(w)*window)[:,None]
            return jnp.sum((prediction-target)**2)/target.shape[1]
        value,grad=jax.value_and_grad(loss)(feature);updates,state=opt.update(grad,state,feature)
        return optax.apply_updates(feature,updates),state,value
    @jax.jit
    def score(feature,xy,w,window,target):
        pred=F.mlp(feature,xy/.75,True)*(jnp.sqrt(w)*window)[:,None]
        return jnp.linalg.norm(pred-target,axis=0)/jnp.linalg.norm(target,axis=0)
    best=np.inf;stale=0
    for it in range(1,a.updates+1):
        feature,state,loss=step(feature,state,q['xy'],q['w'],q['window'],V)
        if it%1000:continue
        coarse=np.asarray(score(feature,q['xy'],q['w'],q['window'],V))
        errors=np.asarray(score(feature,fine['xy'],fine['w'],fine['window'],weighted_fine));worst=float(errors.max())
        row=dict(update=it,loss=float(loss),worst_independent_feature_error=worst,
                 max_scoring_change=float(np.max(abs(errors-coarse))))
        payload=dict(feature=jax.device_get(feature),root=root,teacher_sha256=result['teacher_sha256'],
                     parent_sha256=digest(parent),metrics=row)
        row['sha256']=checkpoint(out/f'feature_{it:05d}.pkl',payload);result['history'].append(row)
        print('local_neural_feature_fit',row,flush=True)
        if worst<best:
            significant=worst<best*.99;stale=0 if significant else stale+1;best=worst
            result['selected_sha256']=checkpoint(out/'neural_features.pkl',payload)
        else:stale+=1
        save(out/'result.json',result)
        if worst<=.005 or (it>=10000 and stale>=5):break
    result.update(status='complete',best_independent_feature_error=best,feature_supervision_pass=best<=.005,
        representation_admitted=False,remaining='Assembled span check, nonlinear child-head training, full representation and PDE gates')
    save(out/'result.json',result)


if __name__=='__main__':main()
