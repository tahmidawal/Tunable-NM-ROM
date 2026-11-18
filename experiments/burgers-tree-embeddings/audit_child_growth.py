"""Frozen shared-child head on a larger initial-callable tree; no online truth use."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint,summary
from build_local_cache import cohort
import data,geometry as geo,local_polynomial as LP,tensor_metric as TM,shared_child_head as H


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--cache',required=True);ap.add_argument('--fits',required=True);ap.add_argument('--dev',required=True)
    ap.add_argument('--nodes',type=int,default=64);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=preflight();mp=Path(a.model);mr=json.loads((mp/'result.json').read_text());assert mr['status']=='complete_shared_child_representation'
    assert digest(mp/'shared_child.pkl')==mr['model_sha256'];model=read_checkpoint(mp/'shared_child.pkl')
    fp=Path(a.fits);fr=json.loads((fp/'result.json').read_text());assert fr['model_sha256']==mr['model_sha256']
    assert digest(fp/'refined_fits.pkl')==fr['fit_sha256'];oldfits=read_checkpoint(fp/'refined_fits.pkl')
    cp=Path(a.cache);cr=json.loads((cp/'result.json').read_text());assert digest(cp/'result.json')==model['cache_sha256']
    basis=jnp.asarray(np.load(cp/'global_basis.npz')['coefficients']);assert len(basis)==128
    refs={r['case']:(p,r) for p,r in cohort(a.dev)};x,w=data.grid(128,True);metric=TM.make_geometry(6);rule=TM.rule()
    params=model['params'];scale=jnp.asarray(model['scale']);k=model['latent'];P=a.nodes;assert P==64 and k==16
    projector=jax.jit(jax.vmap(H.make_project(250,linear_solver='cholesky'),in_axes=(None,0,0,0,0,None,None,None,None,None)))
    result=dict(meta=meta,status='frozen_child_growth_running',nodes=P,latent=k,active_variables=128+P*k,
        head_weights_changed=False,model_sha256=mr['model_sha256'],old_fits_sha256=fr['fit_sha256'],
        allocation='initial-callable swept policy; larger retained tree; no evolved reference input',
        reference_assisted_projection=True,PDE_solve=False,final_test_opened=False,cases=[])
    saved=[];allerrors=[];maxtransfer=0.
    for i,oldrow in enumerate(cr['cases']['dev']):
        label=oldrow['case'];folder,r=refs[label];initial=lambda xy:data.initial(xy,r['parameters'])
        cells=LP.callable_tree(x,w,initial,P,4,'swept');assert len(cells)==P;geo.valid(cells)
        oldpath=cp/oldrow['file'];assert digest(oldpath)==oldrow['sha256'];ov=np.load(oldpath)
        lookup={tuple(c):j for j,c in enumerate(cells)};mapping=np.array([lookup[tuple(c)] for c in ov['cells']])
        gram,tr,mx,my=metric(basis,jnp.asarray(cells),*rule)
        assert float(jnp.max(jnp.abs(tr[mapping]-jnp.asarray(ov['transforms']))))<1e-10
        oldidx=np.concatenate([np.arange(128),128+(mapping[:,None]*36+np.arange(36)).ravel()])
        gramdef=float(jnp.max(jnp.abs(gram[jnp.ix_(jnp.asarray(oldidx),jnp.asarray(oldidx))]-jnp.asarray(ov['gram']))))
        assert gramdef<1e-10
        # References enter only the offline projection after tree construction.
        refpath=folder/r['levels'][-1]['file'];assert digest(refpath)==r['levels'][-1]['sha256']
        bfield=jnp.asarray(np.load(refpath)['coefficients']);norm2=jnp.sum(bfield*bfield,axis=(1,2))*.25
        rhs=TM.rhs(basis,bfield,mx,my);S,b,constant,AinvB,Ainvb=H.schur(gram,rhs,norm2)
        g,h=H.geometry(cells);mask=jnp.ones(P)
        teacher=jnp.linalg.solve(S+1e-8*jnp.eye(S.shape[0]),b.T).T.reshape(11,P,36)
        ze=H.encode(params['encoder'],teacher,g,h,scale)
        assert oldfits['labels'][i]==label
        zold=jnp.zeros((11,P,k)).at[:,mapping].set(jnp.asarray(oldfits['codes'][i]))
        oldlocal=H.head(params['head'],zold,g,h,scale).reshape(11,-1)
        transferred=jnp.concatenate([jnp.asarray(oldfits['coefficients'][i,:,:128]),oldlocal],axis=1)
        oldc=jnp.zeros_like(transferred).at[:,oldidx].set(jnp.asarray(oldfits['coefficients'][i]))
        dc=transferred-oldc
        transfer=float(jnp.max(jnp.sqrt(jnp.maximum(jnp.einsum('ti,ij,tj->t',dc,gram,dc),0)/norm2)))
        assert transfer<1e-9;maxtransfer=max(maxtransfer,transfer)
        def fit(z):return projector(S,b,constant,norm2,z,params['head'],g,h,scale,mask)
        candidates=[fit(zold),fit(ze),fit(jnp.zeros_like(ze))]
        choice=jnp.argmin(jnp.stack([v[1] for v in candidates]),axis=0)
        z=jnp.stack([v[0] for v in candidates])[choice,jnp.arange(11)]
        candidates.append(fit(jnp.concatenate([z[:1],z[:-1]],axis=0)))
        choice=jnp.argmin(jnp.stack([v[1] for v in candidates]),axis=0)
        z,e,s,accepted=[jnp.stack([v[j] for v in candidates])[choice,jnp.arange(11)] for j in range(4)]
        c=H.head(params['head'],z,g,h,scale).reshape(11,-1);root=Ainvb-c@AinvB.T;full=jnp.concatenate([root,c],axis=1)
        saved.append(dict(case=label,cells=cells,transforms=np.asarray(tr),codes=np.asarray(z),coefficients=np.asarray(full)))
        allerrors.append(np.asarray(e));result['cases'].append(dict(case=label,error=np.asarray(e),stationarity=np.asarray(s),
            zero_child_transfer=transfer,old_gram_defect=gramdef,accepted=np.asarray(accepted)))
        if i==0:result['zero_jacobian']=H.zero_audit(params['head'],scale,g,h,S)
        save(out/'result.json',result);print('child_growth',label,float(e.max()),float(s.max()),flush=True)
    result['projection']=summary(np.stack(allerrors));result['max_zero_child_transfer']=maxtransfer
    result['fit_sha256']=checkpoint(out/'growth_fits.pkl',saved)
    ii,tt=np.unravel_index(np.argmax(allerrors),np.asarray(allerrors).shape);v=saved[ii];folder,r=refs[v['case']]
    truthcoef=jnp.asarray(np.load(folder/r['levels'][-1]['file'])['coefficients'][tt]);axis=np.linspace(0,1,4096)
    pred=LP.field_grid(basis,jnp.asarray(v['coefficients'][tt]),v['cells'],v['transforms'],6,axis)[0];truth=data.synth_grid(truthcoef,axis)
    result['actual4096_witness']=dict(case=v['case'],time=float(data.TIMES[tt]),error=float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)),shape=[4096,4096])
    np.savez_compressed(out/'actual4096.npz',prediction=np.asarray(pred),reference=np.asarray(truth),case=v['case'],time=data.TIMES[tt])
    # A successful screen triggers fixed-code all-snapshot independent scoring in the same bounded job.
    dense=[]
    if max(np.max(allerrors),result['actual4096_witness']['error'])<=.005:
        for v in saved:
            folder,r=refs[v['case']];bb=jnp.asarray(np.load(folder/r['levels'][-1]['file'])['coefficients']);axis=np.linspace(0,1,2048)
            ee=[]
            for t in range(11):
                pred=LP.field_grid(basis,jnp.asarray(v['coefficients'][t]),v['cells'],v['transforms'],6,axis)[0];truth=data.synth_grid(bb[t],axis)
                ee.append(float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)))
            dense.append(ee)
        result['dense2048']=summary(np.asarray(dense));result['max_scoring_change']=float(np.max(np.abs(np.asarray(dense)-np.asarray(allerrors))))
    result['representation_gate']=bool(dense) and max(np.max(dense),result['actual4096_witness']['error'])<=.005 and result['max_scoring_change']<=.0002
    result['status']='complete_frozen_child_growth';save(out/'result.json',result)


if __name__=='__main__':main()
