"""Matched-node uniform/prescribed controls with an unchanged shared child head."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint,summary
from build_local_cache import cohort
import data,geometry as geo,local_polynomial as LP,tensor_metric as TM,shared_child_head as H


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    for key in ['model','cache','admission','dev']:ap.add_argument('--'+key,required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight()
    mp=Path(a.model);mr=json.loads((mp/'result.json').read_text());assert digest(mp/'shared_child.pkl')==mr['model_sha256']
    model=read_checkpoint(mp/'shared_child.pkl');assert model['latent']==16
    ar=json.loads((Path(a.admission)/'result.json').read_text());assert ar['representation_gate'] and ar['model_sha256']==mr['model_sha256']
    cp=Path(a.cache);assert digest(cp/'result.json')==model['cache_sha256'];assert digest(cp/'global_basis.npz')==ar['basis_sha256']
    basis=jnp.asarray(np.load(cp/'global_basis.npz')['coefficients']);refs={r['case']:(p,r) for p,r in cohort(a.dev)}
    metric=TM.make_geometry(6);rule=TM.rule();x,w=data.grid(128,True);params=model['params'];scale=jnp.asarray(model['scale'])
    fit=jax.jit(jax.vmap(H.make_project(250,linear_solver='cholesky'),in_axes=(None,0,0,0,0,None,None,None,None,None)))
    result=dict(meta=meta,status='child_allocation_running',model_sha256=mr['model_sha256'],basis_sha256=ar['basis_sha256'],
        root_variables=128,child_variables=16,same_checkpoint_allocations=True,reference_assisted_projection=True,PDE_solve=False,
        final_test_opened=False,controls={})
    for name,P,depth in [('uniform20',20,2),('prescribed20',20,4),('uniform84',84,3),('prescribed84',84,4)]:
        saved=[];errors=[];stations=[];control=dict(active_variables=128+P*16,nodes=P,cases=[])
        result['controls'][name]=control
        for label in model['dev_labels']:
            folder,row=refs[label];initial=lambda xy:data.initial(xy,row['parameters'])
            cells=geo.uniform(depth) if name.startswith('uniform') else LP.callable_tree(x,w,initial,P,depth,'swept')
            assert len(cells)==P;geo.valid(cells);gram,tr,mx,my=metric(basis,jnp.asarray(cells),*rule)
            refpath=folder/row['levels'][-1]['file'];assert digest(refpath)==row['levels'][-1]['sha256']
            bb=jnp.asarray(np.load(refpath)['coefficients']);norm2=.25*jnp.sum(bb*bb,axis=(1,2));rhs=TM.rhs(basis,bb,mx,my)
            S,b,constant,AinvB,Ainvb=H.schur(gram,rhs,norm2);g,h=H.geometry(cells)
            teacher=jnp.linalg.solve(S+1e-8*jnp.eye(len(S)),b.T).T.reshape(11,P,36)
            ze=H.encode(params['encoder'],teacher,g,h,scale)
            def project(z):return fit(S,b,constant,norm2,z,params['head'],g,h,scale,jnp.ones(P))
            candidates=[project(ze),project(jnp.zeros_like(ze))]
            choose=jnp.argmin(jnp.stack([v[1] for v in candidates]),axis=0)
            z=jnp.stack([v[0] for v in candidates])[choose,jnp.arange(11)]
            candidates.append(project(jnp.concatenate([z[:1],z[:-1]],axis=0)))
            choose=jnp.argmin(jnp.stack([v[1] for v in candidates]),axis=0)
            z,e,s,acc=[jnp.stack([v[j] for v in candidates])[choose,jnp.arange(11)] for j in range(4)]
            local=H.head(params['head'],z,g,h,scale).reshape(11,-1);root=Ainvb-local@AinvB.T
            saved.append(dict(case=label,cells=cells,transforms=np.asarray(tr),codes=np.asarray(z),coefficients=np.asarray(jnp.concatenate([root,local],axis=1))))
            errors.append(np.asarray(e));stations.append(np.asarray(s))
            control['cases'].append(dict(case=label,error=np.asarray(e),stationarity=np.asarray(s),accepted=np.asarray(acc)))
            if len(saved)==1:control['representative_zero_jacobian']=H.zero_audit(params['head'],scale,g,h,S)
            save(out/'result.json',result);print('child_allocation',name,label,float(e.max()),float(s.max()),flush=True)
        control.update(projection=summary(np.asarray(errors)),max_stationarity=float(np.max(stations)),fit_sha256=checkpoint(out/f'{name}.pkl',saved))
        ii,tt=np.unravel_index(np.argmax(errors),np.asarray(errors).shape);v=saved[ii];folder,row=refs[v['case']]
        bb=jnp.asarray(np.load(folder/row['levels'][-1]['file'])['coefficients'][tt]);axis=np.linspace(0,1,4096)
        pred=LP.field_grid(basis,jnp.asarray(v['coefficients'][tt]),v['cells'],v['transforms'],6,axis)[0];truth=data.synth_grid(bb,axis)
        control['actual4096_witness']=dict(case=v['case'],time=float(data.TIMES[tt]),error=float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)),shape=[4096,4096])
        np.savez_compressed(out/f'{name}_4096.npz',prediction=np.asarray(pred),reference=np.asarray(truth),case=v['case'],time=data.TIMES[tt])
        control['representation_gate']=False;control['admission']='allocation diagnostic; full dense all-snapshot scoring required separately for any passing candidate'
        save(out/'result.json',result)
    result['status']='complete_frozen_child_allocation_controls';save(out/'result.json',result)


if __name__=='__main__':main()
