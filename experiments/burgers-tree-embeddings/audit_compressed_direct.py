"""Resolved same-checkpoint allocation controls and actual-output qualification."""
import argparse
from common import *
from training_io import checkpoint,read_checkpoint,summary
from build_local_cache import cohort,expand_matrix,reduced_fit
import data,geometry as geo,local_polynomial as LP,tensor_metric as TM


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True)
    ap.add_argument('--dev',required=True);ap.add_argument('--rank',type=int,default=24);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();cache=Path(a.cache)
    raw=json.loads((cache/'result.json').read_text());assert raw['root']==128
    assert digest(cache/'local_POD.pkl')==raw['local_POD_sha256']
    rot=read_checkpoint(cache/'local_POD.pkl')['rotation'];U=jnp.asarray(rot[:,:a.rank])
    basis=jnp.asarray(np.load(cache/'global_basis.npz')['coefficients']);metric=TM.make_geometry(6);rule=TM.rule()
    oldfits=read_checkpoint(cache/f'local_direct{a.rank}_fits.pkl');labels=[r['case'] for r in raw['cases']['dev']]
    refs={r['case']:(f,r) for f,r in cohort(a.dev)};x,w=data.grid(128,True)
    result=dict(meta=meta,status='audit_running',rank=a.rank,root=128,checkpoint_sha256=raw['local_POD_sha256'],
        cache_sha256=digest(cache/'result.json'),same_checkpoint_allocations=True,reference_assisted=True,PDE_solve=False,
        final_test_opened=False,controls={},dense_scoring={})
    configs=[('uniform20',20,2),('prescribed20',20,4),('uniform84',84,3),('prescribed84',84,4),('random52',52,4)]
    for name,nodes,depth in configs:
        errs=[];saved=[]
        for i,label in enumerate(labels):
            folder,r=refs[label];p=folder/r['levels'][-1]['file'];assert digest(p)==r['levels'][-1]['sha256']
            b=jnp.asarray(np.load(p)['coefficients']);norm2=jnp.sum(b*b,axis=(1,2))*.25
            if name.startswith('uniform'):cells=geo.uniform(depth)
            elif name.startswith('random'):
                cells=geo.allocated(x,np.ones(len(x)),w,16+nodes*4,4,depth,random_seed=2026111860+i)
            else:cells=LP.tree(x,w,r['parameters'],nodes,depth,'swept')
            assert len(cells)==nodes;geo.valid(cells)
            gram,tr,mx,my=metric(basis,jnp.asarray(cells),*rule);rhs=TM.rhs(basis,b,mx,my)
            T=expand_matrix(U,nodes,128);c,e=reduced_fit(gram,rhs,norm2,T)
            errs.append(np.asarray(e));saved.append(dict(case=label,coefficients=np.asarray(c@T.T),cells=cells,transforms=np.asarray(tr)))
        result['controls'][name]=dict(projection=summary(np.stack(errs)),active_variables=128+nodes*a.rank)
        checkpoint(out/f'{name}.pkl',saved);save(out/'result.json',result)
        print('allocation',name,float(np.max(errs)),flush=True)
    # Fixed original coefficients: scoring only, with no refit on dense grids.
    fine=[];errors=np.asarray(raw['direct_controls'][str(a.rank)]['projection']['per_snapshot'])
    for i,label in enumerate(labels):
        row=raw['cases']['dev'][i];p=cache/row['file'];assert digest(p)==row['sha256'];v=np.load(p)
        folder,r=refs[label];refpath=folder/r['levels'][-1]['file'];assert digest(refpath)==r['levels'][-1]['sha256']
        b=np.load(refpath)['coefficients'];code=jnp.asarray(oldfits[i]['codes']);c=code@expand_matrix(U,52,128).T
        caseerr=[]
        for t in range(11):
            axis=np.linspace(0,1,2048);pred=LP.field_grid(basis,c[t],v['cells'],v['transforms'],6,axis)[0]
            truth=data.synth_grid(jnp.asarray(b[t]),axis)
            caseerr.append(float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth)))
        fine.append(caseerr);print('dense2048',label,max(caseerr),flush=True)
    result['dense_scoring']['2048']=summary(np.array(fine));result['max_metric_2048_change']=float(np.max(np.abs(np.array(fine)-errors)))
    ii,tt=np.unravel_index(np.argmax(fine),np.asarray(fine).shape);label=labels[ii];v=np.load(cache/raw['cases']['dev'][ii]['file'])
    folder,r=refs[label];b=np.load(folder/r['levels'][-1]['file'])['coefficients'][tt]
    c=jnp.asarray(oldfits[ii]['codes'])@expand_matrix(U,52,128).T
    axis=np.linspace(0,1,4096);pred=LP.field_grid(basis,c[tt],v['cells'],v['transforms'],6,axis)[0];truth=data.synth_grid(jnp.asarray(b),axis)
    e=float(jnp.linalg.norm(pred-truth)/jnp.linalg.norm(truth));diff=np.asarray(pred-truth)
    result['dense_witness']=dict(case=label,time=float(data.TIMES[tt]),relative_l2=e,shape=[4096,4096],
        relative_linf=float(jnp.max(jnp.abs(pred-truth))/jnp.max(jnp.abs(truth))),peak_index=np.unravel_index(np.argmax(abs(diff)),diff.shape),
        change_from_2048=abs(e-fine[ii][tt]))
    np.savez_compressed(out/'actual4096.npz',prediction=np.asarray(pred),reference=np.asarray(truth),case=label,time=data.TIMES[tt])
    gram=jnp.asarray(v['gram']);T=expand_matrix(U,52,128);G=T.T@gram@T
    S=G[128:,128:]-G[128:,:128]@jnp.linalg.solve(G[:128,:128],G[:128,128:]);ev=jnp.linalg.eigvalsh((S+S.T)*.5)
    result['zero_child_jacobian']=dict(zero_change=0.,eigenvalues_after_parent_removal=np.asarray(ev),
        independent_directions=int(jnp.sum(ev>ev[-1]*1e-10)),child_variables=52*a.rank,smallest_to_largest=float(ev[0]/ev[-1]))
    result['representation_gate']=max(np.max(fine),e)<=.005 and result['max_metric_2048_change']<=.0002 and abs(e-fine[ii][tt])<=.0002
    result['status']='complete_compressed_direct_audit';save(out/'result.json',result)
    print('dense4096',e,'gate',result['representation_gate'],flush=True)

if __name__=='__main__':main()
