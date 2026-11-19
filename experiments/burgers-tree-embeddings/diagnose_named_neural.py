"""Named-family transfer diagnosis of retained neural checkpoints; no retraining."""
import argparse
from common import *
import data
from training_io import read_checkpoint,checkpoint
from neural.build_data import resolve
from neural import fields as F,allocations as A
from neural.initialize import missing_field
from train_coordinate_tree import apply,jac


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/'b6namedrefs/output/result.json',ns);refs=json.loads(rp.read_text())
    cp=resolve(ns/'b6namedfd/output/result.json',ns);checks=json.loads(cp.read_text());assert checks['passed']
    pp=resolve(ns/'b6parent/output/neural_parent.pkl',ns);parent_model=read_checkpoint(pp)['params'];parent=parent_model['decoder']
    dr=resolve(ns/'b6refit/output/result.json',ns);diag=json.loads(dr.read_text());selected=diag['best']
    tp=resolve(ns/f"b6treecurv/output/tree_{selected['update']:05d}.pkl",ns);assert digest(tp)==selected['source_sha256']
    tree=read_checkpoint(tp)['network'];xy,w=data.grid(128,True);xf,wf=data.grid(256,True)
    xy,w,xf,wf=map(jnp.asarray,(xy,w,xf,wf));xe,_=data.grid(16,True);ex=jnp.asarray(np.unique(xe[:,0]))
    fitroot=F.make_fit(F.root,120,newton=True);fittree=F.make_fit(apply,60,jacobian=jac)
    decode=jax.jit(apply);root=jax.jit(F.root)
    result=dict(meta=meta,status='running',reference_sha256=digest(rp),crosscheck_sha256=digest(cp),
        parent_sha256=digest(pp),tree_sha256=digest(tp),cases=[],final_test_opened=False,
        classification='Offline transfer diagnosis of failed retained actual nonlinear tree; no method admission or PDE prediction')
    for ref in refs['cases']:
        source=rp.parent/ref['levels'][-1]['file'];assert digest(source)==ref['levels'][-1]['sha256']
        b=jnp.asarray(np.load(source)['coefficients']);norm=jnp.linalg.norm(b.reshape(11,-1),axis=1)/2
        y=data.synth_grid(b,jnp.unique(xy[:,0])).reshape(11,-1);yf=data.synth_grid(b,jnp.unique(xf[:,0])).reshape(11,-1)
        encoded=jax.vmap(lambda y:F.mlp(parent_model['encoder'],y))(data.synth_grid(b,ex).reshape(11,-1))
        cells=jnp.asarray(A.allocate('prescribed',np.asarray(xy),np.asarray(y[0]),2026111956))
        p=dict(tree,cells=cells);rows=[];saved=[];last=jnp.zeros(8);last_tree=jnp.zeros(8)
        for i in range(11):
            candidates=[fitroot(parent,start,xy,y[i],w,norm[i]) for start in [encoded[i],last,jnp.zeros(8)]]
            z,st=min(candidates,key=lambda v:float(v[1][0]));last=z
            candidates=[fitroot(tree['root'],start,xy,y[i],w,norm[i]) for start in [last_tree,z,jnp.zeros(8)]]
            tz,ts=min(candidates,key=lambda v:float(v[1][0]));last_tree=tz
            initial,_=missing_field(tree,tz,cells,xy,y[i],w,norm[i]);zz,status=fittree(p,initial,xy,y[i],w,norm[i])
            py=root(parent,z,xf);ty=decode(p,zz,xf)
            errors=jnp.sqrt(jnp.sum(wf[None]*(jnp.stack([py,ty])-yf[i])**2,axis=1))/norm[i]
            row=dict(time=float(data.TIMES[i]),errors=np.asarray(errors),parent_status=np.asarray(st),tree_status=np.asarray(status),
                scoring_changes=np.asarray(errors-jnp.array([st[0],status[0]])))
            rows.append(row);saved.append(dict(parent=np.asarray(z),tree=np.asarray(zz)))
            if i%2==0:
                np.savez_compressed(out/f"{ref['case']}_time{i:02d}_independent256.npz",xy=np.asarray(xf),reference=np.asarray(yf[i]),parent=np.asarray(py),tree=np.asarray(ty))
        case=dict(case=ref['case'],nu=ref['nu'],definition=ref['definition'],rows=rows,
            worst_parent=max(float(v['errors'][0]) for v in rows),worst_tree=max(float(v['errors'][1]) for v in rows),
            unfinished_parent=sum(v['parent_status'][-1]==0 for v in rows),unfinished_tree=sum(v['tree_status'][-1]==0 for v in rows))
        case['codes_sha256']=checkpoint(out/(ref['case']+'_codes.pkl'),dict(cells=np.asarray(cells),codes=saved))
        result['cases'].append(case);save(out/'result.json',result);print('named_neural_transfer',ref['case'],case['worst_parent'],case['worst_tree'],flush=True)
    result.update(status='complete',representation_admitted=False,pde_admitted=False);save(out/'result.json',result)


if __name__=='__main__':main()
