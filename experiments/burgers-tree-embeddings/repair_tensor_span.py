"""Refit the same rich local span with physically resolved tensor integration."""
import argparse
from common import *
import tensor_metric as TM,local_polynomial as LP,data
from training_io import checkpoint,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--dev',required=True);ap.add_argument('--span',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=json.loads((Path(a.dev)/'result.json').read_text());rows=[r for r in raw['cases'] if not r['case'].startswith('fixture')]
    bases=np.load(Path(a.span)/'pod_basis.npz')['coefficients'];x,w=data.grid(128,True)
    metric=TM.make_geometry(6);rule=TM.rule();rule2=TM.rule(order=32)
    result=dict(meta=preflight(),status='running',prior_coarse_pass_retracted=True,root_controls={},integration='64 aligned panels per axis, 24 Gauss points per panel',final_test_opened=False)
    for root in (128,16):
        basis=jnp.asarray(bases[:root]);saved=[];allerr=[];differences=[]
        for i,r in enumerate(rows):
            p=Path(a.dev)/r['levels'][-1]['file'];assert digest(p)==r['levels'][-1]['sha256'];b=jnp.asarray(np.load(p)['coefficients'])
            cells=LP.tree(x,w,r['parameters'],52,4,'swept');gram,tr,mx,my=metric(basis,jnp.asarray(cells),*rule)
            br=TM.rhs(basis,b,mx,my);norm2=jnp.sum(b*b,axis=(1,2))*.25;c,err=TM.fit(gram,br,norm2)
            if i in (5,16,20,21):
                g2,_,mx2,my2=metric(basis,jnp.asarray(cells),*rule2);br2=TM.rhs(basis,b,mx2,my2)
                diff=max(float(jnp.max(jnp.abs(g2-gram))),float(jnp.max(jnp.abs(br2-br))))
                assert diff<1e-10;differences.append(diff)
            saved.append(dict(case=r['case'],coefficients=np.asarray(c),cells=cells,transforms=np.asarray(tr),errors=np.asarray(err)))
            allerr.append(np.asarray(err));print('resolved',root,r['case'],float(jnp.max(err)),flush=True)
        checkpoint(out/f'fits_root{root}.pkl',saved)
        result['root_controls'][str(root)]=dict(projection=summary(np.asarray(allerr)),max_integration_difference=max(differences),variables=root+52*36)
        save(out/'result.json',result)
    root=min((128,16),key=lambda k:result['root_controls'][str(k)]['projection']['worst']);basis=bases[:root]
    from training_io import read_checkpoint
    saved=read_checkpoint(out/f'fits_root{root}.pkl');errs=np.array([r['errors'] for r in saved]);ii,tt=np.unravel_index(np.argmax(errs),errs.shape);r=saved[ii]
    source=next(row for row in rows if row['case']==r['case']);b=np.load(Path(a.dev)/source['levels'][-1]['file'])['coefficients'][tt]
    dense={}
    for n in (2048,4096):
        axis=np.linspace(0,1,n);pred=LP.field_grid(basis,jnp.asarray(r['coefficients'][tt]),r['cells'],r['transforms'],6,axis)[0]
        ref=data.synth_grid(jnp.asarray(b),axis);error=float(jnp.linalg.norm(pred-ref)/jnp.linalg.norm(ref));dense[str(n)]=error
        if n==4096:np.savez_compressed(out/'actual4096.npz',prediction=np.asarray(pred),reference=np.asarray(ref),case=r['case'],time=data.TIMES[tt])
    result['dense_witness']=dict(root=root,case=r['case'],time=data.TIMES[tt],errors=dense,shape=[4096,4096],PDE_solve=False)
    result['status']='complete_resolved_span_repair';save(out/'result.json',result);print('dense',dense,flush=True)
if __name__=='__main__':main()
