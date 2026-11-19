"""Assembled learned-neural-feature capacity control; not a nonlinear-model claim."""
import argparse
from common import *
import data
from training_io import read_checkpoint,checkpoint
from neural.build_data import resolve
from neural import fields as F, factorized as N, allocations as A


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    fp=resolve(ns/'b6features/output/neural_features.pkl',ns);model=read_checkpoint(fp)
    pp=resolve(ns/'b6parent/output/neural_parent.pkl',ns);parent=read_checkpoint(pp)
    assert digest(pp)==model['parent_sha256']
    dp=pp.parent/'coordinate_data.npz';assert digest(dp)==parent['data_sha256'];d=dict(np.load(dp))
    rp=resolve(ns/'b6treecurv/output/parent_stability.pkl',ns);codes=read_checkpoint(rp)['codes']
    xy,w=jnp.asarray(d['xy']),jnp.asarray(d['w']);xf,wf=data.grid(256,True);xf,wf=jnp.asarray(xf),jnp.asarray(wf)
    features=jax.jit(N.feature_matrix);root=jax.jit(jax.vmap(F.root,in_axes=(None,0,None)))
    result=dict(meta=meta,status='running',feature_sha256=digest(fp),parent_sha256=digest(pp),
        parent_fit_sha256=digest(rp),final_test_opened=False,representation_admitted=False,
        meaning='Direct coefficients in learned neural feature span; necessary capacity control, not nonlinear child performance',allocations=[])
    for kind,depth in [('uniform',3),('prescribed',3),('prescribed',4)]:
        if depth==4 and result['allocations'][-1]['worst_independent_error']<=.005:break
        rows=[];saved=[]
        for ci,source in enumerate(parent['data_record']['groups']['dev']['sources']):
            lo=11*ci;hi=lo+11
            cells=jnp.asarray(A.allocate(kind,d['xy'],d['dev_y'][lo],2026111955+ci,nodes=20,max_depth=depth))
            B=features(model['feature'],cells,xy)
            c,err=N.linear_initialization(model['root'],jnp.asarray(codes[lo:hi]),B,xy,jnp.asarray(d['dev_y'][lo:hi]),w)
            path=resolve(source['file'],ns);assert digest(path)==source['sha256'];b=jnp.asarray(np.load(path)['coefficients'])
            truth=data.synth_grid(b,jnp.unique(xf[:,0])).reshape(11,-1)
            Bf=features(model['feature'],cells,xf)
            pred=root(model['root'],jnp.asarray(codes[lo:hi]),xf)+jnp.einsum('pnr,tnr->tp',Bf,c)
            ef=jnp.sqrt(jnp.sum(wf*(pred-truth)**2,axis=1))/jnp.asarray(d['dev_norm'][lo:hi])
            row=dict(case=source['case'],errors=np.asarray(ef),coarse_errors=np.asarray(err),
                max_scoring_change=float(jnp.max(abs(ef-err))))
            rows.append(row);saved.append(dict(cells=np.asarray(cells),coefficients=np.asarray(c),root_codes=codes[lo:hi]))
            print('neural_feature_span',kind,depth,ci,float(jnp.max(ef)),flush=True)
        summary=dict(kind=kind,max_depth=depth,nodes=20,rows=rows,
            worst_independent_error=max(float(np.max(r['errors'])) for r in rows),
            max_scoring_change=max(r['max_scoring_change'] for r in rows))
        summary['fits_sha256']=checkpoint(out/f'{kind}_depth{depth}.pkl',saved)
        result['allocations'].append(summary);save(out/'result.json',result)
    eligible=[r for r in result['allocations'] if r['kind']=='prescribed' and r['worst_independent_error']<=.005 and r['max_scoring_change']<=.0002]
    result.update(status='complete',capacity_pass=bool(eligible),
        advancement='nonlinear child-head training' if eligible else 'stop final repair: tested frozen-parent feature span does not meet capacity prerequisite',
        limitation='Fixed original-parent embeddings; failure does not prove a universal local neural decoder limitation')
    save(out/'result.json',result)


if __name__=='__main__':main()
