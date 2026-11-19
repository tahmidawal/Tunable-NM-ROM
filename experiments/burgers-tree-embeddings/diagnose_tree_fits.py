"""Separate decoder capacity from stale-code initialization at retained checkpoints."""
import argparse
from common import *
from training_io import read_checkpoint,checkpoint
from neural.build_data import resolve
from neural import fields as F,allocations as A
from neural.initialize import missing_field
from train_coordinate_tree import apply,jac


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta_=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    source=resolve(ns/'b6treecurv/output/result.json',ns);record=json.loads(source.read_text())
    assert record['status']=='complete_neural_tree_subset_diagnostic'
    folder=source.parent;parent=resolve(ns/'b6parent/output/neural_parent.pkl',ns)
    ps=read_checkpoint(parent);dp=parent.parent/'coordinate_data.npz';assert digest(dp)==ps['data_sha256']
    arrays=dict(np.load(dp));d={k:jnp.asarray(v) for k,v in arrays.items()}
    old=read_checkpoint(folder/'parent_stability.pkl')['codes']
    indices=[11*c+j for c in [0,3,13,43] for j in [0,5,10]]
    fitroot=F.make_fit(F.root,120,newton=True);fittree=F.make_fit(apply,60,jacobian=jac)
    decode=jax.jit(apply)
    result=dict(meta=meta_,status='diagnostic_running',source_sha256=digest(source),checkpoints=[],
                final_test_opened=False,representation_admitted=False,
                protocol='Refit current checkpoint parent before full-grid missing-field initialization and unchanged joint fit')
    for history in record['history']:
        path=folder/f"tree_{history['update']:05d}.pkl";assert digest(path)==history['sha256']
        model=read_checkpoint(path);network=model['network'];rows=[]
        for i in indices:
            cells=jnp.asarray(A.allocate('prescribed',arrays['xy'],arrays['dev_y'][11*(i//11)],model['seed']+i//11))
            params=dict(network,cells=cells)
            candidates=[fitroot(network['root'],jnp.asarray(start),d['xy'],d['dev_y'][i],d['w'],d['dev_norm'][i])
                        for start in [old[i],np.zeros(8)]]
            rz,rs=min(candidates,key=lambda v:float(v[1][0]))
            z,loss=missing_field(network,rz,cells,d['xy'],d['dev_y'][i],d['w'],d['dev_norm'][i])
            initial_error=float(jnp.sqrt(loss[-1]))
            fitted,status=fittree(params,z,d['xy'],d['dev_y'][i],d['w'],d['dev_norm'][i])
            tp=dict(network,cells=jnp.asarray(model['cells'][i]))
            train_pred=decode(tp,jnp.asarray(model['training_codes'][i]),d['xy'])
            train_error=float(jnp.sqrt(jnp.sum(d['w']*(train_pred-d['train_y'][i])**2))/d['train_norm'][i])
            rows.append(dict(index=i,root_status=np.asarray(rs),initialized_error=initial_error,
                status=np.asarray(status),codes=np.asarray(fitted),cells=np.asarray(cells),stored_training_error=train_error))
        errors=np.array([r['status'][0] for r in rows]);worst=int(np.argmax(errors));row=rows[worst]
        zzero=np.concatenate([row['codes'][:8],np.zeros(160)])
        JJ=np.array(F.tree_jacobian(network,jnp.asarray(zzero),d['xy'],jnp.asarray(row['cells'])),copy=True)
        JJ*=np.sqrt(arrays['w'])[:,None];q,_=np.linalg.qr(JJ[:,:8]);local=JJ[:,8:]-q@(q.T@JJ[:,8:])
        s=np.linalg.svd(local,compute_uv=False)
        summary=dict(update=history['update'],source_sha256=history['sha256'],worst_error=float(errors.max()),
            median_error=float(np.median(errors)),unfinished=sum(int(r['status'][-1]==0) for r in rows),
            max_root_error=max(float(r['root_status'][0]) for r in rows),
            root_unfinished=sum(int(r['root_status'][-1]==0) for r in rows),
            stored_train_worst=max(r['stored_training_error'] for r in rows),
            child_directions_after_parent=dict(count=int(np.sum(s>s[0]*1e-6)),total=160,minimum_relative=float(s[-1]/s[0])))
        summary['fits_sha256']=checkpoint(out/f"refits_{history['update']:05d}.pkl",rows)
        result['checkpoints'].append(summary);save(out/'result.json',result)
        print('tree_refit_diagnostic',summary,flush=True)
    result.update(status='complete',best=min(result['checkpoints'],key=lambda r:(r['worst_error'],r['median_error'])))
    save(out/'result.json',result)


if __name__=='__main__':main()
