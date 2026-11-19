"""Fixed-checkpoint training/development fit diagnosis; no weight changes."""
import argparse
from common import *
from neural import stable as S,fields as F
from neural.build_data import resolve
from training_io import read_checkpoint,checkpoint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    source=resolve(ns/'b7parentfull/output/result.json',ns);r=json.loads(source.read_text());assert r['status']=='complete_stable_parent'
    cp=source.parent/'stable_parent.pkl';assert digest(cp)==r['selected_sha256'];p=read_checkpoint(cp)
    original=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(original)==p['data_sha256'];d=np.load(original)
    saved=read_checkpoint(source.parent/'development_fits.pkl');network=jax.tree.map(jnp.asarray,p['network']);codes=jnp.asarray(p['training_codes'])
    xy,w=jnp.asarray(d['xy']),jnp.asarray(d['w']);gn=F.make_fit(S.apply,120);newton=F.make_fit(S.apply,120,newton=True)
    result=dict(meta=meta,source_sha256=digest(source),checkpoint_sha256=digest(cp),rows=[],weights_changed=False,final_test_opened=False)
    collected=[]
    for group in ['train','dev']:
        errors=np.asarray(r['training_errors'] if group=='train' else r['independent_errors']);order=np.argsort(errors)
        ids=np.unique(np.concatenate([order[-6:],order[len(order)//2:len(order)//2+2]]))
        for i in ids:
            z0=codes[i] if group=='train' else jnp.asarray(saved['codes'][i])
            y,norm=jnp.asarray(d[group+'_y'][i]),jnp.asarray(d[group+'_norm'][i])
            z,st=gn(network,z0,xy,y,w,norm)
            z2,st2=newton(network,z,xy,y,w,norm)
            row=dict(group=group,index=int(i),stored_error=float(errors[i]),gn_status=np.asarray(st),newton_status=np.asarray(st2))
            result['rows'].append(row);collected.append(np.asarray(z2));save(out/'result.json',result);print('fit_diagnosis',row,flush=True)
    result['codes_sha256']=checkpoint(out/'diagnostic_codes.pkl',dict(rows=result['rows'],codes=collected))
    result['status']='complete';save(out/'result.json',result)


if __name__=='__main__':main()
