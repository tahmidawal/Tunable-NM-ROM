"""Bounded fixed-checkpoint multistart diagnosis; never a new model admission."""
import argparse
from common import *
import data
from neural import stable as S,fields as F
from neural.build_data import resolve
from training_io import read_checkpoint,checkpoint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE'])
    rp=resolve(ns/'b7parentrepair/output/result.json',ns);r=json.loads(rp.read_text());assert r['status']=='complete_stable_parent' and not r['parent_quality_pass']
    cp=rp.parent/'stable_parent.pkl';assert digest(cp)==r['selected_sha256'];p=read_checkpoint(cp)
    fits=rp.parent/'development_fits.pkl';assert digest(fits)==r['all_fits_sha256'];old=read_checkpoint(fits)
    dp=resolve(ns/'b6parent/output/coordinate_data.npz',ns);assert digest(dp)==p['data_sha256'];d=np.load(dp)
    network=jax.tree.map(jnp.asarray,p['network']);codes=jnp.asarray(p['training_codes']);enc=jnp.asarray(p['training_encoder'])
    xy,w=jnp.asarray(d['xy']),jnp.asarray(d['w']);training_fields=jnp.asarray(d['train_y']);xf,wf=data.grid(256,True);xf,wf=jnp.asarray(xf),jnp.asarray(wf)
    fit=F.make_fit(S.apply,60);curv=F.make_fit(S.apply,120,newton=True);decode=jax.jit(S.apply);jacobian=jax.jit(jax.jacfwd(S.apply,argnums=1))
    result=dict(meta=meta,status='running',checkpoint_sha256=digest(cp),source_sha256=digest(rp),rows=[],weights_changed=False,
        classification='Closing fixed-weight starting-point diagnosis only; not a third parent repair or method admission',final_test_opened=False)
    saved=[]
    for i in np.argsort(r['independent_errors'])[-6:][::-1]:
        target=jnp.asarray(d['dev_y'][i]);norm=jnp.asarray(d['dev_norm'][i]);field=jnp.asarray(d['dev_encoder'][i])
        distances=np.asarray(jnp.sum(w[None]*(training_fields-target)**2,axis=1));ids=[];cases=set()
        for k in np.argsort(distances):
            if int(k)//11 in cases:continue
            ids.append(int(k));cases.add(int(k)//11)
            if len(ids)==12:break
        pca=p['pca'];projected=((field-jnp.asarray(pca['center']))@jnp.asarray(pca['basis']))/jnp.asarray(pca['scale'])
        starts=[jnp.asarray(old['codes'][i]),projected,*[codes[k] for k in ids]];statuses=[];zs=[]
        for z0 in starts:
            z,st=fit(network,z0,xy,target,w,norm)
            if float(st[-1])==0:z,st=curv(network,z,xy,target,w,norm)
            zs.append(z);statuses.append(np.asarray(st))
        k=int(np.argmin(np.asarray(statuses)[:,0]));z=zs[k]
        src=p['data_record']['groups']['dev']['sources'][int(i)//11];path=resolve(src['file'],ns);assert digest(path)==src['sha256']
        b=jnp.asarray(np.load(path)['coefficients'][int(i)%11]);truth=data.synth_grid(b,jnp.unique(xf[:,0])).reshape(-1)
        error=float(jnp.sqrt(jnp.sum(wf*(decode(network,z,xf)-truth)**2))/norm)
        tangent=jacobian(network,z,xy)*jnp.sqrt(w)[:,None]/norm
        scales=jnp.linalg.norm(tangent,axis=0);singular=jnp.linalg.svd(tangent/jnp.maximum(scales,1e-14),compute_uv=False)
        conditioning=dict(column_norms=np.asarray(scales),normalized_singular_values=np.asarray(singular),rank_at_relative_1e6=int(jnp.sum(singular>singular[0]*1e-6)),minimum_relative=float(singular[-1]/singular[0]))
        row=dict(tangent_conditioning=conditioning,index=int(i),original_error=r['independent_errors'][int(i)],best_independent_error=error,best_start=k,training_start_indices=ids,statuses=statuses)
        result['rows'].append(row);saved.append(np.asarray(z));save(out/'result.json',result);print('starting_point_diagnosis',int(i),row['original_error'],error,k,flush=True)
    result['codes_sha256']=checkpoint(out/'codes.pkl',saved);result.update(status='complete',representation_admitted=False,pde_admitted=False)
    save(out/'result.json',result)


if __name__=='__main__':main()
