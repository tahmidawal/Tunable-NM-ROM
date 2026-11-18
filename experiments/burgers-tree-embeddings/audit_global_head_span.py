"""Frozen neural-output-span relaxation: separates shape span from nonlinear coding."""
import argparse
from common import *
from training_io import read_checkpoint,summary
import normalized_heads as NH


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--models',required=True);ap.add_argument('--spans',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    result=dict(meta=preflight(),status='running_frozen_global_output_span',models=[],
                weights_changed=False,reference_assisted=True,PDE_solve=False,final_test_opened=False)
    models=a.models.split(',');spans=a.spans.split(',');assert len(models)==len(spans)
    for path,spanpath in zip(models,spans):
        path=Path(path);spanpath=Path(spanpath)
        rec=json.loads((path/'result.json').read_text());sr=json.loads((spanpath/'result.json').read_text())
        assert digest(path/'normalized_parent.pkl')==rec['model_sha256']
        assert digest(spanpath/'reduced_data.npz')==sr['reduced_data_sha256']
        model=read_checkpoint(path/'normalized_parent.pkl');assert model['basis_sha256']==sr['basis_sha256']
        d=np.load(spanpath/'reduced_data.npz');h=jax.tree.map(jnp.asarray,model['params']['head'])
        scale=jnp.asarray(model['scale']);mean=jnp.asarray(model['mean'])
        # Every nonlinear output lies in offset + range(W), regardless of latent fitting.
        W=scale[:,None]*jnp.concatenate([h['linear'],h['net'][-1]['w'].T],axis=1)
        offset=mean+scale*(h['bias']+h['net'][-1]['b'])
        U,s,_=jnp.linalg.svd(W,full_matrices=False);rank=int(jnp.sum(s>s[0]*1e-12));Q=U[:,:rank]
        row=dict(latent=rec['latent'],last_hidden_width=h['net'][-1]['w'].shape[0],
            feature_count=W.shape[0],affine_span_rank=rank,relative_singular_values=np.asarray(s/s[0]),
            model_sha256=rec['model_sha256'],data_sha256=sr['reduced_data_sha256'],panels={})
        for panel in ['train','dev']:
            C=jnp.asarray(d[panel+'_coefficients']);norm=jnp.asarray(d[panel+'_norm']);missing=jnp.asarray(d[panel+'_missing'])
            centered=C-offset;res=centered-(centered@Q)@Q.T
            err=jnp.sqrt(jnp.sum(res*res,axis=1)+missing)/norm
            row['panels'][panel]=summary(err.reshape(-1,11))
            if panel=='dev':
                z=jnp.asarray(model['dev_codes']);pred=NH.head(h,z,mean,scale)
                actual=jnp.sqrt(jnp.sum((pred-C)**2,axis=1)+missing)/norm
                defect=(pred-offset)-((pred-offset)@Q)@Q.T
                row['max_head_distance_outside_affine_span']=float(jnp.max(jnp.linalg.norm(defect,axis=1)/norm))
                row['head_projection']=summary(actual.reshape(-1,11))
                row['lower_bound_violation']=float(jnp.max(err-actual))
                assert row['max_head_distance_outside_affine_span']<1e-9
                assert row['lower_bound_violation']<1e-9
                assert abs(float(jnp.max(actual))-rec['worst_projection'])<1e-9
                row['dev_labels']=rec.get('dev_labels') or sr.get('dev_labels') or sr.get('labels')
                row['legacy_labels_absent']=row['dev_labels'] is None
        result['models'].append(row);save(out/'result.json',result)
        print('head_span',row['latent'],rank,row['panels']['dev']['worst'],rec['worst_projection'],flush=True)
    result['status']='complete_frozen_global_output_span';save(out/'result.json',result)


if __name__=='__main__':main()
