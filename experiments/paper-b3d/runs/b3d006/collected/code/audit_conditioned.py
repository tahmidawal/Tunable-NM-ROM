"""Independent CPU audit of the trained spatial teachers and saved predictions."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from audit import stencil


def trunk(params,coordinates,spec):
    features=[coordinates]
    for frequency in spec.get('trunk_frequencies',[1.,2.,4.]):features.extend([np.sin(np.pi*frequency*coordinates),np.cos(np.pi*frequency*coordinates)])
    x=np.concatenate(features,axis=-1)
    for layer in params['trunk'][:-1]:x=np.tanh(x@layer['w']+layer['b'])
    last=params['trunk'][-1];return (x@last['w']+last['b'])/np.sqrt(spec['rank'])


def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--config',required=True);a=p.parse_args()
    out=Path(a.out);cfg=json.loads(Path(a.config).read_text());raw=np.load(out/'training_fields.npy',mmap_mode='r')
    records=[];max_ref=0.;max_metric=0.
    for seed in (0,1):
        path=out/f'seed{seed}';meta=json.loads((path/'metadata.json').read_text());pre=path/'pretraining'
        info=json.loads((pre/'pretraining.json').read_text());assert not info['development_data_used'] and not info['final_data_used']
        scale=meta['scale'];snapshots=np.asarray(raw[:,1:]/scale).reshape(-1,(cfg['nodes']-2)**3)
        norms=np.repeat(np.sum((raw[:,0]/scale)**2,axis=(1,2,3)),len(cfg['train_steps'])-1)
        teacher=np.load(pre/'training_teacher.npz');B=teacher['spatial_basis'];weighted=snapshots/np.sqrt(norms[:,None])
        assert np.max(np.abs(norms-teacher['denominators']))<1e-8
        orth=float(np.max(np.abs(B.T@B-np.eye(B.shape[1]))));assert orth<1e-6
        covariance=weighted.T@(weighted@B);eigen=float(np.linalg.norm(covariance-B*teacher['eigenvalues'])/np.linalg.norm(covariance));assert eigen<1e-7
        error=np.linalg.norm(snapshots@B@B.T-snapshots,axis=1)/np.sqrt(norms)
        assert np.max(np.abs(error-teacher['projection_errors']))<1e-10
        ck=pickle.loads((pre/'trunk_selected.pkl').read_bytes());n=cfg['nodes'];axis=np.linspace(0,1,n)[1:-1]*2-1
        coords=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3);matrix=trunk(ck['params'],coords,cfg['spec'])
        branch=np.load(pre/'branch_teacher.npz');gram=matrix.T@matrix
        assert np.max(np.abs(gram-branch['gram']))<1e-7
        coefficients=branch['coefficients'].reshape(len(snapshots),cfg['spec']['rank']);bias=np.tile(ck['params']['bias'],cfg['train_trajectories'])
        residual=coefficients@matrix.T+bias[:,None]-snapshots
        normal=float(np.linalg.norm(residual@matrix)/max(np.linalg.norm(residual)*np.linalg.norm(matrix),1e-300));assert normal<1e-9
        learned_error=np.linalg.norm(residual,axis=1)/np.sqrt(norms)
        assert np.max(np.abs(learned_error-branch['learned_trunk_projection_errors']))<1e-10
        dev=json.loads((path/'development.json').read_text());assert dev['expanded_development_cases']==cfg['development_rows']
        for row in dev['rows']:
            ref=np.load(out/f"reference_row{row['row']}.npz");pred=np.load(path/row['artifact'])['fields'];truth=ref['fields']
            err=np.linalg.norm(pred-truth,axis=1)/np.linalg.norm(truth[0]);delta=float(np.max(np.abs(err-row['error_fixed_initial'])))
            assert delta<1e-12 and np.isfinite(pred).all();max_metric=max(max_metric,delta)
            if seed==0:
                for step in range(1,len(truth)):
                    adv,lap=stencil(truth[step],n);defect=truth[step]-truth[step-1]+cfg['dt']*(adv-float(ref['nu'])*lap)
                    max_ref=max(max_ref,float(np.linalg.norm(defect)/np.linalg.norm(truth[step-1])))
        assert abs(max(v['worst_evolved'] for v in dev['rows'])-dev['worst_evolved'])<1e-12
        records.append(dict(seed=seed,teacher_orthogonality_maximum=orth,teacher_eigen_residual=eigen,
            learned_trunk_normal_equation_residual=normal,development_cases=len(dev['rows'])))
    assert max_ref<2e-9
    result=dict(passed=True,seeds=records,maximum_reference_residual=max_ref,maximum_metric_discrepancy=max_metric,
        final_cohort_unopened=True,scope='all saved candidate development fields, initial-norm teacher construction, covariance eigenpairs and learned-trunk coefficient optimality')
    (out/'audit-local.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
