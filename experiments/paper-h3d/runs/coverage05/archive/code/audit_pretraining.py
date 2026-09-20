"""Independent NumPy/SciPy Heat3D teacher and physical coefficient metric audit."""
import pickle
import numpy as np
from audit_panel import reference


def trunk(params,coordinates,spec):
    features=[coordinates]
    for frequency in spec.get('trunk_frequencies',[1.,2.,4.]):
        features.extend([np.sin(np.pi*frequency*coordinates),np.cos(np.pi*frequency*coordinates)])
    value=np.concatenate(features,axis=-1)
    for layer in params['trunk'][:-1]:value=np.tanh(value@layer['w']+layer['b'])
    layer=params['trunk'][-1];return value@layer['w']+layer['b']


def audit(out,record):
    cfg=record['config'];n=cfg['train_intervals'];rows=[]
    settings=[info for info in record['operators'] if (out/'operators'/info['name']/'pretraining').exists()]
    if not settings:return dict(passed=True,recipes=[])
    parameters=np.random.default_rng(cfg['train_seed']).random((cfg['train_count'],5))*[.3,.3,.3,.05,.4]+[.35,.35,.35,.10,.8]
    fields=np.stack([reference(n,p,cfg,False) for p in parameters])
    scale=float(np.sqrt(np.mean(fields[:,0]**2)));x=np.arange(1,n)/n*2-1
    coords=np.stack(np.meshgrid(x,x,x,indexing='ij'),-1).reshape(-1,3)
    for info in settings:
        np.testing.assert_allclose(scale,info['physical_scale'],rtol=1e-12)
        folder=out/'operators'/info['name']/'pretraining';target=(fields[:,1:]/scale).reshape(-1,(n-1)**3)
        denominator=np.sum(target*target,axis=1);saved=np.load(folder/'training_teacher.npz');q=saved['spatial_basis'];eigen=saved['eigenvalues']
        np.testing.assert_allclose(saved['denominators'],denominator,rtol=1e-12,atol=1e-12)
        weighted=target/np.sqrt(denominator[:,None]);coeff=weighted@q
        eigen_defect=float(np.linalg.norm(weighted.T@coeff-q*eigen)/np.linalg.norm(q*eigen));assert eigen_defect<1e-8
        orthogonality=float(np.max(abs(q.T@q-np.eye(q.shape[1]))));assert orthogonality<1e-6
        errors=np.linalg.norm(target@q@q.T-target,axis=1)/np.sqrt(denominator)
        np.testing.assert_allclose(errors,saved['projection_errors'],rtol=1e-7,atol=1e-10)
        selected=pickle.loads((folder/'trunk_selected.pkl').read_bytes());matrix=trunk(selected['params'],coords,selected['spec'])/np.sqrt(q.shape[1])
        teacher=np.load(folder/'branch_teacher.npz');coefficient=teacher['coefficients'].reshape(-1,q.shape[1]);gram=matrix.T@matrix
        gram_defect=float(np.linalg.norm(gram-teacher['gram'])/np.linalg.norm(gram));assert gram_defect<1e-11
        bias=np.tile(selected['params']['bias'],len(fields));residual=coefficient@matrix.T+bias[:,None]-target
        normal_defect=float(np.linalg.norm(residual@matrix)/max(np.linalg.norm(target@matrix),1e-300));assert normal_defect<1e-10
        learned_errors=np.linalg.norm(residual,axis=1)/np.sqrt(denominator)
        np.testing.assert_allclose(learned_errors,teacher['learned_trunk_projection_errors'],rtol=1e-7,atol=1e-10)
        perturbation=np.sin(np.arange(q.shape[1]))[None,:]*.01
        physical=np.sum((perturbation@matrix.T)**2);metric=np.sum((perturbation@np.linalg.cholesky(gram))**2)
        metric_defect=float(abs(physical-metric)/max(physical,1e-300));assert metric_defect<1e-11
        rows.append(dict(name=info['name'],passed=True,training_cases=len(fields),training_output_snapshots=len(target),
            weighted_pod_eigen_residual=eigen_defect,orthogonality=orthogonality,gram_relative_defect=gram_defect,
            least_squares_normal_defect=normal_defect,physical_metric_identity_defect=metric_defect))
    return dict(passed=True,recipes=rows,scope='regenerated training heat solutions, weighted POD teacher, learned trunk least squares and field Gram metric; no development/final teacher')
