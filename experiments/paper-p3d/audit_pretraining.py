"""Independent NumPy/SciPy verification of train-only Poisson POD teachers."""
import pickle
import hashlib
import json
import numpy as np
from scipy.fft import dstn
from audit import family,forcing
from trunk_diagnostic import trunk


def audit(out,record):
    cfg=record['config'];n=cfg['train_intervals'];k=np.arange(1,n)
    one=4*n*n*np.sin(np.pi*k/(2*n))**2;lam=one[:,None,None]+one[None,:,None]+one[None,None,:]
    fields=np.stack([dstn(dstn(forcing(n,p),type=1,norm='ortho')/lam,type=1,norm='ortho')
        for p in family(cfg['train_seed'],cfg['train_count'])])
    a=np.arange(1,n)/n*2-1;coords=np.stack(np.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)
    rows=[];reused=[]
    for info in record['operators']:
        if 'pretraining' not in info:continue
        folder=out/'operators'/info['name']/'pretraining'
        if not folder.exists():
            entry=next(x for x in cfg['operators'] if x['name']==info['name'])
            assert entry.get('reuse') and info.get('immediate_reuse_source')
            origin=info['original_training_source'];assert origin['source_commit'] and origin['job_id']
            name=f"operators/{info['name']}/best.pkl"
            with (out/name).open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
            assert checksum==origin['checkpoint_sha256']
            source=out.parent/cfg['reuse_checkpoint_directory']/f"operators/{info['name']}/SOURCE_RESULT.json"
            with source.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
            assert source_hash==record['reused_checkpoints']['files'][f"operators/{info['name']}/SOURCE_RESULT.json"]
            prior=json.loads(source.read_text());assert prior['complete'] and not prior['final_cohort_opened']
            prior_info=next(x for x in prior['operators'] if x['name']==info['name'])
            assert prior_info['pretraining']==info['pretraining']
            reused.append(dict(name=info['name'],checkpoint_sha256=checksum,original_training_source=origin,
                scope='Previously audited pretraining archive reused by exact checkpoint and source-result hash; teacher algebra not rerun in this allocation.'))
            continue
        target=fields.reshape(len(fields),-1)/info['scales']['output'];denominator=np.sum(target*target,axis=1)
        saved=np.load(folder/'training_teacher.npz');q=saved['spatial_basis'];eigen=saved['eigenvalues']
        assert np.allclose(saved['denominators'],denominator,rtol=1e-12,atol=1e-12)
        weighted=target/np.sqrt(denominator[:,None]);coeff=weighted@q
        eigen_defect=float(np.linalg.norm(weighted.T@coeff-q*eigen)/np.linalg.norm(q*eigen))
        assert eigen_defect<1e-8
        orthogonality=float(np.max(abs(q.T@q-np.eye(q.shape[1]))));assert orthogonality<1e-6
        errors=np.linalg.norm(target@q@q.T-target,axis=1)/np.sqrt(denominator)
        assert np.allclose(errors,saved['projection_errors'],rtol=1e-7,atol=1e-10)
        checkpoint=pickle.loads((folder/'trunk_selected.pkl').read_bytes())
        matrix=trunk(checkpoint['params'],coords,checkpoint['spec'])/np.sqrt(q.shape[1])
        branch=np.load(folder/'branch_teacher.npz');teacher_coeff=branch['coefficients'][:,0]
        gram=matrix.T@matrix
        gram_defect=float(np.linalg.norm(gram-branch['gram'])/np.linalg.norm(gram));assert gram_defect<1e-11
        bias=float(checkpoint['params']['bias'][0]);prediction=teacher_coeff@matrix.T+bias
        residual=prediction-target;normal_defect=float(np.linalg.norm(residual@matrix)/max(np.linalg.norm(target@matrix),1e-300))
        assert normal_defect<1e-10
        learned_errors=np.linalg.norm(residual,axis=1)/np.sqrt(denominator)
        assert np.allclose(learned_errors,branch['learned_trunk_projection_errors'],rtol=1e-7,atol=1e-10)
        # Deterministic coefficient perturbation checks the actual physical metric.
        perturbation=np.sin(np.arange(q.shape[1]))[None,:]*.01
        physical=np.sum((perturbation@matrix.T)**2);metric=np.sum((perturbation@np.linalg.cholesky(gram))**2)
        metric_defect=float(abs(physical-metric)/max(physical,1e-300));assert metric_defect<1e-11
        assert info['pretraining']['development_data_used'] is False and info['pretraining']['final_data_used'] is False
        rows.append(dict(name=info['name'],passed=True,training_cases=len(fields),orthogonality=orthogonality,
            weighted_pod_eigen_residual=eigen_defect,gram_relative_defect=gram_defect,
            least_squares_normal_defect=normal_defect,physical_metric_identity_defect=metric_defect,
            teacher_projection_worst=float(max(errors)),learned_trunk_projection_worst=float(max(learned_errors))))
    return dict(passed=True,recipes=rows,reused_pretraining_records=reused,scope='Fresh recipes: independent regenerated training solutions, weighted POD eigenspace, learned-trunk least squares and physical coefficient metric. Reused recipes: exact prior-checkpoint/source-result provenance only. No retraining or global convergence claim.')
