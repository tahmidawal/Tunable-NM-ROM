"""Bind prospective energy weights to actual original TRAIN POD coefficients.

This is saved-field linear algebra inside a future released training allocation;
no native FE assembly, flow solve, development or validation input is used.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from train_bank import array_hash
from accuracy_v2.train import digest
from fe_pod_a6 import TOL


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def certify(training_path,training_file_sha256,points,weights,teacher,identity,directory,block_points=4096):
    out=Path(directory);out.mkdir(exist_ok=False)
    training_path=Path(training_path)
    if sha(training_path)!=training_file_sha256:raise ValueError('exact original training archive required')
    if identity['split']!='training' or identity['Re']!=[15+1.25*j for j in range(9)]:raise ValueError('actual ordered nine-case training identity required')
    with np.load(training_path,allow_pickle=False) as f:
        reynolds=f['Re'];x=f['points_physical'];w=f['weights_dimensionless'];snapshots=f['velocity_corrections']
    n=len(points);r=teacher.shape[-1]
    if r!=4 or teacher.shape!=(n,2,r) or snapshots.shape!=(9,n,2) or not np.array_equal(reynolds,identity['Re']) or not np.array_equal(x,points) or not np.array_equal(w,weights):raise ValueError('original training/teacher point, component or Re ordering differs')
    if array_hash(x,w,snapshots)!=identity['training_array_sha256'] or digest(teacher)!=identity['arrays']['values']:raise ValueError('original training or teacher field content differs')
    if not np.isfinite(snapshots).all() or not np.isfinite(teacher).all() or np.any(w<=0) or not np.isfinite(w).all():raise ValueError('finite full training fields required')
    if type(block_points) is not int or block_points<=0:raise ValueError('positive block width required')
    gram=np.zeros((r,r));rhs=np.zeros((r,9));cursor=0
    for start in range(0,n,block_points):
        end=min(n,start+block_points);v=teacher[start:end]
        if start!=cursor:raise ValueError('coefficient point coverage changed')
        gram+=np.einsum('n,ncr,ncs->rs',w[start:end],v,v)
        rhs+=np.einsum('n,ncr,snc->rs',w[start:end],v,snapshots[:,start:end,:])
        cursor=end
    assert cursor==n
    np.savez_compressed(out/'projection-inputs.npz',gram=gram,rhs=rhs,Re=reynolds)
    coefficients=np.linalg.solve(gram,rhs);sigma=np.asarray(identity['singular_values'],dtype=np.float64)
    covariance=coefficients@coefficients.T;expected=np.diag(sigma*sigma)
    arrays=out/'coefficients-and-covariance.npz'
    np.savez_compressed(arrays,coefficients=coefficients,gram=gram,rhs=rhs,covariance=covariance,expected_covariance=expected,singular_values=sigma,Re=reynolds)
    gram_error=float(np.linalg.norm(gram-np.eye(r),2));covariance_error=float(np.max(np.abs(covariance-expected))/sigma[0]**2)
    diagonal_relative=np.abs(np.diag(covariance)-sigma*sigma)/(sigma*sigma)
    normalized_correlations=(covariance-expected)/(sigma[:,None]*sigma[None,:])
    finite=all(np.isfinite(a).all() for a in (coefficients,gram,rhs,covariance,expected,normalized_correlations))
    whitened_spectral_error=float(np.linalg.norm(normalized_correlations,2)) if finite else float('inf')
    passed=bool(finite and gram_error<=TOL['gram'] and whitened_spectral_error<=TOL['singular'])
    np.savez_compressed(out/'whitened-covariance-error.npz',error=normalized_correlations)
    def safe(a):
        a=np.asarray(a)
        return a.tolist() if np.isfinite(a).all() else None
    record={'schema':'ns-obstacle-original-training-coefficient-energy-binding-v1','passed':passed,'split':'training','Re':identity['Re'],'mode_order':'first four accepted prepared velocity POD columns, unchanged orientations','training_file_sha256':training_file_sha256,'training_array_sha256':identity['training_array_sha256'],'teacher_identity':identity,'arrays_file':arrays.name,'arrays_sha256':sha(arrays),'coefficients_sha256':digest(coefficients),'gram_spectral_error':safe(gram_error),'covariance_max_error_over_sigma0_squared':safe(covariance_error),'whitened_covariance_spectral_error':safe(whitened_spectral_error),'whitened_covariance_file':'whitened-covariance-error.npz','whitened_covariance_sha256':sha(out/'whitened-covariance-error.npz'),'diagonal_relative_errors':safe(diagonal_relative),'covariance_errors_over_sigma_products':safe(normalized_correlations),'tolerances':{'gram':TOL['gram'],'whitened_covariance_spectral':TOL['singular']},'source_sha256':sha(__file__),'coefficient_semantics':'full-rule L2 projection of actual nine TRAIN homogeneous corrections onto original retained FE POD columns; not online solved coefficients','equivalence_scope':'retained-four-mode training reconstruction with measured covariance approximation; excludes omitted modes and online solve errors','certificate_interpretation':'coefficient-covariance algebra compatibility: relative quadratic-form discrepancy bounded by recorded whitened spectral error; not a physics gate','development_or_validation_read':False}
    (out/'RESULT.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    if not passed:raise ValueError('actual original training coefficient covariance did not certify energy interpretation')
    return record,sha(out/'RESULT.json')
