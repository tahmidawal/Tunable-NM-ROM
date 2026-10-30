"""Synthetic original-coefficient certificate, including retained counterexample."""
from pathlib import Path
import tempfile,json
import numpy as np
from accuracy_v2.training_coefficients import certify,sha
from accuracy_v2.train import digest
from train_bank import array_hash

def check():
    rng=np.random.default_rng(71);n=31;w=np.full(n,1/n);x=rng.normal(size=(n,2))
    q=np.linalg.qr(rng.normal(size=(2*n,4)))[0];v=(q*np.sqrt(n)).reshape(n,2,4)
    sigma=np.array([6.,.35,.025,.002]);rotation=np.linalg.qr(rng.normal(size=(9,4)))[0].T;c=sigma[:,None]*rotation
    snapshots=np.einsum('ncr,rs->snc',v,c);re=[15+1.25*j for j in range(9)]
    identity={'split':'training','Re':re,'singular_values':sigma.tolist(),'training_array_sha256':array_hash(x,w,snapshots),'arrays':{'values':digest(v)}}
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);path=root/'training.npz';np.savez(path,Re=re,points_physical=x,weights_dimensionless=w,velocity_corrections=snapshots)
        record,_=certify(path,sha(path),x,w,v,identity,root/'pass')
        bad=dict(identity,singular_values=(sigma*np.array([1,1,1,1.01])).tolist())
        try:certify(path,sha(path),x,w,v,bad,root/'fail')
        except ValueError:pass
        else:raise AssertionError('tail coefficient mismatch accepted')
        failure=json.loads((root/'fail/RESULT.json').read_text());assert not failure['passed'] and (root/'fail/projection-inputs.npz').exists() and (root/'fail/coefficients-and-covariance.npz').exists()
        return {'passed':True,'synthetic_only':True,'whitened_covariance_spectral_error':record['whitened_covariance_spectral_error'],'tail_mismatch_rejected_and_raw_retained':True}
if __name__=='__main__':print(json.dumps(check(),indent=2))
