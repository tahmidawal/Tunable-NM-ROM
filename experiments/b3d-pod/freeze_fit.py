"""Verify exact retained fitting inputs and freeze the single accepted support before validation."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from analyse import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',required=True);a=ap.parse_args()
    p=Path(a.fit);root=p.parent;here=Path(__file__).resolve().parent
    j=json.loads(p.read_text())
    assert j['complete'] is True and j['accepted'] is True and j['attempts']==1
    assert j['config']['target_points']==16384 and j['config']['max_drops']==3
    assert j['path']['reason']=='taus' or (j['path']['reason']=='kkt' and j['path']['kkt']['ok'] is True)
    assert j['path']['drift_ok'] is True and 0<j['peak_device_bytes']<=120e9
    assert j['path']['banned']<=.01*j['config']['candidate_count']
    for file,key in [('candidate_nodes.npy','candidate_sha256'),('target.npy','target_sha256'),('contributions.npy','matrix_sha256')]:
        assert sha(root/file)==j['contributions'][key],file
    wpath=root/'HR~ecsw~512~k32.npz';assert sha(wpath)==j['support_sha256']
    w=np.load(wpath);nodes,weights=w['nodes'],w['weights']
    cand=np.load(root/'candidate_nodes.npy');b=np.load(root/'target.npy');G=np.load(root/'contributions.npy',mmap_mode='r')
    assert len(nodes)==j['support_points'] and 0<len(nodes)<=16384
    assert len(set(nodes.tolist()))==len(nodes) and np.isfinite(weights).all() and np.all(weights>0)
    ix=np.searchsorted(cand,nodes);assert np.array_equal(cand[ix],nodes)
    r=np.empty(len(b))
    for k in range(0,len(b),128):r[k:k+128]=b[k:k+128]-np.asarray(G[k:k+128])[:,ix]@weights
    rel=float(np.linalg.norm(r)/np.linalg.norm(b))
    assert abs(rel-j['path']['relres'])<=max(1e-10,1e-7*rel),'independent training residual mismatch'
    dst=here/'inputs/HR~ecsw~512~k32.npz';assert not dst.exists(),'support already frozen; do not overwrite'
    shutil.copyfile(wpath,dst)
    m=dict(files={'Phi.npz':j['basis_sha256'],dst.name:j['support_sha256']},points=len(nodes),
           fit_result_path=str(p.resolve()),fit_result_sha256=sha(p),fit_job=j['job_id'],
           source_commit=j['source_commit'],training_matrix_sha256=j['contributions']['matrix_sha256'],
           independent_training_relative_residual=rel,fit_accepted_before_validation=True)
    (here/'inputs/K32-MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m,indent=2))


if __name__=='__main__':main()
