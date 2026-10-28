"""CPU-only metadata fingerprint of retained training targets; no model execution/data upload."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('npz');ap.add_argument('--out',required=True);a=ap.parse_args()
    sha=hashlib.sha256(Path(a.npz).read_bytes()).hexdigest()
    assert sha=='4a7715659fe24b640c68e3c5f8b44e804383d4705aa62efa7dd6a3aaef73ebbb'
    d=np.load(a.npz)
    hashes={k:hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() for k in ('pick','mu_tr','mu_te','traj_tr','t_tr','traj_te','t_te')}
    moments={}
    for k in ('un2_tr','un2_te','fl2_tr','fl2_te','Gram'):
        x=np.asarray(d[k]);moments[k]=[float(x.sum()),float(np.sum(x*x))]
    out=dict(source_npz_sha256=sha,array_hashes=hashes,moments=moments,
             note='Only hashes and aggregate numerical checks; all actual training/diagnostic targets regenerated on cluster.')
    Path(a.out).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
