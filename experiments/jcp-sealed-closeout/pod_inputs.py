"""Regenerate one admitted cohort in an isolated interpreter before loading the POD code."""
import argparse
import sys
import json
from open_sealed import sha,packed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from panel import HERE, check_upstream
from published import Opened

p=argparse.ArgumentParser()
p.add_argument('dim',type=int,choices=[2,3])
p.add_argument('base',type=Path)
p.add_argument('cfg',type=Path)
p.add_argument('dest',type=Path)
a=p.parse_args()
check_upstream(a.base)
opened=Opened(HERE,a.cfg)
root=a.base/'vendor'/'tree'/'experiments'
rows=opened.regenerated(a.dim,SimpleNamespace(V=root))
if a.dim==2:
    np.savez(a.dest,rows=rows)
else:
    np.savez(a.dest,**{k:rows[k] for k in ('B','c','w','rho','A','nu','s_star','m')})

record=dict(dim=a.dim,cohort=opened.descriptor_receipt,input_sha256=sha(a.dest.read_bytes()),
            freeze_sha256=opened.record['freeze_sha256'],published_seed_commitment=opened.record['seed_sha256'])
a.dest.with_suffix('.receipt.json').write_bytes(packed(record))
