"""Actual-checkpoint nested geometry smoke; no scientific timing claim."""
import json
from pathlib import Path
import tempfile
import numpy as np
import nested_head as n
from fresh_models import tree_from_npz

cell=Path(__file__).resolve().parent;endpoints={};initializers={}
for name,run in [('trained_phase','accel08'),('trained_phase40','accel09')]:
    root=cell/'runs'/run/'cluster/out/pilot';endpoints[name]=tree_from_npz(root/f'head_{name}.npz')
    with np.load(root/'fixed_encoder_training.npz') as f:initializers[name]=(f['linear'],f['center'])
with tempfile.TemporaryDirectory(dir=cell/'checks') as temporary:
    _,_,record=n.build(endpoints,initializers,Path(temporary))
print(json.dumps(record,indent=2))
