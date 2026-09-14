"""Sub-minute GPU model-lookup preflight; generate no evaluation inputs."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
cell=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(cell))
import pilot as base
import iterative_replay as previous

cfg=json.loads((cell/'acceleration-confirm-config.json').read_text())
inputs=cell/'stages/accel11/in/dirichlet'
grid=base.Grid(64,'dirichlet','dirichlet')
full,models=previous.load_models(inputs,grid)
bank=models[cfg['primary_method']]
assert base.jax.default_backend()=='gpu' and base.jax.config.x64_enabled
assert bank['p']['linear'].shape==(64,32)
selected=cfg['selection_frozen']['selected_method']
with np.load(inputs/cfg['frozen_head_inputs'][selected]) as head:
    assert head['p/linear'].shape==(64,40)
assert hashlib.sha256((inputs/cfg['frozen_head_inputs'][selected]).read_bytes()).hexdigest()==cfg['selection_frozen']['checkpoint_sha256']
assert all(a['method'] in ('baseline','chol_guard',selected) for a in cfg['arms'])
result=dict(passed=True,jax_backend=base.jax.default_backend(),x64=base.jax.config.x64_enabled,
    resolved_primary_method=cfg['primary_method'],available_model_keys=list(models),
    selected_arm=selected,selected_checkpoint_sha256=cfg['selection_frozen']['checkpoint_sha256'],
    generated_evaluation_cases=0,queries=0)
(cell/'checks/confirmation-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
