"""Actual checkpoint integration smoke; reduced mesh/horizon and tiny fit budget."""
from pathlib import Path
import tempfile
import json
import numpy as np
import dynamics as d

cell=Path(__file__).resolve().parent
inputs=cell/'runs/pilot01/cluster/in/dirichlet'
cfg=json.loads((cell/'dynamics-config.json').read_text())
cfg.update(diagnostic_indices=[0,1],diagnostic_fit_budgets=[4,8])
grid=d.Grid(64)
bank=d.base.rebuild(inputs,grid)
with np.load(inputs/'coordinates.npz') as f:
    # Smoke only: deterministic extended within-bank basis, no scientific PCA claim.
    linear=np.column_stack((f['common_linear'],np.eye(64)[:,:16]));center=f['common_center']
arms=d.arms_for(bank,inputs,linear,center)
par=d.parameter_rows(690602,2)[1]
u0,v0=map(np.asarray,d.localized_initial(grid,par))
u,v,rec,aux=d.linear_query(arms[0],u0,v0,par[5],grid,cfg,bank)
assert rec['completed'] and u.shape==(49,63,63)
truth=d.base.spectral_propagate(d.jnp.asarray(u0),d.jnp.asarray(v0),par[5],d.jnp.arange(49)*.05)
with tempfile.TemporaryDirectory(dir=cell/'checks') as scratch:
    diag=d.fitted_diagnostics(bank,arms,*map(np.asarray,truth),grid,par[5],cfg,Path(scratch)/'diagnostic.npz')
    assert len(diag['arms'])==4 and len(diag['fitting'])==2
print('actual_checkpoint_pipeline_smoke_passed',flush=True)
