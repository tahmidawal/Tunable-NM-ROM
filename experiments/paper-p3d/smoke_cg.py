"""Bounded native panel plus zero-RHS/iteration-cap CG verification."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import iterative_cg as CG
import run
import audit

out=Path(__file__).parent/'runs/cg-smoke'
cfg=json.loads(Path(__file__).with_name('config.json').read_text())
cfg.pop('reuse_checkpoint_directory',None)
cfg.update(train_intervals=8,evaluation_intervals=[8],reference_intervals=[8,16],train_count=12,validation_count=1,
    bank_rank=8,latent_dimensions=[2],bank_steps=3,head_steps=3,weak_tests=32,q_ladder=[0,2],bank_width=16,head_width=16,
    fourier_features=4,bank_batch_states=4,bank_batch_points=64,quadrature_candidates=128,quadrature_fit_rows=64,
    quadrature_decoder_snapshots=4,repetitions=2,burn_seconds=.001,lm_budget=8,field_chunk=1024,oracle_starts=2,
    checkpoint_every=3,operators=[],pod_ranks=[2,4,8],frozen_offline_assets=False,frozen_pod_transfer_control=False)
run.run(cfg,out,smoke=True)
audit.audit(out,out/'audit.json')
f=jnp.zeros((7,7,7));zero,stats=CG.engine(8,1e-6,32)(f);jax.block_until_ready(stats)
assert np.array_equal(zero,np.zeros((7,7,7))) and stats[0]==0 and stats[3]==1
f=jnp.asarray(np.random.default_rng(317).normal(size=(7,7,7)))
x,stats=CG.engine(8,1e-6,1)(f);jax.block_until_ready(stats)
assert stats[0]==1 and stats[3]==0 and stats[4]==1
(out/'edge-checks.json').write_text(json.dumps(dict(zero_rhs_passed=True,iteration_cap_failure_retained=True))+'\n')
