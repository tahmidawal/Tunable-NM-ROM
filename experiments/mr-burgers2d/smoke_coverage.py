"""Sub-minute smoke for the two-seed initial-field coverage trainer."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import jax
import engines as e
import accuracy_coverage_paths as cp
start=time.perf_counter();assert jax.default_backend()=='gpu'
p=e.sc.init_separable(jax.random.PRNGKey(20),4,16,n_ff=4,g_hidden=16,g_layers=1,h_hidden=16,h_layers=1)
Z=np.random.default_rng(1).normal(size=(32,4));cfg=dict(mesh=8,physical_draws=[dict(seed=0,cases=2),dict(seed=1000,cases=2)],cases=4,seed=2,code_fit_budget=3,steps=3,lr=1e-4,batch=4,replay_cases=8,replay_weight=1.)
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:p2,Z2,info=cp.train_head(p,Z,cfg,Path(tmp),lambda x:None)
assert np.array_equal(np.array(info['physical_cases']),np.concatenate((e.params_draw(0,2),e.params_draw(1000,2))))
assert Z2.shape==(36,4) and info['optimized_code_scalars']==16 and info['optimizer_loop_seconds']>0 and info['unwhitening_relative_parity']<1e-12
r=dict(passed=True,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,exact_two_seed_draw=True,unwhitening_relative_parity=info['unwhitening_relative_parity'],seconds=time.perf_counter()-start)
(Path(__file__).parent/'smoke-coverage.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
