"""Bounded one-case native/fine-grid CG trace parity smoke, never a benchmark."""
import os,json,time
from pathlib import Path
import numpy as np
import jax
import common as C
import iterative_cg as I

start=time.perf_counter();root=Path(__file__).resolve().parent
cfg=json.loads((root/'developmentA.json').read_text());assert cfg['evaluation_cohort']=='development'
assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
checks=[]
for n in cfg['evaluation_intervals']:
    xyz=C.coords(n);u=np.asarray(C.initial(xyz,np.array([.43,.51,.62,.12,1.1]))).reshape((n-1,)*3)
    for setting in cfg['iterative_cg_controls']:
        args=(n,cfg,setting['dt'],setting['relative_tolerance'],setting['max_iterations'])
        timed=jax.device_get(I.engine(*args)(u));trace=jax.device_get(I.engine(*args,retain_trace=True)(u))
        np.testing.assert_array_equal(timed['prediction'],trace['prediction'])
        np.testing.assert_array_equal(timed['cg_stats'],trace['cg_stats'])
        assert np.all(timed['cg_stats'][:,2]==1)
        checks.append(dict(intervals=n,setting=setting,exact_prediction_parity=True,exact_stats_parity=True,all_steps_converged=True))
        print('PARITY',n,setting,flush=True)
C.dump(root/'smokes/cg-native-parity.json',dict(passed=True,backend='gpu',x64=True,precision='highest',
    checks=checks,smoke_seconds=time.perf_counter()-start,scope='one synthetic initial field per mesh; native/fine exact trace parity only; no scientific performance result or final-data access'))
