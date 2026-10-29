"""CPU-only algebra smoke check; not ROM timing or a GPU performance result."""
import json
import time
from pathlib import Path
import numpy as np

rng = np.random.default_rng(29102026)
dh = rng.normal(size=(4096, 128, 2))
w = rng.normal(size=(128, 128))
methods = {
    'original_einsum': lambda: np.einsum('nhd,hk->nkd', dh, w),
    'optimized_einsum': lambda: np.einsum('nhd,hk->nkd', dh, w, optimize=True),
    'batched_matmul': lambda: (dh.transpose(0, 2, 1) @ w).transpose(0, 2, 1),
}
ref = methods['original_einsum']()
rows = {}
for name, fn in methods.items():
    result = fn()
    np.testing.assert_allclose(result, ref, rtol=2e-12, atol=2e-12)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    rows[name] = {'seconds': times, 'median_seconds': float(np.median(times)),
                  'max_absolute_gap': float(np.max(np.abs(result-ref)))}
out = {'scope': __doc__, 'seed': 29102026, 'dtype': str(dh.dtype),
       'shape': list(dh.shape), 'numpy_version': np.__version__, 'methods': rows,
       'interpretation': 'Equivalent contraction smoke only. Actual model and full-rule equivalence remain required before changing production evaluation.'}
Path(__file__).with_name('RESULT.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
