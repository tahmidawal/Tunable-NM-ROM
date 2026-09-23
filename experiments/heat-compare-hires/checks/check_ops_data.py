"""Gate: the operators' separable exact flow equals hires-heat core's 2D-DST flow (same grid, same draws)."""
import sys, json
from pathlib import Path
here = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(here / 'ops'), str(here.parent / 'hires-heat')]
import numpy as np, jax.numpy as jnp, torch
import heatops as H, core as C
out = {}
for n in (64, 256):
    draws = H.family(791099, 4)
    a = H.interior_fields(n, draws).cpu().numpy()
    prop = C.make_propagate(2); lam = C.eig_grid(n, 2)
    b = np.stack([np.asarray(prop(C.initial_grid(n, 2, d), lam, jnp.asarray(H.TIMES), H.NU)) for d in draws])
    out[n] = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
print(json.dumps(dict(max_rel_diff=out, passed=all(v < 1e-12 for v in out.values()))))
