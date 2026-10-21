"""Construction diagnostic (DESIGN A5): convergence of the bank Mass / Stiff integrals under Gauss tensor rules.
python diag_mass.py <Rp> <out.json>  (no cohort data; bank only)."""
import json, sys, time
import numpy as np
import jax.numpy as jnp
import hrtab as HT
from hrtab import Q

Rp = int(sys.argv[1])
mdl = Q.Model(ckpt=HT.T2.CKPT)

class _B:
    pass
b = _B(); b.mdl = mdl
res = {}
prev = None
for p in (256, 384, 512, 640, 768, 1024):
    t = time.perf_counter()
    M, K = HT.mass_stiff(b, Rp, f'gauss{p}')
    M, K = np.asarray(M), np.asarray(K)
    row = dict(seconds=time.perf_counter() - t, normM=float(np.linalg.norm(M)), normK=float(np.linalg.norm(K)))
    if prev is not None:
        row['dM_prev'] = float(np.linalg.norm(M - prev[0]) / np.linalg.norm(M))
        row['dK_prev'] = float(np.linalg.norm(K - prev[1]) / np.linalg.norm(K))
        # per-column relative change of K (which columns are unresolved)
        dc = np.linalg.norm(K - prev[1], axis=0) / np.linalg.norm(K, axis=0)
        row['dK_col_q'] = [float(np.quantile(dc, q)) for q in (.5, .9, 1.)]
        row['dK_col_argmax'] = int(np.argmax(dc))
    res[p] = row
    prev = (M, K)
    print(p, row, flush=True)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
