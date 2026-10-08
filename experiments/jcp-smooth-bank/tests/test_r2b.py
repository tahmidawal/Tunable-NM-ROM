"""R2b (DESIGN A1.6): (i) the lane trainer at lam=0 reproduces the vendored original train_autodecoder_v2 (git 5ae420414)
bit for bit on identical inputs (sub-sampled and full phases); (ii) the value path's point indices actually used by the
trainer are identical for lam=0 and lam=0.1 over 50 sub-sampled steps (recorded from the jitted step itself)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'deps')]
import numpy as np
import jax
import smoothtrain as st
import sep_solvers_reference as ref

n = 128
pick, _ = st.state_pick(4, 51, 16384, 5, 0)
U, worst, fp = st.build_data(n, 4, pick)
X = st.grid_coords(n)[st.interior_indices(n)]
arch = dict(n_ff=128, g_hidden=1024, h_hidden=256, ff_scale=4.0)
kw = dict(lr=1e-3, lam_orth=1e-4, weight_decay=1e-5, p_sub=4096, ema_decay=0.999)
leaves = lambda p, Z: jax.tree_util.tree_leaves((p, Z))
maxdiff = lambda A, B: max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(A, B))

# (i) parity with the original, 8 steps of which the last 3 are full-batch
p0, Z0, i0 = ref.train_autodecoder_v2(jax.random.PRNGKey(0), X, U, 16, 512, steps=8, full_last=3, log_every=1, **kw, **arch)
p2, Z2, _ = ref.train_autodecoder_v2(jax.random.PRNGKey(0), X, U, 16, 512, steps=8, full_last=3, log_every=1, **kw, **arch)
noise = maxdiff(leaves(p0, Z0), leaves(p2, Z2))
p1, Z1, i1 = st.train(jax.random.PRNGKey(0), X, U, st.neighbours(n), 1 / (n - 1), 16, 512, 8, full_last=3, lam_sob=0.0,
                      log_every=1, **kw, **arch)
dev = maxdiff(leaves(p0, Z0), leaves(p1, Z1))
print(f'R2b(i) original vs itself {noise:.3e}; lane(lam=0) vs original {dev:.3e}')
assert noise == 0.0 and dev == 0.0, (noise, dev)

# (ii) value-path indices with lam = 0 and lam = 0.1, 50 sub-sampled steps
idx = {}
for lam in (0.0, 0.1):
    rec = (50, [])
    st.train(jax.random.PRNGKey(0), X, U, st.neighbours(n), 1 / (n - 1), 16, 512, 51, full_last=1, lam_sob=lam,
             log_every=1000, record_idx=rec, **kw, **arch)
    idx[lam] = np.stack(rec[1])
same = bool(np.array_equal(idx[0.0], idx[0.1]))
print('R2b(ii) value-path indices identical over 50 steps:', same, idx[0.0].shape)
assert same
print('R2b PASS')
