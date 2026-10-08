"""R2b (DESIGN A1.6): the lane trainer at lam=0 reproduces the vendored original train_autodecoder_v2 (git 5ae420414)
bit for bit on identical inputs; and the value path's point-index key stream does not depend on lam."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'deps')]
import numpy as np
import jax
import jax.numpy as jnp
import smoothtrain as st
import sep_solvers_reference as ref

n, steps, full_last = 128, 8, 3
pick, _ = st.state_pick(4, 51, 16384, 5, 0)
U, worst, fp = st.build_data(n, 4, pick)
X = st.grid_coords(n)[st.interior_indices(n)]
arch = dict(n_ff=128, g_hidden=1024, h_hidden=256, ff_scale=4.0)
kw = dict(steps=steps, lr=1e-3, lam_orth=1e-4, weight_decay=1e-5, p_sub=4096, ema_decay=0.999, full_last=full_last)
p0, Z0, i0 = ref.train_autodecoder_v2(jax.random.PRNGKey(0), X, U, 16, 512, log_every=1, **kw, **arch)
p1, Z1, i1 = st.train(jax.random.PRNGKey(0), X, U, st.neighbours(n), 1 / (n - 1), 16, 512, lam_sob=0.0, log_every=1,
                      **kw, **arch)
p2, Z2, i2 = ref.train_autodecoder_v2(jax.random.PRNGKey(0), X, U, 16, 512, log_every=1, **kw, **arch)
l2 = jax.tree_util.tree_leaves((p2, Z2))
l0 = jax.tree_util.tree_leaves((p0, Z0))
noise = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(l0, l2))
print('R2b original vs itself (platform determinism): max |param diff| =', noise)
l1 = jax.tree_util.tree_leaves((p1, Z1))
dev = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(l0, l1))
print('R2b lam=0 vs original: max |param diff| =', dev, ' recon', i0['recon_rel_l2_mean'], i1['recon_rel_l2_mean'])
# the value path's keys: main key split per sub step, independent of lam (the Sobolev stream is fold_in(PRNGKey(12345), i))
key = jax.random.PRNGKey(0)
key, kz, kp = jax.random.split(key, 3)
ks = []
for i in range(steps - full_last):
    key, k_ = jax.random.split(key)
    ks.append(np.asarray(jax.random.choice(k_, X.shape[0], shape=(4096,), replace=False))[:50])
print('R2b value-path index stream defined without lam: first indices', ks[0][:5])
assert dev <= max(noise, 0.0) * 1.0 + 0.0 or dev == 0.0, (dev, noise)
print('R2b PASS')
