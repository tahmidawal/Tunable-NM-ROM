"""Coordinate bank for Family A3, trained as in the paper's bank phase (pbh_fit.train_spatial_phase, phase 'bank'):
G(x) = bc(x) * MLP(random Fourier features(x)), free per-snapshot coefficients, relative MSE + 1e-4 Gram-identity
penalty, Adam with warmup-cosine schedule, 64 snapshots x 4096 points per step. 3D changes: bc = 64 x(1-x)y(1-y)z(1-z),
Fourier matrix 3 x n_ff.

Usage: bank3d.py <p> <N> <ntrain> <R> <steps> <out.npz> [wall_cap_s]
"""
import sys, os, time, json
import numpy as np, jax, jax.numpy as jnp, optax
jax.config.update("jax_enable_x64", True)
import rom3d
HERE = os.path.dirname(os.path.abspath(__file__)); OD = f"{HERE}/runs/rom3"
p, N, nt, R, steps, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
cap = float(sys.argv[7]) if len(sys.argv) > 7 else 1800.0
NFF, FFS, HID, LR, OW, SB, PB, SEED = 64, 4.0, 512, 1e-3, 1e-4, 64, 4096, 20261111
X = np.load(f"{OD}/snap_p{p}N{N}_{nt}.npy"); x1 = rom3d.node_coords(N, p); n = x1.size
coords = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
key = jax.random.PRNGKey(SEED); kb, k1, k2, k3, kc = jax.random.split(key, 5)
def lin(k, a, b): return (jax.random.normal(k, (a, b), dtype=jnp.float64) * jnp.sqrt(2.0 / a), jnp.zeros((b,), jnp.float64))
params = dict(B=jax.random.normal(kb, (3, NFF), dtype=jnp.float64) * FFS, g=[lin(k1, 2 * NFF, HID), lin(k2, HID, HID), lin(k3, HID, R)])
scale = float(np.sqrt(np.mean(X.astype(np.float64) ** 2)))
def features(pr, xyz):
    ang = 2 * jnp.pi * (xyz @ pr["B"]); h = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1)
    for W, b in pr["g"][:-1]: h = jax.nn.silu(h @ W + b)
    W, b = pr["g"][-1]
    bc = 64.0 * jnp.prod(xyz * (1 - xyz), axis=-1)
    return bc[:, None] * (h @ W + b)
U = jnp.asarray(X); C = jnp.asarray(coords); denom = jnp.mean(U.astype(jnp.float64) ** 2, axis=1)
lat = jax.random.normal(kc, (nt, R), dtype=jnp.float64) * (scale / np.sqrt(R))
sched = optax.warmup_cosine_decay_schedule(0., LR, max(1, int(0.02 * steps)), steps, LR * 0.01)
opt = optax.adam(sched); v = (params, lat); state = opt.init(v)
def loss(v, si, pi):
    pr, lt = v; G = features(pr, C[pi])
    err = lt[si] @ G.T - U[si[:, None], pi[None, :]]
    rec = jnp.mean(jnp.mean(err * err, axis=1) / denom[si])
    gram = G.T @ G / G.shape[0]
    return rec + OW * jnp.mean((gram - jnp.eye(R)) ** 2), rec
@jax.jit
def step(v, state, key):
    key, ks, kp = jax.random.split(key, 3)
    si = jax.random.randint(ks, (SB,), 0, nt); pi = jax.random.randint(kp, (PB,), 0, C.shape[0])
    (val, rec), g = jax.value_and_grad(loss, has_aux=True)(v, si, pi)
    upd, state = opt.update(g, state, v)
    return optax.apply_updates(v, upd), state, key, val, rec
key = jax.random.PRNGKey(SEED + 1); t0 = time.time(); log = []
for it in range(steps):
    v, state, key, val, rec = step(v, state, key)
    if it % 1000 == 0 or it == steps - 1:
        log.append(dict(step=it, loss=float(val), rec=float(rec), seconds=time.time() - t0)); print(log[-1], flush=True)
        if time.time() - t0 > cap: print("wall cap reached"); break
train_seconds = time.time() - t0
feat = jax.jit(features); G = np.concatenate([np.asarray(feat(v[0], C[i:i + 32768])) for i in range(0, C.shape[0], 32768)])
np.savez(out, G=G, train_seconds=train_seconds, steps=it + 1, log=json.dumps(log), backend=jax.default_backend(),
         config=json.dumps(dict(R=R, n_ff=NFF, ff_scale=FFS, hidden=HID, lr=LR, orth=OW, source_batch=SB, point_batch=PB, seed=SEED, ntrain=nt)))
print("done", train_seconds)
