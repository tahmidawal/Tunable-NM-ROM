"""Train bank + head + corrections for Burgers 3D.  usage: PYGPU scripts/train3d.py [--R 128 --k 32 --steps_bank 4000]"""
import argparse, sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.dim3 import interior_coords3
from nmrom.bank import init_bank, train_bank, orthonormalise, bank_eval_chunked
from nmrom.head import init_head, train_head, correction_directions, head
ap = argparse.ArgumentParser()
ap.add_argument("--R", type=int, default=128); ap.add_argument("--k", type=int, default=32)
ap.add_argument("--steps_bank", type=int, default=4000); ap.add_argument("--steps_head", type=int, default=30000)
ap.add_argument("--n_feat", type=int, default=32); ap.add_argument("--scale", type=float, default=1.5)
ap.add_argument("--width", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data")
args = ap.parse_args()
d = np.load(f"{args.data}/burgers3d_train.npz"); snaps = d["snaps"].reshape(d["snaps"].shape[0], -1); N = int(d["N"])
X = interior_coords3(N); print(f"burgers3d: {snaps.shape[0]} snapshots on {N}^3 ({snaps.shape[1]} nodes)", flush=True)
t0 = time.time()
bank = init_bank(jax.random.PRNGKey(args.seed), args.R, args.n_feat, args.scale, args.width, dim=3)
bank, hist = train_bank(bank, X, snaps, steps=args.steps_bank, seed=args.seed)
bank = orthonormalise(bank, X)
G = bank_eval_chunked(bank, X, chunk=131072)
print(f"bank trained in {time.time()-t0:.0f}s; orthonormality err {np.abs(G.T@G-np.eye(args.R)).max():.1e}", flush=True)
Gj = jnp.asarray(G); codes = np.asarray(jnp.asarray(snaps) @ Gj)
recon = np.asarray(jnp.asarray(codes) @ Gj.T)
floor = np.linalg.norm(snaps - recon, axis=1) / np.linalg.norm(snaps, axis=1).clip(1e-12)
print(f"projection floor: median {np.median(floor)*100:.3f}%  worst {floor.max()*100:.3f}%", flush=True)
t0 = time.time()
hp = init_head(jax.random.PRNGKey(args.seed + 1), args.k, args.R, width=192)
hp, Z = train_head(hp, codes, args.k, steps=args.steps_head, seed=args.seed)
C, sing = correction_directions(hp, Z, codes)
pred = np.asarray(jax.vmap(lambda z: head(hp, z))(jnp.asarray(Z)))
herr = np.linalg.norm(pred - codes, axis=1) / np.linalg.norm(codes, axis=1).clip(1e-12)
print(f"head trained in {time.time()-t0:.0f}s; head fit error median {np.median(herr)*100:.3f}% worst {herr.max()*100:.3f}%", flush=True)
ckpt = dict(pde="burgers3d", dim=3, R=args.R, k=args.k, N_train=N, bank={k_: np.asarray(v) for k_, v in bank.items()},
            head={k_: np.asarray(v) for k_, v in hp.items()}, Z=Z, codes=codes, C=C, sing=sing, case=d["case"], t=d["t"], floor=floor, herr=herr)
pickle.dump(ckpt, open(f"{args.data}/burgers3d_model.pkl", "wb")); print("saved", flush=True)
