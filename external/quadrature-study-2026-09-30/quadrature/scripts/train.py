"""Train bank + head + corrections for one PDE.  usage: python scripts/train.py burgers"""
import argparse, sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.grid import interior_coords
from nmrom.bank import init_bank, train_bank, orthonormalise, bank_eval_chunked
from nmrom.head import init_head, train_head, correction_directions, head

ap = argparse.ArgumentParser()
ap.add_argument("pde"); ap.add_argument("--R", type=int, default=128); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--steps_bank", type=int, default=6000); ap.add_argument("--steps_head", type=int, default=20000)
ap.add_argument("--n_feat", type=int, default=32); ap.add_argument("--scale", type=float, default=1.5)
ap.add_argument("--width", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data"); ap.add_argument("--out", default="data")
ap.add_argument("--bank", default="rff", choices=["rff", "siren", "rbf", "spectral", "pod_cubic", "pod_linear"])
ap.add_argument("--suffix", default="")
args = ap.parse_args()

d = np.load(f"{args.data}/{args.pde}_train.npz")
snaps, N = d["snaps"], int(d["N"])
snaps = snaps.reshape(snaps.shape[0], -1)
X = interior_coords(N)
print(f"{args.pde}: {snaps.shape[0]} snapshots on {N}^2", flush=True)

key = jax.random.PRNGKey(args.seed)
t0 = time.time()
from nmrom.bank import init_siren, init_rbf, init_spectral, init_pod_interp
if args.bank == "rff":
    bank = init_bank(key, args.R, args.n_feat, args.scale, args.width)
elif args.bank == "siren":
    bank = init_siren(key, args.R, width=args.width)
elif args.bank == "rbf":
    bank = init_rbf(key, args.R)
elif args.bank == "spectral":
    bank = init_spectral(args.R)
elif args.bank == "pod_cubic":
    bank = init_pod_interp(snaps, N, args.R, cubic=True)
elif args.bank == "pod_linear":
    bank = init_pod_interp(snaps, N, args.R, cubic=False)
if args.bank in ("rff", "siren", "rbf"):
    bank, hist = train_bank(bank, X, snaps, steps=args.steps_bank, seed=args.seed, lr=(1e-3 if args.bank != "siren" else 2e-4))
bank = orthonormalise(bank, X)
G = bank_eval_chunked(bank, X)                       # (n, R), orthonormal on the training mesh
print(f"bank trained in {time.time()-t0:.0f}s; orthonormality err {np.abs(G.T@G-np.eye(args.R)).max():.1e}", flush=True)
codes = (snaps.astype(np.float64) @ G)               # (S, R)
recon = codes @ G.T
floor = np.linalg.norm(snaps - recon, axis=1) / np.linalg.norm(snaps, axis=1).clip(1e-12)
print(f"projection floor: median {np.median(floor)*100:.3f}%  worst {floor.max()*100:.3f}%", flush=True)

t0 = time.time()
hp = init_head(jax.random.PRNGKey(args.seed + 1), args.k, args.R)
hp, Z = train_head(hp, codes, args.k, steps=args.steps_head, seed=args.seed)
C, sing = correction_directions(hp, Z, codes)
pred = np.asarray(jax.vmap(lambda z: head(hp, z))(jnp.asarray(Z)))
herr = np.linalg.norm(pred - codes, axis=1) / np.linalg.norm(codes, axis=1).clip(1e-12)
print(f"head trained in {time.time()-t0:.0f}s; head fit error median {np.median(herr)*100:.3f}% worst {herr.max()*100:.3f}%", flush=True)
print("correction singular values (first 8, every 16th):", np.round(sing[:8], 3), np.round(sing[::16], 3), flush=True)

ckpt = dict(pde=args.pde, R=args.R, k=args.k, N_train=N, arch=args.bank, bank={k: np.asarray(v) for k, v in bank.items()},
            head={k: np.asarray(v) for k, v in hp.items()}, Z=Z, codes=codes, C=C, sing=sing,
            case=d["case"], t=d["t"], floor=floor, herr=herr)
with open(f"{args.out}/{args.pde}_model{args.suffix}.pkl", "wb") as fh:
    pickle.dump(ckpt, fh)
print("saved", flush=True)
