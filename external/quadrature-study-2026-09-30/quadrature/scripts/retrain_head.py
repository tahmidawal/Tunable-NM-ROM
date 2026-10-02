"""Retrain only the head (and corrections) of an existing model with a new latent size.
usage: python scripts/retrain_head.py burgers --k 32 --steps 30000 --width 192 --suffix _k32"""
import argparse, sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.head import init_head, train_head, correction_directions, head
ap = argparse.ArgumentParser()
ap.add_argument("pde"); ap.add_argument("--k", type=int, default=32); ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--width", type=int, default=192); ap.add_argument("--suffix", default="_k32"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data")
args = ap.parse_args()
ckpt = pickle.load(open(f"{args.data}/{args.pde}_model.pkl", "rb"))
codes = ckpt["codes"]
t0 = time.time()
hp = init_head(jax.random.PRNGKey(args.seed + 7), args.k, ckpt["R"], width=args.width)
hp, Z = train_head(hp, codes, args.k, steps=args.steps, seed=args.seed)
C, sing = correction_directions(hp, Z, codes)
pred = np.asarray(jax.vmap(lambda z: head(hp, z))(jnp.asarray(Z)))
herr = np.linalg.norm(pred - codes, axis=1) / np.linalg.norm(codes, axis=1).clip(1e-12)
print(f"head k={args.k} trained in {time.time()-t0:.0f}s; fit error median {np.median(herr)*100:.3f}% worst {herr.max()*100:.3f}%", flush=True)
ckpt.update(k=args.k, head={kk: np.asarray(v) for kk, v in hp.items()}, Z=Z, C=C, sing=sing, herr=herr)
pickle.dump(ckpt, open(f"{args.data}/{args.pde}_model{args.suffix}.pkl", "wb"))
print("saved", flush=True)
