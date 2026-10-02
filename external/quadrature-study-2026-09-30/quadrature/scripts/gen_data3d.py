"""3D Burgers data: training trajectories at 64^3, evaluation references at 64^3 / 128^3 + refined.
usage: PYGPU scripts/gen_data3d.py [--ntrain 64] [--neval 6] [--N_train 64] [--meshes 64,128] [--N_ref 128] [--dt_ref 0.0025]"""
import argparse, sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom
from nmrom.dim3 import BURGERS3D, solve_case3
ap = argparse.ArgumentParser()
ap.add_argument("--ntrain", type=int, default=64); ap.add_argument("--neval", type=int, default=6)
ap.add_argument("--N_train", type=int, default=64); ap.add_argument("--meshes", default="64,128")
ap.add_argument("--N_ref", type=int, default=128); ap.add_argument("--dt_ref", type=float, default=0.0025)
ap.add_argument("--store_every", type=int, default=2); ap.add_argument("--out", default="data")
args = ap.parse_args()
pde = BURGERS3D; meshes = [int(m) for m in args.meshes.split(",")]
rng = np.random.default_rng(1000); t0 = time.time()
snaps, snap_t, snap_case, params = [], [], [], []
for i in range(args.ntrain):
    p = pde.sample_params(rng)
    out = solve_case3(pde, args.N_train, p, store_every=args.store_every)
    snaps.append(out["states"].astype(np.float32)); snap_t.append(out["t_states"]); snap_case.append(np.full(len(out["t_states"]), i)); params.append(p)
    if (i + 1) % 8 == 0: print(f"train {i+1}/{args.ntrain} {time.time()-t0:.0f}s", flush=True)
snaps = np.concatenate(snaps); snap_t = np.concatenate(snap_t); snap_case = np.concatenate(snap_case)
np.savez(f"{args.out}/burgers3d_train.npz", snaps=snaps, t=snap_t, case=snap_case, N=args.N_train)
with open(f"{args.out}/burgers3d_train_params.pkl", "wb") as fh: pickle.dump(params, fh)
print(f"training snapshots {snaps.shape} in {time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(2000); evals = []
for i in range(args.neval):
    p = pde.sample_params(rng); rec = dict(p=p, same_grid={})
    for N in meshes:
        out = solve_case3(pde, N, p); rec["same_grid"][N] = dict(u_out=out["u_out"], t_out=out["t_out"], time=out["time"], its=out["newton_its"])
        print(f"eval {i} N={N} {out['time']:.1f}s", flush=True)
    out = solve_case3(pde, args.N_ref, p, dt=args.dt_ref)
    rec["refined"] = dict(u_out=out["u_out"], t_out=out["t_out"], N=args.N_ref, dt=args.dt_ref, time=out["time"])
    print(f"eval {i} refined {out['time']:.1f}s", flush=True); evals.append(rec)
with open(f"{args.out}/burgers3d_eval.pkl", "wb") as fh: pickle.dump(evals, fh)
print("done", flush=True)
