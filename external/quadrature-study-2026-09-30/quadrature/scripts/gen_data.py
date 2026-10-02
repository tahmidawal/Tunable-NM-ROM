"""Generate training trajectories and evaluation references for one PDE.

usage: python scripts/gen_data.py burgers [--ntrain 96] [--neval 8] [--N_train 256]
       [--meshes 128,256,512,1024] [--N_ref 1024] [--dt_ref 0.00125]
"""
import argparse, sys, os, time, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom
from nmrom.pdes import PDES
from nmrom.fom import solve_case

ap = argparse.ArgumentParser()
ap.add_argument("pde")
ap.add_argument("--ntrain", type=int, default=96)
ap.add_argument("--neval", type=int, default=8)
ap.add_argument("--N_train", type=int, default=256)
ap.add_argument("--meshes", default="128,256,512,1024")
ap.add_argument("--N_ref", type=int, default=1024)
ap.add_argument("--dt_ref", type=float, default=0.00125)
ap.add_argument("--store_every", type=int, default=2)
ap.add_argument("--out", default="data")
args = ap.parse_args()
pde = PDES[args.pde]
meshes = [int(m) for m in args.meshes.split(",")]
os.makedirs(args.out, exist_ok=True)

# ---- training trajectories
rng = np.random.default_rng(1000)
t0 = time.time()
snaps, snap_t, snap_case, params = [], [], [], []
u0_list = []
for i in range(args.ntrain):
    p = pde.sample_params(rng)
    out = solve_case(pde, args.N_train, p, store_every=args.store_every)
    snaps.append(out["states"].astype(np.float32)); snap_t.append(out["t_states"])
    snap_case.append(np.full(len(out["t_states"]), i)); params.append(p)
    if (i + 1) % 8 == 0:
        print(f"train {i+1}/{args.ntrain}  {time.time()-t0:.0f}s", flush=True)
snaps = np.concatenate(snaps); snap_t = np.concatenate(snap_t); snap_case = np.concatenate(snap_case)
np.savez_compressed(f"{args.out}/{args.pde}_train.npz", snaps=snaps, t=snap_t, case=snap_case, N=args.N_train)
with open(f"{args.out}/{args.pde}_train_params.pkl", "wb") as fh:
    pickle.dump(params, fh)
print(f"training snapshots {snaps.shape} in {time.time()-t0:.0f}s", flush=True)

# ---- evaluation cases: same-grid references at every mesh + refined reference
rng = np.random.default_rng(2000)
evals = []
for i in range(args.neval):
    p = pde.sample_params(rng)
    rec = dict(p=p, same_grid={}, )
    for N in meshes:
        out = solve_case(pde, N, p)
        rec["same_grid"][N] = dict(u_out=out["u_out"], t_out=out["t_out"], time=out["time"], its=out["newton_its"])
        print(f"eval {i} N={N} {out['time']:.1f}s", flush=True)
    if not pde.steady:
        out = solve_case(pde, args.N_ref, p, dt=args.dt_ref)
        rec["refined"] = dict(u_out=out["u_out"], t_out=out["t_out"], N=args.N_ref, dt=args.dt_ref, time=out["time"])
        print(f"eval {i} refined N={args.N_ref} dt={args.dt_ref} {out['time']:.1f}s", flush=True)
    else:
        rec["refined"] = dict(u_out=rec["same_grid"][max(meshes)]["u_out"], t_out=np.array([0.0]), N=max(meshes), dt=0.0, time=0.0)
    evals.append(rec)
with open(f"{args.out}/{args.pde}_eval.pkl", "wb") as fh:
    pickle.dump(evals, fh)
print("done", flush=True)
