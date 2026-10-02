"""3D end-to-end rollouts. usage: PYGPU scripts/rollout_study3d.py [--meshes 64,128] [--q_acc 64] [--ncases 6]"""
import argparse, sys, os, time, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.dim3 import *
from nmrom.rom import ReducedProblem
ap = argparse.ArgumentParser()
ap.add_argument("--meshes", default="64,128"); ap.add_argument("--q_acc", type=int, default=64); ap.add_argument("--ncases", type=int, default=6)
ap.add_argument("--dense_max_N", type=int, default=64); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
args = ap.parse_args()
pde = BURGERS3D
ckpt = pickle.load(open(f"{args.data}/burgers3d_model.pkl", "rb")); ckpt["train_params"] = pickle.load(open(f"{args.data}/burgers3d_train_params.pkl", "rb"))
evals = pickle.load(open(f"{args.data}/burgers3d_eval.pkl", "rb"))[:args.ncases]
k = ckpt["k"]; rows = []; dense_out = {}; log = open(f"{args.out}/burgers3d_rollout_study.log", "w")
def say(*a): print(*a, flush=True); print(*a, file=log, flush=True)
def rule_list(N):
    rules = [("dense", 0, "dense", None, "point")]
    stride = N // 16
    rules.append(("mesh_lattice", stride, "mesh", lambda s=stride: mesh_lattice3(N, s), "point"))
    for m in [1024, 2048]:
        f = f"{args.out}/burgers3d_eq_N{N}_m{m}.npz"
        if os.path.exists(f): rules.append(("eq_nnls", m, "mesh", lambda f=f: (lambda d: (d["idx"], d["w"]))(np.load(f)), "point"))
    for p_ in [16, 24]: rules.append(("gauss_tensor", p_, "offmesh", lambda p=p_: gauss_tensor3(p), "point"))
    rules.append(("gauss_tensor", 16, "offmesh", lambda: gauss_tensor3(16), "flux"))
    for lv in [7, 8]: rules.append(("smolyak_cc", lv, "offmesh", lambda l=lv: smolyak3(l), "point"))
    for m in [4096, 32768]:
        rules.append(("sobol", m, "offmesh", lambda mm=m: sobol3(mm), "point")); rules.append(("cbc_lattice", m, "offmesh", lambda mm=m: cbc_lattice3(mm), "point"))
    rules.append(("cbc_lattice", 32768, "offmesh", lambda: cbc_lattice3(32768), "flux"))
    rules.append(("korobov", 4093, "offmesh", lambda: korobov_lattice3(4093), "point"))
    rules.append(("korobov", 16381, "offmesh", lambda: korobov_lattice3(16381), "point"))
    return rules
for N in [int(m) for m in args.meshes.split(",")]:
    for q in [0, args.q_acc]:
        M = 4 * (k + q); mm = MeshModel3(ckpt, pde, N, M); say(f"== burgers3d N={N} q={q} M={M} (offline {mm.offline_time:.1f}s)")
        for fam, par, kind, build, form in rule_list(N):
            if fam == "dense": ev = make_dense_evaluator3(mm)
            elif kind == "mesh": idx, w = build(); ev = make_mesh_evaluator3(mm, idx, w, f"{fam}:{par}")
            else: X, w = build(); ev = make_offmesh_evaluator3(mm, X, w, f"{fam}:{par}", form=form)
            rp = ReducedProblem(mm, q, ev)
            ncase = len(evals) if not (fam == "dense" and N > args.dense_max_N) else 2
            errs_sg, errs_ref, fom_ref, walls, its, exits, hr = [], [], [], [], [], [], []
            for ci, case in enumerate(evals[:ncase]):
                roll = rp.rollout(case)
                if ci == 0: roll = rp.rollout(case)
                err = rollout_errors3(mm, roll, case); U = np.stack([mm.reconstruct(c) for c in roll["out_coeffs"]])
                if fam == "dense": dense_out[(N, q, ci)] = U
                if (N, q, ci) in dense_out:
                    hr.append(np.linalg.norm((U - dense_out[(N, q, ci)]).reshape(len(U), -1), axis=1).max() / np.linalg.norm(pde.u0(mm.X, case["p"])))
                errs_sg.append(err["err_same_grid"].max()); errs_ref.append(err["err_refined"].max()); fom_ref.append(err["fom_err_refined"].max())
                walls.append(roll["wall"]); its.append(np.sum(roll["its"])); exits.append(np.bincount(roll["exits"], minlength=4))
            ex = np.sum(exits, 0); nsteps = int(round(pde.T / pde.dt))
            row = dict(pde="burgers3d", N=N, q=q, M=M, rule=fam, param=par, m=ev["m"], form=form, kind=kind,
                       worst_same_grid=float(np.max(errs_sg)), worst_refined=float(np.max(errs_ref)), fom_worst_refined=float(np.max(fom_ref)),
                       hr_err_worst=float(np.max(hr)) if hr else float("nan"), ms_query=float(np.median(walls) * 1e3),
                       attempts_per_step=float(np.sum(its) / (ncase * nsteps)), ms_per_attempt=float(np.sum(walls) * 1e3 / max(np.sum(its), 1)),
                       exits=ex.tolist(), n_cases=ncase, converged_all=bool(ex[0] == 0 and ex[3] == 0))
            rows.append(row)
            say(f"  {fam:13s} {str(par):6s} {form:5s} m={ev['m']:8d}: worst err same-grid {row['worst_same_grid']*100:7.3f}%  refined {row['worst_refined']*100:7.3f}% (FOM {row['fom_worst_refined']*100:.3f}%)  vs dense {row['hr_err_worst']*100:7.3f}%  {row['ms_query']:9.1f} ms/query  {row['attempts_per_step']:.2f} att/step  {row['ms_per_attempt']:.2f} ms/att  exits {ex.tolist()}")
            json.dump(rows, open(f"{args.out}/burgers3d_rollout_study.json", "w"), indent=1)
say("done")
