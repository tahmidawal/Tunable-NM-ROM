"""End-to-end study: reduced rollouts with every quadrature rule, at every mesh and
correction rank.  Reports error against the same-grid FOM and the refined reference,
LM iterations / exits, and wall time per query and per LM attempt.

usage: python scripts/rollout_study.py burgers [--meshes 128,256,512,1024] [--q_acc 64]
"""
import argparse, sys, os, time, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.pdes import PDES
from nmrom import quadrature as Q
from nmrom.rom import (MeshModel, ReducedProblem, make_dense_evaluator, make_mesh_evaluator,
                       make_offmesh_evaluator, rollout_errors, EXIT)

ap = argparse.ArgumentParser()
ap.add_argument("pde"); ap.add_argument("--meshes", default="128,256,512,1024")
ap.add_argument("--q_acc", type=int, default=64); ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results"); ap.add_argument("--dense_max_N", type=int, default=512)
ap.add_argument("--ncases", type=int, default=6)
ap.add_argument("--model_suffix", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--skip_eq", action="store_true")
ap.add_argument("--rules", default="all", help="comma list of rule families to run, or all")
args = ap.parse_args()
pde = PDES[args.pde]
meshes = [int(m) for m in args.meshes.split(",")]
ckpt = pickle.load(open(f"{args.data}/{args.pde}_model{args.model_suffix}.pkl", "rb"))
ckpt["train_params"] = pickle.load(open(f"{args.data}/{args.pde}_train_params.pkl", "rb"))
evals = pickle.load(open(f"{args.data}/{args.pde}_eval.pkl", "rb"))[:args.ncases]
k = ckpt["k"]
os.makedirs(args.out, exist_ok=True)
rows = []
dense_out = {}
log = open(f"{args.out}/{args.pde}_rollout_study{args.tag}.log", "w")
def say(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)

def rule_list(N):
    """(family, param, kind, builder) for this mesh."""
    rules = [("dense", 0, "dense", None)]
    for stride in [N // 32, N // 64]:
        if stride >= 1:
            rules.append(("mesh_lattice", stride, "mesh", lambda s=stride: Q.mesh_lattice(N, s)))
    for m in ([] if args.skip_eq else [1024, 2048]):
        f = f"{args.out}/{args.pde}_eq_N{N}_m{m}.npz"
        if os.path.exists(f):
            rules.append(("eq_nnls", m, "mesh", lambda f=f: (lambda d: (d["idx"], d["w"]))(np.load(f))))
    for p_ in [16, 32]:
        rules.append(("gauss_tensor", p_, "offmesh", lambda p=p_: Q.gauss_tensor(p)))
    for lv in [6, 8]:
        rules.append(("smolyak_cc", lv, "offmesh", lambda l=lv: Q.smolyak(l, "cc")))
    for m in [1024, 4096]:
        rules.append(("sobol", m, "offmesh", lambda mm=m: Q.sobol(mm, True, 0)))
        rules.append(("halton", m, "offmesh", lambda mm=m: Q.halton(mm, True, 0)))
    for kk in [16, 19]:
        rules.append(("fibonacci", kk, "offmesh", lambda K=kk: Q.fibonacci_lattice(K, seed=0)))
        rules.append(("fibonacci_tent", kk, "offmesh", lambda K=kk: Q.fibonacci_lattice(K, tent=True, seed=0)))
    if args.rules != "all":
        keep = set(args.rules.split(","))
        rules = [r_ for r_ in rules if r_[0] in keep]
    return rules

for N in meshes:
    for q in [0, args.q_acc]:
        M = 4 * (k + q)
        mm = MeshModel(ckpt, pde, N, M)
        say(f"== {args.pde} N={N} q={q} M={M} (offline {mm.offline_time:.1f}s)")
        for fam, par, kind, build in rule_list(N):
            forms = ["point"]
            if kind == "offmesh" and pde.nonlinear_flux is not None and fam in ("gauss_tensor", "sobol", "fibonacci_tent"):
                forms.append("flux")
            for form in forms:
                if fam == "dense":
                    if N > args.dense_max_N and q > 0:
                        continue
                    ev = make_dense_evaluator(mm)
                elif kind == "mesh":
                    idx, w = build(); ev = make_mesh_evaluator(mm, idx, w, f"{fam}:{par}")
                else:
                    X, w = build(); ev = make_offmesh_evaluator(mm, X, w, f"{fam}:{par}", form=form)
                rp = ReducedProblem(mm, q, ev)
                errs_sg, errs_ref, fom_ref, walls, its, accs, exits, med_sg, hr_err = [], [], [], [], [], [], [], [], []
                ncase = len(evals) if not (fam == "dense" and N > args.dense_max_N) else 2
                for ci, case in enumerate(evals[:ncase]):
                    roll = rp.rollout(case)
                    if ci == 0:
                        roll = rp.rollout(case)          # second run: compiled timing
                    err = rollout_errors(mm, roll, case)
                    U = np.stack([mm.reconstruct(c) for c in roll["out_coeffs"]])
                    if fam == "dense":
                        dense_out[(N, q, ci)] = U
                    if (N, q, ci) in dense_out:
                        n0 = np.linalg.norm(pde.u0(mm.X, case["p"])) if not pde.steady else np.linalg.norm(dense_out[(N, q, ci)][0])
                        hr_err.append(np.linalg.norm((U - dense_out[(N, q, ci)]).reshape(len(U), -1), axis=1).max() / n0)
                    errs_sg.append(err["err_same_grid"].max()); errs_ref.append(err["err_refined"].max())
                    med_sg.append(np.median(err["err_same_grid"]))
                    fom_ref.append(err["fom_err_refined"].max()); walls.append(roll["wall"])
                    its.append(np.sum(roll["its"])); accs.append(np.sum(roll["acc"]))
                    exits.append(np.bincount(roll["exits"], minlength=4))
                ex = np.sum(exits, 0)
                row = dict(pde=args.pde, N=N, q=q, M=M, rule=fam, param=par, m=ev["m"], form=form, kind=kind,
                           worst_same_grid=float(np.max(errs_sg)), median_same_grid=float(np.median(med_sg)),
                           worst_refined=float(np.max(errs_ref)), fom_worst_refined=float(np.max(fom_ref)),
                           ms_query=float(np.median(walls) * 1e3), attempts_per_step=float(np.sum(its) / (ncase * max(len(pde.out_times) and int(round(pde.T / pde.dt)), 1))),
                           ms_per_attempt=float(np.sum(walls) * 1e3 / max(np.sum(its), 1)),
                           exits=ex.tolist(), n_cases=ncase, converged_all=bool(ex[0] == 0 and ex[3] == 0),
                           hr_err_worst=float(np.max(hr_err)) if hr_err else float("nan"))
                rows.append(row)
                say(f"  {fam:15s} {str(par):5s} {form:5s} m={ev['m']:7d}: worst err same-grid {row['worst_same_grid']*100:7.3f}%  refined {row['worst_refined']*100:7.3f}% "
                    f"(FOM {row['fom_worst_refined']*100:.3f}%)  vs dense {row['hr_err_worst']*100:7.3f}%  {row['ms_query']:8.1f} ms/query  {row['attempts_per_step']:.2f} att/step  {row['ms_per_attempt']:.2f} ms/att  exits {ex.tolist()}")
                json.dump(rows, open(f"{args.out}/{args.pde}_rollout_study{args.tag}.json", "w"), indent=1)
say("done")
