"""Clean timing study (run when the machine is otherwise idle): ms per residual+Jacobian
evaluation and per full LM time step for dense / mesh / off-mesh evaluation at every mesh,
as a function of the mesh N and of the quadrature count m.
usage: python scripts/timing_study.py burgers [--meshes 128,256,512,1024] [--model_suffix _k32]"""
import argparse, sys, os, time, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nmrom, jax, jax.numpy as jnp
from nmrom.pdes import PDES
from nmrom import quadrature as Q
from nmrom.rom import (MeshModel, ReducedProblem, make_dense_evaluator, make_mesh_evaluator,
                       make_offmesh_evaluator, scalar_params)
ap = argparse.ArgumentParser()
ap.add_argument("pde"); ap.add_argument("--meshes", default="128,256,512,1024"); ap.add_argument("--q_acc", type=int, default=64)
ap.add_argument("--model_suffix", default=""); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--n_rep", type=int, default=30)
args = ap.parse_args()
pde = PDES[args.pde]
ckpt = pickle.load(open(f"{args.data}/{args.pde}_model{args.model_suffix}.pkl", "rb"))
ckpt["train_params"] = pickle.load(open(f"{args.data}/{args.pde}_train_params.pkl", "rb"))
evals = pickle.load(open(f"{args.data}/{args.pde}_eval.pkl", "rb"))
k = ckpt["k"]; p0 = evals[0]["p"]; ps = scalar_params(p0)
nu, r = float(pde.nu(p0)), float(pde.react(p0))
rows = []

def bench(f, n_rep):
    out = f(); jax.block_until_ready(out)
    ts = []
    for _ in range(n_rep):
        t = time.perf_counter(); out = f(); jax.block_until_ready(out); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)

for N in [int(m) for m in args.meshes.split(",")]:
    for q in [0, args.q_acc]:
        M = 4 * (k + q)
        mm = MeshModel(ckpt, pde, N, M)
        b = jnp.asarray(mm.tested_field(pde.source(mm.X, p0)))
        rules = [("dense", 0, make_dense_evaluator(mm))]
        for m_ in [1024, 2048]:
            f = f"{args.out}/{args.pde}_eq_N{N}_m{m_}.npz"
            if os.path.exists(f):
                d = np.load(f); rules.append(("eq_nnls", m_, make_mesh_evaluator(mm, d["idx"], d["w"], "eq")))
        for stride in [N // 32, N // 64]:
            if stride >= 1:
                rules.append(("mesh_lattice", stride, make_mesh_evaluator(mm, *Q.mesh_lattice(N, stride), "lat")))
        for p_ in [16, 32, 48]:
            rules.append(("gauss_tensor", p_, make_offmesh_evaluator(mm, *Q.gauss_tensor(p_), "gl")))
        for lv in [7, 8, 9]:
            rules.append(("smolyak_cc", lv, make_offmesh_evaluator(mm, *Q.smolyak(lv, "cc"), "sm")))
        for m_ in [1024, 4096]:
            rules.append(("sobol", m_, make_offmesh_evaluator(mm, *Q.sobol(m_), "sob")))
        for kk in [16, 19]:
            rules.append(("fibonacci", kk, make_offmesh_evaluator(mm, *Q.fibonacci_lattice(kk, seed=0), "fib")))
        for fam, par, ev in rules:
            rp = ReducedProblem(mm, q, ev)
            zy = jnp.concatenate([jnp.asarray(ckpt["Z"][0]), jnp.zeros(q)])
            c_n = rp.coeffs(zy)
            ms_res = bench(lambda: rp.residual_jit(zy, c_n, b, nu, r, pde.dt, ps), args.n_rep)
            ms_jac = bench(lambda: rp.jac_jit(zy, c_n, b, nu, r, pde.dt, ps), args.n_rep)
            ms_step = bench(lambda: rp.lm_step(zy, jnp.float64(1e-6), c_n, b, nu, r, pde.dt, ps), max(5, args.n_rep // 3))
            its = int(rp.lm_step(zy, jnp.float64(1e-6), c_n, b, nu, r, pde.dt, ps)[2])
            rows.append(dict(pde=args.pde, N=N, q=q, M=M, rule=fam, param=par, m=ev["m"], ms_residual=ms_res, ms_jacobian=ms_jac,
                             ms_lm_step=ms_step, lm_attempts=its))
            print(f"N={N:5d} q={q:3d} {fam:13s} {str(par):5s} m={ev['m']:8d}  residual {ms_res:8.3f} ms  jacobian {ms_jac:8.3f} ms  LM step {ms_step:8.2f} ms ({its} attempts)", flush=True)
        json.dump(rows, open(f"{args.out}/{args.pde}_timing.json", "w"), indent=1)
print("done")
