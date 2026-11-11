"""3D pilot, assembled-matrix CPU comparators: pyamg AMG-PCG, Jacobi-CG, SuperLU (small only).

Usage: time3d_cpu.py <cohort> <out.jsonl> <configs "p:N,..."> [cases]
"""
import json, sys, os, time
import numpy as np, scipy.sparse.linalg as spla, pyamg
import aniso3d as a3
from time_solvers import med, splu, cg
HERE = os.path.dirname(os.path.abspath(__file__)); RD = f"{HERE}/runs/ref3"
cohort, out = sys.argv[1], sys.argv[2]
cfgs = [tuple(map(int, s.split(":"))) for s in sys.argv[3].split(",")]
cases = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(16))
P = a3.sample_params(cohort, 16); ev = a3.Evaluator3(128)
AMG = {"sa_sym": lambda A: pyamg.smoothed_aggregation_solver(A, strength="symmetric", max_coarse=200),
       "sa_evo": lambda A: pyamg.smoothed_aggregation_solver(A, strength=("evolution", {"epsilon": 4.0, "k": 2}), max_coarse=200),
       "rs": lambda A: pyamg.ruge_stuben_solver(A, strength=("classical", {"theta": 0.25}), max_coarse=200),
       "rs_th05": lambda A: pyamg.ruge_stuben_solver(A, strength=("classical", {"theta": 0.5}), max_coarse=200)}
fo = open(out, "a")
for (p, N) in cfgs:
    S = a3.Space(N, p); S.terms
    for i in cases:
        Uref = np.load(f"{RD}/A3_{cohort}_{i:02d}.npy").astype(np.float64)
        t = time.perf_counter(); A = S.matrix(P[i]); b = S.load(P[i]); tasm = time.perf_counter() - t
        base = dict(family="A3", cohort=cohort, case=i, backend="cpu1", p=p, N=N, dofs=S.n ** 3, nnz=int(A.nnz), assemble_ms=1e3 * tasm)
        def add(u, **kw):
            r = {**base, **kw, **ev.errors(ev.field(S, u), Uref)}; fo.write(json.dumps(r) + "\n"); fo.flush()
            print(p, N, i, kw["solver"], kw.get("tol"), kw.get("iters"), "%.1f ms" % kw["total_ms"], "l2w %.4f" % r["l2w"], flush=True)
        if S.n ** 3 <= 40000:
            Ac = A.tocsc(); lu, tf = med(lambda: splu(Ac), 1); u, ts = med(lambda: lu.solve(b), 3)
            add(u, solver="direct_superlu", factor=tf, solve=ts, total_ms=tf["med_ms"] + ts["med_ms"])
        for name, f in AMG.items():
            try: ml, tset = med(lambda: f(A), 1)
            except Exception as e: print("amg fail", name, repr(e)); continue
            M = ml.aspreconditioner(cycle="V")
            for tol in (1e-2, 1e-3, 1e-4):
                (u, it, info), ts = med(lambda: cg(A, b, tol, M, maxiter=300), 3)
                add(u, solver="amg_" + name, tol=tol, iters=it, info=info, setup=tset, solve=ts, total_ms=tset["med_ms"] + ts["med_ms"], opc=float(ml.operator_complexity()))
        dinv = 1.0 / A.diagonal(); M = spla.LinearOperator(A.shape, matvec=lambda v: dinv * v)
        for tol in (1e-2, 1e-3, 1e-4):
            (u, it, info), ts = med(lambda: cg(A, b, tol, M), 3)
            add(u, solver="jacobi_cg", tol=tol, iters=it, info=info, solve=ts, total_ms=ts["med_ms"])
