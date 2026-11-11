"""Warm CPU timings + errors of the comparator solvers on chosen meshes.

Usage: time_solvers.py <family> <cohort> <case-indices e.g. 0,1,2> <meshes e.g. 128:2.5:2.5,256:0:0> <out.json> [solvers]
Assembly is excluded from every time. Errors are against the stored reference (runs/ref).
"""
import json, os, sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import pyamg
import aniso as an

HERE = os.path.dirname(os.path.abspath(__file__))
TOLS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
REP = 5


def med(f, rep=REP):
    f()  # warm
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return r, dict(med_ms=1e3 * float(np.median(ts)), min_ms=1e3 * min(ts), max_ms=1e3 * max(ts))


def splu(A):
    return spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))


def cg(A, b, tol, M=None, maxiter=20000):
    it = [0]
    def cb(_): it[0] += 1
    x, info = spla.cg(A, b, rtol=tol, atol=0.0, M=M, maxiter=maxiter, callback=cb)
    return x, it[0], info


AMG = {
    "sa_sym": lambda A: pyamg.smoothed_aggregation_solver(A, strength="symmetric", max_coarse=200),
    "sa_evo": lambda A: pyamg.smoothed_aggregation_solver(A, strength=("evolution", {"epsilon": 4.0, "k": 2}),
                                                          improve_candidates=[("gauss_seidel", {"sweep": "symmetric", "iterations": 4}), None],
                                                          max_coarse=200),
    "rs": lambda A: pyamg.ruge_stuben_solver(A, strength=("classical", {"theta": 0.25}), max_coarse=200),
    "rs_th05": lambda A: pyamg.ruge_stuben_solver(A, strength=("classical", {"theta": 0.5}), max_coarse=200),
}


def main():
    fam, cohort = sys.argv[1], sys.argv[2]
    cases = [int(s) for s in sys.argv[3].split(",")]
    meshes = [tuple(float(v) for v in m.split(":")) for m in sys.argv[4].split(",")]
    out = sys.argv[5]
    solvers = sys.argv[6].split(",") if len(sys.argv) > 6 else ["direct", "frozen_lu", "amg", "frozen_amg", "cg", "jacobi"]
    ev = an.Evaluator()
    params = an.sample_params(fam, cohort, 16)
    muc = an.centre_param(fam)
    rows = []
    for (N, bx, by) in meshes:
        N = int(N)
        xc, yc, Ac, _ = an.problem(fam, muc, N, bx, by)
        Ac = Ac.tocsc()
        luc = splu(Ac) if "frozen_lu" in solvers else None
        mlc = {k: f(Ac.tocsr()) for k, f in AMG.items()} if "frozen_amg" in solvers else {}
        for i in cases:
            mu = params[i]
            Uref = np.load(f"{HERE}/runs/ref/{fam}_{cohort}_{i:02d}.npy").astype(np.float64)
            t = time.perf_counter(); x, y, A, b = an.problem(fam, mu, N, bx, by); t_asm = time.perf_counter() - t
            Acsc = A.tocsc()
            def score(u):
                return ev.errors(ev.interp(an.to_full(u, N), N, bx, by), Uref)
            base = dict(family=fam, cohort=cohort, case=i, N=N, bx=bx, by=by, assemble_ms=1e3 * t_asm)
            def add(**kw):
                rows.append({**base, **kw}); print(json.dumps(rows[-1]), flush=True)
            if "direct" in solvers:
                lu, tf = med(lambda: splu(Acsc), 3 if N > 256 else REP)
                u, ts = med(lambda: lu.solve(b))
                add(solver="direct_superlu", factor=tf, solve=ts, total_ms=tf["med_ms"] + ts["med_ms"], **score(u))
            if "frozen_lu" in solvers:
                M = spla.LinearOperator(A.shape, matvec=luc.solve)
                for tol in TOLS[:4]:
                    (u, it, info), ts = med(lambda: cg(A, b, tol, M, maxiter=300))
                    add(solver="frozen_lu_pcg", tol=tol, iters=it, info=info, solve=ts, total_ms=ts["med_ms"], **score(u))
                    if info != 0: break
            if "amg" in solvers:
                for name, f in AMG.items():
                    try:
                        ml, tset = med(lambda: f(A), 3)
                    except Exception as e:
                        add(solver="amg_" + name, failed=repr(e)); continue
                    M = ml.aspreconditioner(cycle="V")
                    for tol in TOLS:
                        (u, it, info), ts = med(lambda: cg(A, b, tol, M, maxiter=500))
                        add(solver="amg_" + name, tol=tol, iters=it, info=info, setup=tset, solve=ts,
                            total_ms=tset["med_ms"] + ts["med_ms"], levels=len(ml.levels),
                            opc=float(ml.operator_complexity()), **score(u))
            if "frozen_amg" in solvers:
                for name, ml in mlc.items():
                    M = ml.aspreconditioner(cycle="V")
                    for tol in TOLS[:4]:
                        (u, it, info), ts = med(lambda: cg(A, b, tol, M, maxiter=500))
                        add(solver="frozen_amg_" + name, tol=tol, iters=it, info=info, solve=ts, total_ms=ts["med_ms"], **score(u))
            if "fd" in solvers and fam == "A":
                # fast-diagonalisation preconditioner: diagonal part Kxx Sx(x)My + Kyy Mx(x)Sy inverted exactly
                # (1D generalised eigen-decompositions depend on the mesh only: offline, not timed)
                import scipy.linalg as sla
                def eig1(xn):
                    h = np.diff(xn); S1 = (np.diag(1 / h[:-1] + 1 / h[1:]) - np.diag(1 / h[1:-1], 1) - np.diag(1 / h[1:-1], -1))
                    return sla.eigh(S1, an.mass1d(xn).toarray()[1:-1, 1:-1])
                (lx, Vx), (ly, Vy) = eig1(x), eig1(y)
                a_, b_, c_ = an.tensor(fam, mu, np.array([0.5]), np.array([0.5]))
                den = a_[0, 0] * lx[:, None] + c_[0, 0] * ly[None, :]
                def fdinv(r):
                    return (Vx @ ((Vx.T @ r.reshape(N - 1, N - 1) @ Vy) / den) @ Vy.T).ravel()
                M = spla.LinearOperator(A.shape, matvec=fdinv)
                for tol in TOLS[:4]:
                    (u, it, info), ts = med(lambda: cg(A, b, tol, M, maxiter=2000))
                    add(solver="fd_pcg", tol=tol, iters=it, info=info, solve=ts, total_ms=ts["med_ms"], **score(u))
            if "cg" in solvers:
                for tol in TOLS[:4]:
                    (u, it, info), ts = med(lambda: cg(A, b, tol), 3)
                    add(solver="cg", tol=tol, iters=it, info=info, solve=ts, total_ms=ts["med_ms"], **score(u))
            if "jacobi" in solvers:
                dinv = 1.0 / A.diagonal()
                M = spla.LinearOperator(A.shape, matvec=lambda v: dinv * v)
                for tol in TOLS[:4]:
                    (u, it, info), ts = med(lambda: cg(A, b, tol, M), 3)
                    add(solver="jacobi_cg", tol=tol, iters=it, info=info, solve=ts, total_ms=ts["med_ms"], **score(u))
    json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    main()
