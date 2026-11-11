"""Discretisation-error sweep: reference on the evaluation grid + every comparator mesh (direct solve).

Writes runs/err/<family>_<cohort>_<i>.json and the reference field (float32) to runs/ref/.
Usage: sweep_err.py <cohort> <ncases> <workers> [families]
"""
import json, os, sys, time
import numpy as np, scipy.sparse.linalg as spla
from multiprocessing import Pool
import aniso as an

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
BETAS = [0, 1.5, 2.5]
NREF, BREF = 2048, 2.5


def direct(A, b):
    lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    return lu.solve(b)


def configs(full):
    out = []
    for N in NS:
        for bx in BETAS:
            for by in BETAS:
                if N > 512 and bx != by and not full:
                    continue
                out.append((N, bx, by))
    return out


def run(job):
    fam, cohort, i, mu, full = job
    out = f"{HERE}/runs/err/{fam}_{cohort}_{i:02d}.json"
    if os.path.exists(out):
        return out
    ev = an.Evaluator(NREF, BREF)
    t = time.time()
    x, y, A, b = an.problem(fam, mu, NREF, BREF, BREF)
    Uref = an.to_full(direct(A, b), NREF)
    del A
    np.save(f"{HERE}/runs/ref/{fam}_{cohort}_{i:02d}.npy", Uref.astype(np.float32))
    rec = dict(family=fam, cohort=cohort, index=i, mu=list(map(float, mu)), ref_seconds=time.time() - t, rows=[])
    for (N, bx, by) in configs(full):
        x, y, A, b = an.problem(fam, mu, N, bx, by)
        U = an.to_full(direct(A, b), N)
        e = ev.errors(ev.interp(U, N, bx, by), Uref)
        rec["rows"].append(dict(N=N, bx=bx, by=by, **e))
    json.dump(rec, open(out, "w"))
    return out


if __name__ == "__main__":
    cohort, n, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fams = sys.argv[4].split(",") if len(sys.argv) > 4 else ["A", "B"]
    os.makedirs(f"{HERE}/runs/err", exist_ok=True); os.makedirs(f"{HERE}/runs/ref", exist_ok=True)
    jobs = []
    for i in range(n):
        for fam in fams:
            jobs.append((fam, cohort, i, an.sample_params(fam, cohort, 16)[i], False))
    with Pool(workers) as p:
        for o in p.imap_unordered(run, jobs):
            print(time.strftime("%H:%M:%S"), o, flush=True)
