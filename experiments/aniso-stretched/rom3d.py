"""3D ROM arms for Family A3: POD-Galerkin and coordinate-bank Galerkin span, scored against the same references.

Usage: rom3d.py pod  <p> <N> <ntrain> <cohort> <out.json>
       rom3d.py bank <p> <N> <ntrain> <cohort> <out.json> <bank.npz>
Basis is orthonormalised in the lumped-mass inner product of the truth space and ordered (POD: singular values;
bank: PCA of the training snapshots' projection coefficients in the bank, as in the paper's rotation).
Online = gal3d.Galerkin.online (numpy, one CPU core). Field reconstruction and interpolation are timed separately.
"""
import sys, os, json, time
import numpy as np
import aniso as an, aniso3d as a3, solve3d as s3, gal3d, hq
HERE = os.path.dirname(os.path.abspath(__file__)); OD = f"{HERE}/runs/rom3"
RANKS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def node_coords(N, p):
    xi = (hq.gll(p) + 1) / 2
    x = ((np.arange(N)[:, None] + xi[None, :-1]) / N).ravel()
    return np.r_[x, 1.0][1:-1]                                # interior nodes, (N p - 1,)


def lumped_w(S):
    w1 = np.asarray(S.sp.M.sum(axis=1)).ravel()
    return (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).ravel()


def pod_basis(X, w, rmax):
    t = time.perf_counter(); sw = np.sqrt(w)
    Y = X.astype(np.float64) * sw[None, :]
    G = Y @ Y.T; lam, V = np.linalg.eigh(G); lam, V = lam[::-1], V[:, ::-1]
    r = min(rmax, int((lam > lam[0] * 1e-13).sum()))
    Phi = (Y.T @ V[:, :r]) / np.sqrt(lam[:r])[None, :] / sw[:, None]
    return Phi, np.sqrt(np.maximum(lam, 0)), time.perf_counter() - t


def bank_basis(Gb, X, w, rmax):
    """Gb: bank evaluated at the truth nodes (n^3, R). Returns rotated, mass-orthonormal basis ordered by PCA."""
    t = time.perf_counter(); sw = np.sqrt(w)
    Q, Rg = np.linalg.qr(Gb * sw[:, None])
    A = (X.astype(np.float64) * sw[None, :]) @ Q                  # projection coefficients of training snapshots
    nrm = np.linalg.norm(X.astype(np.float64) * sw[None, :], axis=1)
    _, s, Vt = np.linalg.svd(A / nrm[:, None], full_matrices=False)
    Phi = (Q @ Vt.T[:, :rmax]) / sw[:, None]
    floor = np.sqrt(np.maximum(0, 1 - (A ** 2).sum(1) / nrm ** 2))  # training projection floor at full width
    return Phi, s, time.perf_counter() - t, dict(train_floor_mean=float(floor.mean()), train_floor_max=float(floor.max()),
                                                cond_Rg=float(np.linalg.cond(Rg)))


def main():
    arm, p, N, nt, cohort, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
    S = s3.Solver(N, p); w = lumped_w(S)
    X = np.load(f"{OD}/snap_p{p}N{N}_{nt}.npy")
    snap = json.load(open(f"{OD}/snap_p{p}N{N}_{nt}.json"))
    extra = {}
    if arm == "pod":
        Phi, sv, tb = pod_basis(X, w, max(RANKS))
    else:
        bank = np.load(sys.argv[7]); Phi, sv, tb, extra = bank_basis(bank["G"], X, w, max(RANKS))
        extra.update(bank_train_seconds=float(bank["train_seconds"]), bank_R=int(bank["G"].shape[1]))
    Gal = gal3d.Galerkin(S, Phi, m_load=16)
    ev = a3.Evaluator3(128); P = a3.sample_params(cohort, 16)
    refs = [np.load(f"{HERE}/runs/ref3/A3_{cohort}_{i:02d}.npy").astype(np.float64) for i in range(16)]
    rows = []
    for r in [r for r in RANKS if r <= Phi.shape[1]]:
        g = Gal.restrict(r); Pr = np.ascontiguousarray(Phi[:, :r]); rr = []
        for i in range(16):
            c, tm = g.timed(P[i], 200)
            t = time.perf_counter(); u = Pr @ c; trec = time.perf_counter() - t
            e = ev.errors(ev.field(S.sp, u), refs[i])
            rr.append(dict(arm=arm, cohort=cohort, case=i, rank=r, online=tm, total_ms=tm["med_ms"], recon_truthgrid_ms=1e3 * trec, **e))
        rows += rr
        print("%s rank %-3d l2w mean %.5f max %.5f  online med %.4f ms [%.4f, %.4f]  recon %.2f ms" % (
            arm, r, np.mean([x["l2w"] for x in rr]), np.max([x["l2w"] for x in rr]), np.median([x["total_ms"] for x in rr]),
            min(x["online"]["min_ms"] for x in rr), max(x["online"]["max_ms"] for x in rr), np.median([x["recon_truthgrid_ms"] for x in rr])), flush=True)
    json.dump(dict(arm=arm, truth=dict(p=p, N=N, dofs=S.n ** 3), ntrain=nt, cohort=cohort, snapshots=snap, basis_seconds=tb,
                   galerkin_offline_seconds=Gal.offline_seconds, load_modes=16, load_truncation=Gal.load_trunc,
                   singular_values=sv[:256].tolist(), rows=rows, **extra), open(out, "w"))


if __name__ == "__main__":
    main()
