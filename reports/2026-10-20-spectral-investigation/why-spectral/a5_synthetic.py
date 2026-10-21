"""A5. When does a learned linear basis need far fewer coefficients than sines? Synthetic families (CPU, NumPy).

SYNTHETIC fields, not Burgers solutions: they isolate one property at a time. All live on the 2D scoring lattice
(255^2 interior nodes of x = i/256, zero walls) and are multiplied by the wall envelope sin(pi x) sin(pi y), whose odd
extension is smooth, so that the walls do not cost the sine series anything. Feature width delta.
  F1 front-1p   : planar tanh front 0.5(1 - tanh((x - p)/delta)), p ~ U[.3,.7]           (1 parameter, transport)
  F2 front-2p   : same front along a direction at angle th ~ U[0, pi/2), offset p        (2 parameters, transport)
  F3 bump-2p    : Gaussian bump of width delta, centre ~ U[.3,.7]^2                      (2 parameters, transport)
  F4 bump-width : Gaussian bump at the fixed centre (.5,.5), width ~ U[delta, 4 delta]     (1 parameter, diffusion-like)
Learned basis = POD (SVD) of n_train = 800 training fields: the standard data-driven linear basis (optimal for the
training set in mean square, not for the held-out worst case), a stand-in for a learned linear bank (ours is used
linearly). Sines also as the best anisotropic block (N_x-1)(N_y-1) (a direction-aware spectral solver), K_aniso. Scored on 200 held-out fields:
r_POD(eps) = smallest rank with max held-out relative error <= eps; sines: K_block(eps) = (N-1)^2 for the smallest square
block (the spectral FOM's set) and K_lap(eps) (Laplacian-ordered prefix). rho = K_block / r_POD.
A POD rank above n_train/2 = 400 is reported as None: a reliability cutoff we chose, not a mathematical limit; a
sample-size convergence study would be needed before reading censoring as a property of the family.
Writes analysis/a5_synthetic.json.
"""
import json

import numpy as np
from scipy.fft import dstn

import common as C

n = 255
x = np.arange(1, n + 1) / 256.
X, Y = np.meshgrid(x, x, indexing='ij')
ENV = np.sin(np.pi * X) * np.sin(np.pi * Y)
k = np.arange(1, n + 1)
KX, KY = np.meshgrid(k, k, indexing='ij')
LAP = np.argsort((KX ** 2 + KY ** 2).ravel(), kind='stable')
BLK = np.maximum(KX, KY).ravel()
DELTAS = (0.08, 0.04, 0.02, 0.01, 0.005)
EPS = (1e-2, 1e-3)
NTR, NTE = 800, 200


def field(fam, rng, d):
    if fam == 'F1':
        p = rng.uniform(.3, .7)
        f = 0.5 * (1 - np.tanh((X - p) / d))
    elif fam == 'F2':
        th, p = rng.uniform(0, np.pi / 2), rng.uniform(.3, .7)
        z = (X - .5) * np.cos(th) + (Y - .5) * np.sin(th) - (p - .5)
        f = 0.5 * (1 - np.tanh(z / d))
    elif fam == 'F3':
        cx, cy = rng.uniform(.3, .7, 2)
        f = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * d * d))
    else:
        w = rng.uniform(d, 4 * d)
        f = np.exp(-((X - .5) ** 2 + (Y - .5) ** 2) / (2 * w * w))
    return (f * ENV).ravel()


def first(errs, e):
    idx = np.nonzero(errs <= e)[0]
    return int(idx[0] + 1) if idx.size else None


def main():
    out = dict(deltas=DELTAS, eps=EPS, n_train=NTR, n_test=NTE, rows=[])
    for fam in ('F1', 'F2', 'F3', 'F4'):
        for d in DELTAS:
            rng = np.random.default_rng(20261020)
            A = np.stack([field(fam, rng, d) for _ in range(NTR)], 1)
            T = np.stack([field(fam, rng, d) for _ in range(NTE)], 1)
            Uq, sv, _ = np.linalg.svd(A, full_matrices=False)
            nt = np.linalg.norm(T, axis=0)
            P = Uq.T @ T
            res2 = np.maximum(nt[None] ** 2 - np.cumsum(P ** 2, 0), 0)
            pod = (np.sqrt(res2) / nt[None]).max(1)                           # index r-1, worst held-out
            pod[NTR // 2:] = np.inf                                           # data-limited ranks not trusted
            Cs = np.stack([dstn(T[:, j].reshape(n, n), type=1, norm='ortho').ravel() for j in range(NTE)])
            tot = (Cs ** 2).sum(1)
            lap = (np.sqrt(np.maximum(tot[:, None] - np.cumsum(Cs[:, LAP] ** 2, 1), 0)) / nt[:, None]).max(0)
            Eb = np.stack([np.bincount(BLK - 1, weights=Cs[j] ** 2, minlength=n) for j in range(NTE)])
            blk = (np.sqrt(np.maximum(tot[:, None] - np.cumsum(Eb, 1), 0)) / nt[:, None]).max(0)   # j <-> N = j+2
            # anisotropic blocks: worst held-out tail for every (Nx, Ny)
            worst2 = np.zeros((n, n))
            for j in range(NTE):
                cs2 = np.cumsum(np.cumsum((Cs[j].reshape(n, n)) ** 2, 0), 1)       # kept energy, modes <= (i+1, k+1)
                worst2 = np.maximum(worst2, np.sqrt(np.maximum(tot[j] - cs2, 0)) / nt[j])
            cnt = np.outer(np.arange(1, n + 1), np.arange(1, n + 1))             # (Nx-1)(Ny-1) kept modes
            row = dict(family=fam, delta=d, r_pod={}, K_block={}, N_block={}, K_lap={}, rho={}, K_aniso={}, rho_aniso={})
            for e in EPS:
                rp = first(pod, e)
                j = np.nonzero(blk <= e)[0]
                N = int(j[0] + 2) if j.size else None
                kb = (N - 1) ** 2 if N else None
                row['r_pod'][str(e)], row['N_block'][str(e)], row['K_block'][str(e)] = rp, N, kb
                row['K_lap'][str(e)] = first(lap, e)
                row['rho'][str(e)] = kb / rp if (kb and rp) else None
                feas = worst2 <= e
                ka = int(cnt[feas].min()) if feas.any() else None
                row['K_aniso'][str(e)] = ka
                row['rho_aniso'][str(e)] = ka / rp if (ka and rp) else None
            out['rows'].append(row)
            print(fam, d, row['r_pod'], row['K_block'], row['rho'], row['K_aniso'], flush=True)
    (C.OUT / 'a5_synthetic.json').write_text(json.dumps(out, indent=0))


if __name__ == '__main__':
    main()
