"""spectral-pilot Part B: POD-Galerkin feasibility comparator on the L-shape (DESIGN 4.4, A3.2). PILOT: small, not a
result.

  python rom_b.py [--rmax 384]

POD of the L=512 SBDF2 training snapshots (Euclidean inner product of the interior nodes). Per r: Psi = Phi_r V with
Phi_r^T A Phi_r = V D V^T (A the 5-point Laplacian), so the reduced linear part is nu D; tensor
T[k,i,j] = -sum_g psi_k psi_i (D_h psi_j) with D_h the central (d_x + d_y) of the FD FOM (so ROM-T is the Galerkin
projection of the L=512 FD right-hand side, gate G4); ETDRK4 (as Part A). Errors on the i/64 lattice against the
L=1024 reference; floors by least squares with the sampled basis (A1.4). Writes runs/B/rom.json, ops_r<r>.npz.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

import common as C
import fd_l as F
import romlib as RL
from gen_b import sub
from rom_a import ladder_n

R_LADDER = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
L = 512


def dmatrix(L_, m):
    """Central (d_x + d_y) on the interior nodes of Omega_L (zero outside), as a sparse matrix (the FD arithmetic)."""
    n = int(m.sum())
    idx = -np.ones(m.shape, int)
    idx[m] = np.arange(n)
    I, J = np.nonzero(m)
    rows, cols, vals = [], [], []
    for di, dj, s in ((1, 0, 1.0), (-1, 0, -1.0), (0, 1, 1.0), (0, -1, -1.0)):
        ii, jj = I + di, J + dj
        ok = (ii >= 0) & (ii < L_ - 1) & (jj >= 0) & (jj < L_ - 1)
        ok[ok] = m[ii[ok], jj[ok]]
        rows.append(idx[I[ok], J[ok]])
        cols.append(idx[ii[ok], jj[ok]])
        vals.append(np.full(ok.sum(), s * L_ / 2.0))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def tensor_fd(Phi, Dm):
    U = jnp.asarray(Phi)
    G = jnp.asarray(Dm @ Phi)

    @jax.jit
    def slab(U, G, k):
        return -((U * U[:, k][:, None]).T @ G)
    r = Phi.shape[1]
    T = np.empty((r, r, r))
    for k in range(r):
        T[k] = np.asarray(slab(U, G, k))
    return T


def lat_rows(m, L_=L, lat=64):
    """Row indices (into the interior-node vector) of the i/lat lattice nodes inside Omega_L."""
    idx = -np.ones(m.shape, int)
    idx[m] = np.arange(int(m.sum()))
    I, J = F.lattice_idx(L_, lat)
    return idx[I, J]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rmax', type=int, default=384)
    a = ap.parse_args()
    d = C.RUNS / 'B'
    ref = json.loads((d / 'ref.json').read_text())
    te = np.asarray(ref['test_desc'])
    nu = te[:, 4]
    R = np.load(d / 'ref_lat256.npy')[..., sub(256, 64)]           # (12, 6, P64)
    den = np.sqrt(np.sum(R[:, 0] ** 2, axis=-1))
    X = np.load(d / 'train_flat.npy', mmap_mode='r')                # (96, 21, n_in)
    A, m = F.lap_matrix(L)
    Dm = dmatrix(L, m)
    rows64 = lat_rows(m)
    rep = dict(env=C.env_record(), label=C.PILOT, timings={}, scoring=ref['scoring'])
    t0 = time.perf_counter()
    S = np.asarray(X).reshape(-1, X.shape[-1])
    Phi, sv = RL.pod(S.reshape(len(S), 1, -1), a.rmax)
    Phi48, _ = RL.pod(np.asarray(X[:48]).reshape(48 * 21, 1, -1), a.rmax)
    rep['timings']['pod'] = time.perf_counter() - t0
    rmax = Phi.shape[1]
    rep['rmax_eff'] = rmax
    rep['orth_dev'] = float(np.max(np.abs(Phi.T @ Phi - np.eye(rmax))))
    # sampled-basis floors (lattice norm, least squares), all r on the ladder
    fl = {}
    for tag, B_ in (('pod', Phi), ('pod48', Phi48)):
        out = {}
        for r in [x for x in R_LADDER if x <= rmax]:
            Bl = B_[rows64, :r]
            coef, *_ = np.linalg.lstsq(Bl, R.reshape(-1, R.shape[-1]).T, rcond=1e-12)
            res = R.reshape(-1, R.shape[-1]) - (Bl @ coef).T
            et = np.sqrt(np.sum(res ** 2, axis=-1)).reshape(R.shape[:2]) / den[:, None]
            e = et[:, 1:].max(1)
            out[r] = dict(worst=float(e.max()), median=float(np.median(e)), per_case=e.tolist())
        fl[tag] = out
    rep['floors'] = fl
    t0 = time.perf_counter()
    Tphi = tensor_fd(Phi, Dm)
    rep['timings']['tensor_full'] = time.perf_counter() - t0
    print('B POD + tensor done', rep['timings'], flush=True)
    u0 = np.stack([F.ic(L, dd)[m] for dd in te])                   # (12, n_in) full-node IC
    rom_t, gates = {}, {}
    for r in [x for x in R_LADDER if x <= rmax]:
        t0 = time.perf_counter()
        Lr = Phi[:, :r].T @ (A @ Phi[:, :r])
        Lr = (Lr + Lr.T) / 2
        D, V = np.linalg.eigh(Lr)
        Psi = Phi[:, :r] @ V
        Tp, iu, ju = RL.pack(RL.rotate_tensor(Tphi[:r, :r, :r], V))
        z0 = u0 @ Psi
        Plat = Psi[rows64]                                          # (P64, r)
        coef, *_ = np.linalg.lstsq(Plat, R.reshape(-1, R.shape[-1]).T, rcond=1e-12)
        fr = (np.sqrt(np.sum((R.reshape(-1, R.shape[-1]) - (Plat @ coef).T) ** 2, -1)).reshape(R.shape[:2])
              / den[:, None])[:, 1:].max(1)
        NL = RL.nl_tensor(Tp, iu, ju)

        def run(nn, NL=NL):
            Z = np.asarray(RL.make_rom_batch(NL, D, nn)(jnp.asarray(z0), jnp.asarray(nu)))
            Ul = np.einsum('btr,pr->btp', Z, Plat)
            return F.err_lat(Ul, R), Ul
        rom_t[r] = dict(build_s=time.perf_counter() - t0, rows=ladder_n(run, f'B ROM-T r={r}', den))
        for row in rom_t[r]['rows']:                                # D2 with the A4 tolerance
            assert all((not np.isfinite(e_)) or e_ >= f_ * (1 - 1e-9) for e_, f_ in zip(row['err'], fr)), 'D2'
        np.savez(d / f'ops_r{r}.npz', Psi=Psi, D=D, Tp=Tp, iu=iu, ju=ju, Plat=Plat)
        if r in (32, 128):                                          # G4 (rollouts) and SW4
            I_, J_ = np.nonzero(m)
            Ij, Jj = jnp.asarray(I_), jnp.asarray(J_)

            def _exact(z, o):
                u = jnp.zeros(m.shape).at[Ij, Jj].set(o[0] @ z)
                p = jnp.pad(u, 1)
                adv = u * ((p[2:, 1:-1] - p[:-2, 1:-1]) + (p[1:-1, 2:] - p[1:-1, :-2])) * (L / 2.0)
                return -(o[0].T @ adv[Ij, Jj])
            NLe = RL.NLOp(_exact, (jnp.asarray(Psi),))
            nn = rom_t[r]['rows'][-1]['n']
            Zt = np.asarray(RL.make_rom_batch(NL, D, nn)(jnp.asarray(z0[:2]), jnp.asarray(nu[:2])))
            Ze = np.asarray(RL.make_rom_batch(NLe, D, nn)(jnp.asarray(z0[:2]), jnp.asarray(nu[:2])))
            g4 = float(np.max(np.abs(Zt - Ze)) / np.max(np.abs(Ze)))
            zs = Ze.reshape(-1, r)
            Nv = np.stack([np.asarray(NL(jnp.asarray(z))) for z in zs])
            ks = int(np.argmax(np.max(np.abs(Nv), axis=0)))
            jz = int(np.argmax(np.abs(Nv[:, ks])))
            zf = jnp.asarray(zs[jz])
            Tc = Tp.copy()
            Tc[ks] = 0
            NLc = RL.nl_tensor(Tc, iu, ju)
            full, cor = np.asarray(NL(zf)), np.asarray(NLc(zf))
            dN = full - cor
            nN = float(np.linalg.norm(full))
            avail = bool(np.isfinite(nN) and nN > 0)
            exact = bool(abs(dN[ks] - full[ks]) <= 1e-12 * max(1.0, abs(full[ks]))
                         and np.max(np.abs(np.delete(dN, ks))) <= 1e-14 * max(1.0, abs(full[ks])))
            big = bool(avail and abs(full[ks]) >= 1e-6 * nN)
            Zc = np.asarray(RL.make_rom_batch(NLc, D, nn)(jnp.asarray(z0[:2]), jnp.asarray(nu[:2])))
            gates[r] = dict(G4_rel=g4, G4_pass=bool(g4 <= 1e-9), SW4_exact=exact, SW4_large=big, SW4_available=avail,
                            SW4_fired=bool(avail and exact and big),
                            SW4_rollout_rel=float(np.max(np.abs(Zc - Zt)) / np.max(np.abs(Zt))), n=nn)
            print('B gates', r, gates[r], flush=True)
        C.dump(d / 'rom.json', dict(rep, rom_t=rom_t, gates=gates))
    C.dump(d / 'rom.json', dict(rep, rom_t=rom_t, gates=gates))
    print('B ROM done', flush=True)


if __name__ == '__main__':
    main()
