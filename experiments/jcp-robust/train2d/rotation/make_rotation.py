"""burgers-bank-knob: the offline importance rotation of the frozen R = 512 Burgers bank (training data only).

Nothing is trained. With G the frozen bank on the TRAINING mesh (256^2, the mesh the checkpoint was fitted on),
G = Q_G R_G (thin QR, grid Euclidean metric), and c_i = h_theta(z_i) the head's coefficient vector at every
stored training code z_i (ck['Z_tr'], 131072 codes -- training data only, nothing from any evaluation cohort):

    A = [ (R_G c_i)^T / ||R_G c_i|| ]_i  = U S V_s^T,     T = R_G^{-1} V_s,     L = V_s^T R_G = T^{-1}.

A is the field-space image of the training coefficient vectors, so this is the SVD of G Sigma^{1/2}
(Sigma = empirical second moment of the training coefficient vectors). As in the Poisson lane each sample row is
normalised by its own field norm ||G c_i|| = ||R_G c_i||, so every training state counts equally (the brief asks to
reuse the Poisson lane's construction; the unnormalised variant is computed as a diagnostic only, never deployed). The rotated bank G' = G T = Q_G V_s has orthonormal columns at 256^2, ordered by training energy.
A truncation R' keeps the first R' rotated columns; the model's coefficient vector c becomes a = L[:R'] c, i.e.
the state is G T[:, :R'] L[:R'] c (an oblique projector on coefficient space). At R' = R, T L = I to round-off.

Differences from the Poisson lane (poisson-bank-knob/pbk_core.make_rotation): there the sample rows were
[Q_G^T u_i ; R_G h(z_i)] / ||u_i|| (projections of training fields AND head coefficients, each normalised).
Here only the head coefficients enter (the Burgers training fields are not stored with the checkpoint; the
head coefficients ARE the model's training coefficient vectors); the per-row normalisation is the same.

Writes inputs/rotation_R512.npz (T, L, singular values, R_G) and inputs/rotation_R512.json (diagnostics).

    PYTHONPATH=... python make_rotation.py --checkpoint <pkl> --directions <npz> --out inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import arms as A
import sep_common as sc

LADDER = (32, 64, 128, 256, 384, 512)


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--directions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train-mesh', type=int, default=256)
    a = p.parse_args()
    t0 = time.perf_counter()
    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Z = np.asarray(ck['Z_tr'])
    K, R = Z.shape[1], int(np.asarray(ck['params']['h_lin']).shape[1])
    assert ck['cfg']['N'] == a.train_mesh, ck['cfg']['N']
    G = A.CoordBank(params, K, R).on_grid(a.train_mesh)                   # (255^2, 512)
    Rg = np.asarray(jnp.linalg.qr(G, mode='r'))
    sv_rg = np.linalg.svd(Rg, compute_uv=False)
    hv = jax.jit(jax.vmap(lambda z: sc.head(params, z)))
    Hc = np.concatenate([np.asarray(hv(jnp.asarray(Z[s:s + 8192]))) for s in range(0, len(Z), 8192)])
    A0 = Hc @ Rg.T                                                         # rows (R_G c_i)^T = coords of G c_i in Q_G
    nrm0 = np.linalg.norm(A0, axis=1)
    Am = A0 / nrm0[:, None]                                                # Poisson-lane normalisation (deployed)
    _, s, Vt = np.linalg.svd(Am, full_matrices=False)
    _, s_un, Vt_un = np.linalg.svd(A0, full_matrices=False)               # diagnostic only
    Vs = Vt.T
    T = np.linalg.solve(Rg, Vs)
    Lm = Vs.T @ Rg
    ident = float(np.linalg.norm(Lm @ T - np.eye(R)))
    # the rotated bank at the training mesh must have orthonormal columns (G T = Q_G V_s)
    Gt = np.asarray(G) @ T
    orth = float(np.linalg.norm(Gt.T @ Gt - np.eye(R)))
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    # how much of each field-space quantity the first R' rotated directions capture
    Cq = np.asarray(np.load(a.directions)['C'])
    RC = Rg @ Cq
    VRC = Vs.T @ RC                                                         # rotated coordinates of R_G C
    cap_C = {}
    for q in (16, 48, 112, 240, 256):
        if q <= Cq.shape[1]:
            tot = np.sum(RC[:, :q] ** 2)
            cap_C[str(q)] = {str(r): float(np.sum(VRC[:r, :q] ** 2) / tot) for r in LADDER}
    # per-sample relative truncation error of the head coefficients (the q = 0 representation floor of truncation)
    proj = Am @ Vs
    nrm = np.linalg.norm(Am, axis=1)
    trunc = {}
    proj_un = A0 @ Vt_un.T
    trunc_un = {}
    for r in LADDER:
        e = np.sqrt(np.clip(nrm ** 2 - np.sum(proj[:, :r] ** 2, axis=1), 0, None)) / nrm
        trunc[str(r)] = dict(max=float(e.max()), p99=float(np.quantile(e, .99)), median=float(np.median(e)))
        e = np.sqrt(np.clip(nrm0 ** 2 - np.sum(proj_un[:, :r] ** 2, axis=1), 0, None)) / nrm0
        trunc_un[str(r)] = dict(max=float(e.max()), p99=float(np.quantile(e, .99)), median=float(np.median(e)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'rotation_R512.npz', T=T, L=Lm, singular_values=s, R_G=Rg)
    info = dict(
        construction='SVD of A = [(R_G h(z_i))^T / ||R_G h(z_i)||], z_i = every stored training code (ck["Z_tr"]); '
                     'T = R_G^{-1} V_s, L = V_s^T R_G; per-row normalised as in poisson-bank-knob; training data only',
        training_state_norm_range=[float(nrm0.min()), float(nrm0.max())],
        diagnostic_unnormalised_head_truncation_error=trunc_un,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        directions_sha256=hashlib.sha256(Path(a.directions).read_bytes()).hexdigest(),
        train_mesh=a.train_mesh, training_codes=int(len(Z)), K=int(K), R=int(R),
        R_G_condition_number=float(sv_rg[0] / sv_rg[-1]), L_times_T_identity_deviation=ident,
        rotated_bank_orthonormality_deviation_at_train_mesh=orth,
        cumulative_training_energy={str(r): float(energy[r - 1]) for r in LADDER},
        head_coefficient_truncation_error=trunc,
        correction_directions_energy_captured=cap_C,
        T_sha256=sha_array(T), L_sha256=sha_array(Lm),
        file_sha256=hashlib.sha256((out / 'rotation_R512.npz').read_bytes()).hexdigest(),
        seconds=time.perf_counter() - t0, jax_backend=jax.default_backend())
    (out / 'rotation_R512.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k not in ('head_coefficient_truncation_error',)}, indent=1))
    print('trunc', json.dumps(trunc))
    print('trunc_unnormalised', json.dumps(trunc_un))


if __name__ == '__main__':
    main()
