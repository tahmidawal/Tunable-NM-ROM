"""Diagnostic: WHY does staging stall on the family?  Residual smoothness in
the conditioning variable vs in x.

For each saved decoder (true-z control, auto-decoder K variants), the residual
e_k(x; z_i) = U_i - sum_{l<k} eps_l D_l(x; z_i) over TRAINING samples:
  (a) smoothness in z: Pearson correlation of e_k(.; z_i) with e_k(.; z_j) for
      the nearest neighbours j in WHITENED z-space (Mahalanobis, so the
      measure is reparametrization-invariant), 1-NN and 5-NN means; the same
      for the FIELDS U_i as the reference (fields are smooth in z: NN-corr ~1).
      Residual NN-corr << field NN-corr => the residual is rough in z => no
      smooth function of (x, z) fits it => no per-stage power law.
  (b) frequency in x: spectral centroid (cycles/unit) of e_k vs U.
Provenance: config read from the pkl; family rebuilt from the pkl's seed/N/
counts (asserted equal to the current env-derived module config).
Usage: python ms_diag.py <run_dir_parametric> <run_dir_autodec>
"""
import glob, json, os, pickle, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ms_parametric as mp

print(f"jax_backend={jax.default_backend()}", flush=True)


def check_cfg(cfg):
    for k in ("N", "n_train", "n_val", "seed", "hidden", "n_layers"):
        assert cfg[k] == mp.CONFIG[k], f"config mismatch {k}: pkl {cfg[k]} vs env {mp.CONFIG[k]}"


def whiten(Z):
    Zc = Z - Z.mean(0)
    C = np.cov(Zc.T) + 1e-12 * np.eye(Z.shape[1])
    L = np.linalg.cholesky(np.linalg.inv(C))
    return Zc @ L


def nn_corr(E, Z, k=5):
    Zw = whiten(Z)
    D = ((Zw[:, None, :] - Zw[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    Ec = E - E.mean(1, keepdims=True)
    Ec = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-300)
    c1 = np.array([Ec[i] @ Ec[idx[i, 0]] for i in range(len(E))])
    ck = np.array([np.mean([Ec[i] @ Ec[j] for j in idx[i]]) for i in range(len(E))])
    return float(c1.mean()), float(ck.mean())


def centroid(E, n):
    A = np.abs(np.fft.fft2(E.reshape(-1, n, n))).mean(0)
    fr = np.fft.fftfreq(n, d=1.0 / (n - 1)); FX, FY = np.meshgrid(fr, fr, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2).reshape(-1); A = A.reshape(-1); A[0] = 0
    return float((R * A).sum() / A.sum())


out = {"config": mp.CONFIG}
U, z_true, coords, _ = mp.build_snapshots(mp.N)
U_tr = np.asarray(U[:mp.N_TRAIN]); n = mp.N
Zt = np.asarray(z_true[:mp.N_TRAIN])
c1, c5 = nn_corr(U_tr, Zt)
out["fields_in_true_z"] = {"nn1_corr": c1, "nn5_corr": c5, "spec_centroid": centroid(U_tr, n)}
print("fields (true z):", out["fields_in_true_z"], flush=True)


def resid_after(stages, Z):
    return U_tr - np.asarray(mp.predict_all(stages, jnp.asarray(Z), coords))


p = os.path.join(sys.argv[1], "ms_parametric_stages.pkl")
if os.path.exists(p):
    d = pickle.load(open(p, "rb")); check_cfg(d["config"])
    st = mp.stages_from_np(d["stages"])
    for k in range(1, len(st) + 1):
        E = resid_after(st[:k], Zt); c1, c5 = nn_corr(E, Zt)
        out[f"truez_resid_after_{k}"] = {"nn1_corr": c1, "nn5_corr": c5,
            "spec_centroid": centroid(E, n),
            "global_rel": float(np.linalg.norm(E) / np.linalg.norm(U_tr))}
        print(f"true-z resid after {k}:", out[f"truez_resid_after_{k}"], flush=True)

for p in sorted(glob.glob(os.path.join(sys.argv[2], "ms_autodecoder_K*_stages.pkl"))):
    d = pickle.load(open(p, "rb")); check_cfg(d["config"])
    K = d["config"]["K_LAT"]; st = mp.stages_from_np(d["stages"]); Zl = d["z_tr"]
    out[f"autodec_K{K}_fields_in_latent"] = {"nn1_corr": nn_corr(U_tr, Zl)[0],
                                             "nn5_corr": nn_corr(U_tr, Zl)[1]}
    print(f"autodec K={K} fields in latent space:", out[f"autodec_K{K}_fields_in_latent"], flush=True)
    for k in range(1, len(st) + 1):
        E = resid_after(st[:k], Zl)
        c1z, c5z = nn_corr(E, Zt); c1l, c5l = nn_corr(E, Zl)
        out[f"autodec_K{K}_resid_after_{k}"] = {"nn1_corr_in_true_z": c1z, "nn5_corr_in_true_z": c5z,
            "nn1_corr_in_latent": c1l, "nn5_corr_in_latent": c5l, "spec_centroid": centroid(E, n),
            "global_rel": float(np.linalg.norm(E) / np.linalg.norm(U_tr))}
        print(f"autodec K={K} resid after {k}:", out[f"autodec_K{K}_resid_after_{k}"], flush=True)

json.dump(out, open("ms_diag_report.json", "w"), indent=2)
print("wrote ms_diag_report.json")
