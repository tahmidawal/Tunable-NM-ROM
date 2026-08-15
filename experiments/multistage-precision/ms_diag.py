"""Diagnostic: WHY does staging stall in the parametric setting?

For the trained stage-0 decoder (true-z control and auto-decoder K variants):
residual e_0(x; z_i) = U_i - eps0*D0(x; z_i) over training samples.
  (a) smoothness in z: Pearson correlation of e_0(.; z_i) with e_0(.; z_j) for
      the nearest neighbour j in z-space (and mean over 5-NN); the same for
      the FIELDS U_i themselves as a reference (fields are smooth in z, so
      NN-corr ~ 1).  If residual NN-corr << field NN-corr the residual is
      rough in z -> a fresh z-conditioned net cannot fit it (no power law).
  (b) frequency in x: radial spectrum peak / centroid of e_0 vs of U.
  (c) per-sample fit ceiling: fraction of residual variance a stage-1 net
      would have to explain per sample = 1 (single-function staging works)
      vs the family stage-1 train loss recorded in the report.
Usage: python ms_diag.py <run_dir_parametric> <run_dir_autodec>
"""
import json, os, pickle, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ms_parametric as mp

print(f"jax_backend={jax.default_backend()}", flush=True)
U, z_true, coords = mp.build_snapshots(mp.N)
U_tr = np.asarray(U[:mp.N_TRAIN]); n = mp.N
Zt = np.asarray(z_true[:mp.N_TRAIN])

def nn_corr(E, Z, k=5):
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    Ec = E - E.mean(1, keepdims=True)
    Ec /= (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-300)
    c1 = np.array([Ec[i] @ Ec[idx[i, 0]] for i in range(len(E))])
    ck = np.array([np.mean([Ec[i] @ Ec[j] for j in idx[i]]) for i in range(len(E))])
    return float(c1.mean()), float(ck.mean())

def spec(E):
    A = np.abs(np.fft.fft2(E.reshape(-1, n, n))).mean(0)
    fr = np.fft.fftfreq(n, d=1.0 / (n - 1)); FX, FY = np.meshgrid(fr, fr, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2).reshape(-1); A = A.reshape(-1)
    return float((R * A).sum() / A.sum())

out = {}
out["fields"] = {"nn1_corr": nn_corr(U_tr, Zt)[0], "nn5_corr": nn_corr(U_tr, Zt)[1],
                 "spec_centroid": spec(U_tr)}
print("fields:", out["fields"], flush=True)

def resid_after(stages, Z):
    preds = []
    for s in range(0, len(Z), 64):
        preds.append(np.asarray(jax.vmap(lambda zi: mp.combined_apply(stages, zi, coords))(jnp.asarray(Z[s:s+64]))))
    return U_tr - np.concatenate(preds)

def load_stages(raw):
    return [{"params": jax.tree_util.tree_map(jnp.asarray, s["params"]), "n_freq": s["n_freq"], "eps": s["eps"]} for s in raw]

p = os.path.join(sys.argv[1], "ms_parametric_stages.pkl")
if os.path.exists(p):
    st = load_stages(pickle.load(open(p, "rb")))
    for k in range(1, len(st) + 1):
        E = resid_after(st[:k], Zt)
        c1, c5 = nn_corr(E, Zt)
        out[f"truez_resid_after_{k}"] = {"nn1_corr": c1, "nn5_corr": c5, "spec_centroid": spec(E),
                                         "rel_rms": float(np.sqrt((E**2).mean()) / np.sqrt((U_tr**2).mean()))}
        print(f"true-z resid after {k} stage(s):", out[f"truez_resid_after_{k}"], flush=True)

for p in sorted(__import__("glob").glob(os.path.join(sys.argv[2], "ms_autodecoder_K*_stages.pkl"))):
    d = pickle.load(open(p, "rb")); K = d["z_tr"].shape[1]
    st = load_stages(d["stages"]); Zl = d["z_tr"]
    # latent geometry: NN structure of learned latents vs true params
    for k in range(1, len(st) + 1):
        E = resid_after(st[:k], Zl)
        c1z, c5z = nn_corr(E, Zt); c1l, c5l = nn_corr(E, Zl)
        out[f"autodec_K{K}_resid_after_{k}"] = {"nn1_corr_in_true_z": c1z, "nn5_corr_in_true_z": c5z,
            "nn1_corr_in_latent": c1l, "nn5_corr_in_latent": c5l, "spec_centroid": spec(E),
            "rel_rms": float(np.sqrt((E**2).mean()) / np.sqrt((U_tr**2).mean()))}
        print(f"autodec K={K} resid after {k}:", out[f"autodec_K{K}_resid_after_{k}"], flush=True)
    out[f"autodec_K{K}_latent_vs_true"] = {"field_nn1_corr_in_latent": nn_corr(U_tr, Zl)[0]}
    print(f"autodec K={K} field NN-corr in latent space:", out[f"autodec_K{K}_latent_vs_true"], flush=True)

json.dump(out, open("ms_diag_report.json", "w"), indent=2)
print("wrote ms_diag_report.json")
