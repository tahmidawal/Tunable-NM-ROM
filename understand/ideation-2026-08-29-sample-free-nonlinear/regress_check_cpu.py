import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder")
import numpy as np, jax, jax.numpy as jnp
import b1d_common as b1
N=256; M=32; R=32; K=8
params, Z_tr, cfg = b1.load_pkl("/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/runs/b1dqf/b1ds_n256/out/sep_b1d_scale_n256.pkl")
interior = b1.interior_indices_1d(N); coords = b1.grid_coords_1d(N)[interior]
G = np.asarray(b1.features(params, jnp.asarray(coords)))
kx, Phi, lam = b1.test_modes_1d(N, M)
H_all = np.asarray(b1.head(params, jnp.asarray(Z_tr)))
print("Z_tr", Z_tr.shape, "H rank (svd of centered H, sv ratio at 8/9/16):")
s = np.linalg.svd(H_all - H_all.mean(0), compute_uv=False); print(np.round(s[:20]/s[0], 5))
upw = jax.jit(jax.vmap(lambda u: b1.upwind_adv_field_1d(u, N)))
def targets(H): return np.asarray(upw(jnp.asarray(H @ G.T))) @ Phi     # (S, M)
iu = np.triu_indices(R)
def feats(H): return np.einsum("sj,sk->sjk", H, H)[:, iu[0], iu[1]]        # (S, 528)
rng = np.random.default_rng(1)
idx = rng.choice(len(Z_tr), 4000, replace=False)
Hz = H_all[idx]                                                          # on-manifold
Hh = Hz + 0.1*np.std(H_all)*rng.standard_normal(Hz.shape)                # h-space perturbations
def fit(Hs):
    X=feats(Hs); Y=targets(Hs); W,*_=np.linalg.lstsq(X,Y,rcond=None); return W, np.linalg.norm(X@W-Y)/np.linalg.norm(Y)
Wz, fz = fit(Hz); Wh, fh = fit(Hh)
print(f"fit rel resid: z-only {fz:.2e}  h-pert {fh:.2e}")
tests = {"z-test": H_all[rng.choice(len(Z_tr),500,replace=False)],
         "z+0.02 sigma": None, "h-pert 0.1": None}
Zt = Z_tr[rng.choice(len(Z_tr),500,replace=False)]
tests["z+0.02 sigma"] = np.asarray(b1.head(params, jnp.asarray(Zt + 0.02*rng.standard_normal(Zt.shape))))
tests["h-pert 0.1"] = H_all[rng.choice(len(Z_tr),500,replace=False)] + 0.1*np.std(H_all)*rng.standard_normal((500,R))
for name,Ht in tests.items():
    X=feats(Ht); Y=targets(Ht)
    for lab,W in (("z-only",Wz),("h-pert",Wh)):
        e = np.linalg.norm(X@W-Y,axis=1)/np.linalg.norm(Y,axis=1)
        print(f"  eval {name:14s} model {lab:7s}: median {np.median(e):.2e} p95 {np.quantile(e,.95):.2e} max {e.max():.2e}")
