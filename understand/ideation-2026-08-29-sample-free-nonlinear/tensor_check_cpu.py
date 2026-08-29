# CPU-only, seconds: how wrong is the fixed backward-difference tensor vs the true
# sign-upwind projection Phi^T N(G h) at training codes and at LM-like perturbations?
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder")
import numpy as np, jax, jax.numpy as jnp
import b1d_common as b1
N=256; M=32; R=32; K=8
params, Z_tr, cfg = b1.load_pkl("/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/runs/b1dqf/b1ds_n256/out/sep_b1d_scale_n256.pkl")
interior = b1.interior_indices_1d(N); coords = b1.grid_coords_1d(N)[interior]
G = np.asarray(b1.features(params, jnp.asarray(coords)))          # (n_i, R)
kx, Phi, lam = b1.test_modes_1d(N, M)
dx = 1.0/(N-1)
h_fn = lambda z: np.asarray(b1.head(params, jnp.asarray(z)))
# backward difference of the bank (ghost zero on the left wall)
Gp = np.vstack([np.zeros((1,R)), G])
DmG = (Gp[1:] - Gp[:-1])/dx                                        # (n_i, R)
T = np.einsum("xi,xj,xk->ijk", Phi, G, DmG)                        # (M,R,R)
def true_proj(h):
    u = G @ h
    return Phi.T @ np.asarray(b1.upwind_adv_field_1d(jnp.asarray(u), N)), u
def tensor_proj(h):
    return np.einsum("ijk,j,k->i", T, h, h)
rng = np.random.default_rng(0)
rows=[]
for sig in (0.0, 0.05, 0.2, 0.5):
    errs=[]; negfrac=[]; negmass=[]
    for _ in range(200):
        z = Z_tr[rng.integers(len(Z_tr))] + sig*rng.standard_normal(K)
        h = h_fn(z); t,u = true_proj(h); q = tensor_proj(h)
        errs.append(np.linalg.norm(q-t)/np.linalg.norm(t))
        negfrac.append(np.mean(u<0)); negmass.append(np.linalg.norm(np.minimum(u,0))/np.linalg.norm(u))
    e=np.array(errs)
    print(f"sigma={sig:4.2f}: tensor-vs-upwind rel err median {np.median(e):.2e} p95 {np.quantile(e,.95):.2e} max {e.max():.2e} | "
          f"frac(u<0) mean {np.mean(negfrac):.3f} | neg-mass rel median {np.median(negmass):.1e} max {np.max(negmass):.1e}")
# Jacobian cosine: d/dz of Phi^T N vs tensor, chained through dh/dz
def jac_true(z):
    f = lambda zz: b1.Phi_dummy if False else None
jt = jax.jacfwd(lambda z: jnp.asarray(Phi).T @ b1.upwind_adv_field_1d(jnp.asarray(G) @ b1.head(params, z), N))
jq = jax.jacfwd(lambda z: jnp.einsum("ijk,j,k->i", jnp.asarray(T), b1.head(params,z), b1.head(params,z)))
cos=[]; jrel=[]
for _ in range(50):
    z = jnp.asarray(Z_tr[rng.integers(len(Z_tr))] + 0.05*rng.standard_normal(K))
    A=np.asarray(jt(z)); B=np.asarray(jq(z))
    cos.append(np.sum(A*B)/np.linalg.norm(A)/np.linalg.norm(B)); jrel.append(np.linalg.norm(A-B)/np.linalg.norm(A))
print(f"Jacobian (sigma=0.05): rel err median {np.median(jrel):.2e} max {np.max(jrel):.2e}; cos min {np.min(cos):.4f}")
# where does the tensor error come from? restrict to positive part
z = Z_tr[3]; h=h_fn(z); t,u=true_proj(h); q=tensor_proj(h)
mask = u>0
Nu = np.asarray(b1.upwind_adv_field_1d(jnp.asarray(u), N))
Nq = u*(DmG@h)
print("rel err on u>0 points only:", np.linalg.norm((Nu-Nq)[mask])/np.linalg.norm(Nu), " on u<=0 points:", np.linalg.norm((Nu-Nq)[~mask])/np.linalg.norm(Nu), " n_neg", (~mask).sum())
