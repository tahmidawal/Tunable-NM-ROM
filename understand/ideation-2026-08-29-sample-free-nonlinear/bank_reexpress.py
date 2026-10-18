"""CPU-only diagnostic for the STRUCTURED BANKS angle (no GPU, no training).
Uses the committed N=256 K=8 R=32 checkpoint.
 (a) tensor exactness: Phi^T N_upwind(u) vs h^T T h with fixed D^- on decoded fields
 (b) sine / Chebyshev / B-spline re-expression error of the frozen bank G and of decoded fields
 (c) sine-P projection floor of the FOM truth (fixed trig bank ceiling) vs POD-P floor
 (d) sparsity of the analytic tensor for a pure sine bank with D^-
"""
import os, sys, numpy as np
sys.path.insert(0, "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder")
import jax, jax.numpy as jnp
import b1d_common as b1
N, K, R, M = 256, 8, 32, 32
ck = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/runs/b1dqf/b1ds_n256/out/sep_b1d_scale_n256.pkl"
params, Z_tr, cfg = b1.load_pkl(ck)
interior = b1.interior_indices_1d(N); xi = b1.grid_coords_1d(N)[interior]
G = np.asarray(b1.features(params, jnp.asarray(xi)))           # (254, 32)
kx, Phi, lam = b1.test_modes_1d(N, M)
dx = 1.0/(N-1); ni = G.shape[0]
print("bank G", G.shape, "Z_tr", Z_tr.shape)

# ---------- truth data: 64 train-family trajectories + 8 test (CPU tri Newton) ----
c,w,a,nu = b1.sample_params_1d(0, 64)
U0 = np.stack([b1.blob_ic_1d(N,c[i],w[i],a[i]) for i in range(64)])
roll = b1.make_rollout_1d_tri(N)
sn, worst = roll(jnp.asarray(U0), jnp.asarray(nu)); print("train64 worst res", float(worst))
Utr = np.asarray(sn)[:, :, interior].reshape(-1, ni)           # (64*51, 254)
c,w,a,nu_t = b1.sample_params_1d(1, 8)
U0 = np.stack([b1.blob_ic_1d(N,c[i],w[i],a[i]) for i in range(8)])
sn, worst = roll(jnp.asarray(U0), jnp.asarray(nu_t)); print("test8 worst res", float(worst))
Ute = np.asarray(sn)[:, :, interior].reshape(-1, ni)

def relL2(A, B):  # mean over rows of ||a-b||/||b||
    return float(np.mean(np.linalg.norm(A-B,axis=1)/(np.linalg.norm(B,axis=1)+1e-300)))

# decoder floor on test truth: fit z by least squares in Gram space? use projection onto span(G) (linear ceiling) + actual head fit not needed
P_G = G @ np.linalg.lstsq(G, Ute.T, rcond=None)[0]
print(f"[ceiling] projection of test truth onto span(G) (R=32): {relL2(P_G.T, Ute):.3e}   (decoder floor reported 3.73e-3)")

# ---------- (a) tensor exactness on decoded fields ------------------------------
H = np.asarray(b1.head(params, jnp.asarray(Z_tr[:2048])))       # (2048, 32)
Udec = H @ G.T
def Dm(U):  # backward difference with ghost zeros, on interior rows
    Up = np.pad(U, ((0,0),(1,1))); return (Up[:,1:-1]-Up[:,:-2])/dx
def Dp(U):
    Up = np.pad(U, ((0,0),(1,1))); return (Up[:,2:]-Up[:,1:-1])/dx
def Nup(U):
    return U*np.where(U>0, Dm(U), Dp(U))
Nu_true = Nup(Udec) @ Phi                                        # (S, M)
Nu_back = (Udec*Dm(Udec)) @ Phi
neg_frac = np.mean(Udec < 0); neg_min = Udec.min()
print(f"[sign] decoded train fields: frac points <0 = {neg_frac:.3e}, min u = {neg_min:.3e}, max u = {Udec.max():.3f}")
rel_rows = np.linalg.norm(Nu_true-Nu_back,axis=1)/np.linalg.norm(Nu_true,axis=1)
print(f"[tensor] rel diff Phi^T N_upwind vs Phi^T (u D^- u): mean {rel_rows.mean():.2e} max {rel_rows.max():.2e} median {np.median(rel_rows):.2e}")
# the same with the FOM truth fields (u>=0 exactly there)
print(f"[tensor] on FOM truth fields: max rel diff {np.max(np.linalg.norm(Nup(Utr)@Phi-(Utr*Dm(Utr))@Phi,axis=1)/np.linalg.norm(Nup(Utr)@Phi,axis=1)):.2e}")
# T table built once; check h^T T h == Phi^T(u D^- u)
T = np.einsum("xi,xj,xk->ijk", Phi, G, Dm(G.T).T)               # (M,R,R)
tt = np.einsum("sj,ijk,sk->si", H, T, H)
print(f"[tensor] h^T T h vs grid Phi^T(u D^- u): max rel {np.max(np.linalg.norm(tt-Nu_back,axis=1)/np.linalg.norm(Nu_back,axis=1)):.2e}")
# clipped-positive variant: N(u+) with u+ = max(u,0) -> would be exact; what's the error of using T (no clip) vs upwind on ROM fields at the negative points only
# flux form (u^2/2)_x with D^- : T_flux
Tf = 0.5*np.einsum("xi,xj,xk->ijk", Dp_Phi:=None or Phi, G, G) if False else None
Nflux = Dm(0.5*Udec**2) @ Phi
print(f"[flux] Phi^T D^-(u^2/2) vs Phi^T (u D^- u): mean rel {relL2(Nflux, Nu_back):.2e}  (they are different discretisations; FOM uses the non-conservative one)")

# ---------- (b) re-expression of the frozen bank in structured bases -------------
S = np.sin(np.pi*np.outer(xi[:,0], np.arange(1, ni+1)))         # full discrete sine basis (254 modes)
S /= np.linalg.norm(S,axis=0,keepdims=True)
coef = S.T @ G                                                   # exact (orthogonal) sine coefficients of each bank column
print("\n[sine re-expression of bank] P modes -> rel err of bank cols (worst col) and of decoded test fields")
Hte = np.linalg.lstsq(G, Ute.T, rcond=None)[0].T                # best h for test truth (ceiling-level)
for P in (16, 24, 32, 48, 64, 96, 128, 192):
    Gp = S[:,:P] @ coef[:P]
    colerr = np.linalg.norm(Gp-G,axis=0)/np.linalg.norm(G,axis=0)
    dec = Hte @ Gp.T
    print(f"  P={P:4d}: worst bank col {colerr.max():.2e} mean {colerr.mean():.2e} | decoded-test err vs truth {relL2(dec, Ute):.3e} | vs learned decode {relL2(dec, Hte@G.T):.3e}")
# Chebyshev on [0,1] (with bc mask factored out): fit g~ = G/bc by least squares in Chebyshev, multiply back by bc
bc = 4*xi[:,0]*(1-xi[:,0]); Gt = G/bc[:,None]
t = 2*xi[:,0]-1
print("[Chebyshev re-expression of g~ (bank/bc), bc re-applied]")
for P in (16, 32, 48, 64, 96):
    Cb = np.cos(np.arange(P)[None,:]*np.arccos(t)[:,None])
    Gp = bc[:,None]*(Cb @ np.linalg.lstsq(Cb, Gt, rcond=None)[0])
    colerr = np.linalg.norm(Gp-G,axis=0)/np.linalg.norm(G,axis=0)
    print(f"  P={P:4d}: worst bank col {colerr.max():.2e} | decoded-test err vs truth {relL2(Hte@Gp.T, Ute):.3e}")
# cubic B-splines with uniform knots (bandwidth 4 -> products local)
from scipy.interpolate import BSpline
print("[cubic B-spline re-expression of bank, uniform knots, boundary clamped]")
for nb in (16, 24, 32, 48, 64, 96, 128):
    kn = np.concatenate([[0]*3, np.linspace(0,1,nb-2), [1]*3])
    Bm = BSpline.design_matrix(xi[:,0], kn, 3).toarray()        # (254, nb)
    Bm = Bm[:,1:-1]                                              # drop the two boundary splines (bc=0)
    Gp = Bm @ np.linalg.lstsq(Bm, G, rcond=None)[0]
    colerr = np.linalg.norm(Gp-G,axis=0)/np.linalg.norm(G,axis=0)
    print(f"  nb={nb:4d} (dof {Bm.shape[1]}): worst bank col {colerr.max():.2e} | decoded-test err vs truth {relL2(Hte@Gp.T, Ute):.3e}")

# ---------- (c) fixed trig bank ceiling vs POD ------------------------------------
print("\n[ceilings on test truth] sine-P projection vs POD-P (POD from 64 train trajectories)")
Uc, sv, Vt = np.linalg.svd(Utr, full_matrices=False)
for P in (8, 16, 24, 32, 48, 64, 96, 128):
    es = relL2((Ute@S[:,:P])@S[:,:P].T, Ute)
    ep = relL2((Ute@Vt[:P].T)@Vt[:P], Ute)
    print(f"  P={P:4d}: sine-P floor {es:.3e}   POD-P floor {ep:.3e}")

# ---------- (d) sparsity of the analytic sine-bank tensor with D^- ----------------
print("\n[analytic tensor for a pure sine bank, D^- stencil]  T[i,j,k] = sum_x phi_i s_j D^- s_k")
for P in (32, 64):
    Sp = S[:,:P]; Ts = np.einsum("xi,xj,xk->ijk", Phi, Sp, Dm(Sp.T).T)
    mx = np.abs(Ts).max()
    for thr in (1e-2, 1e-4, 1e-8):
        print(f"  P={P}: entries > {thr:.0e}*max: {np.mean(np.abs(Ts)>thr*mx)*100:5.1f}%  (of {Ts.size})")
    # centered-difference / continuous-derivative version: how sparse?
    Dc = (np.pad(Sp,((1,1),(0,0)))[2:]-np.pad(Sp,((1,1),(0,0)))[:-2])/(2*dx)
    Tc = np.einsum("xi,xj,xk->ijk", Phi, Sp, Dc); mx=np.abs(Tc).max()
    print(f"  P={P}: CENTERED stencil entries > 1e-8*max: {np.mean(np.abs(Tc)>1e-8*mx)*100:5.1f}%")
