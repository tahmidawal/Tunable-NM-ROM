import os, sys, numpy as np
sys.path.insert(0, "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder")
import jax, jax.numpy as jnp
import b1d_common as b1
N, K, R, M = 256, 8, 32, 32
ck = "/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/runs/b1dqf/b1ds_n256/out/sep_b1d_scale_n256.pkl"
params, Z_tr, cfg = b1.load_pkl(ck)
interior = b1.interior_indices_1d(N); xi = b1.grid_coords_1d(N)[interior]
G = np.asarray(b1.features(params, jnp.asarray(xi))); kx, Phi, lam = b1.test_modes_1d(N, M)
dx = 1.0/(N-1); ni = G.shape[0]
H = np.asarray(b1.head(params, jnp.asarray(Z_tr[:2048]))); Udec = H @ G.T
def Dm(U): Up=np.pad(U,((0,0),(1,1))); return (Up[:,1:-1]-Up[:,:-2])/dx
def Dp(U): Up=np.pad(U,((0,0),(1,1))); return (Up[:,2:]-Up[:,1:-1])/dx
def Dc(U): return 0.5*(Dm(U)+Dp(U))
def Nup(U): return U*np.where(U>0, Dm(U), Dp(U))
ref = Nup(Udec) @ Phi
def relrows(A,B): r=np.linalg.norm(A-B,axis=1)/np.linalg.norm(B,axis=1); return f"mean {r.mean():.2e} max {r.max():.2e}"
S = np.sin(np.pi*np.outer(xi[:,0], np.arange(1, ni+1))); S/=np.linalg.norm(S,axis=0,keepdims=True)
C = S.T @ G
print("Nonlinear-term error of the TUCKER re-expression G~S_P C (D^- stencil, exact on the re-expressed bank) vs oracle Phi^T N_up(G h):")
for P in (16,24,32,48,64,96,128):
    Gp = S[:,:P]@C[:P]; U = H@Gp.T
    print(f"  P={P:4d}: {relrows((U*Dm(U))@Phi, ref)}")
print("Modified-FOM tensors on the LEARNED bank (all exact tables, different stencil):")
print(f"  centered  u*D^c u        : {relrows((Udec*Dc(Udec))@Phi, ref)}")
print(f"  flux D^-(u^2/2)          : {relrows(Dm(0.5*Udec**2)@Phi, ref)}")
print(f"  flux D^c(u^2/2)          : {relrows(Dc(0.5*Udec**2)@Phi, ref)}")
al = float(np.max(np.abs(Udec)))
LF = Dc(0.5*Udec**2) - 0.5*al*dx*(Dp(Udec)-Dm(Udec))   # Rusanov/global LF: quadratic + linear (D+ - D-)= dx * second diff
print(f"  global Lax-Friedrichs a={al:.2f}: {relrows(LF@Phi, ref)}")
# magnitude of the nonlinear term relative to the full residual scale: compare to nu*lap term
lap = (Dp(Udec)-Dm(Udec))/dx
for nu in (0.01, 0.03, 0.1):
    print(f"  nu={nu}: ||Phi^T N|| / ||nu Phi^T lap|| = {np.mean(np.linalg.norm(ref,axis=1)/np.linalg.norm(nu*lap@Phi,axis=1)):.2f}")
# NNLS-32 analog: what is the residual-term error of the base_tight rule? reported eq_rel_fit 5.5e-2 (Frobenius over fit states)
# Sparse analytic core with CENTERED stencil on the sine basis: nonzeros per (a,b)
P=48; Sp=S[:,:P]; Tc=np.einsum("xi,xj,xk->ijk", Phi, Sp, Dc(Sp.T).T); mx=np.abs(Tc).max()
nz = np.abs(Tc)>1e-10*mx
print(f"centered sine core P={P}: nonzeros {nz.sum()} of {Tc.size}; max nonzeros per (a,b) slice over i: {nz.sum(0).max()}")
