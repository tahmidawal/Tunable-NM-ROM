import os, sys, numpy as np
sys.path.insert(0,'/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder')
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import b1d_common as b1, b1d_fast_common as fc, b1d_tensor_common as tc
N=int(os.environ['N'])
ck=f'/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-29-b1d-tensor/experiments/separable-decoder/runs/b1dqf/b1ds_n{N}/out/sep_b1d_scale_n{N}.pkl'
su=fc.Setup(ck,N); dx=su.dx; G=np.asarray(su.G_int); Phi=np.asarray(su.Phi_j)
T=tc.build_T(Phi,G,dx); Q=tc.symmetrize(T)
Z=np.asarray(su.Z_tr); H=np.asarray(su.h_fn(jnp.asarray(Z))); U=H@G.T
S=len(U)
Up=np.pad(U,((0,0),(1,1)))
c=Up[:,1:-1]; bwd=(c-Up[:,:-2])/dx; fwd=(Up[:,2:]-c)/dx
Nup=c*np.where(c>0,bwd,fwd); Nbk=c*bwd
lap=(Up[:,2:]-2*c+Up[:,:-2])/dx**2
q_or=Nup@Phi; q_T=0.5*np.einsum('ijk,sj,sk->si',Q,H,H)
# (1) identity: tensor - oracle = -dx Phi^T[1[u<=0] u lap]
pred=-dx*((c<=0)*c*lap)@Phi
print(f"N={N} identity |(q_T-q_or) - pred| max = {np.max(np.abs(q_T-q_or-pred)):.2e} (scale |q| max {np.abs(q_or).max():.2e}); alg identity max {np.max(np.abs(q_T-Nbk@Phi)):.2e}")
nq=np.linalg.norm(q_or,axis=1)+1e-300
mis=np.linalg.norm(q_T-q_or,axis=1)/nq
neg=(U<=0).any(1)
delta=np.maximum(-U.min(1),0)
# bound: |dq_i| <= dx*delta*sum_{u<=0}|Phi_xi||lap_x| ; and a delta^2/W model
B1=dx*delta[:,None]*(np.abs((c<=0)*lap)@np.abs(Phi)); b1n=np.linalg.norm(B1,axis=1)/nq
# what is lap at undershoot points? split lap into 'own' (undershoot bump curvature ~ delta/W^2) vs actual
lap_neg=np.where(c<=0,np.abs(lap),0)
W=(c<=0).sum(1)*dx  # total undershoot width (physical), crude
m=neg&(delta>1e-6)
print(f"  states with u<=0: {neg.sum()}/{S}; delta median {np.median(delta[m]):.2e} max {delta.max():.2e}; undershoot points/state median {np.median((c<=0).sum(1)[m]):.0f}")
print(f"  mismatch rel med {np.median(mis[m]):.2e} max {mis.max():.2e}; first-order bound b1 med {np.median(b1n[m]):.2e} max {b1n.max():.2e}; ratio mis/bound med {np.median(mis[m]/b1n[m]):.2f}")
print(f"  |lap| at u<=0 points: median {np.median(lap_neg[m][lap_neg[m]>0]):.2e} max {lap_neg.max():.2e}; compare delta/W^2 (bump-curvature model) median {np.median(delta[m]/np.maximum(W[m],dx)**2):.2e}; |lap| at u>0 points median {np.median(np.abs(lap[U>0])):.2e}")
sl=np.polyfit(np.log(delta[m]),np.log(mis[m]),1)[0]; print(f"  log-log slope mismatch vs delta: {sl:.2f}")
# scaling with dx: rel mismatch ~ delta * dx * lap / (u u_x) -> report dx*|lap|max
# (3) f32 tensor
Phi32=Phi.astype(np.float32); G32=G.astype(np.float32)
Gm=np.concatenate([np.zeros((1,G32.shape[1]),np.float32),G32[:-1]]); DG32=((G32-Gm)/np.float32(dx)).astype(np.float32)
T32=np.zeros(T.shape,np.float32)
for s in range(0,len(G32),256):
    e=min(s+256,len(G32)); prod=(G32[s:e,:,None]*DG32[s:e,None,:]).reshape(e-s,-1)
    T32+=(Phi32[s:e].T@prod).reshape(T.shape)
Q32=(T32+T32.swapaxes(1,2)).astype(np.float32); H32=H.astype(np.float32)
q32=0.5*np.einsum('ijk,sj,sk->si',Q32,H32,H32)   # f32 build + f32 contraction
q64b32=0.5*np.einsum('ijk,sj,sk->si',Q.astype(np.float32),H32,H32)  # f64 build, f32 contract
q32b64=0.5*np.einsum('ijk,sj,sk->si',T32.astype(np.float64)+T32.swapaxes(1,2).astype(np.float64),H,H)
def relv(a,b): return np.linalg.norm(a-b,axis=1)/(np.linalg.norm(b,axis=1)+1e-300)
for lab,qq in (("f32 build+f32 contract",q32),("f64 build, f32 contract",q64b32),("f32 build, f64 contract",q32b64)):
    r=relv(qq.astype(np.float64),q_T); print(f"  [f32 test] {lab}: q rel err med {np.median(r):.1e} p95 {np.quantile(r,.95):.1e} max {r.max():.1e}  (oracle-vs-tensor mismatch med {np.median(mis):.1e} for scale)")
print(f"  T rel err f32 build: {np.max(np.abs(T32-T))/np.max(np.abs(T)):.1e}; |T| max {np.abs(T).max():.2e}; sum|T h h|/|q| median {np.median(np.einsum('ijk,sj,sk->si',np.abs(T),np.abs(H),np.abs(H))/(np.abs(q_T)+1e-300)):.1e}")
# (2) sensitivity: field change per unit latent; sigma_min(J) for residual->latent propagation
r_or=su.make_full_rw(); rJ=jax.jit(lambda z,p,nu:(r_or(z,p,nu),jax.jacfwd(r_or)(z,p,nu)))
dudz=jax.jit(jax.jacfwd(lambda z: su.G_int@su.h_fn(z)))
rng=np.random.default_rng(1); sens=[]; prop=[]
for i in rng.integers(0,S,24):
    z=jnp.asarray(Z[i]); Jz=np.asarray(dudz(z)); un=np.linalg.norm(U[i])
    smax=np.linalg.svd(Jz,compute_uv=False)[0]; sens.append(smax/un)
    zp=Z[rng.integers(S)]; nu=float(np.exp(rng.uniform(np.log(.01),np.log(.1))))
    r,J=[np.asarray(v) for v in rJ(z,su.prev_of(jnp.asarray(zp)),nu)]
    sv=np.linalg.svd(J,compute_uv=False); prop.append((np.linalg.norm(r)/sv[-1], sv[0]/sv[-1]))
sens=np.array(sens); prop=np.array(prop)
print(f"  field sensitivity sigma_max(du/dz)/|u|: med {np.median(sens):.2f} max {sens.max():.2f} -> latent dev 2e-4 => rel field change up to {2e-4*sens.max():.1e} (N=128 case), 5e-6 => {5e-6*sens.max():.1e}")
print(f"  residual->latent: |r|/sigma_min(J) med {np.median(prop[:,0]):.2e} max {prop[:,0].max():.2e}; cond(J) med {np.median(prop[:,1]):.1e} max {prop[:,1].max():.1e}; x 1e-4 rel residual perturbation => latent {1e-4*np.median(prop[:,0]):.1e} (med)")
print(f"  trust radius {su.TR_DELTA:.4f}; latent scale |z| med {np.median(np.linalg.norm(Z,axis=1)):.3f}")
