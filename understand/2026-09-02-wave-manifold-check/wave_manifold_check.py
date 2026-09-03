"""1D wave u_tt = c^2 u_xx, Dirichlet. Curved manifold u = g(z) (quadratic manifold on a POD bank,
i.e. u = G h(z) with h(z) = [z ; W q(z)] -- a separable decoder with a fixed bank).
Compare latent-stepping formulations:
  A  LSPG on the FOM's Newmark residual, carried decoded fields   (the 2026-08-16 formulation)
  B  Galerkin with J_g at the NEW time level, same residual        (08-16 'galerkin' arm)
  C  Variational (Lagrangian) manifold ROM, Stormer-Verlet:  J_g(z_n)^T [g(z_{n+1}) - 2 g(z_n) + g(z_{n-1}) - c^2 dt^2 L g(z_n)] = 0
  D  Variational midpoint (implicit, symmetric)
  P  POD-K Galerkin Newmark control (linear, same K)
Report: traj error vs FOM, oracle projection floor, energy ratio (central-difference velocity) over the horizon.
"""
import numpy as np, sys, time
from scipy.optimize import least_squares
np.random.seed(0)
N = 128; n = N-2; dx = 1.0/(N-1); c = 1.0
x = np.linspace(0,1,N)[1:-1]
L = (np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1),1) + np.diag(np.ones(n-1),-1))/dx**2   # symmetric, negative
Kst = -L                                                                                       # SPD stiffness
T = 1.0; nsnap = 50; dt_snap = T/nsnap
K = int(sys.argv[1]) if len(sys.argv)>1 else 6
RS = int(sys.argv[2]) if len(sys.argv)>2 else 20          # latent substeps per snapshot
dt = dt_snap/RS
ntrain = 60; ntest = 4
def ic(rng):
    x0 = rng.uniform(0.3,0.7); s = rng.uniform(0.05,0.10)
    return np.exp(-(x-x0)**2/(2*s**2))
# ---------------- FOM: CN on (u,v), 80 substeps, exact energy conservation --------------
def fom(u0, subs=80):
    h = dt_snap/subs; a = (c*h/2)**2
    A = np.eye(n) - a*L; B = np.eye(n) + a*L
    Ainv = np.linalg.inv(A)
    u = u0.copy(); v = np.zeros(n); S=[u.copy()]
    for k in range(nsnap):
        for _ in range(subs):
            # CN: u1 = u + h/2 (v+v1), v1 = v + h/2 c^2 L (u+u1)  ->  (I - a L) u1 = (I + a L) u + h v
            u1 = Ainv @ (B@u + h*v)
            v1 = v + (h*c**2/2)*(L@(u+u1))
            u, v = u1, v1
        S.append(u.copy())
    return np.array(S)
rng = np.random.default_rng(1)
Utr = np.array([fom(ic(rng)) for _ in range(ntrain)])   # (ntrain, 51, n)
rng2 = np.random.default_rng(7)
Ute = np.array([fom(ic(rng2)) for _ in range(ntest)])
X = Utr.reshape(-1, n)
# ---------------- POD bank + quadratic manifold u = V z + W q(z) -----------------------
U_, s_, Vt_ = np.linalg.svd(X, full_matrices=False)
V = Vt_[:K].T
iu = np.triu_indices(K)
def q(z): return np.outer(z,z)[iu]
def dq(z):
    D = np.zeros((len(iu[0]), K))
    for m,(i,j) in enumerate(zip(*iu)):
        D[m,i] += z[j]; D[m,j] += z[i]
    return D
Z = X @ V
Q = np.array([q(z) for z in Z])
Res = X - Z @ V.T
Wt = np.linalg.lstsq(Q, Res, rcond=None)[0]     # (nq, n)
W = Wt.T
# make W orthogonal to V (standard) so the bank is [V, W-part]
W = W - V @ (V.T @ W)
def g(z): return V@z + W@q(z)
def Jg(z): return V + W@dq(z)
# oracle projection floor on the test set
def proj(u, z0):
    r = least_squares(lambda z: g(z)-u, z0, jac=lambda z: Jg(z), method='lm', xtol=1e-12, ftol=1e-12)
    return r.x
def traj_err(S, Sref):
    return np.sqrt(np.mean(np.sum((S-Sref)**2,axis=1)))/np.sqrt(np.mean(np.sum(Sref**2,axis=1)))
floors=[]; podfl=[]
for i in range(ntest):
    P=[]; 
    for k in range(nsnap+1):
        P.append(g(proj(Ute[i,k], Ute[i,k]@V)))
    floors.append(traj_err(np.array(P), Ute[i]))
    podfl.append(traj_err((Ute[i]@V)@V.T, Ute[i]))
print(f"K={K} RS={RS} dt={dt:.1e}  quad-manifold oracle floor {np.mean(floors):.3e}   POD-{K} floor {np.mean(podfl):.3e}")
# ---------------- energy helpers -------------------------------------------------------
def energy(u, v): return 0.5*v@v + 0.5*c**2*(u@(Kst@u))
def energy_ratio_cd(Ufine):   # central-difference velocity from the substep trajectory
    E=[]
    for k in range(1,len(Ufine)-1):
        v=(Ufine[k+1]-Ufine[k-1])/(2*dt); E.append(energy(Ufine[k],v))
    return E[-1]/E[0], np.max(np.abs(np.array(E)/E[0]-1))
a = (c*dt/2)**2
def newton(F, J, z0, tol=1e-11, it=50):
    z=z0.copy()
    for _ in range(it):
        f=F(z); 
        if np.linalg.norm(f)<tol: break
        z = z - np.linalg.solve(J(z), f)
    return z
def run_A(z0):   # LSPG on Newmark residual, carried decoded fields (08-16)
    u0=g(z0); um=u0.copy(); un=u0.copy(); z=z0.copy(); Uf=[u0]
    for k in range(nsnap*RS):
        if k==0:
            R=lambda zz: (g(zz)-u0) - a*(L@(g(zz)+u0)); Jr=lambda zz: (np.eye(n)-a*L)@Jg(zz)
        else:
            R=lambda zz: (g(zz)-2*un+um) - a*(L@(g(zz)+2*un+um)); Jr=lambda zz: (np.eye(n)-a*L)@Jg(zz)
        z = least_squares(R, z, jac=Jr, method='lm', xtol=1e-13, ftol=1e-13, gtol=1e-13).x
        um, un = un, g(z); Uf.append(un)
    return np.array(Uf)
def run_B(z0):   # Galerkin, J_g at the new time
    u0=g(z0); um=u0.copy(); un=u0.copy(); z=z0.copy(); Uf=[u0]
    for k in range(nsnap*RS):
        if k==0: R=lambda zz: (g(zz)-u0) - a*(L@(g(zz)+u0))
        else:    R=lambda zz: (g(zz)-2*un+um) - a*(L@(g(zz)+2*un+um))
        F=lambda zz: Jg(zz).T@R(zz)
        J=lambda zz: Jg(zz).T@((np.eye(n)-a*L)@Jg(zz))     # Gauss-Newton-ish (drops second derivative term)
        z = newton(F,J,z)
        um, un = un, g(z); Uf.append(un)
    return np.array(Uf)
def run_C(z0):   # variational Verlet on the pulled-back Lagrangian; carried latents
    zm=z0.copy(); zn=z0.copy(); Uf=[g(z0)]
    # first step from v0=0: symmetric start  z_1 = z_{-1}  ->  J^T[2g(z1) - 2g(z0) - c^2 dt^2 L g(z0)]=0
    Jn=Jg(zn); gn=g(zn); f=c**2*dt**2*(L@gn)
    F=lambda zz: Jn.T@(2*g(zz)-2*gn-f); J=lambda zz: 2*Jn.T@Jg(zz)
    z1=newton(F,J,zn); zm,zn=zn,z1; Uf.append(g(zn))
    for k in range(1,nsnap*RS):
        Jn=Jg(zn); gn=g(zn); gm=g(zm); f=c**2*dt**2*(L@gn)
        F=lambda zz: Jn.T@(g(zz)-2*gn+gm-f); J=lambda zz: Jn.T@Jg(zz)
        z1=newton(F,J,2*zn-zm); zm,zn=zn,z1; Uf.append(g(zn))
    return np.array(Uf)
def run_D(z0):   # variational midpoint: L_d = dt[ 1/2 |(g(z1)-g(z0))/dt|^2 - V(g((z0+z1)/2)) ]
    def gradV(u): return c**2*(Kst@u)
    zm=z0.copy(); zn=z0.copy(); Uf=[g(z0)]
    def step(zm,zn,first):
        Jn=Jg(zn); gn=g(zn); gm=g(zm)
        if first:
            def F(zz):
                zh=(zn+zz)/2
                return Jn.T@(2*gn-2*g(zz))/dt - dt*Jg(zh).T@gradV(g(zh))
        else:
            zhm=(zm+zn)/2; fm=Jg(zhm).T@gradV(g(zhm))
            def F(zz):
                zh=(zn+zz)/2
                return Jn.T@((gn-gm)-(g(zz)-gn))/dt - (dt/2)*(fm + Jg(zh).T@gradV(g(zh)))
        def J(zz):
            zh=(zn+zz)/2
            return -Jn.T@Jg(zz)/dt*(2 if first else 1) - (dt/2)*(2 if first else 1)*0.5*Jg(zh).T@(c**2*Kst@Jg(zh))
        return newton(F,J,2*zn-zm if not first else zn)
    z1=step(zm,zn,True); zm,zn=zn,z1; Uf.append(g(zn))
    for k in range(1,nsnap*RS):
        z1=step(zm,zn,False); zm,zn=zn,z1; Uf.append(g(zn))
    return np.array(Uf)
def run_P(u0):   # POD-K Galerkin Newmark (linear control)
    Lr=V.T@L@V; Ar=np.eye(K)-a*Lr; Br=np.eye(K)+a*Lr
    z0=V.T@u0; zm=z0.copy(); zn=z0.copy(); Uf=[V@z0]
    z1=np.linalg.solve(Ar, Br@zn); zm,zn=zn,z1; Uf.append(V@zn)
    for k in range(1,nsnap*RS):
        z1=np.linalg.solve(Ar, 2*Br@zn - Ar@zm - 2*zn + 2*zn)   # (z1 - 2zn + zm) - a Lr (z1 + 2 zn + zm) = 0
        z1=np.linalg.solve(Ar, (2*np.eye(K)+2*a*Lr)@zn - Ar@zm)
        zm,zn=zn,z1; Uf.append(V@zn)
    return np.array(Uf)
runs={'A lspg-newmark(08-16)':run_A,'B galerkin-newtime':run_B,'C variational-verlet':run_C,'D variational-midpoint':run_D}
res={k:[] for k in list(runs)+['P pod-galerkin']}
for i in range(ntest):
    u0=Ute[i,0]; z0=proj(u0, u0@V)
    for name,fn in runs.items():
        t0=time.time(); Uf=fn(z0); 
        er=traj_err(Uf[::RS], Ute[i]); Er,Ed=energy_ratio_cd(Uf)
        res[name].append((er,Er,Ed)); 
    Uf=run_P(u0); er=traj_err(Uf[::RS],Ute[i]); Er,Ed=energy_ratio_cd(Uf); res['P pod-galerkin'].append((er,Er,Ed))
print(f"{'arm':28s} {'traj err':>10s} {'E_final/E_0':>12s} {'max|dE/E|':>10s}   (means over {ntest} held-out)")
for k,v in res.items():
    v=np.array(v); print(f"{k:28s} {v[:,0].mean():10.3e} {v[:,1].mean():12.4f} {v[:,2].mean():10.2e}")
print(f"{'oracle manifold floor':28s} {np.mean(floors):10.3e}")
print(f"{'POD-K projection floor':28s} {np.mean(podfl):10.3e}")
