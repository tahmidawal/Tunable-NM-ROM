"""Calibration of the continuum target (DESIGN 5.1), local GB10, 256^2 tests: Gauss p^2 tested advection of the
Gauss-200 least-squares fits of the six dev6 initial bumps (the narrowest states), against the finest p and the flux form.
    python calib_gauss.py <acc|fast> <p1,p2,...>   (last p = reference)"""
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, time, json
import qcore as Q
L=256; mdl=Q.Model(); t0=time.time()
setting=sys.argv[1]; ps=[int(x) for x in sys.argv[2].split(',')]
Rp,M=Q.SETTINGS[setting]['Rp'],Q.SETTINGS[setting]['M']
kx,ky,lam=Q.H.modes_lean(L,M)
# states: Gauss-200 LS fits of dev6 initial fields (t=0 Gaussian bumps, the narrowest early states)
X,wq=Q.offmesh_rule('gauss200'); G,_,_=mdl.values_grads(X,Rp); G=np.asarray(G)
ph=Q.cohort('dev6'); Ws=[]
for c in range(6):
    ui=np.asarray(Q.e.sample_field(jnp.asarray(Q.e.initial(256,ph[c])),jnp.asarray(X),256)); sw=np.sqrt(wq)
    Ws.append(np.linalg.lstsq(G*sw[:,None],ui*sw,rcond=None)[0])
Ws=jnp.asarray(np.array(Ws)); del G
def tested(p,form):
    X,wq=Q.offmesh_rule(f'gauss{p}'); acc=0
    for s in range(0,len(X),16384):
        Xs,ws=X[s:s+16384],wq[s:s+16384]
        d=Q.offmesh_data(mdl,Rp,Xs,ws,L,kx,ky,form)
        acc=acc+np.asarray(jax.vmap(Q.tested_value(form,L),in_axes=(0,None))(Ws,d))
    return acc
res={}
for p in ps:
    res[p]=(tested(p,'point'),tested(p,'flux')); print(setting,p,'done',round(time.time()-t0),flush=True)
ref=res[ps[-1]][0]
rel=lambda a,b: float(np.max(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)))
for p in ps:
    print(setting,p,'point_vs_last',rel(res[p][0],ref),'point_vs_flux',rel(res[p][0],res[p][1]),flush=True)
