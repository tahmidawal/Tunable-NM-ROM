"""CPU fixture for algebraically identical fixed normal-matrix reuse."""
from common import *
import geometry as geo, tensor_metric as TM, discrete_weak as W, packed_fields as PF
from build_local_cache import expand_matrix
rng=np.random.default_rng(83)
basis=np.zeros((2,8,8));basis[0,0,0]=2;basis[1,1,1]=2
rotation=np.linalg.qr(rng.normal(size=(36,2)))[0];cells=geo.uniform(1)
full,tr,_,_=TM.make_geometry(6)(jnp.asarray(basis),jnp.asarray(cells),*TM.rule(panels=16,order=24,N=16))
T=expand_matrix(jnp.asarray(rotation),len(cells),2);gram=T.T@full@T
cache,_=W.prepare(basis,cells,tr,rotation,64,64,gram)
cache={k:cache[k] for k in ['A','indices','lam','gram']}
cache['packed']=PF.prepare_stencil(basis,cells,tr,rotation,64,32)
cached=dict(cache,linear_normal=cache['A'].T@cache['A'])
standard,_=W.make_solver(PF.center,64,theta=.5,advection_tests=PF.tested_advection(64,32))
reused,_=W.make_solver(PF.center,64,theta=.5,advection_tests=PF.tested_advection(64,32),reuse_linear_normal=True)
c=jnp.zeros(len(gram)).at[0].set(.3).at[1].set(.01);d=c;defects=[]
for _ in range(3):
    c,s=standard(c,cache,.00125,.005,1e-8,1e-10)
    d,t=reused(d,cached,.00125,.005,1e-8,1e-10)
    delta=c-d;defect=float(jnp.sqrt(jnp.maximum(delta@gram@delta,0))/jnp.sqrt(c@gram@c));defects.append(defect)
    for v in [np.asarray(s),np.asarray(t)]:
        assert np.all(np.isfinite(v)) and v[1]<=1e-8 and (v[2]<=1e-10 or v[1]<=1e-10),v
assert max(defects)<1e-10,defects
print(json.dumps(dict(status='passed',max_relative_state_change=max(defects),steps=len(defects),
                     full_convergence_both=True,scope='CPU fixture only; no GPU timing or scientific admission'),indent=2))
