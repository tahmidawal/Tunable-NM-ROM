"""Sub-minute toy coverage of training coordinate conversion and strict weak stepping."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import engines as e
import accuracy_paths as ap

start=time.perf_counter();assert jax.default_backend()=='gpu'
A=jnp.asarray([[1.,2.],[3.,-1.],[.5,1.]]);y=jnp.array([1.,2.,-.3])
lm=ap.make_stationary_lm(lambda z:A@z-y,2,50)
v=lm(jnp.zeros(2),(),0.);ref=np.linalg.lstsq(np.asarray(A),np.asarray(y),rcond=None)[0]
assert np.linalg.norm(np.asarray(v[0])-ref)<1e-7 and float(v[4])<1e-6 and int(v[3])==4
p=e.sc.init_separable(jax.random.PRNGKey(20),4,16,n_ff=4,g_hidden=16,g_layers=1,h_hidden=16,h_layers=1)
Z=np.random.default_rng(1).normal(size=(32,4));cfg=dict(mesh=8,physical_seed=0,cases=4,seed=2,code_fit_budget=3,steps=3,lr=1e-4,batch=4,replay_cases=8,replay_weight=1.)
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
    p2,Z2,info=ap.train_head(p,Z,cfg,Path(tmp),lambda x:None)
assert Z2.shape==(36,4) and info['unwhitening_relative_parity']<1e-12
L=12;data,inf=e.build_rom(p,Z,L,16,64,candidate_cap=100,fit_states=4)
cold,_=e.build_gauss_cold(p,data[7],axis_points=6);q=ap.make_rom(p,L,.005,inf['trust_radius'],ic_budget=5,step_budget=3)
u=e.initial(L,e.params_draw(4,1)[0]);value=jax.tree_util.tree_map(np.asarray,q(jnp.asarray(u),.03,data,cold))
assert value[0].shape==(6,13,13) and np.isfinite(value[0]).all() and value[7].shape==(51,4)
z=jnp.asarray(value[7][1]);prev=data[1]@e.sc.head(p,jnp.asarray(value[7][0]));f=lambda z:e.weak(z,prev,.03,data,p,L,.005);r=f(z);J=jax.jacfwd(f)(z)
gn=float(jnp.linalg.norm(J.T@r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300));assert abs(gn-value[8][0])<1e-12
dg,icdg=jax.tree_util.tree_map(np.asarray,ap.make_diagnostics(p,L,.005)(jnp.asarray(value[7]),jnp.asarray(u),.03,data,cold))
assert np.max(abs(dg-value[8]))<1e-12 and abs(icdg-value[9])<1e-12
result=dict(passed=True,backend=jax.default_backend(),x64=jax.config.jax_enable_x64,quadratic_stationarity=float(v[4]),unwhitening_relative_parity=info['unwhitening_relative_parity'],weak_stationarity_delta=abs(gn-value[8][0]),seconds=time.perf_counter()-start)
(Path(__file__).parent/'smoke-accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
