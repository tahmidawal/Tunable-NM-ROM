"""Bounded GPU parity and evolving-state check, no training."""
import json
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import core as c

ck=pickle.loads(Path('inputs/refined_checkpoint.pkl').read_bytes())
p=jax.tree_util.tree_map(jnp.asarray,ck['params']);Z=np.asarray(ck['Z_tr'])
assert jax.default_backend()=='gpu'
gates=c.verify(p,Z)
n=9;k=32;q=4
xyz=c.b3.grid_coords_3d(n)[c.b3.interior_indices_3d(n)]
G=np.asarray(c.b3.features(p,jnp.asarray(xyz)));Q,R=np.linalg.qr(G,mode='reduced')
_,_,_,Phi,lam=c.b3.test_modes_3d(n,160)
C=np.eye(128)[:,:q];H=np.asarray(c.b3.head(p,jnp.asarray(Z[:32])))@R.T
data=c.data_for_basis(Q,Phi,lam,C,R,Z[:32],H)
hp={key:p[key] for key in ['h','h_lin']}
u0=Q@(R@np.asarray(c.b3.head(hp,jnp.asarray(Z[0]))))
query=c.make_query(n,k,q,False,steps=2,fit_budget=20,step_budget=20)
out=jax.tree_util.tree_map(np.asarray,query(jnp.asarray(u0),.05,data,hp))
assert np.isfinite(out[0]).all()
evolution=float(np.linalg.norm(out[0][2]-out[0][1])/np.linalg.norm(out[0][1]))
assert evolution>1e-8
coeff=np.asarray(c.b3.head(hp,jnp.asarray(out[1][:,:k])))@R.T+out[1][:,k:]@C.T
parity=float(np.linalg.norm(coeff@Q.T-out[0])/np.linalg.norm(out[0]))
assert parity<1e-11
gates.update(evolution=evolution,decoded_state_parity=parity,iterations=out[2].tolist(),
             reasons=out[3].tolist(),gradients=out[4].tolist(),actual_M=len(lam))
lin_data=c.data_for_basis(Q[:,:8],Phi,lam,np.zeros((8,0)),R,Z[:32],H)
lin_query=c.make_query(n,0,8,True,steps=2,fit_budget=20,step_budget=20)
lin=jax.tree_util.tree_map(np.asarray,lin_query(jnp.asarray(u0),.05,lin_data,hp))
assert np.isfinite(lin[0]).all()
linear_ic=float(np.linalg.norm(lin[0][0]-Q[:,:8]@(Q[:,:8].T@u0))/np.linalg.norm(u0))
assert linear_ic<1e-12 and lin[1].shape==(3,8)
gates.update(linear_initial_projection_relative=linear_ic,linear_endpoint_state_dimension=lin[1].shape[1],
             backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
Path('checks').mkdir(exist_ok=True)
Path('checks/smoke.json').write_text(json.dumps(gates,indent=2)+'\n')
print(json.dumps(gates,indent=2))
