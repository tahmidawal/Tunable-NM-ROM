"""Bounded shape/precision/projection/history and weighted-training smoke."""
from pathlib import Path
import json
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R
import operator_adapter as O
import dataset_adapter as A
from operators import models3d as M
from extra03 import with_coordinates


def main():
    n=8;g=F.geometry(n);pars=F.parameters(77,8);U=np.stack([F.initial(n,p) for p in pars]);ds=dict(initial=U,viscosity=pars[:,-1:],targets=np.repeat(U[:,None],5,1));stats=A.training_statistics(ds);xx,yy=A.common_model_arrays(ds,stats);xx=with_coordinates(xx)
    checks=[]
    for spec in [dict(kind='deeponet3d',width=2,levels=2,pool_bins=2,rank=8,trunk_width=8,periodic=True,coordinate_channels=[-3,-2,-1],trunk_frequencies=[2.]),dict(kind='transolver3d',width=8,depth=1,heads=2,slices=4,patch=2,reference_grid=2,periodic=True,coordinate_channels=[-3,-2,-1],coordinate_bounds=[0.,1.])]:
        p=M.init_model(jax.random.PRNGKey(5),spec,7,15);q=O.make_query(spec,n,True)
        value=q(jnp.asarray(U[0]),pars[0,-1],p,jnp.asarray(stats['input_mean']),jnp.asarray(stats['input_std']),jnp.asarray(stats['output_scale']),g)
        value=np.asarray(value);assert value.shape==(6,3,n,n,n) and value.dtype==np.float64 and np.array_equal(value[0],U[0]);assert all(a.dtype==jnp.float64 for a in jax.tree_util.tree_leaves(p))
        projected=np.asarray(jax.vmap(F.project_field,in_axes=(0,None))(jnp.asarray(value[1:]),g));assert np.linalg.norm(projected-value[1:])<1e-10
        grad=jax.grad(lambda pp:jnp.sum(M.apply_model(pp,jnp.asarray(xx[:1]),spec)**2))(p);assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves(grad));checks.append(spec['kind'])
    P,scores,_=D.pod_gpu(U,8)
    out=Path('experiments/ns3d/checks/extra03');out.mkdir(parents=True,exist_ok=True)
    p,z,info,c=D.train_free_bank(U,n,2,8,101,2,20,out/'free.pkl',scores,batch=2,width=16,n_ff=8,spatial_batch=16,snapshot_variance=np.mean(U**2,axis=(1,2,3,4)))
    G=np.asarray(D.bank(p,D.coords(n),g,n));Q,Rb,ct,_,_=D.whiten(G,U)
    p,z,info=D.train_head(p,z,ct,Rb,2,20,101,out/'head.pkl',dict(n=n),snapshot_variance=np.mean(U**2,axis=(1,2,3,4)))
    Phi,lam,ids=R.test_modes(n,32);T=R.build_tensor(G,n,ids);C=np.linalg.solve(Rb,np.eye(8)[:,:2]);theta={key:p[key] for key in ('h','h_lin')};args=tuple(jax.tree_util.tree_map(jnp.asarray,x) for x in (G,Q,Rb,Phi.T@G,T,lam,C,theta,z))
    ordinary=R.make_run(.001,2,1,2,2,budget=2);retained=R.make_run(.001,2,1,2,2,budget=2,retain_states=True)
    aa=ordinary(jnp.asarray(U[0]),pars[0,-1],*args);bb=retained(jnp.asarray(U[0]),pars[0,-1],*args)
    assert np.allclose(aa[0],bb[0],rtol=1e-12,atol=1e-12) and bb[3].shape==(3,4)
    rec=np.asarray(D.head(theta,bb[3][:,:2])+bb[3][:,2:]@C.T)@G.T;assert np.allclose(rec,np.asarray(bb[0]),rtol=1e-12,atol=1e-12)
    record=dict(passed=True,checks=checks+['weighted bank/head training','state-history parity and reconstruction'],backend=jax.default_backend(),f64=True)
    (out/'smoke.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
