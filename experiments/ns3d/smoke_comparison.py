"""Under-minute numerical check of new bank, coefficient dynamics and operator adapter."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R
import ns3d_independent as I
import operator_adapter as O
from operators import models3d as M
from operators import training as OT
from comparison import raw_projection_check
from pilot import write

out=Path(__file__).resolve().parent/'checks/comparison';out.mkdir(parents=True,exist_ok=True)
n=8;geom=F.geometry(n)
U=np.stack([F.initial(n,p) for p in F.parameters(103,12)])
P,scores,singular=D.pod_gpu(U,8)
params,z,binfo,c=D.train_free_bank(U,n,2,8,14,4,30,out/'free.pkl',scores,batch=4,width=16,n_ff=8,spatial_batch=64)
G=np.asarray(D.bank(params,D.coords(n),geom,n));Q,Rb,C,perp,w=D.whiten(G,U)
pcheck=raw_projection_check(params,C,U,n);assert pcheck['passed']
params,z,hinfo=D.train_head(params,z,C,Rb,5,20,14,out/'head.pkl',dict(n=n))
fit=D.representation(params,G,Rb,C,perp,U,z,starts=2,budget=15)
assert np.all(np.isfinite(fit['head_error']))
L,T=R.build_galerkin(P,n)
rng=np.random.default_rng(15);coeff=rng.normal(size=8)
u=(P@coeff).reshape(3,n,n,n)
expected=P.T@I.physical(I.advective(I.transform(u),I.setup(n))).ravel()
actual=np.einsum('ijk,j,k->i',T,coeff,coeff)
parity=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
assert parity<1e-10
energy=float(abs(coeff@actual));assert energy<1e-10
nu0=.005;horizon=.2
exact=solve_ivp(lambda t,c: np.einsum('ijk,j,k->i',T,c,c)+nu0*(L@c),[0,horizon],coeff,rtol=1e-12,atol=1e-14).y[:,-1]
errors=[]
for steps in (8,16):
    run=R.make_galerkin_run(horizon/steps,steps,steps)
    field=np.asarray(run(jnp.asarray(u),nu0,jnp.asarray(P),jnp.asarray(L),jnp.asarray(T)))[-1]
    errors.append(float(np.linalg.norm(P.T@field-exact)))
print("GALERKIN_TEMPORAL_ERRORS",errors,flush=True)
assert errors[0]/errors[1]>3.5
operator_results={}
for spec in [dict(kind='fno3d',width=2,modes=[2,2,2],depth=1,padding=0),dict(kind='unet3d',width=2,levels=2,periodic=True)]:
    mp=M.init_model(jax.random.PRNGKey(6),spec,4,15)
    query=O.make_query(spec,n,True)
    pred=np.asarray(query(jnp.asarray(U[0]),.005,mp,jnp.zeros((1,4,1,1,1)),jnp.ones((1,4,1,1,1)),jnp.asarray(1.),geom))
    assert pred.shape==(6,3,n,n,n) and np.isfinite(pred).all()
    assert np.array_equal(pred[0],U[0])
    div=np.asarray(F.diagnostics(jnp.asarray(pred),geom)[2]);assert max(div)<1e-10
    assert all(a.dtype==jnp.float64 for a in jax.tree_util.tree_leaves(mp))
    x=np.moveaxis(np.concatenate((U[:2],np.full((2,1,n,n,n),.005)),axis=1),1,-1)
    y=np.moveaxis(np.tile(U[:2],(1,5,1,1,1)),1,-1)
    dn=np.repeat(np.sum(U[:2]**2,axis=(1,2,3,4))[:,None],5,axis=1)
    tc=dict(steps=2,wall_seconds=20,batch_size=1,seed=91,learning_rate=.001,validation_every=2,components_per_output=3)
    _,info=OT.train(x,y,x,y,spec,tc,out/spec['kind'],lambda p,o:write(o,p),O.checkpoint,dn,dn)
    operator_results[spec['kind']]=dict(finite=True,max_divergence=float(max(div)),training_steps=info['steps_completed'])
record=dict(passed=True,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64),
            projection_nonexpansion=pcheck,galerkin_independent_parity=parity,galerkin_energy_defect=energy,
            galerkin_CNAB2_errors=errors,galerkin_order_ratio=errors[0]/errors[1],operators=operator_results,
            representation_finite=True,bank_steps=binfo['steps'])
write(record,out/'result.json');print(json.dumps({k:v for k,v in record.items() if k!='projection_nonexpansion'},indent=2))
