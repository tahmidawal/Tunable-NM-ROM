"""Bounded f64 checks: independent physics attention, all axes, patch identities."""
import sys,json,os,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
from operators import models3d as M
from operators import extra_models3d as E
assert jax.default_backend()=='gpu'
assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
rng=np.random.default_rng(920323)
p=E.init_physics(jax.random.PRNGKey(0),8,2,4)
x=rng.normal(size=(2,2,13,4));f=rng.normal(size=x.shape)
def softmax(a):
    e=np.exp(a-a.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
pn=jax.tree_util.tree_map(np.asarray,p)
w=softmax((x@pn['slice']['w']+pn['slice']['b'])/np.clip(pn['temperature'],.1,5))
t=np.einsum('bhnd,bhns->bhsd',f,w)/(w.sum(axis=2)[...,None]+1e-5)
q=t@pn['q']['w'];k=t@pn['k']['w'];v=t@pn['v']['w']
ref=np.einsum('bhsd,bhns->bhnd',softmax(q@np.swapaxes(k,-1,-2)/2)@v,w)
y=np.asarray(jax.jit(E.physics_core)(p,jnp.asarray(x),jnp.asarray(f)))
core_error=np.linalg.norm(y-ref)/np.linalg.norm(ref);assert core_error<1e-12

# Compare the complete convolutional structured-3D module against pinned upstream
# PyTorch on CPU with IDENTICAL f64 parameters (all JAX execution remains on GPU).
import torch
torch.set_num_threads(1)
module_spec=importlib.util.spec_from_file_location('upstream_physics',Path(__file__).parent/'upstream/Physics_Attention.py')
up=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(up)
net=up.Physics_Attention_Structured_Mesh_3D(8,heads=2,dim_head=4,dropout=0.,slice_num=4,H=4,W=5,D=6).double()
def assign(target,value):target.data.copy_(torch.from_numpy(np.array(value,copy=True)))
for name,label in [('in_project_x','x'),('in_project_fx','fx')]:
    layer=getattr(net,name);assign(layer.weight,pn[label]['w'].transpose(4,3,0,1,2));assign(layer.bias,pn[label]['b'])
assign(net.in_project_slice.weight,pn['slice']['w'].T);assign(net.in_project_slice.bias,pn['slice']['b'])
for label in ('q','k','v'):assign(getattr(net,'to_'+label).weight,pn[label]['w'].T)
assign(net.to_out[0].weight,pn['out']['w'].T);assign(net.to_out[0].bias,pn['out']['b']);assign(net.temperature,pn['temperature'])
xx=rng.normal(size=(2,4,5,6,8))
expected=net(torch.from_numpy(xx.reshape(2,-1,8))).detach().numpy().reshape(xx.shape)
actual=np.asarray(jax.jit(lambda p,x:E._physics(p,x,2,False))(p,jnp.asarray(xx)))
upstream_error=np.linalg.norm(actual-expected)/np.linalg.norm(expected);assert upstream_error<1e-12

checks=[]
for kind in ('deeponet3d','transolver3d'):
 for periodic in (False,True):
    shape=(7,8,9);axes=[np.arange(n)/n if periodic else np.arange(1,n+1)/(n+1) for n in shape]
    coords=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)*2-1
    xx=np.concatenate((rng.normal(size=(2,*shape,3)),np.broadcast_to(coords,(2,*shape,3))),axis=-1)
    spec=dict(kind=kind,width=4,rank=8,trunk_width=8,levels=2,pool_bins=2,depth=1,heads=2,slices=4,patch=2,reference_grid=2,periodic=periodic)
    p=M.init_model(jax.random.PRNGKey(1),spec,6,6)
    loss,grad=jax.jit(jax.value_and_grad(lambda p,x:jnp.mean((M.apply_model(p,x,spec)-.2)**2)))(p,jnp.asarray(xx))
    assert all(a.dtype==jnp.float64 and np.isfinite(a).all() for a in jax.tree_util.tree_leaves(grad))
    yy=np.asarray(jax.jit(lambda p,x:M.apply_model(p,x,spec))(p,jnp.asarray(xx)))
    assert yy.shape==(2,*shape,6) and yy.dtype==np.float64 and np.isfinite(yy).all()
    assert np.linalg.norm(yy[0]-yy[1])>1e-12,'network must depend on supplied field'
    checks.append(dict(kind=kind,periodic=periodic,parameters=M.parameter_count(p),loss=float(loss)))

# Patch fields must round-trip exactly, including odd axes and both padding rules.
for periodic in (False,True):
    spec=dict(patch=2,reference_grid=2,periodic=periodic)
    tok,pads=E._patches(jnp.asarray(xx[...,:3]),jnp.asarray(xx[...,3:]),spec)
    tok=np.asarray(tok)[...,:24];b,nx,ny,nz,_=tok.shape
    recovered=tok.reshape(b,nx,ny,nz,2,2,2,3).transpose(0,1,4,2,5,3,6,7).reshape(b,2*nx,2*ny,2*nz,3)
    recovered=recovered[:,pads[0][0]:pads[0][0]+7,pads[1][0]:pads[1][0]+8,pads[2][0]:pads[2][0]+9]
    assert np.array_equal(recovered,xx[...,:3])
result=dict(jax_backend='gpu',x64=True,precision='highest',numpy_physics_relative=core_error,upstream_structured_3d_relative=upstream_error,checks=checks,patch_roundtrip=True)
dest=Path(sys.argv[1]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
