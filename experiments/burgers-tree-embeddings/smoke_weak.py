from common import *
import operators as O
import ast

# Verify against the exact committed vendor arithmetic, without its unrelated imports.
tree=ast.parse((HERE/'vendor/engines.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='spatial')
ns={'jnp':jnp};exec(compile(ast.Module(body=[node],type_ignores=[]),'verified_vendor_spatial','exec'),ns)
L=16;P,lam,x=O.modes(L,64);x=jnp.asarray(x)
def feature(x):
    return jnp.stack([jnp.sin(jnp.pi*i*x[:,0])*jnp.sin(jnp.pi*j*x[:,1]) for i,j in [(1,1),(2,1),(1,3),(3,2)]],axis=-1)
G=feature(x);c=jnp.array([.7,-.4,.3,-.2]);prev=jnp.array([.5,-.1,.25,-.18]);u=G@c
adv,lap=O.spatial(u,L);vadv,vlap=ns['spatial'](u,L)
np.testing.assert_array_equal(adv,vadv);np.testing.assert_array_equal(lap,vlap)
offset=jnp.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])/L
G5=feature((x[:,None]+offset[None]).reshape(-1,2)).reshape(len(x),5,4)
nu=.005;dt=.005;A=P.T@G
r=O.tested(c,A@prev,A,lam,G5,P,L,nu,dt)
dense=(P.T@(u-G@prev+dt*(adv-nu*lap)))/(1+dt*nu*lam)
err=float(jnp.max(jnp.abs(r-dense)));assert err<1e-13
save(HERE/'smoke/weak.json',dict(meta=meta(),L=L,M=64,m=len(x),fine_stencil_spacing=1/L,
       sign_changing_fixture=True,vendor_spatial_bitwise_equal=True,tested_dense_discrepancy=err,
       status='passed_arithmetic_fixture_only; no large-mesh or dynamics admission'))
print('weak residual fixture passed',err)
