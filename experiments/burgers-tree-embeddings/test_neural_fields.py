"""Sub-minute fixtures for real neural child directions and physical nonlinear fits."""
from common import *
from neural import fields as F
import data

xy,w=data.grid(9,True);xy=jnp.asarray(xy);w=jnp.asarray(w)
keys=jax.random.split(jax.random.PRNGKey(815),2)
p=dict(root=F.init_mlp(keys[0],[5,12,12,1],True),child=F.init_mlp(keys[1],[7,12,12,1],True))
cells=jnp.array([[1.,0.,0.],[1.,1.,1.]]);z=jnp.array([.1,.2,-.1,0.,0.,0.,0.])
fun=lambda z:F.tree(p,z,xy,cells,root_k=3,child_k=2)
zero_change=float(jnp.max(jnp.abs(fun(z)-F.root(p['root'],z[:3],xy))))
assert zero_change==0
J=jax.jacfwd(fun)(z);Q,_=jnp.linalg.qr(jnp.sqrt(w)[:,None]*J[:,:3])
local=jnp.sqrt(w)[:,None]*J[:,3:];local=local-Q@(Q.T@local)
s=jnp.linalg.svd(local,compute_uv=False);assert float(s[-1]/s[0])>1e-6
v=jnp.arange(1,8,dtype=jnp.float64)/10
automatic=jax.jvp(fun,(z,),(v,))[1];fd=(fun(z+1e-6*v)-fun(z-1e-6*v))/2e-6
jvp_error=float(jnp.linalg.norm(automatic-fd)/jnp.linalg.norm(automatic));assert jvp_error<1e-7
walls=jnp.array([[0.,.2],[1.,.7],[.4,0.],[.9,1.]])
assert float(jnp.max(jnp.abs(F.tree(p,z+.1,walls,cells,3,2))))==0
def curved(_,z,xy):
    x,y=xy[:,0],xy[:,1]
    return z[0]*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)+(z[1]+.2*z[0]**2)*jnp.sin(2*jnp.pi*x)*jnp.sin(jnp.pi*y)+.3*jnp.sin(z[1])*jnp.sin(jnp.pi*x)*jnp.sin(2*jnp.pi*y)
target=curved(None,jnp.array([.3,-.2]),xy);norm=jnp.sqrt(jnp.sum(w*target**2))
fit=F.make_fit(curved);code,status=fit(None,jnp.zeros(2),xy,target,w,norm)
assert float(status[0])<1e-8 and bool(status[-1])
print(json.dumps(dict(status='passed',zero_child_change=zero_change,useful_child_directions=4,
    minimum_singular_ratio=float(s[-1]/s[0]),jvp_relative_error=jvp_error,
    nonlinear_fit_status=np.asarray(status).tolist(),nonlinear_code=np.asarray(code).tolist())))
