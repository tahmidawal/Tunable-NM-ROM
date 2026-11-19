"""Curvature-aware fitting of an unattainable target with a known stationary code."""
from common import *
from neural import fields as F
import data

xy,w=data.grid(9,True);xy=jnp.asarray(xy);w=jnp.asarray(w)
def field(_,z,xy):
    x,y=xy[:,0],xy[:,1]
    return (z[0]*jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y)+z[1]*jnp.sin(2*jnp.pi*x)*jnp.sin(jnp.pi*y)
            +.8*(z[0]**2+z[1]**2)*jnp.sin(jnp.pi*x)*jnp.sin(2*jnp.pi*y))
known=jnp.array([.3,-.2]);J=jax.jacfwd(lambda z:field(None,z,xy))(known)
v=jnp.sin(jnp.pi*xy[:,0])*jnp.sin(2*jnp.pi*xy[:,1])
normal=v-J@jnp.linalg.solve(J.T@(w[:,None]*J),J.T@(w*v))
target=field(None,known,xy)+.04*normal;norm=jnp.sqrt(jnp.sum(w*target**2))
fit=F.make_fit(field,newton=True)
z,status=fit(None,jnp.array([.1,-.1]),xy,target,w,norm)
assert status[-1]==1 and jnp.linalg.norm(z-known)<1e-7 and status[0]>.01
print(json.dumps(dict(status='passed',code_error=float(jnp.linalg.norm(z-known)),fit_status=np.asarray(status).tolist())))
