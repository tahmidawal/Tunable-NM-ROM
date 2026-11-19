"""Dense arithmetic, nonlinear Jacobian and stationary-step fixtures (CPU <1min)."""
from common import *
from neural import weak as W
import operators


def field(params,z,xy):
    modes=jnp.array([[1,1],[2,1],[1,2]])
    basis=jnp.prod(jnp.sin(jnp.pi*xy[:,None,:]*modes[None]),axis=-1)
    # Curved output manifold, including signed fields to test both stencil branches.
    coefficients=jnp.array([z[0],z[1],.2*z[0]*z[1]])
    return basis@coefficients


L=16;Q=16;M=32;cache=W.prepare(L,Q,M);z=jnp.array([.3,-.7]);dt=.00125;nu=.005
mass,adv,previous,residual=W.make_operators(field,L,Q)
phi,lam,xy=operators.modes(L,M);u=field(None,z,jnp.asarray(xy))
aa,lap=operators.spatial(u,L)
mass_error=float(jnp.max(abs(mass(None,z,cache)-phi.T@u/L)))
advection_error=float(jnp.max(abs(adv(None,z,cache)-phi.T@aa/L)))
diffusion_error=float(jnp.max(abs(cache['lam']*mass(None,z,cache)+phi.T@lap/L)))
assert max(mass_error,advection_error,diffusion_error)<1e-12
old=previous(None,z,cache,dt,nu);norm=jnp.linalg.norm(u)/L
trial=z+jnp.array([.001,-.002]);ut=field(None,trial,jnp.asarray(xy));at,lt=operators.spatial(ut,L)
dense=(phi.T@(ut-u+.5*dt*(aa+at-nu*(lap+lt)))/L)/(1+.5*dt*nu*lam)/norm
residual_error=float(jnp.max(abs(residual(None,trial,old,cache,dt,nu,norm)-dense)))
assert residual_error<1e-12
fun=lambda zz:residual(None,zz,old,cache,dt,nu,norm)
direction=jnp.array([.2,.6]);eps=1e-6
derivative=jax.jvp(fun,(trial,),(direction,))[1]
finite=(fun(trial+eps*direction)-fun(trial-eps*direction))/(2*eps)
jvp_error=float(jnp.linalg.norm(derivative-finite)/jnp.linalg.norm(derivative));assert jvp_error<1e-8
step=W.make_step(field,L,Q)
returned,status=step(None,z,cache,dt,nu)
tight,tight_status=step(None,z,cache,dt,nu,1e-10,1e-12)
assert status[-1]==1 and tight_status[-1]==1
change=float(jnp.linalg.norm(field(None,returned,cache['xy'])-field(None,tight,cache['xy']))/jnp.linalg.norm(u))
assert change<1e-8
base=jnp.array([1.,1e-8]);delta=jnp.array([0.,-1e-10])
naive=float(base@base-(base+delta)@(base+delta))
stable=float(-2*base@delta-delta@delta)
assert naive==0 and stable>0
assert abs(stable-1.99e-18)<1e-32
print(json.dumps(dict(cancellation_naive=naive,cancellation_stable=stable,mass_error=mass_error,advection_error=advection_error,diffusion_error=diffusion_error,
                     residual_error=residual_error,jvp_error=jvp_error,status=np.asarray(status).tolist(),
                     tight_status=np.asarray(tight_status).tolist(),tight_field_change=change)))
