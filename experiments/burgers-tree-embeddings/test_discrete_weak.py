"""Small exact-grid arithmetic/Jacobian fixture for the fixed-tree solver."""
from common import *
import discrete_weak as W,operators,geometry,tensor_metric as TM,local_polynomial as LP

def main():
    L=32;root=4;rank=4;M=80;cells=geometry.uniform(1)
    base=jnp.zeros((root,16,16)).at[jnp.arange(root),jnp.array([0,1,0,1]),jnp.array([0,0,1,1])].set(2.)
    G,T,_,_=TM.make_geometry(6)(base,jnp.asarray(cells),*TM.rule(N=32))
    U=jnp.eye(36,rank)
    from build_local_cache import expand_matrix
    trans=expand_matrix(U,len(cells),root);G=trans.T@G@trans
    cache,slices=W.prepare(base,cells,T,U,L,M,G);field=W.make_field(slices,rank)
    c=jnp.zeros(root+len(cells)*rank).at[0].set(.2).at[root:].set(.001)
    u=field(c,cache);ref=LP.field_grid(base,c@trans.T,cells,T,6,np.arange(1,L)/L)[0]
    assert float(jnp.max(jnp.abs(u-ref)))<1e-12
    phi,lam,_=operators.modes(L,M);exact=phi.T@u.reshape(-1)/L
    assert float(jnp.max(jnp.abs(exact-cache['A']@c)))<1e-12
    assert float(jnp.max(jnp.abs(W.dst2(u)[cache['indices'][0],cache['indices'][1]]/L-exact)))<1e-12
    step,residual=W.make_solver(field,L);dt=.0025;nu=.005;prior=.99*exact;norm=.2
    r=residual(c,prior,cache,dt,nu,norm);adv,lap=operators.spatial(u,L)
    dense=(exact-prior+dt*(phi.T@(adv-nu*lap)/L))/(1+dt*nu*lam)/norm
    assert float(jnp.max(jnp.abs(r-dense)))<1e-12
    direction=jnp.cos(jnp.arange(len(c)));eps=1e-6
    jv=jax.jvp(lambda z:residual(z,prior,cache,dt,nu,norm),(c,),(direction,))[1]
    fd=(residual(c+eps*direction,prior,cache,dt,nu,norm)-residual(c-eps*direction,prior,cache,dt,nu,norm))/(2*eps)
    assert float(jnp.linalg.norm(jv-fd)/jnp.linalg.norm(fd))<1e-7
    out,status=step(c,cache,dt,nu,1e-8,1e-10)
    assert jnp.all(jnp.isfinite(out)) and float(status[1])<1e-7,status
    print('PASS exact dense weak-upwind / eigen-diffusion / field / Jacobian / step',np.asarray(status),flush=True)

if __name__=='__main__':main()
