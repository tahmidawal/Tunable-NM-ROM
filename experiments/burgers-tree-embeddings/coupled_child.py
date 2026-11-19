"""Joint root/child online coordinates; the PDE only sees the assembled coefficient field."""
from common import *
import shared_child_head as H


def coefficients(z,cache):
    root=len(cache['basis']);P=len(cache['child_width']);k=cache['child_head']['linear'].shape[1]
    local=H.head(cache['child_head'],z[root:].reshape(P,k),cache['child_geometry'],cache['child_width'],cache['child_scale'])
    return jnp.concatenate([z[:root],local.reshape(-1)])


def child_jacobian(z,cache):
    root=len(cache['basis']);P=len(cache['child_width']);k=cache['child_head']['linear'].shape[1]
    return jax.vmap(jax.jacfwd(lambda zz,g,h:H.head(cache['child_head'],zz,g,h,cache['child_scale'])))(
        z[root:].reshape(P,k),cache['child_geometry'],cache['child_width'])


def tested_mass_jacobian(z,cache):
    root=len(cache['basis']);P=len(cache['child_width']);J=child_jacobian(z,cache);M=len(cache['A'])
    local=jnp.einsum('mpr,prk->mpk',cache['A'][:,root:].reshape(M,P,-1),J).reshape(M,-1)
    return jnp.concatenate([cache['A'][:,:root],local],axis=1)


def coefficient_jacobian(z,cache):
    root=len(cache['basis']);J=child_jacobian(z,cache);P,R,k=J.shape
    full=jnp.zeros((root+P*R,root+P*k)).at[:root,:root].set(jnp.eye(root))
    rows=root+jnp.arange(P)[:,None,None]*R+jnp.arange(R)[None,:,None]
    cols=root+jnp.arange(P)[:,None,None]*k+jnp.arange(k)[None,None,:]
    return full.at[rows,cols].set(J)


def make_initial_fit(field,maxiter=250):
    project=H.make_project(maxiter,linear_solver='cholesky')
    @jax.jit
    def fit(u0,cache):
        Q=len(u0)+1;root=len(cache['basis']);P=len(cache['child_width']);R=cache['child_head']['linear'].shape[0]
        G=cache['input_gram'];zero=jnp.zeros(len(G));_,adj=jax.vjp(lambda c:field(c,cache),zero)
        rhs=adj(u0)[0]/Q**2;norm2=jnp.sum(u0*u0)/Q**2
        S,b,constant,AinvB,Ainvb=H.schur(G,rhs[None],norm2[None],root=root)
        teacher=jnp.linalg.solve(S+1e-8*jnp.eye(len(S)),b[0]).reshape(P,R)
        ze=H.encode(cache['child_encoder'],teacher,cache['child_geometry'],cache['child_width'],cache['child_scale'])
        def solve(z):return project(S,b[0],constant[0],norm2,z,cache['child_head'],cache['child_geometry'],cache['child_width'],cache['child_scale'],jnp.ones(P))
        encoded=solve(ze);cold=solve(jnp.zeros_like(ze));which=encoded[1]<=cold[1]
        z=jnp.where(which,encoded[0],cold[0]);local=H.head(cache['child_head'],z,cache['child_geometry'],cache['child_width'],cache['child_scale']).reshape(-1)
        parent=Ainvb[0]-AinvB@local;code=jnp.concatenate([parent,z.reshape(-1)]);c=coefficients(code,cache)
        error=jnp.linalg.norm(field(c,cache)-u0)/jnp.linalg.norm(u0)
        J=coefficient_jacobian(code,cache);gradient=J.T@(G@c-rhs)/norm2
        diag=jnp.maximum(jnp.sum(J*(G@J),axis=0)/norm2,1e-12)
        station=jnp.max(jnp.abs(gradient)/jnp.sqrt(diag))
        return code,error,station,jnp.array([encoded[3],cold[3]])
    return fit
