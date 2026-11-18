"""Geometry-conditioned, zero-preserving shared child coefficients and physical LM."""
from common import *
import decoder as dec


def geometry(cells):
    cells=jnp.asarray(cells);h=2.**(-cells[...,0])
    center=(cells[...,1:]+.5)*h[...,None]
    return jnp.concatenate([2*center-1,-cells[...,:1]/4],axis=-1),h


def head(params,z,g,h,scale):
    g=jnp.broadcast_to(g,z.shape[:-1]+(3,))
    q=jnp.concatenate([z,g],axis=-1)
    q0=jnp.concatenate([jnp.zeros_like(z),g],axis=-1)
    return h[...,None]*scale*(z@params['linear'].T+dec.mlp(params['net'],q)-dec.mlp(params['net'],q0))


def encode(params,c,g,h,scale):
    y=c/(h[...,None]*scale)
    g=jnp.broadcast_to(g,y.shape[:-1]+(3,))
    q=jnp.concatenate([y,g],axis=-1)
    q0=jnp.concatenate([jnp.zeros_like(y),g],axis=-1)
    return y@params['linear']+dec.mlp(params['net'],q)-dec.mlp(params['net'],q0)


def initialize(local,h,norm,k,key,width=128):
    """Training coefficients only; scalar physical loss weights determine the PCA skip."""
    y=local/h[...,None]
    normalized=(local/norm[...,None,None]).reshape(-1,local.shape[-1])
    _,_,vh=jnp.linalg.svd(normalized,full_matrices=False);V=vh[:k].T
    initial=y@V
    sd=jnp.maximum(jnp.std(initial.reshape(-1,k),axis=0),1e-8)
    scale=jnp.maximum(jnp.std(y.reshape(-1,y.shape[-1]),axis=0),1e-8)
    scale=jnp.maximum(scale,jnp.max(scale)*1e-3)
    kd,ke=jax.random.split(key)
    hp=dict(linear=V*sd[None]/scale[:,None],net=dec.init_mlp(kd,[k+3,width,width,local.shape[-1]],.01))
    ep=dict(linear=scale[:,None]*V/sd[None],net=dec.init_mlp(ke,[local.shape[-1]+3,width,width,k],.01))
    return dict(head=hp,encoder=ep),scale


@jax.jit
def schur(gram,rhs,norm2):
    root=128
    A=gram[:root,:root];B=gram[:root,root:]
    AinvB=jnp.linalg.solve(A,B);Ainvb=jnp.linalg.solve(A,rhs[:,:root].T).T
    S=gram[root:,root:]-B.T@AinvB;S=(S+S.T)*.5
    b=rhs[:,root:]-Ainvb@B
    constant=norm2-jnp.sum(rhs[:,:root]*Ainvb,axis=1)
    return S,b,constant,AinvB,Ainvb


def make_project(maxiter=50,cgiter=60):
    """Full overlapping physical metric; independent codes, root refitted by elimination."""
    @jax.jit
    def project(S,b,constant,norm2,z0,params,g,h,scale,mask):
        P,R=h.shape[0],b.shape[0]//h.shape[0]
        diagonal_blocks=S.reshape(P,R,P,R)[jnp.arange(P),:,jnp.arange(P),:]
        def coef(z):return (head(params,z,g,h,scale)*mask[:,None]).reshape(-1)
        def objective(z):
            c=coef(z)
            return (c@S@c-2*b@c+constant)/norm2
        def jac(z):
            return jax.vmap(jax.jacfwd(lambda zz,gg,hh:head(params,zz,gg,hh,scale)),in_axes=(0,0,0))(z,g,h)*mask[:,None,None]
        def body(_,state):
            z,loss,damping,accepted,done=state
            c=coef(z);J=jac(z)
            r=(S@c-b).reshape(P,R)/norm2
            gradient=jnp.einsum('prk,pr->pk',J,r)
            diag=jnp.maximum(jnp.einsum('prk,prs,psk->pk',J,diagonal_blocks,J)/norm2,1e-12)
            def action(v):
                jv=jnp.einsum('prk,pk->pr',J,v).reshape(-1)
                Sjv=(S@jv).reshape(P,R)/norm2
                return jnp.einsum('prk,pr->pk',J,Sjv)+(damping*diag+1e-12)*v
            delta,_=jax.scipy.sparse.linalg.cg(action,-gradient,tol=1e-6,atol=1e-12,maxiter=cgiter,
                                             M=lambda v:v/((1+damping)*diag+1e-12))
            trial=z+delta;newloss=objective(trial)
            good=(newloss<loss)&jnp.isfinite(newloss)&(~done)
            stationary=jnp.max(jnp.abs(gradient)/jnp.sqrt(diag))<1e-8
            return (jnp.where(good,trial,z),jnp.where(good,newloss,loss),
                    jnp.clip(jnp.where(good,damping/3,damping*5),1e-10,1e12),accepted+good.astype(jnp.int32),done|stationary)
        initial=(z0,objective(z0),jnp.float64(1e-3),jnp.int32(0),jnp.bool_(False))
        z,loss,damping,accepted,done=jax.lax.fori_loop(0,maxiter,body,initial)
        c=coef(z);J=jac(z);gradient=jnp.einsum('prk,pr->pk',J,(S@c-b).reshape(P,R)/norm2)
        diag=jnp.maximum(jnp.einsum('prk,prs,psk->pk',J,diagonal_blocks,J)/norm2,1e-12)
        station=jnp.max(jnp.abs(gradient)/jnp.sqrt(diag))
        return z,jnp.sqrt(jnp.maximum(loss,0)),station,accepted
    return project


def zero_audit(params,scale,g,h,S):
    k=params['linear'].shape[1];P=len(h);R=len(scale);z=jnp.zeros((P,k))
    zero=head(params,z,g,h,scale)
    J=jax.vmap(jax.jacfwd(lambda zz,gg,hh:head(params,zz,gg,hh,scale)),in_axes=(0,0,0))(z,g,h)
    H=jnp.einsum('prk,prqs,qsl->pkql',J,S.reshape(P,R,P,R),J).reshape(P*k,P*k)
    ev=jnp.linalg.eigvalsh((H+H.T)*.5);largest=jnp.maximum(ev[-1],1e-30)
    return dict(zero_child_max=float(jnp.max(jnp.abs(zero))),after_parent_removal_eigenvalues=np.asarray(ev),
                independent_directions=int(jnp.sum(ev>largest*1e-10)),child_variables=P*k,
                smallest_to_largest_eigenvalue=float(ev[0]/largest))
