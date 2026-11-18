"""Dense fine-grid diagnostic of the incumbent scaled tested weak-upwind equations.

FFT tests avoid storing a grid-by-test matrix. This is exact dense integration,
not a coarse-physics or continuous-flux substitute and not yet hyper-reduced.
"""
from common import *
import operators


def dst1(x):
    # Arithmetic copied from the already-vendored engines.dst1; see provenance.
    n=x.shape[-1];zero=jnp.zeros(x.shape[:-1]+(1,),x.dtype)
    ext=jnp.concatenate((zero,x,zero,-x[...,::-1]),axis=-1)
    return -jnp.fft.fft(ext,axis=-1).imag[...,1:n+1]/jnp.sqrt(2.*(n+1))


def dst2(x):return dst1(dst1(x).swapaxes(-1,-2)).swapaxes(-1,-2)


def selected_modes(L,M):
    k=np.arange(1,L);v=4*L**2*np.sin(np.pi*k/(2*L))**2
    lam=v[:,None]+v[None];ix=np.argsort(lam.ravel(),kind='stable')[:M]
    kx,ky=np.unravel_index(ix,lam.shape)
    return np.stack([kx,ky]),lam.ravel()[ix]


def raw_axis(x,center,h,side=6):
    v=(x-center)/(.75*h);p=[np.ones_like(x),v]
    for n in range(2,side):p.append(((2*n-1)*v*p[-1]-(n-1)*p[-2])/n)
    t=np.clip((np.abs((x-center)/h)-.5)/.25,0.,1.)
    window=1-(10*t**3-15*t**4+6*t**5)
    return np.stack(p[:side],axis=-1)*(4*x*(1-x)*window)[:,None]


def prepare(basis,cells,transforms,rotation,L,M,physical_gram):
    """All feature arrays are runtime arguments; only compact slice metadata is static."""
    basis=np.asarray(basis);cells=np.asarray(cells);axis=np.arange(1,L)/L
    idx,lam=selected_modes(L,M);root=len(basis);N=basis.shape[-1]
    SX=np.sin(np.pi*axis[:,None]*np.arange(1,N+1)[None]);axes=[];slices=[];T=[];Alocal=[]
    maxmode=int(idx.max())+1
    for cell,transform in zip(cells,transforms):
        depth,ix,iy=cell;h=2.**(-depth);c=(np.array([ix,iy])+.5)*h
        parts=[];locations=[];mom=[]
        for center in c:
            ids=np.where(abs(axis-center)<=.75*h)[0];lo,hi=int(ids[0]),int(ids[-1])+1;xx=axis[lo:hi]
            raw=raw_axis(xx,center,h);parts.append(jnp.asarray(raw));locations.append((lo,hi))
            sine=np.sin(np.pi*xx[:,None]*np.arange(1,maxmode+1)[None])
            mom.append(jnp.asarray(sine.T@raw/L))
        transform=jnp.asarray(transform)@jnp.asarray(rotation);T.append(transform);axes.append(tuple(parts));slices.append(tuple(locations))
        aa=2*jnp.einsum('mi,mj->mij',mom[0][idx[0]],mom[1][idx[1]]).reshape(M,36)@transform
        Alocal.append(aa)
    kx,ky=idx;inside=(kx<N)&(ky<N)
    Ar=.5*basis[:,np.minimum(kx,N-1),np.minimum(ky,N-1)].T*inside[:,None]
    A=jnp.concatenate([jnp.asarray(Ar)]+Alocal,axis=1)
    cache=dict(basis=jnp.asarray(basis),sine=jnp.asarray(SX),axes=tuple(axes),transform=tuple(T),
               A=A,indices=jnp.asarray(idx),lam=jnp.asarray(lam),gram=jnp.asarray(physical_gram))
    return cache,slices


def make_field(slices,rank):
    def field(c,cache):
        root=cache['basis'].shape[0]
        B=jnp.einsum('k,kij->ij',c[:root],cache['basis'])
        S=cache['sine'];u=S@B@S.T
        for i,((x0,x1),(y0,y1)) in enumerate(slices):
            raw=cache['transform'][i]@c[root+i*rank:root+(i+1)*rank]
            X,Y=cache['axes'][i];patch=X@raw.reshape(6,6)@Y.T
            u=u.at[x0:x1,y0:y1].add(patch)
        return u
    return field


def make_solver(field,L,iterations=30,cg_iterations=80):
    def residual(c,previous,cache,dt,nu,norm):
        u=field(c,cache);adv,_=operators.spatial(u,L)
        advtests=dst2(adv.reshape(L-1,L-1))[cache['indices'][0],cache['indices'][1]]/L
        denom=1+dt*nu*cache['lam']
        return (cache['A']@c-previous/denom+dt*advtests/denom)/norm
    @jax.jit
    def step(c0,cache,dt,nu,station_tolerance,update_tolerance):
        previous=cache['A']@c0;norm=jnp.sqrt(jnp.maximum(c0@cache['gram']@c0,1e-30))
        func=lambda c:residual(c,previous,cache,dt,nu,norm)
        H0=(cache['A'].T@cache['A'])/norm**2;diag=jnp.maximum(jnp.diag(H0),1e-12)
        def body(_,state):
            c,loss,damp,accepted,done,lastupdate,lastlinear,used=state
            def advance(state):
                c,loss,damp,accepted,done,lastupdate,lastlinear,used=state
                r,jv=jax.linearize(func,c);_,vjp=jax.vjp(func,c)
                grad=vjp(r)[0]
                def action(v):return vjp(jv(v))[0]+(damp*diag+1e-12)*v
                C=jnp.linalg.cholesky(H0+jnp.diag(damp*diag+1e-12))
                def precondition(v):return jax.scipy.linalg.cho_solve((C,True),v)
                delta,_=jax.scipy.sparse.linalg.cg(action,-grad,M=precondition,tol=1e-6,atol=1e-12,maxiter=cg_iterations)
                linear=jnp.linalg.norm(action(delta)+grad)/jnp.maximum(jnp.linalg.norm(grad),1e-30)
                update=jnp.sqrt(jnp.maximum(delta@cache['gram']@delta,0))/norm
                trial=c+delta;rr=func(trial);newloss=rr@rr
                station=jnp.max(jnp.abs(grad)/jnp.sqrt(diag))
                good=(newloss<loss)&jnp.isfinite(newloss)
                # A small step alone is not convergence; preserve the stationarity test.
                finished=(station<station_tolerance)&((update<update_tolerance)|(station<station_tolerance*.01))
                return (jnp.where(good,trial,c),jnp.where(good,newloss,loss),
                        jnp.clip(jnp.where(good,damp/3,damp*5),1e-10,1e10),accepted+good.astype(jnp.int32),
                        finished,update,linear,used+1)
            return jax.lax.cond(done,lambda s:s,advance,state)
        r=func(c0);initial=(c0,r@r,jnp.float64(1e-3),jnp.int32(0),jnp.bool_(False),jnp.float64(0),jnp.float64(0),jnp.int32(0))
        c,loss,damp,accepted,done,update,linear,used=jax.lax.fori_loop(0,iterations,body,initial)
        rr=func(c);_,vjp=jax.vjp(func,c);grad=vjp(rr)[0];station=jnp.max(jnp.abs(grad)/jnp.sqrt(diag))
        return c,jnp.array([jnp.sqrt(loss),station,update,linear,used,accepted],dtype=jnp.float64)
    return step,residual


def make_initial_fit(field):
    @jax.jit
    def fit(u0,cache):
        L=u0.shape[0]+1;K=cache['A'].shape[1];z=jnp.zeros(K)
        _,adj=jax.vjp(lambda c:field(c,cache),z)
        rhs=adj(u0)[0]/L**2
        def action(c):return adj(field(c,cache))[0]/L**2+1e-10*c
        C=jnp.linalg.cholesky(cache['gram']+1e-8*jnp.eye(K))
        c,_=jax.scipy.sparse.linalg.cg(action,rhs,tol=1e-9,atol=1e-13,maxiter=200,
                                      M=lambda v:jax.scipy.linalg.cho_solve((C,True),v))
        error=jnp.linalg.norm(field(c,cache)-u0)/jnp.linalg.norm(u0)
        linear=jnp.linalg.norm(action(c)-rhs)/jnp.linalg.norm(rhs)
        return c,error,linear
    return fit
