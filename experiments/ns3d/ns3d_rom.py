"""Exact discrete quadratic weak NS3D tensor and current bank/head/q stepping."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
from ns2d_rom import make_lm


def test_modes(n,m):
    """Solenoidal vector Fourier modes, Euclidean orthonormal, lowest |k| first."""
    maxk=int(np.ceil(n/3))-1
    candidates=[(a*a+b*b+c*c,a,b,c) for a in range(-maxk,maxk+1)
                for b in range(-maxk,maxk+1) for c in range(-maxk,maxk+1)
                if (a>0 or (a==0 and b>0) or (a==0 and b==0 and c>0))]
    candidates.sort()
    xyz=np.asarray(D.coords(n)).reshape(n,n,n,3)
    columns=[];lams=[];ids=[]
    for square,a,b,c in candidates:
        wave=np.asarray([a,b,c],dtype=float)
        axis=np.eye(3)[np.argmin(np.abs(wave))]
        first=np.cross(wave,axis);first/=np.linalg.norm(first)
        second=np.cross(wave/np.sqrt(square),first)
        angle=2*np.pi*np.einsum('...d,d->...',xyz,wave)
        for pol in (first,second):
            for fn,label in ((np.cos,'cos'),(np.sin,'sin')):
                field=pol[:,None,None,None]*fn(angle)*np.sqrt(2/n**3)
                columns.append(field.ravel());lams.append((2*np.pi)**2*square)
                ids.append(dict(wave=[a,b,c],polarization=pol.tolist(),kind=label))
                if len(columns)==m:
                    return np.stack(columns,axis=1),np.asarray(lams),ids
    raise ValueError('too many smooth vector tests for this mesh')


def build_tensor(G,n,ids,reverse=False):
    """T_mjk=<phi_m,G_j cross curl G_k>; Fourier extraction avoids dense tests."""
    G=jnp.asarray(G)
    rank=G.shape[1]
    fields=G.T.reshape(rank,3,n,n,n)
    geom=F.geometry(n)
    omega=jax.vmap(lambda u:F.ifft(1j*F.cross(geom[0],F.fft(u))))(fields)
    indices=np.asarray([x['wave'] for x in ids],dtype=int)%n
    pol=jnp.asarray([x['polarization'] for x in ids])
    cosine=jnp.asarray([x['kind']=='cos' for x in ids])

    @jax.jit
    def column(fields,one_omega,indices,pol,cosine):
        products=jax.vmap(lambda u:F.cross(u,one_omega))(fields)
        spectra=F.fft(products)
        selected=spectra[:,:,indices[:,0],indices[:,1],indices[:,2]]
        vector_value=jnp.einsum('rcm,mc->rm',selected,pol)
        real=jnp.where(cosine[None,:],vector_value.real,-vector_value.imag)
        return (np.sqrt(2*n**3)*real).T

    result=np.empty((len(ids),rank,rank),dtype=np.float64)
    order=range(rank-1,-1,-1) if reverse else range(rank)
    for k in order:
        result[:,:,k]=np.asarray(column(fields,omega[k],jnp.asarray(indices),pol,cosine))
    return result


def contract(T,c):
    return jnp.einsum('mjk,j,k->m',T,c,c)


def dense_projection(Phi,G,c,n):
    u=(jnp.asarray(G)@c).reshape(3,n,n,n)
    adv=F.ifft(F.nonlinear(F.fft(u),F.geometry(n))).ravel()
    return jnp.asarray(Phi).T@adv


def tensor_checks(G,n,m=32):
    Phi,lam,ids=test_modes(n,m)
    T=build_tensor(G,n,ids)
    rng=np.random.default_rng(190)
    errors=[];derivative=[]
    for _ in range(3):
        c=jnp.asarray(rng.normal(size=G.shape[1]))
        direction=jnp.asarray(rng.normal(size=G.shape[1]))
        expected=dense_projection(Phi,G,c,n)
        actual=contract(jnp.asarray(T),c)
        errors.append(float(jnp.linalg.norm(actual-expected)/jnp.maximum(jnp.linalg.norm(expected),1e-300)))
        dactual=jnp.einsum('mjk,j,k->m',jnp.asarray(T+T.swapaxes(1,2)),c,direction)
        dexpected=jax.jvp(lambda cc:dense_projection(Phi,G,cc,n),(c,),(direction,))[1]
        derivative.append(float(jnp.linalg.norm(dactual-dexpected)/jnp.maximum(jnp.linalg.norm(dexpected),1e-300)))
    reverse=build_tensor(G,n,ids,reverse=True)
    order_error=float(np.linalg.norm(T-reverse)/max(np.linalg.norm(T),1e-300))
    orth=float(np.linalg.norm(Phi.T@Phi-np.eye(m))/np.sqrt(m))
    return dict(passed=max(errors+derivative+[order_error,orth])<1e-9,
                contraction_errors=errors,jacobian_errors=derivative,
                build_order_error=order_error,test_orthogonality=orth)


def make_run(dt,nsteps,out_every,k,q,linear=False,budget=80,gtol=1e-7,cold_starts=4):
    """Complete query: full initial projection, latent fit, evolution, dense output.

    All large arrays and model parameters are explicit arguments. Linear endpoint
    uses coefficient coordinates only, not redundant latent plus full-bank q.
    Returns fields and per-step fit/solve convergence records.
    """
    assert nsteps%out_every==0

    def coefficient(w,theta,C):
        return w if linear else D.head(theta,w[:k])+C@w[k:]

    def weak(w,previous,nu,A,T,lam,C,theta):
        new=coefficient(w,theta,C)
        mid=(new+previous)/2
        return (A@(new-previous)-dt*(contract(T,mid)-nu*lam*(A@mid)))/(1+dt*nu*lam/2)

    def cold(w,target,Rb,C,theta):
        return Rb@(coefficient(w,theta,C)-target)

    evolve_lm=make_lm(weak,budget,gtol=gtol)
    cold_lm=make_lm(cold,max(200,budget),gtol=gtol)

    @jax.jit
    def run(u0,nu,G,Qb,Rb,A,T,lam,C,theta,training_codes):
        target=jnp.linalg.solve(Rb,Qb.T@u0.ravel())
        if linear:
            w=target
            cold_info=(jnp.asarray(0.),jnp.int32(0),jnp.int32(4),jnp.asarray(0.))
        else:
            H=D.head(theta,training_codes)
            scores=jnp.sum(((H-target)@Rb.T)**2,axis=1)
            zstarts=training_codes[jnp.argsort(scores)[:cold_starts]]
            starts=jnp.concatenate((zstarts,jnp.zeros((cold_starts,q))),axis=1)
            if q:
                projected=(target-D.head(theta,zstarts))@Rb.T@(Rb@C)
                starts=starts.at[:,k:].set(projected)
            fit=jax.vmap(lambda ww:cold_lm(ww,(target,Rb,C,theta),
                                          1e-12*jnp.linalg.norm(u0)))(starts)
            best=jnp.argmin(fit[1])
            w=fit[0][best]
            cold_info=(fit[1][best],fit[2][best],fit[3][best],fit[4][best])
        initial=G@coefficient(w,theta,C)

        def step(w,_):
            previous=coefficient(w,theta,C)
            next_w,rn,it,reason,gn=evolve_lm(w,(previous,nu,A,T,lam,C,theta),0.)
            return next_w,(rn,it,reason,gn)

        def block(w,_):
            next_w,info=jax.lax.scan(step,w,None,length=out_every)
            return next_w,(G@coefficient(next_w,theta,C),info)

        _,(fields,info)=jax.lax.scan(block,w,None,length=nsteps//out_every)
        fields=jnp.concatenate((initial[None],fields))
        return fields,cold_info,tuple(x.reshape(-1) for x in info)
    return run


def build_galerkin(G,n):
    """Exact quadratic Galerkin operators in an orthonormal solenoidal basis.

    These classical controls have no least-squares residual with M=k. They
    integrate the coefficient ODE with CNAB2, rather than manufacturing a slow
    fully implicit FOM. The pressure projection is orthogonal to the tests.
    """
    fields=jnp.asarray(G.T.reshape(G.shape[1],3,n,n,n))
    geom=F.geometry(n)
    omega=jax.vmap(lambda u:F.ifft(1j*F.cross(geom[0],F.fft(u))))(fields)
    lap=jax.vmap(lambda u:F.ifft(-geom[1]*F.fft(u)))(fields).reshape(G.shape[1],-1).T
    linear=np.asarray(jnp.asarray(G).T@lap)
    @jax.jit
    def column(G,fields,one_omega):
        product=jax.vmap(lambda u:F.cross(u,one_omega))(fields)
        return G.T@product.reshape(fields.shape[0],-1).T
    rank=G.shape[1]
    tensor=np.empty((rank,rank,rank),dtype=np.float64)
    for k in range(rank):
        tensor[:,:,k]=np.asarray(column(jnp.asarray(G),fields,omega[k]))
    return linear,tensor


def make_galerkin_run(dt,nsteps,out_every):
    assert nsteps%out_every==0
    @jax.jit
    def run(u0,nu,G,L,T):
        c0=G.T@u0.ravel()
        identity=jnp.eye(L.shape[0])
        half=jnp.linalg.cholesky(identity-.5*dt*nu*L)
        full=jnp.linalg.cholesky(identity-dt*nu*L)
        def solve(factor,b):
            return jax.scipy.linalg.cho_solve((factor,True),b)
        def step(carry,index):
            c,old=carry
            current=contract(T,c)
            def first():
                pred=solve(full,c+dt*current)
                return solve(half,c+.5*dt*(nu*(L@c)+current+contract(T,pred)))
            def normal():
                return solve(half,c+.5*dt*nu*(L@c)+dt*(1.5*current-.5*old))
            new=jax.lax.cond(index==0,first,normal)
            return (new,current),None
        def block(carry,index):
            new,_=jax.lax.scan(step,carry,index*out_every+jnp.arange(out_every))
            return new,G@new[0]
        _,fields=jax.lax.scan(block,(c0,jnp.zeros_like(c0)),jnp.arange(nsteps//out_every))
        return jnp.concatenate(((G@c0)[None],fields))
    return run
