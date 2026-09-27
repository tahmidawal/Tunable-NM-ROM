"""Burgers multiresolution kernels; intervals L imply (L+1)^2 total nodes.

The state is the (L-1)^2 interior, with zero Dirichlet boundary. The FOM
discretization matches burgers2d_film, but its Helmholtz preconditioner uses
FFT DST-I instead of dense sine matrix products. All large data are arguments.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import scipy.optimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'separable-decoder'))
import sep_common as sc


def params_draw(seed, count):
    """Exactly the incumbent sequential RNG draws/ranges; descriptors only make data."""
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                     r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                     np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)


def coords(L, interior=True):
    x = np.arange(1, L) / L if interior else np.arange(L + 1) / L
    return np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)


def initial(L, physical):
    c = coords(L, False)
    cx, cy, w, a, _ = physical
    u = (a * np.exp(-((c[:, 0]-cx)**2 + (c[:, 1]-cy)**2)/(2*w*w))).reshape(L+1,L+1)
    u[[0,-1], :] = 0.; u[:, [0,-1]] = 0.
    return u


def spatial(u, L):
    p = jnp.pad(u.reshape(L-1,L-1), 1)
    c, xm, xp, ym, yp = p[1:-1,1:-1],p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:]
    adv = c * L * (jnp.where(c>0,c-xm,xp-c) + jnp.where(c>0,c-ym,yp-c))
    lap = L**2 * (xm+xp+ym+yp-4*c)
    return adv.reshape(-1), lap.reshape(-1)


def residual(u, prev, nu, dt, L):
    adv, lap = spatial(u, L)
    return u-prev+dt*(adv-nu*lap)


def dst1(x):
    """Orthonormal, self-inverse DST-I on the final axis, odd-extension FFT."""
    n = x.shape[-1]
    zero = jnp.zeros(x.shape[:-1]+(1,), x.dtype)
    ext = jnp.concatenate((zero,x,zero,-x[...,::-1]), axis=-1)
    return -jnp.fft.fft(ext, axis=-1).imag[...,1:n+1] / jnp.sqrt(2.*(n+1))


def dst2(x):
    return dst1(dst1(x).swapaxes(-1,-2)).swapaxes(-1,-2)


def helmholtz(v, nu, dt, L):
    k = jnp.arange(1,L,dtype=jnp.float64)
    lam = 4*L**2*jnp.sin(jnp.pi*k/(2*L))**2
    x = dst2(v.reshape(L-1,L-1)) / (1+dt*nu*(lam[:,None]+lam[None,:]))
    return dst2(x).reshape(-1)


def output_field(u, L, target):
    a = jnp.pad(u.reshape(L-1,L-1),1)
    if target == L:
        return a
    # Aligned bilinear interpolation: coarse/fine boundaries and nested nodes exact.
    x = jnp.arange(target+1,dtype=jnp.float64)*L/target
    lo = jnp.minimum(x.astype(jnp.int32),L-1); f=x-lo
    a = (1-f[:,None])*a[lo,:] + f[:,None]*a[lo+1,:]
    return (1-f[None,:])*a[:,lo] + f[None,:]*a[:,lo+1]


def make_fom(L, dt, target=None, horizon=.25, output_spacing=.05, max_newton=20):
    target = L if target is None else target
    substeps=int(round(output_spacing/dt)); nout=int(round(horizon/output_spacing))
    assert abs(substeps*dt-output_spacing)<1e-12 and target%L==0

    def step(prev, nu, ntol, ltol):
        scale=jnp.maximum(jnp.linalg.norm(prev),1e-300)
        def cond(s):
            return (s[2]>ntol*scale)&(s[1]<max_newton)&jnp.isfinite(s[2])
        def body(s):
            u,it,rn=s; r=residual(u,prev,nu,dt,L)
            def jv(v):
                return jax.jvp(lambda q:residual(q,prev,nu,dt,L),(u,),(v,))[1]
            delta,_=jax.scipy.sparse.linalg.bicgstab(jv,-r,tol=ltol,maxiter=200,
                M=lambda v:helmholtz(v,nu,dt,L))
            cand=u+delta; rn2=jnp.linalg.norm(residual(cand,prev,nu,dt,L))
            return cand,it+1,rn2
        u,it,rn=jax.lax.while_loop(cond,body,(prev,jnp.int32(0),jnp.linalg.norm(residual(prev,prev,nu,dt,L))))
        return u,(it,rn/scale)

    def query(input_field, nu, ntol, ltol):
        stride=target//L
        u0=input_field[::stride,::stride][1:-1,1:-1].reshape(-1)
        def block(u,_):
            u,(it,rn)=jax.lax.scan(lambda u,_:step(u,nu,ntol,ltol),u,None,length=substeps)
            return u,(output_field(u,L,target),it,rn)
        _,(fields,it,rn)=jax.lax.scan(block,u0,None,length=nout)
        return jnp.concatenate((output_field(u0,L,target)[None],fields)),it.reshape(-1),rn.reshape(-1)
    return jax.jit(query), jax.jit(step)


def modes(L,M):
    k=np.arange(1,L)
    kx,ky=np.meshgrid(k,k,indexing='ij')
    lam=4*L**2*(np.sin(np.pi*kx/(2*L))**2+np.sin(np.pi*ky/(2*L))**2)
    ind=np.argsort(lam.ravel(),kind='stable')[:M]
    kx,ky=kx.ravel()[ind],ky.ravel()[ind]
    c=coords(L)
    phi=(2./L)*np.sin(np.pi*c[:,0,None]*kx)*np.sin(np.pi*c[:,1,None]*ky)
    return phi,lam.ravel()[ind],np.stack((kx,ky),axis=1)


def gj_solve(A,b):
    """Existing b1d_fast_common Gauss-Jordan algebra, for regularized SPD normals."""
    n=A.shape[0]; mat=jnp.concatenate((A,b[:,None]),1); rows=jnp.arange(n)
    for k in range(n):
        row=mat[k]/mat[k,k]
        mat=mat-jnp.where(rows==k,0.,mat[:,k])[:,None]*row[None,:]
        mat=jnp.where((rows==k)[:,None],row[None,:],mat)
    return mat[:,n]


def make_lm(fun,K,budget,trust=np.inf,stall=1e-3,linear='gj'):
    solve=gj_solve if linear=='gj' else jnp.linalg.solve
    def lm(z0,args,tol):
        r=fun(z0,*args); J=jax.jacfwd(fun)(z0,*args); rn=jnp.linalg.norm(r)
        # reason: 0 budget; 1 residual tolerance; 2 stationary/stall; 3 nonfinite/rejected.
        reason=jnp.where(jnp.isfinite(rn),jnp.where(rn<=tol,1,0),3).astype(jnp.int32)
        def body(s):
            z,r,J,rn,lam,it,reason=s
            H=J.T@J; g=J.T@r
            dz=solve(H+lam*jnp.diag(jnp.diag(H)+1e-30),-g)
            ok=jnp.all(jnp.isfinite(dz))&(jnp.linalg.norm(dz)<=trust)
            zn=z+jnp.where(ok,dz,0.); rn2=jnp.linalg.norm(fun(zn,*args))
            accept=ok&jnp.isfinite(rn2)&(rn2<rn)
            r2,J2=jax.lax.cond(accept,lambda:(fun(zn,*args),jax.jacfwd(fun)(zn,*args)),lambda:(r,J))
            tiny=ok&(jnp.linalg.norm(dz)<=1e-12*(1+jnp.linalg.norm(z)))
            reason=jnp.where(accept&(rn2<=tol),1,jnp.where(tiny|(accept&((rn-rn2)/rn<stall)),2,
                    jnp.where((~accept)&(lam>=1e12),3,0))).astype(jnp.int32)
            return jnp.where(accept,zn,z),r2,J2,jnp.where(accept,rn2,rn),jnp.where(accept,jnp.maximum(lam/3,1e-12),jnp.minimum(lam*10,1e12)),it+1,reason
        z,r,J,rn,lam,it,reason=jax.lax.while_loop(lambda s:(s[5]<budget)&(s[6]==0),body,
            (z0,r,J,rn,jnp.asarray(1e-6),jnp.int32(0),reason))
        return z,rn,it,reason
    return lm


def nnls_capped(G,b,max_support):
    """Greedy positive active-set selection, exact nonnegative refit after each add.

    Same Lawson-Hanson support criterion as blat_common; scipy NNLS solves the
    restricted problem, bounded support is followed by a full-row final refit.
    """
    support=[]; w=np.empty(0); residual_=b.copy()
    for _ in range(5*max_support):
        grad=G.T@residual_
        if support: grad[support]=-np.inf
        idx=int(np.argmax(grad))
        if grad[idx]<=1e-10*max(np.linalg.norm(b),1e-300) or len(support)>=max_support:break
        support.append(idx)
        w,_=scipy.optimize.nnls(G[:,support],b,maxiter=10*len(support))
        active=w>1e-14; support=list(np.asarray(support)[active]);w=w[active]
        residual_=b-G[:,support]@w
    return np.asarray(support,dtype=int),w


def build_rom(params,Z,L,M,m,eq_seed=20259,candidate_cap=8192,fit_states=64):
    """Rebuild mesh-dependent exact linear terms and decoder-output NNLS rule."""
    t0=time.perf_counter();R=params['h_lin'].shape[1];K=Z.shape[1]
    assert M>=4*K and m>=4*M
    dec=sc.SeparableDecoder(params,K,R)
    G=dec.feat_at(coords(L),chunk=8192);jax.block_until_ready(G)
    Phi,lam,mode_ids=modes(L,M);P=jnp.asarray(Phi);A=P.T@G
    rng=np.random.default_rng(eq_seed)
    cp=np.sort(rng.choice((L-1)**2,min(candidate_cap,(L-1)**2),replace=False))
    fit=np.sort(rng.choice(len(Z),min(fit_states,len(Z)),replace=False))
    h=jax.jit(lambda z:sc.head(params,z))
    adv=jax.jit(lambda G,z:spatial(G@h(z),L)[0])
    rows=[];targets=[]
    for i in fit:
        n=adv(G,jnp.asarray(Z[i]));targets.append(np.asarray(P.T@n));rows.append(Phi[cp].T*np.asarray(n)[cp])
    design=np.concatenate(rows);b=np.concatenate(targets)
    scale=np.linalg.norm(design,axis=1)+1e-300;design/=scale[:,None];b/=scale
    supp,w=nnls_capped(design,b,m)
    if len(supp)<m:
        rest=np.setdiff1d(np.arange(len(cp)),supp)
        supp=np.concatenate((supp,rest[np.argsort(-np.abs(design[:,rest]).mean(0))[:m-len(supp)]]))
    w,_=scipy.optimize.nnls(design[:,supp],b,maxiter=10*len(supp))
    pos=cp[supp];ij=np.stack(np.unravel_index(pos,(L-1,L-1)),1)+1
    offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    gst=dec.feat_at(((ij[:,None,:]+offsets[None,:,:])/L).reshape(-1,2)).reshape(m,5,R)
    Pq=jnp.asarray(Phi[pos]*w[:,None])
    # Fixed bounded cold-start sampling from the dense input. It is a least-squares
    # initial-state fit, not random PDE collocation. No physical descriptors enter.
    ix=np.unique(np.rint(np.linspace(1,L-1,min(48,L-1))).astype(int))
    ii,jj=np.meshgrid(ix,ix,indexing='ij');ic_pos=((ii-1)*(L-1)+jj-1).ravel()
    Gi=G[jnp.asarray(ic_pos)]/np.sqrt(len(ic_pos));gram=Gi.T@Gi
    chol=jnp.linalg.cholesky(gram+1e-12*jnp.trace(gram)/R*jnp.eye(R))
    # Select among training t=0 codes when lineage is recorded; otherwise all train codes.
    stride=max(1,len(Z)//8192);candidate_Z=jnp.asarray(Z[::stride]);H=sc.head(params,candidate_Z)
    cH=H@chol;norm=jnp.sum(cH*cH,axis=1)
    radius=float(np.max(np.linalg.norm(Z-Z.mean(0),axis=1)))
    data=(G,A,jnp.asarray(lam),gst,Pq,Gi,chol,candidate_Z,H,norm,jnp.asarray(ii.ravel()),jnp.asarray(jj.ravel()))
    info=dict(intervals=L,total_nodes=(L+1)**2,interior_unknowns=(L-1)**2,boundary_nodes=4*L,K=K,R=R,M=M,m=m,
        setup_seconds=time.perf_counter()-t0,eq_relative_fit=float(np.linalg.norm(design[:,supp]@w-b)/np.linalg.norm(b)),
        eq_seed=eq_seed,eq_fit_rows=fit.tolist(),eq_candidate_cap=candidate_cap,eq_actual_candidates=len(cp),
        eq_indices=pos.tolist(),eq_weights=w.tolist(),cold_points=len(ic_pos),cold_candidates=len(candidate_Z),
        trust_radius=.01*radius,array_bytes=sum(x.nbytes for x in jax.tree_util.tree_leaves(data)),
        mode_ids=mode_ids.tolist())
    return data,info


def weak(z,prev,nu,data,params,L,dt):
    _,A,lam,G5,Pq,*_=data;h=sc.head(params,z)
    us=jnp.einsum('msr,r->ms',G5,h);c,xp,xm,yp,ym=[us[:,i] for i in range(5)]
    adv=c*L*(jnp.where(c>0,c-xm,xp-c)+jnp.where(c>0,c-ym,yp-c))
    ah=A@h
    return (ah-prev+dt*(Pq.T@adv+nu*lam*ah))/(1+dt*nu*lam)


def make_rom(params,L,dt,trust,stall=1e-3,budget=30,linear='gj',ic_starts=1):
    K=params['h_lin'].shape[0];substeps=int(round(.05/dt));assert abs(substeps*dt-.05)<1e-12
    def fitfun(z,y,data):return data[6].T@sc.head(params,z)-y
    ic_lm=make_lm(fitfun,K,60,stall=1e-7,linear=linear)
    step_lm=make_lm(lambda z,p,nu,data:weak(z,p,nu,data,params,L,dt),K,budget,trust,stall,linear)
    def query(u0,nu,data):
        G,A,lam,G5,Pq,Gi,chol,Zc,Hc,norm,ii,jj=data
        ui=u0[ii,jj]/jnp.sqrt(len(ii));b=Gi.T@ui
        y=jax.scipy.linalg.solve_triangular(chol,b,lower=True)
        score=norm-2*Hc@b;inds=jnp.argsort(score)[:ic_starts]
        fits=jax.vmap(lambda z:ic_lm(z,(y,data),0.))(Zc[inds])
        best=jnp.argmin(fits[1]);z0=fits[0][best]
        def step(carry,_):
            z,zprev=carry;p=A@sc.head(params,z);ze=z+(z-zprev)
            r0=jnp.linalg.norm(weak(z,p,nu,data,params,L,dt));re=jnp.linalg.norm(weak(ze,p,nu,data,params,L,dt))
            zi=jnp.where(jnp.isfinite(re)&(re<r0),ze,z)
            z2,rn,it,reason=step_lm(zi,(p,nu,data),1e-9*jnp.linalg.norm(ui)*np.sqrt(len(ii)))
            return (z2,z),(z2,rn,it,reason)
        def block(carry,_):
            carry,details=jax.lax.scan(step,carry,None,length=substeps)
            return carry,(carry[0],details[1],details[2],details[3])
        _,(Z,rn,it,reason)=jax.lax.scan(block,(z0,z0),None,length=5)
        Z=jnp.concatenate((z0[None],Z))
        fields=jax.vmap(lambda z:output_field(G@sc.head(params,z),L,L))(Z)
        return fields,it.reshape(-1),rn.reshape(-1),reason.reshape(-1),Z,fits[2][best],fits[3][best]
    return jax.jit(query)


def burn(seconds=.4):
    a=jnp.ones((512,512),jnp.float64)*.01
    kernel=jax.jit(lambda a:a@a/512+.01)
    t=time.perf_counter()
    while time.perf_counter()-t<seconds:a=kernel(a);jax.block_until_ready(a)


def errors(field,reference,Lobs=256):
    field=np.asarray(field);reference=np.asarray(reference)
    L=field.shape[-1]-1
    assert L%Lobs==0
    f=field[:,::L//Lobs,::L//Lobs]
    n0=np.linalg.norm(reference[0]);nr=np.linalg.norm(reference.reshape(len(reference),-1),axis=1)
    e=np.linalg.norm((f-reference).reshape(len(reference),-1),axis=1)
    return dict(fixed_initial_per_time=(e/n0).tolist(),current_relative_per_time=(e/np.maximum(nr,1e-300)).tolist(),
                fixed_initial_max=float(np.max(e/n0)),current_relative_max=float(np.max(e/np.maximum(nr,1e-300))),
                absolute_rms_max=float(np.max(e)/(Lobs+1)))
