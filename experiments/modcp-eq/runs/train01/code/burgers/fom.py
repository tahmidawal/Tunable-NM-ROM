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

