"""Burgers 2D full-order kernels, copied verbatim (only the unused ROM parts and the sep_common import dropped)
from experiments/ops-all/g2d/src/burgers/experiments/mr-burgers2d/{engines.py,iterative_paths.py} @ 822f27166:
params_draw, coords, initial, spatial, residual, dst1, dst2, helmholtz, output_field, and iterative_paths.make_fom
with the FFT preconditioner (= the `fft_tight` reference solver when called with ntol 1e-6, ltol 1e-8, dt 0.005)."""
from __future__ import annotations
import numpy as np
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


def output_field(u, L):
    return jnp.pad(u.reshape(L-1,L-1),1)


def make_fom(L, dt=.005, max_newton=20):
    """iterative_paths.make_fom(L, dt, 'fft'), minus the diagnostic outputs."""
    def query(u0,nu,ntol,ltol):
        def pre(v):
            return helmholtz(v,nu,dt,L)
        def step(prev,_):
            scale=jnp.maximum(jnp.linalg.norm(prev),1e-300)
            def body(state):
                u,it,rn=state;r=residual(u,prev,nu,dt,L)
                def jv(v):return jax.jvp(lambda q:residual(q,prev,nu,dt,L),(u,),(v,))[1]
                delta,_=jax.scipy.sparse.linalg.bicgstab(jv,-r,tol=ltol,maxiter=200,M=pre)
                cand=u+delta
                return cand,it+1,jnp.linalg.norm(residual(cand,prev,nu,dt,L))
            state=(prev,jnp.int32(0),jnp.linalg.norm(residual(prev,prev,nu,dt,L)))
            u,it,rn=jax.lax.while_loop(lambda s:(s[2]>ntol*scale)&(s[1]<max_newton)&jnp.isfinite(s[2]),body,state)
            return u,(it,rn/scale)
        nsub=int(round(.05/dt))
        def block(u,_):
            u,(it,rn)=jax.lax.scan(step,u,None,length=nsub)
            return u,(output_field(u,L),it,rn)
        initial=u0[1:-1,1:-1].reshape(-1)
        _,(f,it,rn)=jax.lax.scan(block,initial,None,length=5)
        return jnp.concatenate((u0[None],f)),it.reshape(-1),rn.reshape(-1)
    return jax.jit(query)
