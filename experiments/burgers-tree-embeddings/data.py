"""Exact new-family generator; final solutions are guarded independently of seeds."""
import numpy as np
from common import jnp

SEEDS = dict(train=2026111801, dev=2026111802, final_test=2026111803)
TIMES = np.arange(11, dtype=np.float64) * .025
NU = .005

def parameters(cohort, count=None, frozen=False):
    if cohort == 'final_test' and not frozen:
        raise PermissionError('Final test parameters/fields remain sealed until method freeze.')
    n = count if count is not None else dict(train=64, dev=16, final_test=64)[cohort]
    rows = []
    for i in range(n):
        r = np.random.default_rng(np.random.SeedSequence([SEEDS[cohort], i]))
        if i % 4 == 3:
            rows.append([1., 0., 0., 1., 0., r.uniform(.4, .8)])
        else:
            cx, cy = r.uniform(.25, .65), r.uniform(.25, .65)
            sigma = np.exp(r.uniform(np.log(.035), np.log(.075)))
            rows.append([0., cx, cy, sigma, r.uniform(.8, 1.4), .15])
    return np.asarray(rows, dtype=np.float64)

def alignment():
    rows=[]
    for sig in (.035, .055):
        for delta in (-.01, 0., .01):
            for cx,cy in ((.5+delta,.35),(.5+delta,.5-delta)):
                rows.append([0.,cx,cy,sig,1.2,.15])
    return np.asarray(rows, dtype=np.float64)

def fixtures():
    return np.array([[0., .35, .35, .045, 1.2, .15], [1.,0.,0.,1.,0.,.8]])

def initial(x, p):
    """The callable is the online input. Parameters do not go to allocation policies."""
    x=jnp.asarray(x); p=jnp.asarray(p)
    env=jnp.sin(jnp.pi*x[...,0])**2*jnp.sin(jnp.pi*x[...,1])**2
    pulse=p[4]*jnp.exp(-((x[...,0]-p[1])**2+(x[...,1]-p[2])**2)/(2*p[3]**2))
    return env*(p[5]+(1-p[0])*pulse)

def grid(n, gauss=False):
    if gauss:
        x,w=np.polynomial.legendre.leggauss(n); x=(x+1)/2; w=w/2
    else:
        x=np.linspace(0,1,n); w=np.full(n,1/(n-1)); w[[0,-1]]*=.5
    xy=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
    return xy, np.outer(w,w).ravel()

def initial_coefficients(p, N, panels=64):
    """Composite Gauss integration, separable envelope/pulse; not a sampled IC fit."""
    from vendor import pscore as P
    x,w=P.composite_gl(panels,32)
    base=np.sin(np.pi*x)**2
    g=P.sine_integrals(base,x,w,N)
    b=p[5]*np.outer(g,g)
    if p[0]==0:
        gx=P.sine_integrals(base*np.exp(-(x-p[1])**2/(2*p[3]**2)),x,w,N)
        gy=P.sine_integrals(base*np.exp(-(x-p[2])**2/(2*p[3]**2)),x,w,N)
        b+=p[4]*np.outer(gx,gy)
    return b

def synth(b, xy):
    k=jnp.arange(1,b.shape[-1]+1,dtype=jnp.float64)
    sx=jnp.sin(jnp.pi*xy[:,0,None]*k)
    sy=jnp.sin(jnp.pi*xy[:,1,None]*k)
    return jnp.einsum('pi,...ij,pj->...p',sx,b,sy)

def synth_grid(b, x):
    k=jnp.arange(1,b.shape[-1]+1,dtype=jnp.float64)
    S=jnp.sin(jnp.pi*jnp.asarray(x)[:,None]*k)
    return jnp.einsum('ai,...ij,bj->...ab',S,b,S)
