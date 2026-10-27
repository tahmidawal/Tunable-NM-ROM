"""Small numerical tests of the actual closures; no production data or model is loaded."""
import inspect
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jcp-qualify'))
import p1core as original
import prefixcore as candidate
import jax
import jax.numpy as jnp


def actual_evaluators(nl, nlJ, r, rc, trust=1.):
    """Capture closures passed to the unchanged LM factory; restore before execution."""
    captured = []
    factory = candidate.make_lm
    def capture(evalJ, *args, **kwargs):
        captured.append(evalJ)
        return factory(evalJ, *args, **kwargs)
    candidate.make_lm = capture
    try:
        candidate.make_evolve(nl, nlJ, 'LSPG', r, trust, prefix_mode='affine2', coarse_width=rc)
    finally:
        candidate.make_lm = factory
    assert len(captured) == 2
    return captured


def run():
    assert jax.config.jax_enable_x64
    assert inspect.getsource(original.make_lm) == inspect.getsource(candidate.make_lm)
    rng = np.random.default_rng(20261027)
    r, rc, m, M = 6, 2, 13, 11
    d = {k: jnp.asarray(rng.normal(size=shape)) for k, shape in
         [('A', (M,r)), ('Gq', (m,r)), ('Gs', (m,r)), ('Psi', (m,M))]}
    d.update(lam=jnp.arange(1., M+1), hmask=jnp.zeros(M))
    nl = lambda c, d: d['Psi'].T @ ((d['Gq']@c)*(d['Gs']@c))
    def nlj(c,d,T):
        u,s=d['Gq']@c,d['Gs']@c
        return nl(c,d), d['Psi'].T@(s[:,None]*d['Gq']+u[:,None]*d['Gs'])
    full, narrow = actual_evaluators(nl,nlj,r,rc)
    w=jnp.asarray(rng.normal(size=r))*.1
    z=jnp.asarray(rng.normal(size=rc))*.02
    vals={}
    for scheme,co in [('BE',jnp.asarray(original.BE_CO)),('CN',jnp.asarray([1.,-1.,0.,.5,.5]))]:
        hist=tuple(jnp.asarray(rng.normal(size=M)) for _ in range(3))
        args=(hist, .03, d, co, .005)
        for label,base,zz in [('affine',w,z),('pure',jnp.zeros_like(w),w[:rc])]:
            bundle=(args,d['Gq']@base,d['Gs']@base,d['A']@base)
            c=base.at[:rc].add(zz)
            rv,Jc=narrow(zz,bundle)
            rf,Jf=full(c,args)
            independent_residual=original.lspg_residual(nl,co,.005,.03,d,hist)
            Ja=jax.jacfwd(lambda v:independent_residual(base.at[:rc].add(v)))(zz)
            key=scheme+'_'+label
            vals[key]={'residual':float(jnp.max(jnp.abs(rv-rf))),
                         'column_jacobian':float(jnp.max(jnp.abs(Jc-Jf[:,:rc]))),
                         'autodiff_jacobian':float(jnp.max(jnp.abs(Jc-Ja)))}
            assert max(vals[key].values()) < 1e-11, vals
    # Positive and negative transfer tests with nonzero fine tail.
    pad=jnp.pad(z,(0,r-rc))
    vals['injection']=float(jnp.max(jnp.abs(d['Gq']@pad-d['Gq'][:,:rc]@z)))
    vals['tail_loss']=float(jnp.linalg.norm(d['Gq']@w-d['Gq']@jnp.pad(w[:rc],(0,r-rc))))
    assert vals['injection'] < 1e-13 and vals['tail_loss'] > .01
    sch=original.sched('CNR',.05,1e-3,1e-9)
    ini=w*.01
    a=jax.jit(original.make_evolve(nl,nlj,'LSPG',r,1.,budget=50,nrec=6,nstep=5))(ini,.03,1.,d,sch)
    b=jax.jit(candidate.make_evolve(nl,nlj,'LSPG',r,1.,budget=50,nrec=6,nstep=5,prefix_mode='none',coarse_width=rc))(ini,.03,1.,d,sch)
    vals['baseline_states']=float(jnp.max(jnp.abs(a[1]['states']-b[1]['states'])))
    assert vals['baseline_states'] < 1e-13
    assert np.array_equal(a[1]['step_it'],b[1]['step_it'])
    assert np.array_equal(a[1]['step_reason'],b[1]['step_reason'])
    # Force the actual coarse factory to return a finite, numerically failed proposal.
    # Its rejection must preserve all baseline decisions and propagated fine damping.
    real_factory=candidate.make_lm
    def force_failed_coarse(evalJ,budget,*args,**kwargs):
        if budget==2:
            return lambda z,bundle,tol,gtol,lam:(z+1.,jnp.float64(1.),jnp.int32(1),jnp.int32(3),
                                               jnp.float64(1.),jnp.int32(1),lam*100.)
        return real_factory(evalJ,budget,*args,**kwargs)
    candidate.make_lm=force_failed_coarse
    try:
        forced=candidate.make_evolve(nl,nlj,'LSPG',r,1.,budget=50,nrec=6,nstep=5,prefix_mode='affine2',coarse_width=rc)
    finally:
        candidate.make_lm=real_factory
    f=jax.jit(forced)(ini,.03,1.,d,sch)
    assert np.array_equal(f[1]['coarse_accept'],np.zeros(5))
    np.testing.assert_allclose(f[1]['states'],a[1]['states'],rtol=0,atol=1e-13)
    assert all(np.array_equal(f[1][k],a[1][k]) for k in ('step_reason','step_it','exits','rej'))
    vals['failed_coarse_restores_baseline']=True
    return {'pass':True,'backend':jax.default_backend(),'x64':True,'checks':vals}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
