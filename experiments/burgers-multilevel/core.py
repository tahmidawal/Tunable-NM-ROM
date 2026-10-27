"""Fixed-fine-objective subspace experiments; baseline time loop is vendored p1core.
All arrays are runtime JIT arguments. No mesh, test or quadrature coarsening.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jcp-dims'))
import p1core as P
import jax
import jax.numpy as jnp
import numpy as np


def guarded(x, candidate, residual, args):
    old = jnp.linalg.norm(residual(x, args))
    new = jnp.linalg.norm(residual(candidate, args))
    accept = jnp.isfinite(new) & jnp.all(jnp.isfinite(candidate)) & (new < old)
    return jnp.where(accept, candidate, x), accept


def step_factory(nl, nlJ, rf, rc, trust, arm, budget=600):
    assert arm in ('direct', 'nested', 'twolevel')
    def res(c, args):
        hist, nu, data, co, dt = args
        return P.lspg_residual(nl, co, dt, nu, data, hist)(c)
    def ej(c, args):
        hist, nu, data, co, dt = args
        val, dn = nlJ(c, data, None)
        aw = data['A'] @ c
        S = 1 / (co[0] + dt * co[3] * nu * data['lam'])
        rr = co[0]*aw + co[1]*hist[0] + co[2]*hist[1] + dt*(co[3]*(val+nu*data['lam']*aw)+co[4]*hist[2])
        return S*rr, data['A'] + (dt*co[3]*S)[:,None]*dn
    def cej(z, extra):
        args, offset = extra
        hist, nu, data, co, dt = args
        c = offset.at[:rc].add(z)
        # Restricted derivative directly, without forming full-width nonlinear Jacobian.
        u, s = data['Gq'] @ c, data['Gs'] @ c
        val = data['Psi'].T @ (u*s)
        dn = data['Psi'].T @ (s[:,None]*data['Gq'][:,:rc]+u[:,None]*data['Gs'][:,:rc])
        aw = data['A'] @ c
        S = 1 / (co[0]+dt*co[3]*nu*data['lam'])
        rr = co[0]*aw+co[1]*hist[0]+co[2]*hist[1]+dt*(co[3]*(val+nu*data['lam']*aw)+co[4]*hist[2])
        return S*rr, data['A'][:,:rc]+(dt*co[3]*S)[:,None]*dn
    fine = P.make_lm(ej,budget,trust,1e-10)
    pre = P.make_lm(ej,1,trust,1e-10)
    coarse = P.make_lm(cej,4,trust,1e-10)
    def active_stage(x,args,tol,gtol,lam):
        before=x
        preit=jnp.int32(0); prej=jnp.int32(0)
        if arm == 'twolevel':
            x,_,preit,_,_,prej,lam=pre(x,args,tol,gtol,lam)
        prestate=x
        offset = jnp.zeros_like(x) if arm=='nested' else x
        z0 = x[:rc] if arm=='nested' else jnp.zeros(rc)
        rr,jj=ej(x,args)
        already=(jnp.linalg.norm(jj.T@rr)/(jnp.linalg.norm(jj)*jnp.linalg.norm(rr)+1e-300)<=gtol)
        def skipcoarse():
            return z0,jnp.linalg.norm(rr),jnp.int32(0),jnp.int32(4),jnp.float64(0),jnp.int32(0),lam
        z,rn,it,reason,gn,rej,lm=jax.lax.cond(already,skipcoarse,lambda:coarse(z0,(args,offset),0.,1e-3,lam))
        candidate=offset.at[:rc].add(z)
        chosen,accepted=guarded(x,candidate,res,args)
        return chosen,lam,(preit,prej,it,rej,accepted,before,prestate,candidate,chosen, jnp.int32(2)+(preit+1 if arm=='twolevel' else 0), jnp.where(already,0,it+1))
    def stage(x,args,tol,gtol,lam):
        r,j=ej(x,args)
        gamma=jnp.linalg.norm(j.T@r)/(jnp.linalg.norm(j)*jnp.linalg.norm(r)+1e-300)
        done=(gamma<=gtol)|(jnp.linalg.norm(r)<=tol)
        def skip():
            z=jnp.int32(0)
            return x,lam,(z,z,z,z,jnp.bool_(False),x,x,x,x,jnp.int32(1),z)
        return jax.lax.cond(done,skip,lambda:active_stage(x,args,tol,gtol,lam))
    def run(x,args,tol,gtol,lam):
        if arm=='direct':
            out=fine(x,args,tol,gtol,lam)
            meta=jnp.array([0,0,0,0,0,0,0],dtype=jnp.int32)
        else:
            x,lam,s=stage(x,args,tol,gtol,lam)
            out=fine(x,args,tol,gtol,lam)
            pi,pj,ci,cj,ac,*_=s
            fj,cjcount=s[-2:]
            meta=jnp.array([pi,pj,ci,cj,ac,fj,cjcount],dtype=jnp.int32)
        return out,meta
    return dict(run=run,res=res,ej=ej,cej=cej,stage=stage)


def make_query(Q,rf,rc,L,trust,arm,budget=600):
    """Reuse exact inherited time loop with a local LM replacement, instrumented work.
    Construct via source transformation pinned by explicit assertions. Original file is untouched.
    """
    import inspect
    src=inspect.getsource(P.make_evolve)
    assert src.count('lm = make_lm(evalJ, budget, trust, ridge, solver, clip)')==1
    src=src.replace('lm = make_lm(evalJ, budget, trust, ridge, solver, clip)',
                    "ml = step_factory(nl, nlJ, Rp, RC, trust, ARM, budget)\n    lm = ml['run']")
    src=src.replace('w2, rn, it, reason, gn, rej, lamn = lm(wi, args, tol, sch[\'gtol\'], lam0)',
                    "(w2, rn, it, reason, gn, rej, lamn), mlwork = lm(wi, args, tol, sch['gtol'], lam0)")
    src=src.replace('stats0 = dict(it_sum=', 'stats0 = dict(mlwork=jnp.zeros((nstep,7),jnp.int32), damping=jnp.zeros(nstep), it_sum=')
    src=src.replace("st = dict(it_sum=st['it_sum']", "st = dict(mlwork=st['mlwork'].at[k].set(mlwork), damping=st['damping'].at[k].set(lamn), it_sum=st['it_sum']")
    ns=dict(P.__dict__,step_factory=step_factory,RC=rc,ARM=arm)
    exec(compile(src,'<instrumented-p1core>','exec'),ns)
    nl,nlJ=Q.tested_value('point',L),Q.tested_jac('point',L)
    evolve=ns['make_evolve'](nl,nlJ,'LSPG',rf,trust,budget=budget,nrec=51,nstep=50)
    original=P.make_split_query(Q,'point','LSPG',rf,L,trust,nrec=51,nstep=50,budget=budget)
    def query(u0,nu,data,cold,sch):
        w0,scale=original['init'](u0,cold)
        W,st=evolve(w0,nu,scale,data,sch)
        return dict(fields=original['decode'](W,data),W=W,w0=w0,stats=st)
    return jax.jit(query), original
