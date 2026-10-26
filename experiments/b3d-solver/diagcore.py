"""One-step fresh-restart diagnostics. Original trajectory kernels remain unchanged in vendor/."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'/'speed3d'))
import c3common as C3
import qn3
import t2core as T2
import jax
import jax.numpy as jnp

def step_functions(scheme='BDF2', dt=.025):
    """Residual/Jacobian and all production predictor candidates at exactly the saved history."""
    def parts(W,k,nu,data):
        vals=T2.SCHEMES[scheme]
        co=[jnp.where(k<vals[5],a,b) for a,b in zip(T2.BE_CO,vals[:5])]
        a0,a1,a2,b0,b1=co
        A,lam=data['A'],data['lam']
        cn,cm,cm2,cm3=[W[jnp.maximum(k-j,0)] for j in range(4)]
        adv=lambda c: ((data['B']@c)*(data['D']@c))@data['P']
        S=1/(a0+dt*b0*nu*lam)
        Fn=adv(cn)+nu*lam*(A@cn)
        hist=a1*(A@cn)+a2*(A@cm)+dt*b1*Fn
        return co,S,hist,cn,cm,cm2,cm3
    def residual(c,args):
        W,k,nu,data=args
        co,S,hist,*_=parts(W,k,nu,data)
        a0,a1,a2,b0,b1=co
        Ac=data['A']@c
        adv=((data['B']@c)*(data['D']@c))@data['P']
        return S*(a0*Ac+hist+dt*b0*(adv+nu*data['lam']*Ac))
    def rj(c,args):
        W,k,nu,data=args
        co,S,hist,*_=parts(W,k,nu,data)
        a0,a1,a2,b0,b1=co
        Ju=C3.OM.contract_offmesh(data,c)
        Ac=data['A']@c
        r=S*(a0*Ac+hist+dt*b0*(.5*Ju@c+nu*data['lam']*Ac))
        return r,data['A']+(dt*b0*S)[:,None]*Ju
    def candidates(args):
        W,k,nu,data=args
        co,S,hist,cn,cm,cm2,cm3=parts(W,k,nu,data)
        adv=lambda c:((data['B']@c)*(data['D']@c))@data['P']
        advt=jnp.where(k>=1,2*adv(cn)-adv(cm),adv(cn))
        v=S*(hist+dt*co[3]*advt)
        p4=-jax.scipy.linalg.solve_triangular(data['RA'],v@data['QA'],lower=False)
        return jnp.stack([cn,2*cn-cm,3*cn-3*cm+cm2,4*cn-6*cm+4*cm2-cm3,p4])
    return residual,rj,candidates

def make_lm(rj, trust, gtol, budget=600):
    """Fresh full-Jacobian LM, matching t2core arithmetic, with explicit work traces/counters."""
    def run(w0,args):
        r,J=rj(w0,args); rn=jnp.linalg.norm(r); g=J.T@r
        ratio=lambda gg,jj,rr: jnp.linalg.norm(gg)/(jnp.linalg.norm(jj)*rr+1e-300)
        gamma=ratio(g,J,rn)
        reason=jnp.where(~jnp.isfinite(rn),3,jnp.where(gamma<=gtol,4,0)).astype(jnp.int32)
        trace=jnp.full((budget,6),jnp.nan)
        s0=(w0,r,J,g,rn,jnp.float64(1e-6),jnp.int32(0),reason,jnp.int32(0),jnp.int32(0),trace)
        def body(s):
            w,r,J,g,rn,mu,it,reason,rejects,clips,trace=s
            H=J.T@J; dd=jnp.diag(H)+1e-30
            L=jax.scipy.linalg.cho_factor(H+jnp.diag(mu*dd),lower=True)
            dw=jax.scipy.linalg.cho_solve(L,-g)
            nz=jnp.linalg.norm(dw); clipped=nz>trust
            dw=dw*jnp.where(clipped,trust/(nz+1e-300),1.)
            finite=jnp.all(jnp.isfinite(dw))
            wn=w+jnp.where(finite,dw,0.)
            r2,J2=rj(wn,args); rn2=jnp.linalg.norm(r2)
            accept=finite&jnp.isfinite(rn2)&(rn2<rn)
            sel=lambda x,y:jnp.where(accept,x,y)
            w3,r3,J3,rn3=sel(wn,w),sel(r2,r),sel(J2,J),sel(rn2,rn)
            g3=J3.T@r3; gn=ratio(g3,J3,rn3)
            tiny=finite&(jnp.linalg.norm(dw)<=1e-14*(1+jnp.linalg.norm(w)))
            reason=jnp.where(gn<=gtol,4,jnp.where(tiny,2,jnp.where((~accept)&(mu>=1e14),3,0))).astype(jnp.int32)
            trace=trace.at[it].set(jnp.array([rn3,gn,mu,nz,clipped,accept]))
            mun=jnp.where(accept,jnp.maximum(mu/3,1e-12),jnp.minimum(mu*10,1e14))
            return w3,r3,J3,g3,rn3,mun,it+1,reason,rejects+(~accept).astype(jnp.int32),clips+clipped.astype(jnp.int32),trace
        w,r,J,g,rn,mu,it,reason,rejects,clips,trace=jax.lax.while_loop(lambda s:(s[6]<budget)&(s[7]==0),body,s0)
        return dict(c=w,rn=rn,gamma=ratio(g,J,rn),mu=mu,it=it,reason=reason,rejects=rejects,clips=clips,
                    residual_evals=it+1,jacobian_evals=it+1,trace=trace)
    return jax.jit(run)

def check_step(W,k,nu,data,pidx,trust):
    """Compare baseline return with tight/tighter fresh solves and independent derivative controls."""
    res,rj,candidate_fn=step_functions()
    args=(W,jnp.int32(k),jnp.float64(nu),data)
    start=jax.jit(candidate_fn)(args)[pidx]
    runs={level:jax.block_until_ready(make_lm(rj,trust,gt)(start,args))
          for level,gt in [('tight',1e-7),('tighter',1e-8)]}
    c=W[k+1];r,J=jax.jit(rj)(c,args)
    rr=jax.jit(res)(c,args)
    v=jnp.sin(jnp.arange(c.size,dtype=jnp.float64)+1);v=v/jnp.linalg.norm(v)
    jv=jax.jit(lambda x,v,args:jax.jvp(lambda y:res(y,args),(x,),(v,))[1])(c,v,args)
    s=jnp.linalg.svd(J,compute_uv=False)
    diagnostics=dict(baseline_residual=jnp.linalg.norm(r),baseline_gamma=jnp.linalg.norm(J.T@r)/(jnp.linalg.norm(J)*jnp.linalg.norm(r)+1e-300),
                     residual_agreement=jnp.linalg.norm(r-rr)/(jnp.linalg.norm(rr)+1e-300),
                     jvp_agreement=jnp.linalg.norm(J@v-jv)/(jnp.linalg.norm(jv)+1e-300),
                     singular_min=s[-1],singular_max=s[0],condition=s[0]/s[-1])
    return runs,jax.block_until_ready(diagnostics)
