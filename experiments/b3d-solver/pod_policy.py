"""One predeclared stronger production stopping policy; all POD equations and gates unchanged."""
import jax
import jax.numpy as jnp

def stronger(subject,gtol):
    assert gtol==1e-4
    s=dict(subject);q=s['query'];data=s['data'];sch=dict(s['sch'],gtol=jnp.float64(gtol))
    original_validate=s['validate'];ev=s['evolution_query'];ce=s['fresh_certifier']
    def run(u,nu):return q(u,nu,data,sch)
    def evolve(c0,nu,scale):return ev(c0,nu,scale,data,sch)
    def certify(out,nu):
        # Fresh certificate uses unchanged original production Q1 threshold; gtol controls the solve only.
        return subject['certify'](out,nu)
    def validate(u,nu,prod=None,need_tighter=True,n0=None):
        if prod is None:prod=jax.block_until_ready(run(u,nu))
        detail=original_validate(u,nu,prod=prod,need_tighter=need_tighter,n0=n0)
        detail['production_stopping_gtol']=gtol
        return detail
    s.update(name=subject['name']+'_g1e-4',sch=sch,run=run,evolve=evolve,certify=certify,validate=validate,
             production_gtol=gtol,metadata=dict(subject['metadata'],production_stopping_gtol=gtol))
    assert float(s['sch']['gtol'])==gtol
    return s

def build(POD,phi_path,include_original_qualified=False):
    old=POD.build(phi_path);new=[stronger(s,1e-4) for s in old]
    if include_original_qualified:new=[s for s in old if s['steps'] in (15,40)]+new
    return new
