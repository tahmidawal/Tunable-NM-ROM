"""Frozen full-residual POD subjects for a paired frontier driver; no E0/cohort access.

run/initialize/evolve/decode/certify/validate are ordinary Python adapters passing
large arrays to already-jitted explicit-argument kernels. Never jit the adapters.
Use each subject's query/data/sch for direct compiled calls if needed.
"""
from __future__ import annotations
import numpy as np
import interface as I

jax,jnp,H,R=I.jax,I.jnp,I.H,I.R
SCHEDULES=(('BDF2',.025),('BDF2',.0125),('BDF2',.00625),('CN',.01),('BDF2',1/60))


def _subjects_from_data(data,trust,metadata,L=64):
    assert L in (5,64)
    r=int(data['Phi'].shape[1]);assert data['Phi'].shape[0]==(L-1)**3
    if L==64:assert r==512
    gram=np.asarray(jax.block_until_ready(data['Phi'].T@data['Phi']))
    orth=float(np.max(np.abs(gram-np.eye(r))));assert orth<=1e-10

    def distance(a,b,n0):
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape==(6,r) and np.isfinite(n0) and n0>0
        if not (np.isfinite(a).all() and np.isfinite(b).all()):return float('inf')
        diff=a[1:]-b[1:]
        return float(np.sqrt(np.maximum(np.sum((diff@gram)*diff,axis=1),0)).max()/n0)

    def one(scheme,dt):
        steps=int(round(.25/dt))
        q,ini,ev,dec,ce,sched=I.functions_for(trust,steps,'common',L=L)
        native,_,_,_,ncert,nsched=I.functions_for(trust,steps,'native',L=L)
        sch=sched(scheme,dt,'prod')
        st=nsched(scheme,dt,'tight');stt=nsched(scheme,dt,'tighter')
        assert int(sch['steps'])==steps and len(sch['slot'])==steps+1

        def run(u0,nu):return q(u0,nu,data,sch)
        def initialize(u0):return ini(u0,data)
        def evolve(c0,nu,scale):return ev(c0,nu,scale,data,sch)
        def decode(W):return dec(W,data)
        def certify(out,nu):
            gamma=jax.block_until_ready(ce(out['C'],nu,data,sch))
            return dict(R.certify(out,gamma,sch,'prod'),fresh_gamma=np.asarray(gamma)[:steps])
        def validate(u0,nu,prod=None,need_tighter=True,n0=None):
            u0=jnp.asarray(u0);assert u0.shape==((L-1)**3,)
            actual=float(np.linalg.norm(np.asarray(u0)))
            n0=actual if n0 is None else float(n0)
            assert np.isfinite(n0) and n0>0 and abs(n0/actual-1)<=1e-12
            prod=jax.block_until_ready(run(u0,nu)) if prod is None else jax.block_until_ready(prod)
            pc=certify(prod,nu)
            ot=jax.block_until_ready(native(u0,nu,data,st))
            gt=jax.block_until_ready(ncert(ot['C'],nu,data,st))
            tc=R.certify(ot,gt,st,'tight')
            detail=dict(production_certificate=pc,tight_certificate=tc,
                        prod_to_tight_distance=distance(prod['W'],ot['W'],n0),n0=n0,steps=steps,
                        Wtight=np.asarray(ot['W']),Ctight=np.asarray(ot['C']),gamma_tight=np.asarray(gt)[:steps],
                        need_tighter=bool(need_tighter),tight_tighter_distance=None,tighter_certificate=None)
            if need_tighter:
                ott=jax.block_until_ready(native(u0,nu,data,stt))
                gtt=jax.block_until_ready(ncert(ott['C'],nu,data,stt))
                detail.update(tighter_certificate=R.certify(ott,gtt,stt,'tighter'),
                              tight_tighter_distance=distance(ot['W'],ott['W'],n0),
                              Wtighter=np.asarray(ott['W']),Ctighter=np.asarray(ott['C']),gamma_tighter=np.asarray(gtt)[:steps])
            # The frontier driver aggregates pooled stationarity and prescribed tighter-case coverage.
            # Do not turn a single case's nonstationarity into a configuration-level qualification verdict here.
            return detail
        return dict(name=f'POD{r}_none_{scheme}_{steps}steps',scheme=scheme,dt=dt,steps=steps,r=r,
                    kind='POD-full-residual',hyper_reduced=False,metadata=dict(metadata,orthonormality=orth),
                    run=run,initialize=initialize,evolve=evolve,decode=decode,certify=certify,validate=validate,
                    query=q,initialization_query=ini,evolution_query=ev,decode_query=dec,fresh_certifier=ce,
                    data=data,sch=sch,native_query=native,tight_sch=st,tighter_sch=stt)
    return [one(s,dt) for s,dt in SCHEDULES]


def build(phi_path):
    _,_,_,_,data,_,meta=I.setup(phi_path)
    return _subjects_from_data(data,meta['trust_radius'],meta,L=64)
