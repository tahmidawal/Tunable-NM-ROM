"""Tiny synthetic controls; CPU smoke, no scientific performance claims."""
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import diagcore as DC
import t2core as T2
import qn3
import qn_policy
import p1core as P1
import canonical_cohort as CC
import hashlib

def main():
    rng=np.random.default_rng(7); r,M,m=4,10,24
    A=jnp.asarray(rng.normal(size=(M,r)));B=jnp.asarray(.2*rng.normal(size=(m,r)))
    D=jnp.asarray(.2*rng.normal(size=(m,r)));P=jnp.asarray(.2*rng.normal(size=(m,M)))
    QA,RA=jnp.linalg.qr(A,mode='reduced')
    d=dict(A=A,B=B,D=D,P=P,lam=jnp.arange(1,M+1,dtype=jnp.float64),QA=QA,RA=RA)
    W=jnp.asarray(.1*rng.normal(size=(5,r)));nu=jnp.float64(.01)
    res,rj,cands=DC.step_functions(); errors=[]; matches=[]
    for k in [0,1,3]:
        args=(W,jnp.int32(k),nu,d);c=cands(args)[1]
        rv,J=rj(c,args);Jad=jax.jacfwd(lambda x:res(x,args))(c)
        err=float(jnp.linalg.norm(J-Jad)/(jnp.linalg.norm(Jad)+1e-300));errors.append(err)
        assert err<1e-13
        assert float(jnp.linalg.norm(rv-res(c,args)))<1e-13
        orig=T2.make_lm(rj,600,1.,1e-10)
        ref=jax.jit(orig)(c,args,0.,1e-7,1e-6)
        got=DC.make_lm(rj,1.,1e-7)(c,args)
        delta=float(jnp.linalg.norm(ref[0]-got['c']));matches.append(delta)
        assert delta<1e-14
        assert int(got['residual_evals'])==int(got['it'])+1
        assert int(got['jacobian_evals'])==int(got['it'])+1
    ev=qn3.make_qn_evolve(r,'BDF2',.025,4,1.,4)
    out=jax.jit(ev)(W[:1],jnp.asarray([.01]),d)
    control=qn_policy.make_qn_evolve(r,'BDF2',.025,4,1.,4,slow_threshold=6)
    twin=jax.jit(control)(W[:1],jnp.asarray([.01]),d)
    assert all(np.array_equal(np.asarray(out[k]),np.asarray(twin[k])) for k in out), 'policy extension baseline changed'
    g=np.asarray(qn3.make_certify('BDF2',.025,4)(out['W_all'][:,0],nu,d))
    fresh=[]
    for k in range(4):
        args=(out['W_all'][:,0],jnp.int32(k),nu,d)
        rv,J=rj(out['W_all'][k+1,0],args)
        fresh.append(float(jnp.linalg.norm(J.T@rv)/(jnp.linalg.norm(J)*jnp.linalg.norm(rv)+1e-300)))
    assert np.allclose(g,fresh,rtol=1e-7,atol=1e-12)
    # An incorrectly signed BDF history must be detected by the direct residual control.
    args=(W,jnp.int32(3),nu,d); c=W[4]
    wrong=res(c,args)-.025*(A@W[2])
    assert float(jnp.linalg.norm(wrong-res(c,args)))>1e-5
    G=rng.normal(size=(25,r));L=np.linalg.cholesky(G.T@G);diff=rng.normal(size=(6,r))
    assert np.allclose(np.linalg.norm(diff@L,axis=1),np.linalg.norm(diff@G.T,axis=1),rtol=1e-14)
    d['hmask']=(d['lam']>jnp.median(d['lam'])).astype(jnp.float64)
    ev40=P1.make_evolve(DC.C3.nl3,DC.C3.nlJ3,'LSPG',r,1.,nrec=41)
    sch40=P1.sched('BDF2',.00625,1e-7,0.)
    w40,stats40=jax.jit(ev40)(W[0],nu,jnp.float64(1.),d,sch40)
    assert np.array_equal(np.asarray(w40),np.asarray(stats40['states'])[::8])
    assert stats40['states'].shape==(41,r)
    assert np.isfinite(np.asarray(stats40['states'])).all()
    root=Path(__file__).resolve().parent
    fixture=np.load(root/'results/qpilot02/raw/output/regenerated_cohort.npz',allow_pickle=False)
    table={k:fixture[k] for k in fixture.files};table['sha256']=str(table['sha256'])
    reference=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/store')
    manifest={p.strip():h for line in (root/'vendor/e0/store/MANIFEST.sha256').read_text().splitlines() if line.strip() for h,p in [line.split(None,1)]}
    normalisers=[]
    for j in range(64):
        rel=f'3d/val64/ST/c{j:03d}.json';raw=(reference/rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest[rel]
        normalisers.append(json.loads(raw)['desc']['s_star'])
    expected='b7993291f4f6cf0e60d68169ff1ecd07ca3d26f47d8c61a89a5f3f3ef8be9b4c'
    restored,na=CC.restore(table,normalisers,expected)
    assert na['changed_cases']==[1] and na['maximum_ulp']==1. and restored['sha256']==expected
    bad=dict(table,s_star=np.asarray(normalisers).copy());bad['s_star'][1]=np.nextafter(np.nextafter(normalisers[1],np.inf),np.inf)
    bad['sha256']=CC.table_hash(bad)
    try:CC.restore(bad,normalisers,expected)
    except AssertionError:pass
    else:raise AssertionError('two-ULP normaliser change admitted')
    bad=dict(restored,c=restored['c'].copy());idx=int(np.flatnonzero(restored['B']<3)[0])
    bad['c'][idx,2,0]=np.nextafter(bad['c'][idx,2,0],np.inf);bad['sha256']=CC.table_hash(bad)
    try:CC.restore(bad,normalisers,expected)
    except AssertionError:pass
    else:raise AssertionError('inactive raw-draw mutation admitted')
    out=dict(passed=True,x64=jax.config.x64_enabled,backend=jax.default_backend(),
             policy_extension_baseline_bitwise=True,metric_factor_control=True,
             recording_40_step_slots=True,
             actual_failed_cohort_restored=True,two_ulp_rejected=True,
             inactive_draw_mutation_rejected=True,
             derivative_relative=errors,lm_original_match=matches,fresh_certificate_difference=float(np.max(np.abs(g-fresh))))
    dest=Path(__file__).resolve().parent/'audits'/'small-controls.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(out))

if __name__=='__main__':main()
