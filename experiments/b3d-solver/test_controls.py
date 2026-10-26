"""Tiny synthetic controls; CPU smoke, no scientific performance claims."""
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import diagcore as DC
import t2core as T2
import qn3

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
    out=dict(passed=True,x64=jax.config.x64_enabled,backend=jax.default_backend(),
             derivative_relative=errors,lm_original_match=matches,fresh_certificate_difference=float(np.max(np.abs(g-fresh))))
    dest=Path(__file__).resolve().parent/'audits'/'small-controls.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(out))

if __name__=='__main__':main()
