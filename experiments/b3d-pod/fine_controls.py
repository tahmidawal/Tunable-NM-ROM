"""Tiny all-node ECSW/full equivalence for the forty-step parameterized A3 path."""
from pathlib import Path
import json
import numpy as np
import hr_interface as I
import replay as X

jax,jnp,H,R,B=I.jax,I.jnp,I.H,I.R,I.B


def main():
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    before=dict(H.NMAX);rng=np.random.default_rng(20261027);L,r=5,4
    Phi=jnp.asarray(np.linalg.qr(rng.normal(size=((L-1)**3,r)))[0])
    full=dict(Phi=Phi,LPhi=H.lap_cols(Phi,3,L),DPhi=H.dsum_cols(Phi,3,L))
    data,_=B.ecsw_data(Phi,np.arange(len(Phi)),np.ones(len(Phi)),3,L);hr=dict(data,Phi=Phi)
    u=Phi@jnp.asarray(rng.normal(size=r)*.03);nu=jnp.float64(.02)
    q,_,_,_,ce,schedule=I.functions_for('none',.5,40,L=L)
    h,_,_,_,ch,hs=I.functions_for('ecsw',.5,40,L=L)
    sch=dict(schedule('BDF2',.00625),gtol=jnp.float64(.0001));ssh=dict(hs('BDF2',.00625),gtol=jnp.float64(.0001))
    a=jax.block_until_ready(q(u,nu,full,sch));b=jax.block_until_ready(h(u,nu,hr,ssh))
    c_difference=float(np.max(np.abs(np.asarray(a['C'])-np.asarray(b['C']))))
    ga=np.asarray(ce(a['C'],nu,full,sch));gb=np.asarray(ch(b['C'],nu,hr,ssh))
    m=X.metric_summary(jax.block_until_ready(X.make_metrics(L,r,40)(b['C'],nu,full,hr,ssh)))
    metrics_agree=float(np.max(np.abs(np.asarray(m['full_gamma'])-np.asarray(ch(b['C'],nu,hr,ssh)))))
    C=b['C']+jnp.asarray(rng.normal(size=(41,r))*.02)
    corrupt=dict(hr,sw=hr['sw']*jnp.sqrt(2.))
    wrong=X.metric_summary(jax.block_until_ready(X.make_metrics(L,r,40)(C,nu,full,corrupt,ssh)))
    schedule_ok=bool(int(sch['steps'])==40 and int(sch['nstart'])==1 and len(sch['slot'])==41
        and np.array_equal(np.flatnonzero(np.asarray(sch['slot'])>=0),np.arange(0,41,8)))
    out=dict(backend='gpu',precision='f64/highest',L=L,rank=r,steps=40,
        C_shape=list(b['C'].shape),all_node_C_maxabs=c_difference,
        all_node_delta_g_max=max(m['delta_g']),whole_gamma_fresh_difference=metrics_agree,
        corrupted_scale_detected=bool(max(wrong['delta_g'])>.001),schedule_ok=schedule_ok,
        global_buffer_unchanged=before==H.NMAX,
        full_valid=R.certify(a,ga,sch)['valid'],sampled_valid=R.certify(b,gb,ssh)['valid'])
    out['pass_']=bool(c_difference<1e-12 and max(m['delta_g'])<1e-10 and metrics_agree<1e-10
        and out['corrupted_scale_detected'] and schedule_ok and out['global_buffer_unchanged']
        and out['full_valid'] and out['sampled_valid'])
    print(json.dumps(out,indent=2),flush=True);assert out['pass_']
    (Path(__file__).parent/'checks/fine-controls.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
