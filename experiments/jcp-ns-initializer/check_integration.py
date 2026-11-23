"""Old-development-case API/compiled-query parity, before fresh-case generation."""
import argparse,json
from pathlib import Path
import support as S
import fit_api as A
import query_adapter as Q
import legacy_phase as P
import frozen_contract as FC
np,jnp,H,F,SR=S.np,S.jnp,S.H,S.F,S.SR

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    config=S.HERE/'configs/fresh-phase.json';FC.verify(json.loads(config.read_text()),config)
    assert S.jax.default_backend()=='gpu' and S.jax.config.jax_enable_x64
    G,_=S.learned_basis(64);basis=jnp.asarray(G)
    ops,checks=SR.build_operators_fast(G,64,1168,check_modes=8)
    params,codes=S.load_head(S.HERE/'assets/head_k8_n64.npz');candidates=np.asarray(H.head_apply(params,jnp.asarray(codes)))
    par=F.parameters(202609202,1)[0];initial=F.initial(64,par);geom=F.geometry(64)
    physical=S.query_input(jnp.asarray(initial),geom);norm=float(np.linalg.norm(np.asarray(physical)))
    field,fit=A.fit_initial_field(physical,basis,params,codes,candidates,1e-9)
    query=Q.make_query(n=64,rank=64,latent=8,dt=.01,steps=20,output_every=4)
    oa=tuple(jnp.asarray(ops[k]) for k in ('A','T','lam','Dd'))
    result=query(physical,float(par[-1]),basis,oa[0][:584],oa[1][:584],oa[2][:584],oa[3][:,:584],
                 params,codes,candidates,jnp.float64(1e-7),jnp.float64(1e-9))
    truth=np.asarray(F.make_solver(.000125,1600,80)(physical,float(par[-1]),geom))
    _,prepare,_=SR.make_frozen_step(.01,64,64,1,1e-6)
    wide=prepare(float(par[-1]),*oa)
    selected=prepare(float(par[-1]),oa[0][:584],oa[1][:584],oa[2][:584],oa[3][:,:584])
    host=P.rollout('head',fit['coefficients'],fit['code'],fit['center'],params,selected,wide,.01,64,8,20,1e-7,250,
                   basis,truth,geom,norm,codes,candidates,False,584)
    gaps=np.linalg.norm((result['fields']-host['fields'][::4]).reshape(6,-1),axis=1)/norm
    error=max(x['relative_error'] for x in host['metrics'][1:]);bound=min(1e-6,.01*error)
    code_gap=float(np.linalg.norm(np.asarray(fit['code'])-np.asarray(result['initializer']['code'])))
    passed=bool(fit['qualified'] and result['qualified'] and all(r['reason']=='stationarity' for r in host['records'])
                and gaps.max()<=bound and code_gap==0.)
    args.out.mkdir(parents=True,exist_ok=False)
    A.save_invocation(args.out/'host_initializer',fit,params)
    A.save_invocation(args.out/'query_initializer',result['initializer'],params)
    S.dump(args.out/'summary.json',{'passed':passed,'scope':'exposed development case0, seed202609202; no new-case outcome',
        'per_output_gap':gaps,'bound':bound,'returned_code_gap':code_gap,'query_qualified':result['qualified'],
        'host_steps_stationary':all(r['reason']=='stationarity' for r in host['records']),
        'operator_checks':checks,'query_step_info':result['step_info'],'timing_claim':False})
    if not passed:raise RuntimeError('integration fixture failed before fresh case generation')
    print('old-development integration fixture PASS',flush=True)

if __name__=='__main__':main()
