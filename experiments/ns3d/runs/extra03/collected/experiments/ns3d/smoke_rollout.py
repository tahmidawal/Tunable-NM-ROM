"""Bounded complete neural/q and linear-bank rollout check on a tiny grid."""
import json
import pickle
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R


def main():
    out=Path(__file__).resolve().parent/'checks'
    with (out/'smoke_head.pkl').open('rb') as stream:
        ck=pickle.load(stream)
    params=ck['params'];z=ck['codes']
    n,k,rank=8,2,8
    G=np.asarray(D.bank(params,D.coords(n),F.geometry(n),n))
    Qb,Rb=np.linalg.qr(G,mode='reduced')
    Phi,lam,ids=R.test_modes(n,32)
    T=R.build_tensor(G,n,ids)
    coef=np.asarray(D.head(params,jnp.asarray(z)))
    C=np.linalg.solve(Rb,np.eye(rank))
    u0=(G@coef[0]).reshape(3,n,n,n)
    results={}
    for label,q,linear in [('head',0,False),('q2',2,False),('linear',0,True)]:
        run=R.make_run(.001,2,1,k,q,linear=linear,budget=20,cold_starts=2)
        value=run(jnp.asarray(u0),.005,jnp.asarray(G),jnp.asarray(Qb),jnp.asarray(Rb),
                  jnp.asarray(Phi.T@G),jnp.asarray(T),jnp.asarray(lam),jnp.asarray(C[:,:q]),
                  params,jnp.asarray(z))
        fields,cold,info=tuple(value)
        fields=np.asarray(fields)
        record=dict(finite=bool(np.all(np.isfinite(fields))),
                    evolved_change=float(np.linalg.norm(fields[-1]-fields[0])),
                    cold=[np.asarray(x).item() for x in cold],
                    iterations=np.asarray(info[1]).tolist(),reasons=np.asarray(info[2]).tolist(),
                    normalized_gradient=np.asarray(info[3]).tolist())
        results[label]=record
    results['passed']=all(x['finite'] and x['evolved_change']>1e-8 for x in results.values())
    (out/'local_rollout.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    if not results['passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
