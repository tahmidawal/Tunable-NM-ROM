"""Bounded local GPU training/projection/tensor smoke; no scientific result."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_model as D
import ns3d_rom as R


def main():
    output=Path(__file__).resolve().parent/'checks'
    output.mkdir(exist_ok=True)
    n,k,rank=8,2,8
    pars=F.parameters(19,4)
    U=np.stack([F.initial(n,p) for p in pars])
    params,z,info=D.train_bank(U,n,k,rank,20,3,20,output/'smoke_bank.pkl',batch=4,width=16)
    G=np.asarray(D.bank(params,D.coords(n),F.geometry(n),n))
    Q,Rb,C,perp,whitening=D.whiten(G,U)
    params,z,hinfo=D.train_head(params,z,C,Rb,5,10,21,output/'smoke_head.pkl',
                                dict(n=n,k=k,r=rank))
    correction,singular=D.correction_directions(params,Rb,C,z)
    tensor=R.tensor_checks(G,n,m=16)
    divergence=np.asarray(F.diagnostics(jnp.asarray(G.T.reshape(rank,3,n,n,n)),F.geometry(n))[2])
    result=dict(backend=jax.default_backend(),f64=bool(jax.config.jax_enable_x64),
                training=info,head=hinfo,whitening=whitening,tensor=tensor,
                max_divergence=float(np.max(divergence)),
                frozen_frequencies=bool(np.array_equal(params['B'],np.round(params['B']))),
                correction_orthogonality=float(np.linalg.norm(
                    (Rb@correction).T@(Rb@correction)-np.eye(correction.shape[1]))))
    result['passed']=bool(result['backend']=='gpu' and result['f64'] and tensor['passed']
                          and result['max_divergence']<1e-10 and result['frozen_frequencies'])
    (output/'local_model.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
