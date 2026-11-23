"""Small regression fixtures for scientific gates, not an experiment."""
import os
import numpy as np
import shared_initializer as I
import support as S
import check_arithmetic as C
from arithmetic_audit import NetworkAudit

def main():
    assert I.jax.default_backend()=='gpu' and I.jax.config.jax_enable_x64
    eps=np.finfo(np.float64).eps
    # An easier stationary solution outside the unchanged objective tie cannot win.
    infos=[{'cost':1.,'reason':'stagnation'},{'cost':1.+4096*eps,'reason':'stationarity'}]
    assert I.select_fit_candidate(infos)[0]==0
    # Repeated tiny allowed increases may not accumulate past the original endpoint.
    current={'cost':1.+900*eps,'gradient':1e-7,'finite':True}
    trial={'cost':1.+1800*eps,'gradient':1e-9,'finite':True}
    assert not I.acceptance(current,trial,1e-8,1.,1.)[0]
    # SVD truncation never changes the unpreconditioned stationarity definition.
    res=I.jnp.array([1.,1.]);jac=I.jnp.array([[1.,0.],[0.,0.]])
    _,singular,cutoff,kept=I.svd_direction(res,jac)
    assert np.asarray(kept).sum()==1
    assert float(I.statistics(res,jac)[1])>0.7
    # Exercise a capped/rejected polishing path and its fresh status directly.
    params,codes=S.load_head(S.HERE/'assets/head_k8_n64.npz')
    target=I.H.head_apply(params,I.jnp.asarray(codes[0]))
    seed=np.asarray(codes[0]).copy()+0.1
    _,_,before=I.snapshot(seed,target,params)
    code,info=I._stage(seed,target,params,1e-11,before['cost'],'svd')
    _,_,after=I.snapshot(code,target,params)
    assert info['reason']!='stationarity' or after['gradient']<=1e-11
    assert info['original_endpoint_guard']
    assert len(info['history'])<=I.POLICY['svd_iterations']*I.POLICY['svd_halvings']
    assert C.check_saturation()['passed']
    for filename in ('head_k8_n64.npz','pod_head_k8_n64.npz'):
        pp,zz=S.load_head(S.HERE/'assets'/filename)
        assert C.check_network(pp,zz[0])['passed']
        tt=np.asarray(I.H.head_apply(pp,I.jnp.asarray(zz[0])))+1e-3
        audit=NetworkAudit(pp).compare([{'seed_index':0,'kind':'fixture','code':zz[0]}],zz[0],tt,False)
        assert audit['passed'] and audit['decimal_required']
    print('initializer policy fixtures PASS',flush=True)
    output=os.environ.get('FIXTURE_OUTPUT')
    if output:S.dump(output,{'passed':True,'backend':I.jax.default_backend(),
        'x64':I.jax.config.jax_enable_x64,'policy':I.POLICY,'fixture':'selector/endpoint/rank/fresh-stationarity/cap'})

if __name__=='__main__':main()
