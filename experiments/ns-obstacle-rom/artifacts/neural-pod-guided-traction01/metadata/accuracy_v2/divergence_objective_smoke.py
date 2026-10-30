"""Small deterministic array/finite-difference test; no fitted model or data access."""
import json
import numpy as np
from accuracy_v2.divergence_objective import block_objective, validate_weight

def check():
    rng=np.random.default_rng(7319);n=17;r=4
    v=rng.normal(size=(n,2,r));g=rng.normal(size=(n,2,2,r))
    t=rng.normal(size=v.shape);tg=rng.normal(size=g.shape);w=rng.uniform(.1,1,n)
    nv=np.einsum('n,ncr,ncr->r',w,t,t);ng=np.einsum('n,ncdr,ncdr->r',w,tg,tg)
    importance=np.array([.6,.25,.1,.05]);lam=validate_weight(1.7)
    def calc(v,g,sl=slice(None),weight=lam):
        return block_objective(v[sl],g[sl],t[sl],tg[sl],w[sl],nv,ng,importance,weight,np)
    total,metrics=calc(v,g)
    expected=np.sum(importance*(np.einsum('n,ncr,ncr->r',w,v-t,v-t)/nv+np.einsum('n,ncdr,ncdr->r',w,g-tg,g-tg)/ng))
    assert calc(v,g,weight=0)[0]==expected
    parts=[calc(v,g,slice(0,5))[1],calc(v,g,slice(5,None))[1]]
    for key,value in metrics.items():np.testing.assert_allclose(parts[0][key]+parts[1][key],value,rtol=3e-15,atol=1e-14)
    exact=block_objective(t,tg,t,tg,w,nv,ng,importance,lam,np)[1]
    assert exact['combined_objective']==0 and np.all(exact['teacher_divergence_norm_squared_by_mode']>0)
    # Directional derivative of the combined gradient regression objective.
    direction=rng.normal(size=g.shape);err=g-tg
    trace=err[:,0,0,:]+err[:,1,1,:]
    analytic=2*w[:,None,None,None]*err*importance[None,None,None,:]/ng[None,None,None,:]
    correction=2*lam*w[:,None]*trace*importance[None,:]/ng[None,:]
    analytic[:,0,0,:]+=correction;analytic[:,1,1,:]+=correction
    eps=1e-5;fd=(calc(v,g+eps*direction)[0]-calc(v,g-eps*direction)[0])/(2*eps)
    gap=abs(fd-np.sum(analytic*direction));assert gap<1e-9
    for invalid in (True,-1,float('nan'),float('inf'),None):
        try:validate_weight(invalid)
        except ValueError:pass
        else:raise AssertionError('invalid weight accepted')
    return {'passed':True,'points':n,'modes':r,'directional_derivative_gap':gap,'lambda_zero_base_exact':True,'block_additivity':True,'nonzero_teacher_divergence_zero_matching_loss':True,'actual_training_or_development_data_used':False,'execution_release':False}

if __name__=='__main__':print(json.dumps(check(),indent=2))
