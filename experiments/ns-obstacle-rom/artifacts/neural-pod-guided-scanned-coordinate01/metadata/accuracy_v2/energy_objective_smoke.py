"""Small synthetic ensemble identity test; no real coefficients/data read."""
import json
import numpy as np
from accuracy_v2.energy_objective import training_energy_weights,block_energy_objective

def check():
    rng=np.random.default_rng(7331);n=13;r=4;snapshots=9
    t=rng.normal(size=(n,2,r));g=rng.normal(size=(n,2,2,r));v=t+.2*rng.normal(size=t.shape);dg=g+.2*rng.normal(size=g.shape);w=rng.uniform(.1,1,n)
    sigma=np.array([6.2,.35,.025,.0018]);q,_=np.linalg.qr(rng.normal(size=(snapshots,r)));coeff=sigma[:,None]*q.T
    nv=np.einsum('n,ncr,ncr->r',w,t,t);ng=np.einsum('n,ncdr,ncdr->r',w,g,g)
    weights=training_energy_weights(sigma,nv,ng);lam=.7
    loss,parts=block_energy_objective(v,dg,t,g,w,nv,ng,weights['value'],weights['gradient_and_trace'],lam,np)
    ev=np.einsum('ncr,rs->ncs',v-t,coeff);tv=np.einsum('ncr,rs->ncs',t,coeff)
    eg=np.einsum('ncdr,rs->ncds',dg-g,coeff);tg=np.einsum('ncdr,rs->ncds',g,coeff)
    denv=np.einsum('n,ncs,ncs->',w,tv,tv);deng=np.einsum('n,ncds,ncds->',w,tg,tg)
    trace=eg[:,0,0,:]+eg[:,1,1,:]
    directv=np.einsum('n,ncs,ncs->',w,ev,ev)/denv;directg=np.einsum('n,ncds,ncds->',w,eg,eg)/deng;directd=np.einsum('n,ns,ns->',w,trace,trace)/deng
    np.testing.assert_allclose([parts['value_component_objective'],parts['gradient_component_objective'],parts['supervised_divergence_objective_unweighted']],[directv,directg,directd],rtol=2e-14,atol=1e-15)
    assert abs(loss-(directv+directg+lam*directd))<1e-14
    # Non-orthogonal coefficient changes are outside this equivalence claim.
    covariance_gap=np.max(np.abs(coeff@coeff.T-np.diag(sigma*sigma)))
    extreme=training_energy_weights([1e300,1e-300],[1e-300,1e300],[1e300,1e-300])
    assert all(np.isfinite(x).all() and np.isclose(x.sum(),1) for x in extreme.values())
    return {'passed':True,'retained_modes':r,'synthetic_training_snapshots':snapshots,'coefficient_covariance_absolute_gap':float(covariance_gap),'value_ensemble_identity_gap':float(abs(parts['value_component_objective']-directv)),'gradient_ensemble_identity_gap':float(abs(parts['gradient_component_objective']-directg)),'trace_ensemble_identity_gap':float(abs(parts['supervised_divergence_objective_unweighted']-directd)),'actual_model_or_data_read':False,'optimizer_integrated':False,'same_as_solved_field_loss_claim':False}
if __name__=='__main__':print(json.dumps(check(),indent=2))
