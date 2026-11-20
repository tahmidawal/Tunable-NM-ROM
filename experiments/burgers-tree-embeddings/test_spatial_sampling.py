"""Exact-expectation and small Monte Carlo fixtures; no model training."""
import json
import numpy as np
from neural import spatial_sampling as S
import data
from neural.expanded_cases import definitions

rng=np.random.default_rng(2026111990)
w=np.array([0.,.01,.04,.2,.6,.8,.3,.05])
residual=np.array([[0.,0.,0.,0.,4.,.2,0.,0.],np.zeros(8),[1e200]*8])
error=np.array([[.1,.3,-.2,.7,.8,-.1,.4,.2],[.4]*8,[.2,-.5,.1,.3,-.2,.6,.1,-.3]])
sampler=S.prepare(w,residual,alpha=.5)
p=sampler['probabilities'];norm=np.array([.8,1.2,.6])
importance=np.divide(w[None,:],p,out=np.zeros_like(p),where=p>0)
objective=np.sum(w*error**2,axis=1)/norm**2
expected=np.sum(p*importance*error**2,axis=1)/norm**2
assert np.max(abs(expected-objective))<1e-14
assert np.max(importance)<=2*w.sum()+1e-14
assert np.allclose(p[1],w/w.sum(),rtol=0,atol=1e-15)
assert np.allclose(p[2],w/w.sum(),rtol=0,atol=1e-15)
sample=S.draw(rng,sampler,np.arange(3),200000)
values=sample['importance_weights']*error[np.arange(3)[:,None],sample['indices']]**2/norm[:,None]**2
estimate=values.mean(axis=1)
population_variance=np.sum(p*(importance*error**2/norm[:,None]**2-objective[:,None])**2,axis=1)
se=np.sqrt(population_variance/values.shape[1])
assert np.all(abs(estimate-objective)<=6*se+1e-14)
assert np.all(sample['indices']!=0) and np.isfinite(sample['importance_weights']).all()
primary=data.parameters('train',128);expanded=[r['family'] for r in definitions('train')]
strata=S.stratify(primary,expanded)
all_indices=np.concatenate(list(strata['groups'].values()))
assert np.array_equal(np.sort(all_indices),np.arange(160)) and len(set(all_indices))==160
assert all(v==.2 for v in strata['probabilities'].values())
assert all(primary[i,3]<=.05 for i in strata['groups']['narrow_pulse'])
assert all(primary[i,3]>.05 for i in strata['groups']['broad_pulse'])
for _ in range(100):
    name,ids=S.draw_stratum(rng,strata,4)
    assert len(set(ids))==4 and set(ids)<=set(strata['groups'][name])
for bad in [lambda:S.prepare(w,residual,alpha=1.),lambda:S.prepare(w,residual,cohort='dev'),
            lambda:S.stratify(primary,expanded,cohort='final_test')]:
    try:bad()
    except ValueError:pass
    else:raise AssertionError('Invalid sampling specification accepted')
print(json.dumps(dict(exact_expectation_max_error=float(np.max(abs(expected-objective))),
    monte_carlo_estimates=estimate.tolist(),exact_objectives=objective.tolist(),
    monte_carlo_standard_errors=se.tolist(),zero_residual_fallback=True,overflow_safe=True,
    bounded_importance_weights=True,training_only_guard=True,
    actual_training_strata_counts=strata['counts'],registered_default_probabilities=strata['probabilities'],
    per_trajectory_probabilities=strata['per_trajectory_probability'])))
