"""Tiny synthetic controls for phase equivalence and fixed-population cost/matching logic."""
from pathlib import Path
import copy
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/speed3d'))
import jax
import jax.numpy as jnp
import psjax
import ps_phases
from front_metrics import summarize
from pod_policy import stronger

def main():
    assert jax.config.x64_enabled
    b0=jnp.asarray(np.random.default_rng(7).normal(size=(4,4,4))*.001)
    q=psjax.make_ps_query(5,.025);ev,dec,S=ps_phases.make(5,.025)
    a=np.asarray(q(b0,.05));b=np.asarray(dec(ev(b0,.05),S))
    rel=float(np.linalg.norm(a-b)/np.linalg.norm(a));assert rel<=1e-12
    z=dict(cohort=dict(eligible_cases=[0,1]),subjects={s:dict(config=dict(kind=k),qualification=dict(status='qualified')) for s,k in [('r','bank'),('f','PS')]},
        case_results={s:{str(j):dict(valid=True,score=dict(e=e,epsilon=.1)) for j in [0,1]} for s,e in [('r',4.),('f',2.)]},timing_repetitions=[])
    for s,v in [('r',2.),('f',1.)]:
        for j in (0,1):z['timing_repetitions'].append(dict(subject=s,case=j,role='B',given_native_seconds=v,descriptor_seconds=v,timing_admitted=True,native_timing_admitted=True))
    summarize(z);assert z['matching']['subjects']['r']['PS']['descriptor']['selected']=='f'
    missing=copy.deepcopy(z);missing['timing_repetitions']=[r for r in missing['timing_repetitions'] if not(r['subject']=='f' and r['case']==1)]
    summarize(missing);assert missing['matching']['subjects']['r']['PS']['descriptor']['selected'] is None
    balanced=copy.deepcopy(z)
    balanced['timing_repetitions']=[r for r in balanced['timing_repetitions'] if r['subject']!='f']
    for j,values in [(0,[1.,1.,1.]),(1,[9.])]:
        for v in values:balanced['timing_repetitions'].append(dict(subject='f',case=j,role='B',given_native_seconds=v,descriptor_seconds=v,timing_admitted=True,native_timing_admitted=True))
    summarize(balanced);assert balanced['summaries']['f']['costs']['descriptor']['median_seconds']==5.
    probe=jax.jit(lambda u,nu,data,sch:dict(value=u*data['scale']+sch['gtol']))
    evolve=jax.jit(lambda c,nu,scale,data,sch:c*0+sch['gtol'])
    original=dict(name='synthetic',query=probe,data=dict(scale=jnp.float64(1)),sch=dict(gtol=jnp.float64(1e-3)),
        evolution_query=evolve,fresh_certifier=None,metadata={},certify=lambda out,nu:out,
        validate=lambda u,nu,prod=None,need_tighter=True,n0=None:dict(probe=prod['value']))
    policy=stronger(original,1e-4)
    assert float(policy['run'](jnp.float64(0),.1)['value'])==1e-4
    assert float(policy['evolve'](jnp.float64(0),.1,1))==1e-4
    assert float(policy['validate'](jnp.float64(0),.1)['probe'])==1e-4
    assert float(original['sch']['gtol'])==1e-3
    out=dict(passed=True,synthetic_only=True,x64=jax.config.x64_enabled,PS_phase_fused_relative=rel,
             incomplete_case_coverage_rejects_cost_match=True,case_balanced_median_control=True,POD_stronger_policy_actual_schedule_binding=True)
    dest=HERE/'audits/front-small-controls.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
