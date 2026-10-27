"""Tiny synthetic controls for phase equivalence and fixed-population cost/matching logic."""
from pathlib import Path
import copy
import json
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/speed3d'))
import jax
import jax.numpy as jnp
import psjax
import ps_phases
from front_metrics import summarize
from pod_policy import stronger
from work_counts import fixed_sweep,quasi_newton
import fs3
from projection_budget import ProjectionBudget
import pod_identity as ID
import frozen_tables as FT

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
    qcount=quasi_newton([2,3],[1,0],[4,0],batch_size=1)
    assert qcount['residual_evaluations']==18 and qcount['gradient_evaluations']==7 and qcount['advection_value_evaluations']==22
    assert quasi_newton([2],[0],[3])['residual_evaluations'] is None
    assert quasi_newton([2],[0],[4],batch_size=2)['gradient_evaluations'] is None
    actual_probe=HERE/'results/podtune04/raw/output/identity_probe_POD512_none_BDF2_40steps_c000.npz'
    probe_result=__import__('json').loads((actual_probe.parent/'result.json').read_text())['identity_probes']['POD512_none_BDF2_40steps|case0']
    z=np.load(actual_probe,allow_pickle=False);n0=probe_result['current_score']['normalisation']
    identity=ID.original_policy(z['C'],z['old_C'],z['fields'],z['old_decoded_fields'],z['Gram'],n0,probe_result['current_score']['e'],probe_result['prior_score']['e'])
    assert identity['passed'] and not np.array_equal(z['C'],z['old_C'])
    changed=np.asarray(z['C']).copy();changed[0,0]+=2*ID.FIELD_TOL*n0/np.sqrt(z['Gram'][0,0])
    assert not ID.original_policy(changed,z['old_C'],z['fields'],z['old_decoded_fields'],z['Gram'],n0,probe_result['current_score']['e'],probe_result['prior_score']['e'])['passed']
    calls=[]
    def counted_contract(data,c):
        jax.debug.callback(lambda _:calls.append(1),c,ordered=True)
        return jnp.einsum('mij,j->mi',data['T'],c)
    d=dict(A=jnp.asarray([[1.,.1],[.1,1.],[.2,.3]]),lam=jnp.asarray([1.,2.,3.]),T=jnp.ones((3,2,2))*.01)
    actual=[]
    for aeff in (0,2):
        calls.clear();ev=fs3.make_fs_evolve(2,counted_contract,'CN',.025,1,1e-3,.05,aeff,4,unroll=False)
        out=jax.block_until_ready(ev(jnp.asarray([.2,.3]),.02,1e-12,d));jax.effects_barrier()
        derived=fixed_sweep(np.asarray(out['it']),aeff,1,'CN')
        assert len(calls)==derived['jacobian_contractions'],(len(calls),derived)
        actual.append(dict(adaptive_steps=aeff,observed_contractions=len(calls),derived=derived))
    rng=np.random.default_rng(19);G=rng.normal(size=(7,3));R=rng.normal(size=(6,7));W=rng.normal(size=(6,3));n0=np.linalg.norm(R[0])
    with tempfile.TemporaryDirectory(dir=HERE/'audits') as td:
        cx=SimpleNamespace(out=Path(td),cfg=dict(phase_cases=[0]),ref=lambda j:(R,dict(judge=dict(n0=n0)),{}))
        d=dict(GTb=(jnp.asarray(G.T[:1]),jnp.asarray(G.T[1:])),L=jnp.asarray(np.linalg.cholesky(G.T@G)))
        sub=dict(kind='bank',st=SimpleNamespace(bk=dict(sha256='synthetic-only'),lean=d))
        output={};budget=ProjectionBudget(cx,output)
        b=budget.measure(sub,0,W,(np.linalg.norm(W@G.T-R,axis=1)/n0).tolist())
        exact=np.linalg.lstsq(G,R.T,rcond=None)[0].T
        observed=budget.cache[('bank_synthetic-only_3',0)][0]
        oracle_relative=float(np.linalg.norm(exact-observed)/(np.linalg.norm(exact)+1e-300));assert oracle_relative<=1e-12
        assert max(abs(x) for x in b['pythagorean_squared_error_defect'])<=1e-12
    frozen_k=np.asarray([[1,1,1],[2,1,1],[1,2,1]],np.int64);frozen_checks=[]
    for n in (5,9):
        x=fs3.C.interior_coords(n);features=np.prod(np.sin(np.pi*x[:,None,:]*frozen_k[None,:,:]),axis=2)*np.sqrt(8)
        tables,meta=FT.assemble(n,(jnp.asarray(features.T[:1]),jnp.asarray(features.T[1:])),frozen_k)
        assert np.allclose(tables['A'],np.eye(3)*(n-1)**1.5,rtol=1e-12,atol=1e-12)
        checks=FT.direct_checks(n,tables,frozen_k);frozen_checks.append(dict(n=n,mode_hash=meta['mode_hash'],**checks))
    assert frozen_checks[0]['mode_hash']==frozen_checks[1]['mode_hash']
    out=dict(passed=True,synthetic_only=True,x64=jax.config.x64_enabled,PS_phase_fused_relative=rel,
             incomplete_case_coverage_rejects_cost_match=True,case_balanced_median_control=True,POD_stronger_policy_actual_schedule_binding=True,
             FS_logical_contract_runtime_controls=actual,QN_singleton_counter_scope_guard=True,
             projection_oracle_relative=oracle_relative,projection_orthogonality_Pythagoras_pass=True,
             actual_POD_active_identity_pass=identity,POD_identity_above_bound_rejected=True,frozen_mode_assembly_controls=frozen_checks)
    dest=HERE/'audits/front-small-controls.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
