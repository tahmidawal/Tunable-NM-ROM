"""Mechanical collection checks. This is not the independent scientific results audit."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from work_counts import fixed_sweep,quasi_newton
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=True),(a,b)
def distance(a,b,L,n0):return float(np.linalg.norm((a-b)@L,axis=1)[1:].max()/n0)
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    root=HERE/'results'/a.attempt;raw=root/'raw/output';context=json.loads((raw/'result.json').read_text())
    mode=context['config']['mode'];name='front_result.json' if mode=='front' else 'mesh_result.json';rp=raw/name;z=json.loads(rp.read_text())
    assert z['execution']['complete'] or a.allow_partial
    assert z['execution']['source_commit']==context['source_commit']==(root/'raw/COMMIT.txt').read_text().strip()
    inv={r['invocation_id']:r for r in z['timing_repetitions']};assert len(inv)==len(z['timing_repetitions'])
    brackets={}
    for r in inv.values():
        close(r['descriptor_seconds'],r['given_native_seconds']+r['input_prep_seconds'])
        assert r['descriptor_seconds']>=0 and r['given_native_seconds']>=0 and r['input_prep_seconds']>=0
        brackets.setdefault(r['bracket'],[]).append(r)
    for rows in brackets.values():
        assert sorted(r['role'] for r in rows)==['A1','A2','B'];r={x['role']:x for x in rows}
        for metric,ratio_key,mask in [('descriptor_seconds','anchor_drift_ratio','timing_admitted'),('given_native_seconds','native_anchor_drift_ratio','native_timing_admitted')]:
            ratio=r['A2'][metric]/r['A1'][metric]
            for row in rows:
                close(ratio,row[ratio_key]);assert row[mask]==(1/1.1<=ratio<=1.1)
    counts=dict(case_records=0,coefficient_arrays=0,logical_work_records=0,own_quadrature_records=0)
    metrics={}
    for sid,cases in z['case_results'].items():
        info=z['subjects'][sid] if mode=='front' else z['queries'][sid];cfg=info['config'] if mode=='front' else info
        for j,row in cases.items():
            counts['case_records']+=1
            if row['score'].get('finite',True):close(row['score']['e'],max(row['score']['e_t'][1:]))
            if row.get('invocation_id'):
                r=inv[row['invocation_id']];assert r['role']=='B' and r['case']==int(j)
                assert row['score']==r['score'] and row['output_sha256']==r['output_sha256'] and row['coefficient_sha256']==r['coefficient_sha256']
            f=raw/row['raw']['path'];assert sha(f)==row['raw']['sha256']
            arrays=np.load(f,allow_pickle=False)
            if 'C' in arrays:
                C,W=arrays['C'],arrays['W'];assert C.dtype==W.dtype==np.float64 and W.shape[0]==6
                assert ah(C)==row['coefficient_sha256'];keep=int(round(.05/cfg['dt']));close(C[::keep],W)
                counts['coefficient_arrays']+=1
            work=row['work'];rec=work.get('raw_step_records',{})
            if 'q' in rec:
                expected=quasi_newton(rec['q'],rec['n_bt'],rec['reason'])
                for k,v in expected.items():
                    if k not in ('count_kind','missing_counter_reason'):assert work[k]==v,(sid,j,k)
                assert work['jacobian_contractions']==sum(rec['n_exec']);counts['logical_work_records']+=1
            elif cfg.get('solver')=='fs':
                aeff=info['effective_parameters']['FS_a_eff'] if mode=='front' else __import__('math').ceil(6*.01/cfg['dt'])
                expected=fixed_sweep(rec['it'],aeff,cfg['work_policy'],cfg['scheme'])
                for k,v in expected.items():
                    if isinstance(v,int):assert work[k]==v,(sid,j,k)
                counts['logical_work_records']+=1
            elif mode=='front' and cfg['kind']=='POD':
                n=cfg['steps'];it=int(np.asarray(rec['it'])[:n].sum())
                assert work['nonlinear_iterations']==it and work['residual_jacobian_pairs']==n+it and work['residual_evaluations']==4*n+it
                counts['logical_work_records']+=1
            elif mode=='front' and cfg['kind']=='PS':
                n=int(round(.25/cfg['spec'][2]));assert work['time_steps']==n and work['nonlinear_evaluations']==4*n
                counts['logical_work_records']+=1
            elif mode=='front' and cfg['kind']=='FD':
                assert work['nonlinear_iterations']==sum(work['raw_stage_Newton'])
                assert work['residual_evaluations'] is None and work['JVP_evaluations'] is None
                counts['logical_work_records']+=1
            if mode=='mesh':
                bank=cfg['bank']
                if bank not in metrics:metrics[bank]=dict(np.load(raw/'data/operators'/f'{bank}.npz',allow_pickle=False))
                L=metrics[bank]['L65'];n0=row['common_initial_reference_norm']
                close(distance(W,arrays['tight_W'],L,n0),row['common_distance_tight'])
                close(distance(arrays['tight_W'],arrays['tighter_W'],L,n0),row['common_tight_tighter_distance'])
                nn=129 if bank=='W1024' else cfg['N'];LN=metrics[bank]['L'+str(nn)]
                close(distance(W,arrays['tight_W'],LN,row['native_initial_norm']),row['native_distance_tight'])
                g=arrays['fresh_gamma'];assert row['nonstationary']==int(np.sum(~(g<=1e-3)))
        if mode=='mesh' and info.get('Q1'):
            assert sorted(info['Q1']['case_ids'])==sorted(map(int,cases)) and info['Q1']['case_count']==len(cases)
    if mode=='mesh':
        for sid,query in z['queries'].items():
            controls=[query['Q2']] if 'Q2' in query else []
            controls+=list(query.get('anchor_Q2',{}).values())
            for q in controls:
                for r in q['cases']:
                    f=raw/r['raw']['path'];assert sha(f)==r['raw']['sha256'];v=np.load(f,allow_pickle=False)
                    num=np.linalg.norm(v['adv_target']-v['adv_successor'],axis=1);den=np.linalg.norm(v['adv_successor'],axis=1)
                    rho=np.divide(num,den,out=np.full_like(num,np.inf),where=den>0);rho=np.where((den==0)&(np.linalg.norm(v['adv_target'],axis=1)==0),0.,rho)
                    close(rho,r['rho']);close(rho,v['rho']);counts['own_quadrature_records']+=1
    out=dict(passed=True,complete=z['execution']['complete'],mode=mode,source_result=dict(path=str(rp.relative_to(HERE)),sha256=sha(rp)),
        checker_sha256=sha(Path(__file__)),counts=counts,timing_rows=len(inv),brackets=len(brackets),
        scope='Mechanical source/array/timing/linkage/logical-work/Gram-distance checks only; no independent E0 decode or fresh residual replay. Bulk field collection hashes are verified by collector.',
        independent_scientific_audit='still required')
    (root/'MECHANICAL-CHECK.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
