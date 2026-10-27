"""Create the exact full64 manifest only from audited fixed diagnostic outputs."""
from pathlib import Path
import hashlib
import json
import argparse
import re

HERE=Path(__file__).resolve().parent
def pin(p):return dict(path=str(p.relative_to(HERE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def bank(name,r,M,rule,solver,scheme,dt,k,archived=True):
    steps=round(.25/dt);key=f'{name}|{rule}|{solver}|{scheme}|{dt:.8g}|'+('k' if solver=='qn' else 'K')+str(k)
    d=dict(id=f'{name}_{rule}_{solver}_{scheme}_{steps}',kind='bank',setting=name,r=r,M=M,rule=rule,solver=solver,scheme=scheme,dt=dt,work_policy=k)
    if archived:d['archived_key']=key
    else:d['successors']=['gl64','gl80']
    return d

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pod-attempt',default='podtune05');args=parser.parse_args()
    assert re.fullmatch('[a-z0-9-]+',args.pod_attempt)
    audits=json.loads((HERE/'AUDIT-STATUS.json').read_text())
    for attempt in ('tdiag02',args.pod_attempt,'qcheck801'):
        assert audits['jobs'][attempt]['results'].startswith('READY'),(attempt,'independent results gate not READY')
    tp=HERE/'results/tdiag02/raw/output/result.json';pp=HERE/f'results/{args.pod_attempt}/raw/output/result.json'
    t,p=[json.loads(q.read_text()) for q in (tp,pp)]
    assert t['complete'] and p['complete'] and t['cohort_sha256']==p['cohort_sha256']
    subjects=[bank('W256',256,1027,'lat8192','qn','BDF2',.025,4),bank('M2_512',512,2052,'gl32','qn','BDF2',.05/3,4),
        bank('W512k8',512,4097,'lat16384','fs','CN',.01,1),bank('W768',768,3076,'gl40','fs','CN',.01,1),
        bank('W1024',1024,4097,'gl56','fs','CN',.01,1)]
    cases=[0,5,30,43];rejected={}
    coarse=[t['runs'][f'W1024|gl56|BDF2|0.025|candidate|case{j}']['score'] for j in cases]
    lower=max(max(s['e']-s['eps_ref'],0) for s in coarse)
    repaired=[t['runs'][f'W1024|gl56|BDF2|0.025|candidate|case{j}'] for j in cases]
    repair_Q1=all(r['valid'] and r['anchor_usable'] and r['distance_tight']<=2.5e-4 for r in repaired) and sum(r['nonstationary'] for r in repaired)<=.01*sum(r['steps'] for r in repaired)
    repair_Q2=all(t['quadrature'][f'0.025|{lev}|case{j}']['quadrature_subset_pass'] for lev in ('candidate','tight','tighter') for j in cases)
    if repair_Q1 and repair_Q2:subjects.append(bank('W1024',1024,4097,'gl56','qn','BDF2',.025,1,False))
    else:rejected['coarse_repair']=dict(Q1_subset=repair_Q1,Q2_subset=repair_Q2)
    for dt in (.0125,.00625):
        rows=[t['runs'][f'W1024|gl56|BDF2|{dt:.8g}|candidate|case{j}'] for j in cases]
        q1=all(r['valid'] and r['anchor_usable'] and r['distance_tight']<=2.5e-4 for r in rows) and sum(r['nonstationary'] for r in rows)<=.01*sum(r['steps'] for r in rows)
        q2=all(t['quadrature'][f'{dt:.8g}|{lev}|case{j}']['quadrature_subset_pass'] for lev in ('candidate','tight','tighter') for j in cases)
        if q1 and q2 and dt==.00625:subjects.append(bank('W1024',1024,4097,'gl56','qn','BDF2',dt,1,False))
        else:rejected[str(dt)]=dict(Q1_subset=q1,Q2_subset=q2,status='diagnostic retained; final two-arm freeze chooses coarse repair and finest representative')
    pod=list(p['frozen_picks'])
    # Preserve the original qualified POD endpoint even when a fair-grid endpoint wins development.
    if 'POD512_none_BDF2_15steps' not in pod:pod.append('POD512_none_BDF2_15steps')
    for name in pod:
        row=p['runs'][name+'|case0']
        subjects.append(dict(id=name,kind='POD',subject_name=name,r=512,scheme=row['scheme'],dt=row['dt'],steps=row['steps'],
                             production_gtol=row['production_stopping_gtol'],production_tolf=1e-9,solver='common'))
    for N,dt in [(14,.025),(16,.025),(24,.01),(32,.005)]:subjects.append(dict(id=f'PS_N{N}_{round(.25/dt)}',kind='PS',spec=['ps',N,dt]))
    for dt,level in [(.0125,1),(.005,0)]:subjects.append(dict(id=f'FD_CC_L64_{round(.25/dt)}_t{level}',kind='FD',spec=['fd','CC',64,dt,level]))
    out=dict(seed=923801,mode='front',cases=list(range(64)),subjects=subjects,anchor=subjects[1]['id'],
        timing_seed=20261026,timing_reps=3,max_timing_reps=5,phase_repetitions=3,burn_seconds=.1,phase_cases=cases,finer_check_cases=list(range(8)),
        memory_peak_limit_bytes=120_000_000_000,selection_sources=dict(time=pin(tp),POD=pin(pp),audits=pin(HERE/'AUDIT-STATUS.json')),
        rejected_time_candidates=rejected,scope='exposed validation cohort; finite frozen tested set, not sealed confirmation')
    (HERE/'configs/front64.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(dict(subjects=len(subjects),selected_new_time=[s['id'] for s in subjects if s.get('successors')],POD=pod,rejected_time=rejected)))

if __name__=='__main__':main()
