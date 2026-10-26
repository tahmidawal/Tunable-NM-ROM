"""Generate the coordinator's small numerical export directly from preserved job outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ledger=json.loads((HERE/'JOB-LEDGER.json').read_text())
    audits=json.loads((HERE/'AUDIT-STATUS.json').read_text())
    out=dict(audits=audits,status='diagnostic subset; consult per-job independent audit verdicts',scope='validation diagnostics; no new frontier or comparator win claim',
             gpu_hours=sum(e.get('gpu_hours',0.) for e in ledger['entries']),budget_gpu_hours=ledger['budget_gpu_hours'],
             jobs=ledger['entries'],observations=[],same_history=[],archive_identities=[],quadrature=[],
             next_open=['Independent results audit of D-S pilot','Remaining W768/W1024 cases0,5,30,43 with complete adaptive-anchor distance',
                        'D-Q gl80 missing finer-rule check on first8','Full64 paired same-invocation frontier validation after diagnostic review'])
    for entry in ledger['entries']:
        rp=HERE/'results'/entry['attempt']/'raw/output/result.json'
        if not rp.exists():continue
        z=json.loads(rp.read_text());source=dict(path=str(rp.relative_to(HERE)),sha256=sha(rp),job_id=entry['job_id'],source_commit=z['source_commit'],gpu=z['gpu'])
        for key,v in z['runs'].items():
            if 'nonlinear_iterations' not in v:
                out['quadrature'].append(dict(key=key,source=source,**v));continue
            obs=dict(key=key,source=source,scope=z['scope'],score=v['score'],fresh_nonstationary_steps=v['nonstationary'],
                     nonlinear_iterations=v['nonlinear_iterations'],jacobian_contractions=v['jacobian_contractions'],
                     residual_evaluations=v['residual_evaluations'],historical_reproduction_relative=v.get('historical_reproduction_relative'),
                     distance_historical_tight=v.get('distance_historical_tight'),distance_gate_diagnostic=v.get('distance_gate_diagnostic'),
                     valid=v.get('valid',v['finite']),Q1_full_population='not established',Q2='historical gl40 evidence; not recertified by this single-case pilot')
            obs['timing_seconds']=[]
            obs['timing_admission']='pilot descriptive; drift gate not applied' if entry['attempt']=='spilot02' else 'bracket drift ratio in [1/1.1,1.1]'
            obs['timing_all_admitted']=True
            for tk,trips in z['timing'].items():
                sn,policy,case=key.split('|')
                if not tk.startswith(sn+'|') or not tk.endswith('|'+case):continue
                for trip in trips:
                    for r in trip:
                        if r['arm']==policy:
                            obs['timing_seconds'].append(r['seconds'])
                            obs['timing_all_admitted'] &= r.get('timing_admitted',False)
            obs['query_median_seconds']=float(np.median(obs['timing_seconds'])) if obs['timing_seconds'] else None
            obs['max_evolved_error_time_index']=int(np.argmax(v['score']['e_t'][1:])+1)
            out['observations'].append(obs)
        out['same_history'] += [dict(key=k,source=source,**v) for k,v in z['restarts'].items()]
        out['archive_identities'] += [dict(key=k,source=source,**v) for k,v in z['archive_identity'].items()]
    (HERE/'EXPORT.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(dict(observations=len(out['observations']),same_history=len(out['same_history']),gpu_hours=out['gpu_hours'])))

if __name__=='__main__':main()
