"""Tabulate reference-resolved numerical error pairs on the two native POD meshes."""
import argparse
import json
import math
from pathlib import Path
from collections import Counter
from frozen_decisions import RULES
from open_sealed import sha
def classify(pod,other,ref):
    if not ref['eligible']: return 'ineligible_reference'
    if pod is None or other is None or not math.isfinite(pod) or not math.isfinite(other): return 'missing'
    eps=ref['eps_ref']
    if not isinstance(eps,(int,float)) or not math.isfinite(eps) or eps<0: raise ValueError('Invalid uncertainty')
    if abs(pod-other)<=2*eps: return 'uncertainty_overlap'
    return 'pod_lower' if pod<other else 'other_lower'
def compare(bank_path,pod_path):
    bank_raw=bank_path.read_bytes()
    raw=pod_path.read_bytes()
    bank=json.loads(bank_raw)
    cfg=bank['config']; d=cfg['dim']; ref=bank['references']
    if cfg['mesh']!={2:256,3:65}[d]: raise ValueError('Not native mesh')
    n={2:128,3:64}[d]
    samples=[json.loads(x) for x in raw.splitlines()]
    cases=[p for p in samples if p['rep']==0]; pod={p['case']:p for p in cases}
    if len(pod)!=n or len(cases)!=n or sorted(pod)!=list(range(n)): raise ValueError('POD case roster mismatch')
    outputs={}
    for category,vals,label in [('bank',bank['rows'],'arm'),('full_order',bank['fom_rows'],'fom')]:
        lookup={}
        for row in vals:
            name=row[label]
            if category=='bank' and name not in (*RULES[d],cfg['conv_arm']): continue
            if row['case'] in lookup.setdefault(name,{}): raise ValueError('Duplicate case/subject')
            lookup[name][row['case']]=row
        outputs[category]={}
        for name,rows in lookup.items():
            if sorted(rows)!=list(range(n)): raise ValueError('Missing comparator case')
            bykind={}
            for k in ('S','ST'):
                counts=Counter(); valid=Counter()
                for i,p in pod.items():
                    r=rows[i]; meta=ref[f'{i}|{k}']
                    if k in p and p[k]['eligible']!=meta['eligible']: raise ValueError('Status mismatch')
                    e=classify(p.get(k,{}).get('e'),r[k]['e'],meta)
                    counts[e]+=1
                    if p['finite'] and p['qualification']['valid'] and (r['finite'] and not r['hard_fail'] if category=='bank' else r['converged']):
                        valid[e]+=1
                bykind[k]={'all_pairs':dict(counts),'both_valid_case_flags':dict(valid)}
            outputs[category][name]=bykind
    return {'dim':d,'mesh':cfg['mesh'],'case_count':n,'bank_sha256':sha(bank_raw),'pod_sha256':sha(raw),
            'quantity':'Native POD paired numerical errors; per-case eligibility and delta > 2 eps_ref. Solver flags provided separately.',
            'subjects':outputs}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);p.add_argument('pod',type=Path);a=p.parse_args()
    print(json.dumps(compare(a.bank,a.pod),indent=2,sort_keys=True,allow_nan=False))
