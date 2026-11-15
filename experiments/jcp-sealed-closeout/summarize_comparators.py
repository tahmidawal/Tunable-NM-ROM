"""Describe retained full-solver and native POD outcomes without selection on sealed data."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from statistics import median
from open_sealed import COHORTS, sha

def stats(values):
    finite=[float(x) for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    return dict(total=len(values),finite=len(finite),missing_or_nonfinite=len(values)-len(finite),
                median=median(finite) if finite else None,maximum=max(finite) if finite else None)

def timing(rep,cfg):
    panel=rep['timing']
    tables={}
    fixed=cfg['timing']; cases=rep['cases'][:fixed['cases']]
    if panel['cases']!=cases: raise ValueError('Wrong timed cases')
    for phase,key,nrep in [('main','invocations',fixed['reps']),('confirm','confirm_invocations',fixed['confirm_reps'])]:
        groups=defaultdict(dict)
        for x in panel[key]:
            k=x['kind']+'|'+x['name']; j=(x['case'],x['rep'])
            if j in groups[k] or x['seconds']<=0 or not math.isfinite(x['seconds']):
                raise ValueError('Invalid timing repetition')
            groups[k][j]=x['seconds']
        for v in groups.values():
            if set(v)!={(i,r) for i in cases for r in range(nrep)}: raise ValueError('Missing timed case/repetition')
        tables[phase]={k:{i:median(v[i,r] for r in range(nrep)) for i in cases} for k,v in groups.items()}
    main,confirm=tables['main'],tables['confirm']
    drift={}
    for k in confirm:
        m=panel['summary'][k]['median_ms']; c=panel['confirm_summary'][k]['median_ms']
        drift[k]=max(c/m,m/c)
    return dict(raw_main_repetitions=len(panel['invocations']),raw_confirm_repetitions=len(panel['confirm_invocations']),
                gates=rep['gates']['G_time'],main_confirm_ratio_fold=drift,
                qualified=bool(rep['gates']['G_time']['outputs_pass'] and rep['gates']['G_time']['drift_pass'] and max(drift.values())<=1.10),
                main_per_case_median_seconds=main,confirm_per_case_median_seconds=confirm)

def score_rows(rows,kind,refs,n):
    ix={row['case']:row for row in rows}
    if len(rows)!=n or sorted(ix)!=list(range(n)): raise ValueError('Missing physical cases')
    eligible=[i for i in range(n) if refs[f'{i}|{kind}']['eligible']]
    def e(i):
        if kind not in ix[i]: return None
        if bool(ix[i][kind]['eligible'])!=bool(refs[f'{i}|{kind}']['eligible']): raise ValueError('Reference eligibility mismatch')
        return ix[i][kind]['e']
    return {'all':stats([e(i) for i in range(n)]),'eligible':stats([e(i) for i in eligible])}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--pod',type=Path)
    args=ap.parse_args()
    braw=args.bank.read_bytes(); bank=json.loads(braw)
    cfg=bank['config']; dim=cfg['dim']; co,n=COHORTS[dim]
    if cfg['cohort']!=co or not bank['complete']: raise ValueError('Not a complete sealed panel')
    refs=bank['references']
    result={'job_id':bank['job_id'],'result_sha256':sha(braw),'dim':dim,'mesh':cfg['mesh'],'foms':{},'timing':timing(bank,cfg)}
    for name in [f['name'] for f in cfg['foms']]:
        rows=[r for r in bank['fom_rows'] if r['fom']==name]
        result['foms'][name]={**{k:score_rows(rows,k,refs,n) for k in ('ST','S')},'unconverged_count':sum(not x['converged'] for x in rows)}
    if args.pod:
        if cfg['mesh']!={2:256,3:65}[dim]: raise ValueError('POD controls are native meshes only')
        meta=json.loads((args.pod/'COMPLETE.json').read_bytes())
        if meta['n']!=n or meta['backend']!='gpu' or not meta['x64'] or meta['precision']!='highest': raise ValueError('Invalid POD run')
        raw=(args.pod/'result.jsonl').read_bytes(); rows=[json.loads(line) for line in raw.splitlines()]
        expected={(i,r) for i in range(n) for r in range(5 if i<6 else 1)}
        if len(rows)!=len(expected) or {(x['case'],x['rep']) for x in rows}!=expected: raise ValueError('Missing POD repetitions')
        cases=[x for x in rows if x['rep']==0]
        info={k:score_rows(cases,k,refs,n) for k in ('ST','S')}
        info.update(meta=meta,result_sha256=sha(raw),finite_fields=sum(x['finite'] for x in cases),
              valid_cases=sum(x['qualification']['valid'] for x in cases),
              cases_with_nonstationary_steps=sum(x['qualification']['nonstat']>0 for x in cases),
              nonstationary_steps=sum(x['qualification']['nonstat'] for x in cases),
              steps=sum(x['qualification']['steps'] for x in cases),
              all_case_query_seconds=stats([x['seconds'] for x in cases]),timing_repeats=len(rows))
        result['pod']=info
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__': main()
