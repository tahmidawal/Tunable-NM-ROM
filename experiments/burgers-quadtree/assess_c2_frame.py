"""Summarize every registered frame recipe; no unseen data access."""
from pathlib import Path
import json,numpy as np
from collections import defaultdict
H=Path(__file__).resolve().parent;rows=[]
for label,path in [('diffusive_transferred','results/c2frame/output/result.json'),('initial_transferred','results/c2trainframe/output/transfer/result.json'),('initial_aligned','results/c2trainframe/output/result.json')]:
 r=json.loads((H/path).read_text());assert r['complete'];groups=defaultdict(list)
 for x in r['rows']:groups[x['r'],x['k'],x['kind']].append(x)
 for (coarse,local,kind),cases in groups.items():
  assert len(cases)==32
  changes=[max(abs(np.asarray(x['error128'])[[2,4,6,8,10]]-np.asarray(x['error256'])[[2,4,6,8,10]])) for x in cases]
  worst=max(cases,key=lambda x:x['evolved']);rows.append(dict(stage=label,coarse=coarse,local=local,kind=kind,worst_evolved=worst['evolved'],worst_case=worst['case'],evolved_scoring_change=max(changes),cases_below_half_percent=sum(x['evolved']<=.005 for x in cases),cases_below_one_percent=sum(x['evolved']<=.01 for x in cases)))
best={label:min([x for x in rows if x['stage']==label],key=lambda x:x['worst_evolved']) for label in sorted({x['stage'] for x in rows})};result=dict(rows=rows,best=best,contract='all development trajectories, same physical coefficient fit rescored on finer grid; representation only, no PDE or speed claim; initial-wall changes reported separately in raw data')
(H/'evidence/c2frame-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(best,indent=2))
