"""Retain initial ladder outcomes, including failed physical criteria; no admission."""
from pathlib import Path
import argparse
import json
import sys

p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--exit-code',type=int,required=True);p.add_argument('--failed-stage',required=True);a=p.parse_args()
root=a.results
report={'physical_admission':False,'training_released':False,'run_exit_code':a.exit_code,'last_stage':a.failed_stage,'cases':{},'comparisons':{},'completed_measurements':False}
for kind in ('dfg','manufactured'):
    for level in range(3):
        name=f'{kind}_h{level}';f=root/name/'result.json'
        if f.exists():report['cases'][name]=json.loads(f.read_text())
        failure=root/name/'FAILED.json'
        if failure.exists():report['cases'].setdefault(name,{})['failure']=json.loads(failure.read_text())
    f=root/f'comparison_{kind}'/'comparison.json'
    if f.exists():report['comparisons'][kind]=json.loads(f.read_text())
    f=root/f'comparison_{kind}'/'FAILED.json'
    if f.exists():report['comparisons'][kind]={'failure':json.loads(f.read_text())}
report['completed_measurements']=bool(a.exit_code==0 and a.failed_stage=='completed' and len(report['cases'])==6 and len(report['comparisons'])==2 and all('failure' not in x for x in report['comparisons'].values()))
report['limits']=['Initial h study only; p, tolerance, quadrature, domain and outlet controls remain required.',
                  'All coarse/fine physical diagnostics are retained, including failures; no threshold is changed.',
                  'Producer point cap differs from downstream ROM caps, which remain unchanged.']
(root/'ladder_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'completed_measurements':report['completed_measurements'],'physical_admission':False}),flush=True)
sys.exit(0 if report['completed_measurements'] else 1)
