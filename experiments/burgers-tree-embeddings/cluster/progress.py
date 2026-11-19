"""Read-only live progress; partial cases are never counted as completed gates."""
from pathlib import Path
import argparse,json,math,shlex,subprocess
H=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('names',nargs='+');args=ap.parse_args()
rows=json.loads((H/'cluster/ledger.json').read_text())
for name in args.names:
    row=next(v for v in rows if v['name']==name)
    remote=row.get('archived_remote',row['remote'])
    text=subprocess.check_output(['ssh','tufts-login','cat '+shlex.quote(remote+'/output/result.json')],text=True)
    r=json.loads(text);all_cases=r.get('cases',[])
    complete=[c for c in all_cases if isinstance(c,dict) and len(c.get('methods',{}))==2 and
        all('physical_accuracy_pass' in v and 'actual4096_final_error' in v for v in c['methods'].values())]
    values=[v for c in complete for v in c['methods'].values()]
    bad=sum(not(all(math.isfinite(z) for z in s) and s[1]<=1e-8 and (s[2]<=1e-10 or s[1]<=1e-10)) for v in values for s in v['step_status'])
    result=dict(name=name,job=row['id'],source_status=r['status'],fully_scored_cases=len(complete),
        partial_cases=len(all_cases)-len(complete),physical_failures=sum(not v['physical_accuracy_pass'] for v in values),
        nonconverged_steps=bad,
        max_change=max((max(v.get('previous_state_relative_change',[0])) for v in values),default=None),
        max_error_with_allowance=max((max(v['physical_error'][1:]+[v['actual4096_final_error']])+v['reference_allowance_max'] for v in values),default=None),
        scope='Live provisional progress; retention and independent final audits still required')
    print(json.dumps(result),flush=True)
