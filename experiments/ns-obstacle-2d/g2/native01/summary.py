"""One producer/reload API control. No reference or training admission."""
from pathlib import Path
import argparse
import json
import sys

parser=argparse.ArgumentParser()
parser.add_argument('results',type=Path)
parser.add_argument('--exit-code',required=True,type=int)
parser.add_argument('--failed-stage',required=True)
args=parser.parse_args()
root=args.results
report={'scope':'one coarse native producer and fresh-process binary reload control',
        'physical_admission':False,'training_released':False,'passed':False,
        'run_exit_code':args.exit_code,'last_stage':args.failed_stage,'observations':{}}
for name in ('preflight.json','dfg/result.json','dfg/reload_checks.json',
             'dfg/reload_process.json','dfg/essential_trace_certificate.json',
             'dfg/affine_gradient_orientation.json','dfg/geometry_checks.json',
             'dfg/FAILED.json'):
    path=root/name
    if path.is_file():
        value=json.loads(path.read_text())
        if name=='preflight.json':value={'passed':value.get('passed')}
        report['observations'][name]=value
stage_file=root/'stage_exit_codes.jsonl'
report['stage_exit_codes']=[json.loads(line) for line in stage_file.read_text().splitlines()] if stage_file.exists() else []
if args.exit_code==0 and args.failed_stage=='completed':
    observed=report['observations']
    report['passed']=bool(observed.get('preflight.json',{}).get('passed') is True
        and observed.get('dfg/reload_checks.json',{}).get('passed') is True
        and observed.get('dfg/affine_gradient_orientation.json',{}).get('passed') is True
        and 'dfg/result.json' in observed and 'dfg/FAILED.json' not in observed)
report['limits']=['Successful serialization is not physical accuracy or reference admission.',
                  'Coarse DFG force/divergence/balance/probe discrepancies remain diagnostic.',
                  'Downstream adapter needs separate reviewed binary-format binding; no training manifest is created.']
(root/'native_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':report['passed'],'physical_admission':False,'last_stage':args.failed_stage}),flush=True)
sys.exit(0 if report['passed'] else 1)
