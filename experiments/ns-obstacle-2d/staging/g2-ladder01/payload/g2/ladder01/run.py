"""Execute the frozen initial spatial ladder; never admit reference data."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results'
plan=json.loads((HERE/'PLAN.json').read_text())
started=time.monotonic()
progress={'physical_admission':False,'training_released':False,'completed':[],'failed':None}

def save():
    (OUT/'ladder_progress.json').write_text(json.dumps(progress,indent=2)+'\n')

def execute(label,command):
    remaining=plan['internal_budget_seconds']-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal allocation budget exhausted')
    timeout=min(plan['per_process_timeout_seconds'],remaining)
    before=time.monotonic()
    with (OUT/(label+'.stdout.log')).open('w') as stdout,(OUT/(label+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=timeout,check=False)
    record={'name':label,'command':command,'returncode':result.returncode,'seconds':time.monotonic()-before}
    progress['completed'].append(record);save()
    if result.returncode:raise RuntimeError(f'{label} failed with exit {result.returncode}')

try:
    for case in plan['cases']:
        execute(case['name'],[sys.executable,str(ROOT/'fom.py'),'--config',case['configuration'],'--out',str(OUT/case['name'])])
    for kind in ('dfg','manufactured'):
        entries={}
        for level in range(3):
            name=f'{kind}_h{level}';folder=OUT/name
            entries[name]={'directory':str(folder),'export_manifest_sha256':hashlib.sha256((folder/'EXPORT-MANIFEST.json').read_bytes()).hexdigest()}
        comparison={'cases':entries,'rule_case':f'{kind}_h2','comparisons':[
            {'case':f'{kind}_h{i}','reference':f'{kind}_h{i+1}','purpose':'successive h refinement on fixed full h2 rule'} for i in range(2)]}
        path=OUT/f'{kind}-comparison-plan.json';path.write_text(json.dumps(comparison,indent=2)+'\n')
        execute(f'comparison_{kind}',[sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/f'comparison_{kind}')])
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
