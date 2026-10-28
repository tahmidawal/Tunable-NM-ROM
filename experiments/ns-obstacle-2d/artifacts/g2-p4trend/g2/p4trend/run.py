"""Prospective P4 own-state h1/h2/h3 qualification, no h4 or admission."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
started=time.monotonic();progress={'completed':[],'failed':None,'physical_admission':False}
def save():(OUT/'p4_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
def execute(name,command):
    remaining=1100-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal job budget exhausted')
    t=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=min(600,remaining))
    progress['completed'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-t,'command':command});save()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    cases={}
    for level in (1,2,3):
        name=f'dfg_h{level}';folder=OUT/name
        execute(name,[sys.executable,str(ROOT/'fom.py'),'--config',f'g2/p4trend/{name}.json','--out',str(folder)])
        cases[name]={'directory':str(folder),'export_manifest_sha256':hashlib.sha256((folder/'EXPORT-MANIFEST.json').read_bytes()).hexdigest()}
    plan={'cases':cases,'rule_case':'dfg_h3','comparisons':[{'case':f'dfg_h{i}','reference':f'dfg_h{i+1}','purpose':'P4/g4 own-state successive differences on complete fixed P4 h3 rule'} for i in (1,2)]}
    path=OUT/'comparison-plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    execute('comparison_dfg',[sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/'comparison_dfg')])
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
