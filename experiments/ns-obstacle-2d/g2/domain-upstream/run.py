"""Single same-resolution P4 domain control; explicit bounded resources."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
started=time.monotonic();progress={'completed':[],'failed':None,'physical_admission':False}
def save():(OUT/'controls_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
def execute(name,command):
    remaining=1100-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal job budget exhausted')
    t=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=min(900,remaining))
    progress['completed'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-t,'command':command});save()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    for name in ('case',):
        execute(name,[sys.executable,str(ROOT/'fom.py'),'--config',f'g2/domain-upstream/{name}.json','--out',str(OUT/name)])
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
