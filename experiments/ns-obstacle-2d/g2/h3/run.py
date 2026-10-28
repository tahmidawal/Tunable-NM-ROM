"""One h3 extension and complete fixed-h3-rule successive differences."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
started=time.monotonic();progress={'completed':[],'failed':None,'physical_admission':False}
def save():(OUT/'h3_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
def execute(name,command):
    remaining=1100-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal job budget exhausted')
    t=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=min(300,remaining))
    progress['completed'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-t,'command':command});save()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    inputs=json.loads((ROOT/'g2/h3/INPUTS.json').read_text())
    for kind in ('dfg','manufactured'):
        name=kind+'_h3';execute(name,[sys.executable,str(ROOT/'fom.py'),'--config',f'g2/h3/{name}.json','--out',str(OUT/name)])
    for kind in ('dfg','manufactured'):
        cases={name:{'directory':str(ROOT/row['directory']),'export_manifest_sha256':row['export_manifest_sha256']} for name,row in inputs['cases'].items() if name.startswith(kind+'_')}
        name=kind+'_h3';folder=OUT/name;cases[name]={'directory':str(folder),'export_manifest_sha256':hashlib.sha256((folder/'EXPORT-MANIFEST.json').read_bytes()).hexdigest()}
        plan={'cases':cases,'rule_case':name,'comparisons':[{'case':f'{kind}_h{i}','reference':f'{kind}_h{i+1}','purpose':'two successive differences on complete fixed h3 rule'} for i in (1,2)]}
        path=OUT/(kind+'-plan.json');path.write_text(json.dumps(plan,indent=2)+'\n')
        execute('comparison_'+kind,[sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/('comparison_'+kind))])
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
