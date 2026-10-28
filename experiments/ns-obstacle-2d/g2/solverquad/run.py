"""Same immutable P4 mesh, tighter solver and higher quadrature, full-rule comparison."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
started=time.monotonic();progress={'completed':[],'failed':None,'physical_admission':False}
def save():(OUT/'controls_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
def execute(name,command):
    remaining=1100-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal job budget exhausted')
    tick=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=min(600,remaining))
    progress['completed'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-tick,'command':command});save()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    saved=json.loads((ROOT/'g2/solverquad/CASE-MANIFESTS.json').read_text());cases={'p4_g4_h3':{'directory':str(ROOT/'inputs/p4_g4_h3'),'export_manifest_sha256':saved['p4_g4_h3']}}
    for name in ('solver_100x','quadrature_plus4'):
        folder=OUT/name;execute(name,[sys.executable,str(ROOT/'fom.py'),'--config',f'g2/solverquad/{name}.json','--out',str(folder)])
        cases[name]={'directory':str(folder),'export_manifest_sha256':hashlib.sha256((folder/'EXPORT-MANIFEST.json').read_bytes()).hexdigest()}
    plan={'cases':cases,'rule_case':'quadrature_plus4','comparisons':[{'case':name,'reference':'p4_g4_h3','purpose':'numerical sensitivity against adopted P4; common original-P4 denominator on complete higher quadrature rule'} for name in ('solver_100x','quadrature_plus4')]}
    path=OUT/'comparison-plan.json';path.write_text(json.dumps(plan,indent=2)+'\n');execute('comparison_solver_quad',[sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/'comparison_solver_quad')])
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
