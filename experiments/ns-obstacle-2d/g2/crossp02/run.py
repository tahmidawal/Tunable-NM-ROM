"""Complete adopted-P4-rule p/geometry evidence plus explicit native controls."""
from pathlib import Path
import json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
started=time.monotonic();progress={'completed':[],'failed':None,'physical_admission':False}
def save():(OUT/'crossp_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
def execute(name,command,cap):
    remaining=1100-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal job budget exhausted')
    tick=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=min(cap,remaining))
    progress['completed'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-tick,'command':command});save()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    records=json.loads((ROOT/'g2/crossp02/CASE-MANIFESTS.json').read_text())
    cases={name:{'directory':str(ROOT/'inputs'/name),'export_manifest_sha256':digest} for name,digest in records.items()}
    plan={'cases':{k:v for k,v in cases.items() if k!='native_output_coarse'},'rule_case':'p4_g4_h3','comparisons':[{'case':name,'reference':'p4_g4_h3','purpose':'raw p contrast with adopted P4 denominator; conservative envelope uses maximum of both'} for name in ('p3_g3_h3','p3_g4_h3')]}
    path=OUT/'comparison-plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    execute('comparison_p_geometry',[sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/'comparison_p_geometry')],600)
    for name,case,script in [('immutable_lift','p4_g4_h3','immutable_lift_control.py'),('cached_outputs','native_output_coarse','fom_output_control.py')]:
        path=OUT/(name+'-plan.json');path.write_text(json.dumps({'case':cases[case]},indent=2)+'\n')
        execute(name,[sys.executable,str(ROOT/'g2'/script),'--plan',str(path),'--out',str(OUT/name)],300)
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
