"""Retained-state full original-domain comparisons; no PDE rerun or admission."""
from pathlib import Path
import json,subprocess,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
started=time.monotonic();progress={'completed':[],'failed':None,'physical_admission':False,'new_PDE_solves':0}
def save():(OUT/'comparison_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
def execute(name,command):
    remaining=1100-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('internal job budget exhausted')
    tick=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=min(600,remaining))
    progress['completed'].append({'name':name,'returncode':result.returncode,'seconds':time.monotonic()-tick,'command':command});save()
    result.check_returncode()
try:
    records=json.loads((ROOT/'g2/domain-comparison01/CASE-MANIFESTS.json').read_text())
    cases={name:{'directory':str(ROOT/'inputs'/name),'export_manifest_sha256':digest} for name,digest in records.items()}
    pairs=[{'case':name,'reference':'original','purpose':'full original-domain field/force sensitivity; finite-inlet DFG and length independence remain separate'} for name in ('upstream','downstream','symmetric')]
    pairs.append({'case':'symmetric','reference':'downstream','purpose':'separate outlet-model sensitivity on extended outlet, evaluated on complete original domain; downstream reference denominator explicitly retained'})
    values={name:json.loads((Path(case['directory'])/'result.json').read_text()) for name,case in cases.items()}
    scalar_rows=[]
    for pair in pairs:
        a,b=pair['case'],pair['reference'];force=[]
        for offset in (0,4):
            fa,fb=[values[k]['metrics']['forces']['physical_symmetric_order'+str(values[k]['configuration']['diagnostic_order']+offset)]['coefficient'] for k in (a,b)]
            numerator=abs(fa[0]-fb[0]);denominator=abs(fb[0]);original=abs(values['original']['metrics']['forces']['physical_symmetric_order'+str(values['original']['configuration']['diagnostic_order']+offset)]['coefficient'][0])
            if not np.isfinite([*fa,*fb,denominator,original]).all() or min(denominator,original)<=0:raise ValueError('invalid physical force denominator')
            force.append({'diagnostic_order_offset':offset,'case_coefficients':fa,'reference_coefficients':fb,'drag_absolute_difference':numerator,'reference_drag_denominator':denominator,'drag_relative_difference':numerator/denominator,'original_drag_denominator':original,'drag_difference_over_original':numerator/original,'transverse_absolute_difference':abs(fa[1]-fb[1])})
        pa,pb=[values[k]['metrics']['forces']['pressure_probes'] for k in (a,b)]
        if not(pa['valid'] and pb['valid']) or pa['points']!=pb['points']:raise ValueError('exact common pressure poles required')
        scalar_rows.append({**pair,'physical_force_differences':force,'pressure_probes_case':pa,'pressure_probes_reference':pb,'pressure_drop_absolute_difference':abs(pa['difference']-pb['difference'])})
    (OUT/'raw_domain_scalars.json').write_text(json.dumps({'physical_admission':False,'new_PDE_solves':0,'failed_bars_not_suppressed':True,'comparisons':scalar_rows,'meshing_disclosure':'Each extension was independently meshed with unchanged physical maxh/obstacle_h and FE/geometry orders; no assertion of an embedded identical inner mesh.','scope':'The canonical DFG result is distinct from any project-defined extended-domain family.'},indent=2)+'\n')
    plan={'cases':cases,'rule_case':'original','comparisons':pairs};path=OUT/'comparison-plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    execute('comparison_original_domain',[sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/'comparison_original_domain')])
except Exception as exc:
    progress['failed']={'error':repr(exc),'traceback':traceback.format_exc()};save();raise
save()
