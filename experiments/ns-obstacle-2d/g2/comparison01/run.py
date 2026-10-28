"""Exact retained inputs: first two failed points, then both full comparisons."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'g2')]
import ngsolve as ng
from exact_lookup import ExactPointLocator
from compare_fields import load_candidate

OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
inputs=json.loads((ROOT/'g2/comparison01/INPUTS.json').read_text())
original=ROOT/'original-failure.json'
assert hashlib.sha256(original.read_bytes()).hexdigest()==inputs['original_failure_sha256']
assert inputs['original_failed_points']==[r['point'] for r in json.loads(original.read_text())['errors']]
entries={name:{'directory':str(ROOT/row['directory']),'export_manifest_sha256':row['export_manifest_sha256']} for name,row in inputs['cases'].items()}
folder,cfg,_=load_candidate(entries['dfg_h0'])
ng.SetNumThreads(cfg['threads'])
with ng.TaskManager():
    mesh=ng.Mesh(str(folder/'mesh.vol.bin'));locator=ExactPointLocator(mesh,ng,cfg['D'])
    affine=ng.CF((2*ng.x+3*ng.y,-5*ng.x+7*ng.y))
    evidence=[]
    for xy in inputs['original_failed_points']:
        mip,owners,certificate,fallback=locator.locate(xy)
        value=np.asarray(affine(mip),dtype=np.float64)
        expected=np.array([2*xy[0]+3*xy[1],-5*xy[0]+7*xy[1]])
        gap=float(np.max(np.abs(value-expected)))
        row={'requested_point':xy,'certificate':certificate,'fallback':fallback,'affine_value':value.tolist(),'expected_affine_value':expected.tolist(),'affine_max_error':gap}
        evidence.append(row)
        (OUT/'two_point_control.json').write_text(json.dumps({'points':evidence,'passed':False,'physical_admission':False},indent=2)+'\n')
        if not certificate['valid'] or fallback is None or gap>1e-10:raise ArithmeticError('exact-point native inverse control failed')
    (OUT/'two_point_control.json').write_text(json.dumps({'points':evidence,'passed':True,'physical_admission':False},indent=2)+'\n')
for kind in ('dfg','manufactured'):
    cases={name:entry for name,entry in entries.items() if name.startswith(kind+'_')}
    plan={'cases':cases,'rule_case':kind+'_h2','comparisons':[{'case':f'{kind}_h{i}','reference':f'{kind}_h{i+1}','purpose':'unchanged complete h2 rule; exact-coordinate lookup repair'} for i in range(2)]}
    path=OUT/(kind+'-plan.json');path.write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'g2/compare_fields.py'),'--plan',str(path),'--out',str(OUT/('comparison_'+kind))],check=True,timeout=300)
(OUT/'comparison_control_complete.json').write_text(json.dumps({'completed':True,'physical_admission':False,'new_PDE_solves':0,'original_failed_comparison_retained':True},indent=2)+'\n')
