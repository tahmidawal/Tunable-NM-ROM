"""Metadata-only fixtures; imports no JAX and runs no scientific computation."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/'screen3d.py').read_text())
selector=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='select')
scope={'np':np}
exec(compile(ast.Module(body=[selector],type_ignores=[]),'screen_selector_fixture','exec'),scope)
select=scope['select']
rows=[]
for lc in (16,32):
    for source in ('A','B'):
        for category in ('ordinary','bank','pod_matched','pod_stronger'):
            for variant,errors in [('median',[1.,.1,.1,.1]),('worst',[.1001]*4)]:
                for case,error in enumerate(errors):
                    desc=dict(category=category,power=2,rank=128,ridge=0.,prior='field')
                    rows.append(dict(Lc=lc,source=source,cohort='dev4',case=case,
                        descriptor=desc,method=f'{category}_{variant}',e=error,e_t=[0.]+[error]*5,
                        eps_ref=.001,input_eligible=(source=='A' or case!=0),status='complete',
                        initial_error=0.,scores=[],repetitions=[[]]*6,sensitivity=[[]]*6,
                        approximation_gap=[],timing_policy='accuracy-only'))
choices,details=select(rows)
assert all(name.endswith('_median' if '/A/' in key else '_worst') for key,name in choices.items())
assert all(value['winner']['qualified_spatial_input']==('/A/' in key) for key,value in details.items())
failed=[dict(r,e=None,status='setup-failure') for r in rows]
assert all(v is None for v in select(failed)[0].values())
bad=[dict(r) for r in rows];bad[1]['case']=0
try:select(bad)
except AssertionError:pass
else:raise AssertionError('duplicate development case accepted')
try:select([dict(r,cohort='val64') for r in rows])
except AssertionError:pass
else:raise AssertionError('validation rows accepted by selector')

spec=importlib.util.spec_from_file_location('export_fixture',HERE/'export3d.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(dir=HERE,prefix='fixture-') as directory:
    root=Path(directory)
    for lc in (16,32):
        (root/f'L{lc}/plans').mkdir(parents=True)
        (root/f'L{lc}/plans/setup.json').write_text(json.dumps(dict(plans={})))
    (root/'execution.json').write_text(json.dumps(dict(phase='val')))
    (root/'selection.json').write_text(json.dumps(dict(selected=choices)))
    (root/'all_scores.json').write_text(json.dumps(rows))
    sys.argv=['export3d.py','--input',str(root)];module.main()
    output=json.loads((root/'EXPORT.json').read_text())
    for row in output['cases']:
        assert row['approximate_pod_gap_consequential'] is None
    for comparison in output['selected_comparisons']:
        assert comparison['finite_cases']==4
        if comparison['source']=='A': assert comparison['resolved_wins'] is None
        else: assert comparison['qualified_cases']==3
        assert comparison['ratio_of_cohort_medians']==1.
print('metadata fixtures passed: frozen source-independent export, selectors, coverage, unavailable diagnostics')
