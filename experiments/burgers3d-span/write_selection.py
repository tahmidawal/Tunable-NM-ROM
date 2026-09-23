"""Freeze the validation selection (A3 panels) into selection.json: per mesh the accurate and fast arm names and
specs and the validation FOM. Must be committed before the held-out configs are generated."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tag = sys.argv[1]
out = dict(source=f'checks/select-{tag}-n*.json', meshes={})
for n in (33, 65, 129):
    s = json.loads((HERE / 'checks' / f'select-{tag}-n{n}.json').read_text())
    spec = lambda k: None if k is None else {kk: v for kk, v in s['arms'][k]['spec'].items()
                                             if kk in ('kind', 'rule', 'Rp', 'K', 'gtol', 'solver', 'dt')}
    out['meshes'][str(n)] = dict(accurate=s['accurate'], fast=s['fast'], fom=s['fom'],
                                 accurate_spec=spec(s['accurate']), fast_spec=spec(s['fast']),
                                 stopping_rule_pass=s['stopping_rule_pass'])
(HERE / 'selection.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
