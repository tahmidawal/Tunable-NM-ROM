"""FD1 structural/negative gate checks only; no numerical admission or data fit."""
import copy,json,sys,tempfile
from pathlib import Path
from qualification import GATE,expected_scope,require_qualification
from fe_adapter import read_training_manifest
from train_bank import train
scope=expected_scope()
# This incomplete structural fixture has no reviewed hashes/data and cannot pass
# the actual training reader or trainer. It is never written as an admission.
structural={'gate':GATE,'passed':True,'qualification_scope':scope}
assert require_qualification(structural)==structural
invalid=[{},dict(structural,gate='G2-steady'),dict(structural,passed=False),dict(structural,qualification_scope={})]
for key,value in [('scope_contract_sha256','0'*64),('amendment_id','other'),('original_G2_status','passed'),
                  ('domain_sensitivity_status','passed'),('domain_independence_admitted',True),
                  ('domain_independence_admitted',0),('geometry_population_and_numerical_thresholds_unchanged',False)]:
 row=copy.deepcopy(structural);row['qualification_scope'][key]=value;invalid.append(row)
for row in invalid:
 try:require_qualification(row)
 except (RuntimeError,ValueError):pass
 else:raise AssertionError('invalid scope accepted')
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'not-an-admission.json';path.write_text(json.dumps(structural))
 try:read_training_manifest(path)
 except RuntimeError:pass
 else:raise AssertionError('scope alone admitted data')
 try:train(None,None,None,None,dict(structural,split='training'),evidence_directory=Path(directory)/'training')
 except RuntimeError:pass
 else:raise AssertionError('scope alone permitted training')
 assert not (Path(directory)/'training').exists()
assert 'ngsolve' not in sys.modules and 'jax' not in sys.modules
print('PASS FD1 scope/failure metadata bound; original G2/amendment-only inputs rejected before native/JAX/data admission')
