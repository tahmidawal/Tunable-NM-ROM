"""Explicit FD1 qualification binding; the scope amendment is never data admission."""
from pathlib import Path
import hashlib,json

GATE='G2_FIXED_FINITE_CHANNEL_V1'
EXPORT_RELEASE='G2_fixed_finite_channel_training_export'
SCOPE_SHA256='31ec75738dc47f8d960acf2f51dd355caf497a2681a3d499cc00faa138b28f76'
SCOPE_KEYS=('amendment_id','original_G2_status','domain_sensitivity_status',
 'domain_independence_admitted','required_scope','geometry_population_and_numerical_thresholds_unchanged')

def expected_scope():
 path=Path(__file__).with_name('FD1-SCOPE-CONTRACT.json');raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=SCOPE_SHA256:raise ValueError('frozen FD1 scope contract changed')
 contract=json.loads(raw);base=json.loads(path.with_name('MODEL-CONTRACT.json').read_text())
 if (contract['schema']!='ns-obstacle-reference-scope-amendment-v1' or
     contract['status']!='user_approved_not_reference_admission' or contract['qualification_gate']!=GATE or
     contract['base_campaign_sha256']!=base['central_campaign_sha256'] or
     contract['base_design_sha256']!=base['central_design_sha256']):
  raise ValueError('FD1 scope/base contract mismatch')
 return dict(scope_contract_sha256=SCOPE_SHA256,**{k:contract[k] for k in SCOPE_KEYS})

def require_qualification(record):
 if not isinstance(record,dict) or record.get('gate')!=GATE or record.get('passed') is not True:
  raise RuntimeError('actual independently reviewed FD1 numerical qualification required; original G2 remains not admitted')
 expected=expected_scope();actual=record.get('qualification_scope')
 if (not isinstance(actual,dict) or set(actual)!=set(expected) or
     any(type(actual[k]) is not type(v) or actual[k]!=v for k,v in expected.items())):
  raise ValueError('explicit exact FD1 scope/failure metadata required')
 return {'gate':GATE,'passed':True,'qualification_scope':dict(actual)}
