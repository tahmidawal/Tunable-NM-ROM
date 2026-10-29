"""Generate the complete retained domain diagnostic; FD1 is not original-G2 success."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
base=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'fields':a.results/'comparison_original_domain/comparison.json','scalars':a.results/'raw_domain_scalars.json','scope':base/'reviews/FD1-SCOPE-CONTRACT.json','design':base/'DESIGN.md'}
f=json.loads(paths['fields'].read_text());s=json.loads(paths['scalars'].read_text());scope=json.loads(paths['scope'].read_text())
assert f['full_rule_preserved'] and not f['physical_admission'] and not s['physical_admission']
assert scope['original_G2_status']=='not_admitted_domain_sensitivity_failed'
scalars={(r['case'],r['reference']):r for r in s['comparisons']};rows=[]
# Unchanged original DESIGN section5 domain/outlet bars; FD1 changes claim/admission scope only.
bars={'velocity_relative':.002,'drag_relative':.002,'lift_absolute':.0002}
for r in f['comparisons']:
    q=scalars[(r['case'],r['reference'])];force=max(q['physical_force_differences'],key=lambda x:x['diagnostic_order_offset'])
    v={'velocity_relative':r['velocity_relative']['value'],'drag_relative':force['drag_relative_difference'],'lift_absolute':force['transverse_absolute_difference']}
    assert all(math.isfinite(x) and x>=0 for x in v.values()) and r['point_count']>0
    assert not r['pressure_shift_fitted'] and not r['unshifted_pressure_field_eligible_for_same_domain_gate']
    rows.append({'case':r['case'],'reference':r['reference'],**v,'passes_original_sensitivity_bars':all(v[k]<=bars[k] for k in bars),'pressure_drop_absolute_difference':q['pressure_drop_absolute_difference'],'point_count':r['point_count'],'force_rule_offset':force['diagnostic_order_offset']})
assert len(rows)==4 and len(scalars)==4
result={'status':'retained_domain_diagnostic_not_reference_admission','scope_amendment':scope['amendment_id'],'original_G2_status':scope['original_G2_status'],'domain_independence_admitted':False,'original_sensitivity_bars':bars,'rows':rows,'input_hashes':{k:sha(v) for k,v in paths.items()},'generator_sha256':sha(Path(__file__)),'interpretation':'Original inlet sensitivity remains failed. Finite-channel numerical qualification is separate under FD1. Unshifted cross-domain pressure levels are diagnostic, not an admission test.','meshing_disclosure':s['meshing_disclosure']}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
