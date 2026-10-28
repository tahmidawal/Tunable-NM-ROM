from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['reference_export.py','scalars.py','shared_lift.py','oriented_normals.py','preflight.py','dependencies.py','DEPENDENCIES.json','g2/compare_fields.py','g2/exact_lookup.py','g2/comparison01/run.py','g2/comparison01/summary.py','g2/comparison01/compare.sh','g2/comparison01/INPUTS.json']
rows=[{'source':n,'destination':n,'sha256':sha(B/n),'bytes':(B/n).stat().st_size} for n in names]
for source,dest in [('g2/comparison01/PROVISION.json','REPAIR.json'),('artifacts/g2-ladder02/results/comparison_dfg/dfg_h0-validity.json','original-failure.json')]:
 rows.append({'source':source,'destination':dest,'sha256':sha(B/source),'bytes':(B/source).stat().st_size})
inputs=json.loads((D/'INPUTS.json').read_text())
for name,row in inputs['cases'].items():
 folder=B/row['original_relative_directory'];m=json.loads((folder/'EXPORT-MANIFEST.json').read_text())
 for filename in list(m['files'])+['EXPORT-MANIFEST.json']:
  p=folder/filename;digest=sha(p)
  if filename in m['files']:assert digest==m['files'][filename]
  rows.append({'source':str(p.relative_to(B)),'destination':row['directory']+'/'+filename,'sha256':digest,'bytes':p.stat().st_size})
(D/'SOURCE-CANDIDATE.json').write_text(json.dumps({'status':'review_candidate_not_released','files':rows,'new_PDE_solves':0,'physical_admission':False,'resource':{'gpus':1,'cpus':16,'memory_GiB':64,'wall_seconds':1200,'comparison_subprocess_seconds':300,'complete_rule_points':266756,'input_bytes':sum(r['bytes'] for r in rows),'extra_locator_adjacency':'two vertex-neighbor rings only; bounded inverse iteration12 per candidate; no full-mesh exhaustive fallback','memory_basis':'previous complete6case+failedcomparison batch MaxRSS1335680KiB; retained-case evaluation stores few fields/certificates, no factorization;64GiB envelope unchanged'}},indent=2)+'\n')
print(len(rows),sum(r['bytes'] for r in rows),sha(D/'SOURCE-CANDIDATE.json'))
