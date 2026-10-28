from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['fom.py','reference_export.py','shared_lift.py','scalars.py','oriented_normals.py','preflight.py','dependencies.py','DEPENDENCIES.json','benchmark.json','manufactured_witness.json','g2/campaign-A6.json','g2/DESIGN-A6.md','g2/compare_fields.py','g2/exact_lookup.py','g2/h3/run.py','g2/h3/summary.py','g2/h3/h3.sh','g2/h3/INPUTS.json','g2/h3/dfg_h3.json','g2/h3/manufactured_h3.json']
rows=[{'source':n,'destination':n,'sha256':sha(B/n),'bytes':(B/n).stat().st_size} for n in names]
p=B/'g2/h3/PROVISION.json';rows.append({'source':str(p.relative_to(B)),'destination':'REPAIR.json','sha256':sha(p),'bytes':p.stat().st_size})
inputs=json.loads((D/'INPUTS.json').read_text())
for name,row in inputs['cases'].items():
 folder=B/row['original_relative_directory'];m=json.loads((folder/'EXPORT-MANIFEST.json').read_text())
 for filename in list(m['files'])+['EXPORT-MANIFEST.json']:
  p=folder/filename;digest=sha(p)
  if filename in m['files']:assert digest==m['files'][filename]
  rows.append({'source':str(p.relative_to(B)),'destination':row['directory']+'/'+filename,'sha256':digest,'bytes':p.stat().st_size})
proposal=json.loads((B/'g2/H3-RESOURCE-PROPOSAL.json').read_text())
(D/'SOURCE-CANDIDATE.json').write_text(json.dumps({'status':'review_candidate_no_execution_release','files':rows,'new_PDE_solves':2,'physical_admission':False,'resource_proposal_sha256':sha(B/'g2/H3-RESOURCE-PROPOSAL.json'),'resource':proposal,'comparison_budget':{'per_kind_hard_timeout_seconds':300,'complete_runner_seconds':1100,'allocation_seconds':1200,'measured_predecessor_complete_comparison_job_seconds':136,'predecessor_job_id':6614742,'note':'both full266756point comparisons plus native controls and dependency setup completed in136s; separate comparison process times were not recorded; proposed1.067Mpoint comparisons guarded at300s each','solver_planning_plus_two_comparison_timeouts_seconds':proposal['planning_total_seconds_not_guarantee']+600},'actual_h3_rule':'entire positive mapped rule; h1/h2/h3 differences on identical points; no subset/nudge/shift'},indent=2)+'\n')
print(len(rows),sha(D/'SOURCE-CANDIDATE.json'))
