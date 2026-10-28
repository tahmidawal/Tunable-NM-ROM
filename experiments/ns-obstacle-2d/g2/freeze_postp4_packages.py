from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=['fom.py','reference_export.py','shared_lift.py','oriented_normals.py','scalars.py','dependencies.py','preflight.py','DEPENDENCIES.json','benchmark.json','manufactured_witness.json','g2/campaign-A6.json']
for group in ('solverquad','domain-upstream','domain-downstream','domain-symmetric'):
 G=B/'g2'/group;rows=[]
 def add(src,dst):
  p=B/src;rows.append({'source':src,'destination':dst,'sha256':sha(p),'bytes':p.stat().st_size})
 files=base+[f'g2/{group}/'+name for name in ('run.py','run.sh','summary.py')]
 if group=='solverquad':
  folder='artifacts/g2-p4trend/results/dfg_h3';m=B/folder/'EXPORT-MANIFEST.json';(G/'CASE-MANIFESTS.json').write_text(json.dumps({'p4_g4_h3':sha(m)},indent=2)+'\n')
  for name in ['EXPORT-MANIFEST.json',*json.loads(m.read_text())['files']]:add(folder+'/'+name,'inputs/p4_g4_h3/'+name)
  files+=['g2/compare_fields.py','g2/exact_lookup.py','g2/solverquad/CASE-MANIFESTS.json','g2/solverquad/solver_100x.json','g2/solverquad/quadrature_plus4.json']
 else:files+=[f'g2/{group}/case.json']
 provision=G/'PROVISION.json'
 provision.write_text(json.dumps({'fresh_attempt':'g2-solverquad01' if group=='solverquad' else 'g2-'+group,'purpose':'explicit dependency installation namespace binding','no_shared_environment_mutation':True,'automatic_retry':False},indent=2)+'\n')
 add(str(provision.relative_to(B)),'REPAIR.json')
 for name in files:add(name,name)
 record={'status':'source_review_candidate_no_execution_release','files':rows,'physical_admission':False,'resource_proposal_sha256':sha(B/'g2/POST-P4-RESOURCE-PROPOSAL.json'),'bounds':{'allocation_seconds':1200,'runner_seconds':1100,'process_seconds':600 if group=='solverquad' else 900,'allocation_GiB':64 if group=='solverquad' else 128,'memory_stop_GiB':48 if group=='solverquad' else 112,'volume_points_hard_cap':2000000 if group=='solverquad' else 3000000,'cpus':16},'scientific_scope':'same immutable P4 mesh with 100x solver tightening and +4 assembly/diagnostic quadrature; all higher-rule points and adopted-P4 denominators' if group=='solverquad' else 'one separate same-resolution domain case; no coarsening or new h level; symmetric is a separately labeled BVP'}
 (G/'SOURCE-CANDIDATE.json').write_text(json.dumps(record,indent=2)+'\n');print(group,len(rows),sha(G/'SOURCE-CANDIDATE.json'))
