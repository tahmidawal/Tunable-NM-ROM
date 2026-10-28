from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];G=B/'g2/domain-comparison01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
source=(B/'g2/domain-upstream/run.sh').read_text().replace('domain-upstream','domain-comparison01').replace('#SBATCH --mem=128G','#SBATCH --mem=64G').replace('P4_domain_upstream','retained_original_domain_comparison');(G/'run.sh').write_text(source);(G/'summary.py').write_text((B/'g2/domain-upstream/summary.py').read_text())
dump(G/'PROVISION.json',{'fresh_attempt':'g2-domain-comparison01','purpose':'retained-state original-domain comparisons with all failed bars retained','no_shared_environment_mutation':True,'automatic_retry':False})
folders={'original':'artifacts/g2-p4trend/results/dfg_h3',**{name:f'artifacts/g2-domain-{name}/results/case' for name in ('upstream','downstream','symmetric')}}
rows=[];manifests={}
def add(src,dst):
 p=B/src;rows.append({'source':src,'destination':dst,'sha256':sha(p),'bytes':p.stat().st_size})
for name,folder in folders.items():
 m=B/folder/'EXPORT-MANIFEST.json';manifests[name]=sha(m)
 for member in ['EXPORT-MANIFEST.json',*json.loads(m.read_text())['files']]:add(folder+'/'+member,'inputs/'+name+'/'+member)
dump(G/'CASE-MANIFESTS.json',manifests)
for path in ['reference_export.py','shared_lift.py','oriented_normals.py','scalars.py','dependencies.py','preflight.py','DEPENDENCIES.json','g2/compare_fields.py','g2/exact_lookup.py',*[f'g2/domain-comparison01/{x}' for x in ('run.py','run.sh','summary.py','CASE-MANIFESTS.json')]]:add(path,path)
add('g2/domain-comparison01/PROVISION.json','REPAIR.json')
dump(G/'SOURCE-CANDIDATE.json',{'status':'source_review_candidate_no_execution_release','files':rows,'physical_admission':False,'new_PDE_solves':0,'bounds':{'allocation_seconds':1200,'runner_seconds':1100,'process_seconds':600,'allocation_GiB':64,'volume_points_hard_cap':1300000,'cpus':16},'scope':'All original P4 h3 physical points/weights across original/upstream/downstream/symmetric states. No subset, shifted pressure, new solve, threshold waiver or suppression of failed physical bars. Distinct pairwise and original-drag denominators retained; exact common-pole pressure drops.'})
print(len(rows),sha(G/'SOURCE-CANDIDATE.json'))
