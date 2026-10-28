"""Freeze saved-state readout proof without authorizing execution."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];G=B/'g2/export-batch01';G.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
folder='artifacts/g2-p4trend/results/dfg_h3';manifest=B/folder/'EXPORT-MANIFEST.json';remote='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/g2-export-batch01'
dump(G/'plan.json',{'case':{'directory':remote+'/inputs/p4_g4_h3','export_manifest_sha256':sha(manifest)}})
dump(G/'PROVISION.json',{'fresh_attempt':'g2-export-batch01','purpose':'same-state scalar/batched full-rule readout equivalence and diagnostic cost repetitions','no_shared_environment_mutation':True,'automatic_retry':False})
script=(B/'g2/domain-upstream/run.sh').read_text().replace('domain-upstream','export-batch01').replace('#SBATCH --mem=128G','#SBATCH --mem=64G').replace('P4_domain_upstream','saved_state_batch_readout')
(G/'run.sh').write_text(script)
(G/'summary.py').write_text((B/'g2/domain-upstream/summary.py').read_text())
(G/'run.py').write_text('''from pathlib import Path
import json,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
record={'physical_admission':False,'new_PDE_solves':0,'completed':False};tick=time.monotonic()
try:
    command=[sys.executable,str(ROOT/'g2/export_batch_control.py'),'--plan',str(ROOT/'g2/export-batch01/plan.json'),'--out',str(OUT/'export_batch_control')]
    with (OUT/'control.stdout.log').open('w') as stdout,(OUT/'control.stderr.log').open('w') as stderr:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=900)
    record.update(command=command,returncode=result.returncode,completed=result.returncode==0)
    result.check_returncode()
except Exception as exc:
    record.update(error=repr(exc),traceback=traceback.format_exc());raise
finally:
    record['seconds']=time.monotonic()-tick;(OUT/'control_progress.json').write_text(json.dumps(record,indent=2)+'\\n')
''')
rows=[]
def add(src,dst):
 p=B/src;rows.append({'source':src,'destination':dst,'sha256':sha(p),'bytes':p.stat().st_size})
for name in ['EXPORT-MANIFEST.json',*json.loads(manifest.read_text())['files']]:add(folder+'/'+name,'inputs/p4_g4_h3/'+name)
for name in ['reference_export.py','shared_lift.py','oriented_normals.py','scalars.py','dependencies.py','preflight.py','DEPENDENCIES.json','g2/compare_fields.py','g2/exact_lookup.py','g2/export_batch_control.py',*[f'g2/export-batch01/{x}' for x in ('run.py','run.sh','summary.py','plan.json')]]:add(name,name)
add('g2/export-batch01/PROVISION.json','REPAIR.json')
record={'status':'source_review_candidate_no_execution_release','files':rows,'physical_admission':False,'new_PDE_solves':0,'bounds':{'allocation_seconds':1200,'process_seconds':900,'allocation_GiB':64,'memory_stop_GiB':48,'volume_points_hard_cap':1300000,'cpus':16,'output_planning_GiB':2},'scope':'Three alternating scalar/batched same-invocation repetitions on original full P4 volume and native boundary rules; original coefficients and all returned arrays retained losslessly. Native arrays are proof; timings concern readout only. Optional producer batching remains default-off until independent runtime acceptance.'}
dump(G/'SOURCE-CANDIDATE.json',record);print(len(rows),sha(G/'SOURCE-CANDIDATE.json'))
