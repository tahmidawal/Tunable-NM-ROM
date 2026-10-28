"""Freeze reviewable source/config byte hashes; no cluster action."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
files=['oriented_normals.py','scalars.py','preflight.py','benchmark.json','fom.py','reference_export.py','shared_lift.py','dependencies.py','smoke_summary.py','manufactured_witness.json','DEPENDENCIES.json','REPAIR.json','PREFLIGHT-REPAIR.json','SCALAR-REPAIR.json','campaign-A2.json','g2/campaign-A6.json','cluster/smoke.sh']
files+=sorted(str(p.relative_to(root)) for p in (root/'configs').glob('*.json'))
records=[{'path':name,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),'bytes':(root/name).stat().st_size} for name in files]
(root/'SOURCE-SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in records))
(root/'INPUT-MANIFEST.json').write_text(json.dumps({'status':'review_candidate_no_execution_release','files':records,
 'source_parent_when_manifest_generated':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'execution_source_commit_binding':'SOURCE_COMMIT.txt is created from exact source commit at staging and separately recorded in staging manifest',
 'command':'sbatch --gres=gpu:h200:1 --chdir=<fresh unique namespace attempt> cluster/smoke.sh',
 'allocation':{'partition':'gpu','gpus':1,'gpu_type_submit_override':'h200','job_name':'ctol_nsobs2d_g1','cpus':16,'mem_GiB':64,'wall_seconds':1200},
 'case_order':['channel','manufactured','dfg'],'no_training':True,'no_reference_admission':True},indent=2)+'\n')
