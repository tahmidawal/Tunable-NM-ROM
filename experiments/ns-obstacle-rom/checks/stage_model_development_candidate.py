"""Reversible source/metadata staging and actual warm-input hashing; no execution."""
from pathlib import Path
import subprocess,json,hashlib,shutil
B=Path(__file__).resolve().parents[1];R=B.parents[1];P=B/'staging/model-development01';root=R.parent/'2026-10-27-ns-obstacle/experiments/ns-obstacle/reviews';job='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/model-development01';py='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
consumer=P/'consumer';shutil.copytree(B/'staging/bank-fit01/consumer',consumer)
p=root/'fine-preparation01/DEVELOPMENT-CONSUMER-RELEASE.json';assert sha(p)=='081ad11b1e5d6b10baea8f9797891b0859d59e0d951ebbe468d77c2767740793';shutil.copyfile(p,consumer/p.name)
reviews=P/'reviews';reviews.mkdir();
for name in ['MODEL-REVIEW.json','INDEPENDENT-ORDERING-RESULT.json']:shutil.copyfile(root/'bank-fit01'/name,reviews/name)
assert sha(reviews/'MODEL-REVIEW.json')=='5010336542cdd8367105d5052cc71e54f9b024ff94ee1fe7522192733604bc0c'
inv=json.loads((P/'RUNTIME-INVENTORY.json').read_text());files=[]
for row in inv['files']:
 p=B/row['source'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];files.append((p,row['destination']))
files.append((P/'RUNTIME-INVENTORY.json','RUNTIME-INVENTORY.json'))
for base,dest in [(consumer,'inputs/consumer'),(reviews,'inputs/reviews')]:
 for p in sorted(base.rglob('*')):
  if p.is_file():files.append((p,dest+'/'+str(p.relative_to(base))))
code=f'''from pathlib import Path
import subprocess,json
r=Path({job!r});assert not r.exists();base=r.parent.parent;names=['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027'];usage={{n:int(subprocess.check_output(['du','-sb',str(base/n)]).split()[0]) for n in names}};projected=sum(usage.values())+12*2**30+256*2**20;assert projected<50*2**30;r.mkdir();print(json.dumps({{'namespace_bytes':usage,'prospective12GiB_envelope_plus_misc_bytes':projected}}))
'''
guard=json.loads(subprocess.check_output(['ssh','tufts-login',py,'-'],input=code.encode()));dirs=sorted({str(Path(n).parent) for _,n in files if str(Path(n).parent)!='.'});subprocess.run(['ssh','tufts-login','mkdir','-p',*[job+'/'+x for x in dirs]],check=True)
for p,n in files:subprocess.run(['scp','-q',str(p),'tufts-login:'+job+'/'+n],check=True)
code=f'''from pathlib import Path
import os,sys,json,hashlib,subprocess
r=Path({job!r});os.chdir(r);sys.path.insert(0,str(r));import stage2d
base=r.parent;inputs={{'prepared':str(base/'bank-fit01/inputs/prepared'),'preparation_release':str(r/'inputs/consumer/DEVELOPMENT-CONSUMER-RELEASE.json'),'bank':str(base/'bank-fit01/results/fit/bank.npz'),'training_evidence':str(base/'bank-fit01/results/fit/bank.npz.training'),'probes':str(base/'fine-preparation01/results/preparation/probes.npz'),'development':str(base/'development-materialize02/results/population/development.json')}}
c=stage2d.build_candidate('development',inputs,'STAGE-CANDIDATE.json');assert 'jax' not in sys.modules and 'ngsolve' not in sys.modules
from pilot2d import read_preparation_metadata
meta,inner=read_preparation_metadata(inputs['prepared'],inputs['preparation_release'])
for key,field in [('bank','bank_sha256'),('probes','probes_sha256'),('development','development_manifest_sha256')]:assert stage2d.sha_file(inputs[key])==inner[field]
assert json.loads(Path(inputs['development']).read_text())['reference_admission_sha256']==meta['admission_evidence']['reference_admission_sha256']
n=1036497;invocations=(3*2+3*4)*2*(4+1);field_bytes=invocations*n*3*8;rows=1+2+9*(2+2*2+64);fit_bytes=6*rows*sum([320,640,1280,2560])*8;raw=field_bytes+fit_bytes+2**30;each=raw+(raw+99)//100+128*2**20
usage={{name:int(subprocess.check_output(['du','-sb',str(base.parent/name)]).split()[0]) for name in ['ns-obstacle-rom-20261027','ns-obstacle-2d-20261027','ns-obstacle-3d-20261027']}};envelope=12*2**30
assert 2*each<envelope and sum(usage.values())+envelope+256*2**20<50*2**30
resource={{'schema':'ns-obstacle-actual-model-development-resource-v1','execution_released':False,'allocation_seconds':3360,'prior_materialization_reserved_seconds':240,'prepared_input_bytes':sum(x['bytes'] for x in c['inputs']['prepared']['files'].values()),'prepared_input_storage':'existing immutable warm tree; no new5.50GB copy','full_rule_points':n,'both_bank_and_POD_assumed_eligible_for_storage':True,'full_field_invocation_upper_count':invocations,'full_field_raw_bytes_without_dedup_credit':field_bytes,'selected_moment_column_upper_bytes':fit_bytes,'other_evidence_and_cache_margin_bytes':2**30,'per_raw_or_archive_envelope_with1percent_and128MiB':each,'combined_raw_archive_envelope_bytes':envelope,'namespace_bytes':usage,'misc_outstanding_bytes':256*2**20,'maximum_global_bytes':sum(usage.values())+envelope+256*2**20,'all_existing_bank_and_reconstruction_files_and_archives_in_current_total':True,'local_collection_separately_measured_and_released':True,'native_or_jax_imported':False,'training_or_model_evaluation_executed':False}}
Path('RESOURCE-PROPOSAL.json').write_text(json.dumps(resource,indent=2)+'\\n');print(json.dumps({{'candidate_sha256':stage2d.sha_file('STAGE-CANDIDATE.json'),'resource_sha256':stage2d.sha_file('RESOURCE-PROPOSAL.json'),'resource':resource}}))
'''
receipt=json.loads(subprocess.check_output(['ssh','tufts-login',py,'-'],input=code.encode()))
for name in ['STAGE-CANDIDATE.json','RESOURCE-PROPOSAL.json']:subprocess.run(['scp','-q','tufts-login:'+job+'/'+name,str(P/name)],check=True)
(P/'CANDIDATE-STAGING.json').write_text(json.dumps({'guard':guard,'runtime_files':len(inv['files']),'additional_staged_files':len(files)-len(inv['files']),'no_native_or_model_evaluation':True,**receipt},indent=2)+'\n');print(json.dumps(receipt,indent=2))
