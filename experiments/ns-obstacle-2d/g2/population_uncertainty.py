"""Package reviewed per-Re metrics only after actual complete FD1 admission.

This copies numerical constituents; it neither grants admission nor writes the
independent uncertainty review required by the ROM consumer.
"""
from pathlib import Path
import argparse,hashlib,json,sys
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from reference_scope import FD1_GATE,FD1_SCOPE,validate_scope

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
def confined(name):
 path=(B/name).resolve();require(path.is_relative_to(B),'evidence outside reference lane');return path

def assemble(admission_path,candidate_path):
 admission=read(admission_path);candidate=read(candidate_path);plan=read(B/'g2/population01/PLAN.json')
 require(admission.get('schema')=='ns-obstacle-FD1-reference-admission-v1'and admission.get('gate')==FD1_GATE and admission.get('passed')is True,'actual complete FD1 admission required')
 require(validate_scope(admission.get('qualification_scope'))==FD1_SCOPE,'exact fixed-channel scope required')
 require(admission.get('candidate_sha256')==sha(candidate_path)and candidate.get('qualification_gate')==FD1_GATE and candidate.get('qualification_scope')==FD1_SCOPE and not candidate.get('missing_parameter_groups'),'admitted complete candidate binding differs')
 expected=[(g['Re'],g['split'])for g in plan['groups']];rows={(r['Re'],r['split']):r for r in candidate['rows']};adopted={(r['Re'],r['split']):r for r in admission['admitted_references']}
 require(len(rows)==len(candidate['rows'])==15 and len(adopted)==len(admission['admitted_references'])==15 and set(rows)==set(adopted)==set(expected),'all15 exact parameter roles required')
 population={s:[g['Re']for g in plan['groups']if g['split']==s]for s in ('train','development','validation')}
 require(admission.get('population')==population,'admitted population differs')
 identity=None;noise_rows={}
 for key,row in rows.items():
  for field in ('export_manifest_sha256','config_sha256','mesh_sha256','layout_sha256','solution_sha256'):
   require(adopted[key][field]==row['adopted_reference'][field],'adopted state identity differs')
  descriptor=row['noise_ledger'];path=confined(descriptor['path']);require(sha(path)==descriptor['sha256'],'reviewed noise ledger changed');noise=read(path)
  require((noise['Re'],noise['split'])==key,'per-parameter noise identity differs')
  current={name:noise[name]for name in ('volume_rule_sha256','volume_rule_hash_algorithm','physical_rule_identity')}
  if identity is None:identity=current
  require(current==identity,'uncertainty rules or normalization identities differ')
  noise_rows[key]={'Re':key[0],'split':key[1],'metrics':noise['metrics'],'noise_ledger_sha256':descriptor['sha256'],'noise_ledger_path':descriptor['path'],'geometry_checked_at_P3':noise['geometry_checked_at_P3'],'P4_g5_interaction_measured':noise['P4_g5_interaction_measured']}
 qualification={'gate':FD1_GATE,'passed':True,'qualification_scope':FD1_SCOPE};records={}
 for split in ('development','validation'):
  records[split]={'schema':'ns-obstacle-reference-conditional-uncertainty-v1','split':split,'reference_qualification':qualification,'reference_admission_sha256':sha(admission_path),'qualification_candidate_sha256':sha(candidate_path),**identity,'rows':[noise_rows[(re,split)]for re in population[split]],'conditional_not_rigorous':True,'original_domain_gate':'failed','generator_sha256':sha(Path(__file__)),'independent_uncertainty_review_required':True,'training_released':False}
 return records

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();records=assemble(args.admission,args.candidate);out=args.out.resolve();require(out.is_relative_to(B),'output outside reference lane');out.mkdir(parents=True,exist_ok=True);paths={split:out/(split+'.json')for split in records};require(not any(path.exists()for path in paths.values()),'never overwrite uncertainty artifacts')
 for split,record in records.items():
  with paths[split].open('x')as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(json.dumps({split:sha(path)for split,path in paths.items()}))
if __name__=='__main__':main()
