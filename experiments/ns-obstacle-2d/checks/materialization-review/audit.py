"""Read-only independent semantic review of actual nine-case materialization."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
B=Path(__file__).resolve().parents[2]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def require(x,message):
 if not x:raise AssertionError(message)
def digest(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps({'dtype':a.dtype.str,'shape':list(a.shape)},sort_keys=True).encode());h.update(memoryview(a).cast('B'));return h.hexdigest()
def desc(a):return {'dtype':a.dtype.str,'shape':list(a.shape),'content_sha256':digest(a)}
def run(root):
 admission_path=B/'g2/population01/complete15/REFERENCE-ADMISSION.json';truth=read(admission_path);candidate=read(B/'g2/population01/complete15/QUALIFICATION-CANDIDATE.json');truth_sha=sha(admission_path)
 require(sha(root/'FD1-REFERENCE-ADMISSION.json')==truth_sha,'actual admission differs')
 q={k:truth[k]for k in ('gate','passed','qualification_scope')};require(q['passed'] is True,'truth not admitted');scope=q['qualification_scope']['scope_contract_sha256']
 inventory={}
 for line in (root/'COLLECTION-SHA256SUMS').read_text().splitlines():
  h,name=line.split('  ',1);p=(root/name).resolve();require(p.is_relative_to(root.resolve()),'inventory escape');require(sha(p)==h,'original member hash: '+name);inventory[name]=h
 require(len(inventory)==1113,'regular inventory coverage')
 retention=read(root/'RETENTION.json');whole=hashlib.sha256()
 for item in retention['chunks']:
  p=root/item['path'];require(sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'archive chunk')
  with p.open('rb')as f:
   while block:=f.read(1024**2):whole.update(block)
 require(whole.hexdigest()=='490ade93edb5c65b3fd7f3000987bec7588fc4e693fd049a5f7e9166e994e30f','whole archive')
 native=read(root/'primary_native/SOURCE.json');require(set(native['files'])=={'fom.py','reference_export.py','shared_lift.py','preflight.py','scalars.py','oriented_normals.py','immutable_rule.py','reference_primary.py','reference_scope.py'},'native source closure')
 for n,h in native['files'].items():require(sha(root/'primary_native'/n)==h,'native helper '+n)
 source=hashlib.sha256(json.dumps({'entry':sha(root/'materialize_reference2d.py'),'native':sha(root/'primary_native/SOURCE.json'),'qualification':sha(root/'qualification.py'),'scope':scope,'adapter':sha(root/'fe_adapter.py')},sort_keys=True).encode()).hexdigest()
 copied=('mesh.vol.bin','boundary_rule.json','solution.sol','result.json','fe_layout.json','reload_checks.json','mesh_identity.json','normal_orientation.json','reload_normal_orientation.json')
 rows=[];expected_re=[15+1.25*j for j in range(9)];plan=read(root/'BATCH-PLAN.json');require([x['Re']for x in plan['cases']]==expected_re,'training order')
 for item in plan['cases']:
  re=item['Re'];case=root/'results'/item['id'];original=root/item['original_directory'];record=read(case/'MATERIALIZATION.json');execution=read(case/'MATERIALIZATION-EXECUTION.json');release_path=root/'results'/(item['id']+'-RELEASE.json');release=read(release_path);adopted=next(x for x in truth['admitted_references']if x['Re']==re);reference=B/next(x for x in candidate['rows']if x['Re']==re)['adopted_reference']['directory']
  require(adopted['split']=='train' and record['Re']==re and release['Re']==re,'Re/split')
  require(record['reference_qualification']==q and record['reference_admission_sha256']==truth_sha and record['source_sha256']==source,'record qualification/source')
  require(release['source_sha256']==source and release['reference_admission_sha256']==truth_sha and execution['release_sha256']==sha(release_path),'actual release')
  require(record['status']=='exact_content_replay_passed_pending_independent_admission' and execution['status']=='exact_reconstruction_completed_pending_independent_review','actual completion')
  require(record['physical_admission']is False and record['historical_ZIP_identity_claimed']is False and record['original_files_mutated']is False,'derivation scope')
  for key,name in [('export_manifest_sha256','EXPORT-MANIFEST.json'),('config_sha256','config.json'),('mesh_sha256','mesh.vol.bin'),('solution_sha256','solution.sol')]:require(sha(original/name)==sha(reference/name)==adopted[key],'adopted '+name)
  require(read(original/'fe_layout.json')['sha256']==adopted['layout_sha256'],'internal layout')
  require(record['original_export_sha256']==adopted['export_manifest_sha256'] and record['original_config_sha256']==adopted['config_sha256'],'original record identity')
  for name in copied:require(sha(case/name)==sha(original/name)==sha(reference/name),'unchanged certificate '+name)
  config=read(original/'config.json');want=dict(config,release='G2_fixed_finite_channel_training_export',split='training',scope_contract_sha256=scope);require(read(case/'config.json')==want and sha(case/'config.json')==record['derived_config_sha256'],'role-only config')
  if re!=20:
   primary=read(reference/'PRIMARY-REFERENCE.json');require(sha(original/'PRIMARY-REFERENCE.json')==sha(reference/'PRIMARY-REFERENCE.json')==record['primary_manifest_sha256'],'primary record identity');require(sha(original/primary['primary_file'])==primary['primary_sha256']==record['primary_coefficients_sha256'],'primary coefficients');expected=primary['derived_full_rule_arrays'];require(record['immutable_geometry_rules']==primary['immutable_geometry_rules'],'immutable rules')
   replay=record['reconstruction'];require(all(replay[k]is True for k in ['no_PDE_solve','fresh_native_state_from_primary','portable_closure_only','runtime_integrity_verified']),'fresh portable native replay')
   for rule in replay['rules'].values():require(rule['mapping']['all_points_preserved'] and rule['mapping']['point_adjustments']==rule['mapping']['discarded_points']==0 and rule['evaluation']['all_points_evaluated'],'complete mapping')
   require(replay['rules']['volume']['mapping']['point_count']==1036497,'volume count')
  else:
   require(record['source_storage_mode']=='legacy_expanded_Re20' and record['original_full_rule_archive_sha256']==sha(reference/'full_rule.npz')==sha(original/'full_rule.npz'),'legacy original container')
   with np.load(reference/'full_rule.npz',allow_pickle=False)as z:expected={k:desc(z[k])for k in z.files}
   require(record['reconstruction']['no_PDE_solve']is True and record['reconstruction']['new_native_replay_performed']is False,'legacy replay labeling')
  require(record['original_derived_array_descriptors']==expected,'original semantic descriptors')
  with np.load(case/'full_rule.npz',allow_pickle=False)as z:
   require(set(z.files)==set(expected),'exact full-rule member set');actual={}
   for name in z.files:
    a=z[name];require(np.isfinite(a).all(),'finite field '+name);actual[name]=desc(a)
   require(z['points'].shape==(1036497,2) and z['velocity_gradient'].shape==(1036497,2,2),'field dimensions')
  require(actual==expected==record['actual_derived_array_descriptors'],'all actual field content bits')
  require(sha(case/'full_rule.npz')==record['new_full_rule_archive_sha256'],'new container identity')
  export=read(case/'EXPORT-MANIFEST.json');require(export['materialization_sha256']==sha(case/'MATERIALIZATION.json') and export['original_export_sha256']==adopted['export_manifest_sha256'],'new export lineage')
  for name,h in export['files'].items():require(sha(case/name)==h,'derived export '+name)
  rows.append({'schema':'ns-obstacle-fd1-materialization-review-v1','passed':True,'Re':re,'split':'training','materialization_sha256':sha(case/'MATERIALIZATION.json'),'export_manifest_sha256':sha(case/'EXPORT-MANIFEST.json'),'reference_admission_sha256':truth_sha,'reference_qualification':q,'source_sha256':source,'semantic_arrays_verified':len(actual),'source_storage_mode':record['source_storage_mode'],'original_adopted_identities_verified':True,'role_only_config_verified':True,'unchanged_certificates_verified':list(copied),'new_native_or_PDE_execution_by_reviewer':False})
 require(len(rows)==9,'nine cases');require((root/'slurm-6632930.err').stat().st_size==0,'scheduler stderr');log=(root/'slurm-6632930.out').read_text();require('jax_backend=gpu'in log,'GPU log');return {'schema':'ns-obstacle-nine-materialization-independent-review-v1','passed':True,'artifact_directory':str(root),'retention_sha256':sha(root/'RETENTION.json'),'archive_sha256':whole.hexdigest(),'original_regular_files_verified':len(inventory),'reviewer_source_sha256':sha(Path(__file__)),'reference_admission_sha256':truth_sha,'cases':rows,'new_native_or_PDE_execution_by_reviewer':False,'model_training_or_validation_performed':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=run(a.root.resolve());require(a.out.resolve().is_relative_to(B),'owned review output');require(not a.out.exists(),'no overwrite');a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'cases':len(result['cases']),'arrays':sum(x['semantic_arrays_verified']for x in result['cases']),'sha256':sha(a.out)}))
