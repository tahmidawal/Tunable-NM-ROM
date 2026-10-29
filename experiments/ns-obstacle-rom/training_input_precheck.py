"""Read-only nine-training runtime inventory; no admission, staging or reconstruction."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
import numpy as np
from fe_adapter import FILES
ROOT=Path(__file__).parent
IDS=('re15','re16p25','re17p5','re18p75','re20','re21p25','re22p5','re23p75','re25')
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(ref):
 assert sha(ref/'g2/source_inputs.py')=='bdde5ce4f17448235fa2e37d8d34c19f399d72b0294b0458127b196799e842cc'
 rows=[];archives={};immutable={};pending=[];versions=None;total=0;outputs=0
 for identifier in IDS:
  frozen=ref/'g2/population01/frozen'/identifier;candidate=frozen/'QUALIFICATION-CANDIDATE.json';raw=json.loads(candidate.read_text());row=raw['rows'][0];adopted=row['adopted_reference'];folder=ref/adopted['directory']
  export_path=folder/'EXPORT-MANIFEST.json';export=json.loads(export_path.read_text())
  assert sha(export_path)==adopted['export_manifest_sha256']
  for key,name in [('config_sha256','config.json'),('mesh_sha256','mesh.vol.bin'),('solution_sha256','solution.sol')]:assert sha(folder/name)==adopted[key]
  assert json.loads((folder/'fe_layout.json').read_text())['sha256']==adopted['layout_sha256']
  assert row['split']=='train'
  if versions is None:versions=export['versions']
  assert versions==export['versions']
  primary=row['Re']!=20;expected=dict(export['runtime_files']) if primary else {n:export['files'][n] for n in FILES if n!='EXPORT-MANIFEST.json'}
  expected['EXPORT-MANIFEST.json']=sha(export_path);files=[]
  for name,digest in expected.items():
   p=folder/name;assert p.is_file() and sha(p)==digest
   files.append({'source':str(p.relative_to(ref)),'name':name,'sha256':digest,'bytes':p.stat().st_size})
  closure_path=frozen/'ADOPTED-RUNTIME-CLOSURE.json';closure=None
  if primary:
   record=json.loads((folder/'PRIMARY-REFERENCE.json').read_text());assert record==export['storage_contract']
   sources=dict(export['producer_source_sha256']);sources.update(record['source_sha256'])
   for name,digest in sources.items():assert sha(ROOT/'primary_native'/name)==digest
   for descriptor in record['immutable_geometry_rules'].values():
    path=folder/descriptor['manifest_path'];assert expected[descriptor['manifest_path']]==descriptor['manifest_sha256'];cache=json.loads(path.read_text())
    for name,digest in {cache['arrays_file']:cache['arrays_sha256'],**cache.get('support_files',{})}.items():assert expected[str((path.parent/name).relative_to(folder))]==digest
   derived=sum(int(np.prod(d['shape']))*np.dtype(d['dtype']).itemsize for d in record['derived_full_rule_arrays'].values())
   if closure_path.exists():
    closure=json.loads(closure_path.read_text());assert closure['export_manifest_sha256']==sha(export_path)
    assert {r['destination']:r['sha256'] for r in closure['files']}==expected
    for item in closure['files']:
     if 'archive_source' in item:
      immutable[item['sha256']]={'source':item['source'],'bytes':item['bytes'],'archive_source':item['archive_source']}
      a=item['archive_source'];p=ref/a['manifest_source'];assert sha(p)==a['manifest_sha256'];archive=json.loads(p.read_text())
      archives[a['manifest_sha256']]={'path':str(p),'archive_sha256':archive['archive_sha256'],'archive_bytes':archive['archive_bytes']}
   else:pending.append(identifier+': immutable runtime closure declaration')
  else:
   with zipfile.ZipFile(folder/'full_rule.npz')as z:derived=sum(i.file_size for i in z.infolist())
   cold=ref/'artifacts/g2-p4trend/GIT-COLD-ARCHIVE.json';a=json.loads(cold.read_text());archives[sha(cold)]={'path':str(cold),'archive_sha256':a['archive_sha256'],'archive_bytes':a['archive_bytes']}
   checksums=ref/'artifacts/g2-p4trend/COLLECTION-SHA256SUMS'
   ledger={line.split('  ',1)[1].removeprefix('./'):line.split('  ',1)[0] for line in checksums.read_text().splitlines()}
   for item in files:
    member='results/dfg_h3/'+item['name'];assert ledger[member]==item['sha256']
    item['archive_source']={'manifest_source':str(cold.relative_to(ref)),'manifest_sha256':sha(cold),'member':'g2-p4trend/'+member}
    item['original_collection_checksums_sha256']=sha(checksums)
  copied=sum((folder/name).stat().st_size for name in FILES if name not in ('full_rule.npz','EXPORT-MANIFEST.json'))
  output=derived+copied;outputs+=output;size=sum(f['bytes']for f in files);total+=size
  rows.append({'Re':row['Re'],'split':'train','candidate_sha256':sha(candidate),'adopted_reference':adopted,'materialization_route':'coefficient_primary' if primary else 'legacy_expanded_Re20',
    'runtime_files':files,'warm_runtime_bytes':size,'derived_array_and_copied_input_projection_bytes':output,
    'primary_source_vendor_matches':True if primary else 'not applicable: legacy lossless-copy route; original producer source preserved',
    'closure':{'path':str(closure_path),'sha256':sha(closure_path)} if closure else None})
 assert [r['Re'] for r in rows]==[15+1.25*j for j in range(9)]
 recover=sum(a['archive_bytes']for a in archives.values());allowance=64*2**20
 return {'schema':'ns-obstacle-nine-training-materialization-input-precheck-v1','source_sha256':sha(__file__),
  'physical_admission_created':False,'full_reference_admission_consumed':False,'executable_packet_frozen':False,'staging_performed':False,'native_reconstruction_performed':False,
  'rows':rows,'recorded_runtime_versions':versions,'all_consumed_warm_bytes_checksum_verified':True,
  'missing_runtime_bytes':[],'pending_closure_declarations':pending,'unique_full_recovery_archives':list(archives.values()),
  'verified_warm_option':{'existing_helper_commit':'0bd9b6cb0','existing_helper_mirror':'72c6bf049',
   'source_inputs_sha256':'bdde5ce4f17448235fa2e37d8d34c19f399d72b0294b0458127b196799e842cc',
   'opt_in_required':True,'existing_working_files_only':True,'conservative_reservations_unchanged':True,
   'checks':'committed cold manifest SHA; confined non-symlink working path; exact length and SHA; verified hit receipt; original cold fallback',
   'unique_archive_backed_primary_runtime_files':list(immutable.values()),'unique_archive_backed_primary_runtime_bytes':sum(r['bytes'] for r in immutable.values()),
   'no_duplicate_nine_case_cache_created':True},
  'resources':{'actual_warm_input_bytes':total,'additional_fresh_staging_copy_bytes':total,
   'all_unique_SourceInputs_recovery_temporaries_bytes':recover,'staging_plus_recovery_before_other_reservations_bytes':total+recover,
   'derived_outputs_no_compression_projection_bytes':outputs+allowance,'projection_overhead_allowance_bytes':allowance,
   'remote_inputs_plus_derived_outputs_before_native_environment_bytes':total+outputs+allowance,
   'minimum_disk_reserve_bytes':10*2**30,'current_local_free_bytes':shutil.disk_usage(ROOT).free,
   'allocation_seconds_reserved':450,'remaining_fine_preparation_seconds_reserved':1350,'total_preparation_seconds':1800,
   'native_environment_and_git_transport_not_in_totals':True},
  'pending_bindings':['genuine full15 FD1 admission and independent review','dataset-local common-lift admission after genuine full truth',
   'final source/mirror commit freeze for runtime closures','exact materialization release for each actual Re and single450s job',
   'measured staging/Git/other-active reservation and pinned native environment packet'],
  'retention_requirement':'Keep every declared source member directly recoverable from pinned Git or explicit full-archive SourceInputs mapping; do not rely on delta fallback.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();record=check(a.reference_root)
 a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('rows','unique_full_recovery_archives')},indent=2))
