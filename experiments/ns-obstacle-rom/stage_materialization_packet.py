"""Stage a released packet with exact reviewed warm reads and unchanged cold fallback."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,tempfile
import materialize_batch2d as batch
ROOT=Path(__file__).parent

def owned_inputs(source_inputs,reference_repo,inventory):
 class OwnedScratchInputs(source_inputs.SourceInputs):
  def __enter__(self):
   self.temporary=tempfile.TemporaryDirectory(prefix='source-recovery-',dir=ROOT)
   return self
 return OwnedScratchInputs(reference_repo,inventory['producer_prefix'],inventory['producer_git_commit'],verify_warm_archives=True)

def stage(packet,release_path,reference_repo):
 packet=Path(packet).resolve();plan=json.loads((packet/'BATCH-PLAN.json').read_text());release=json.loads(Path(release_path).read_text());inventory=json.loads((packet/'INPUT-INVENTORY.json').read_text())
 batch.released_cases(plan,release,packet/'BATCH-PLAN.json',packet/'INPUT-INVENTORY.json',packet/'FD1-REFERENCE-ADMISSION.json')
 if release.get('staging_source_sha256')!=batch.sha(__file__) or inventory.get('staging_source_sha256')!=batch.sha(__file__):raise ValueError('exact reviewed stager required')
 for key,name in [('source_inputs_sha256','source_inputs.py'),('archive_recovery_sha256','git_archive_recovery.py')]:
  if batch.sha(ROOT/'reference_inputs'/name)!=inventory[key] or release.get(key)!=inventory[key]:raise ValueError('reviewed exact staging helper release required')
 if release.get('verify_warm_archives') is not True or release.get('conservative_recovery_reservations_unchanged') is not True:raise ValueError('explicit reviewed warm opt-in required')
 other=release.get('local_other_reserved_bytes')
 if type(other)is not int or other<0:raise ValueError('explicit concurrent local reservation required')
 bound=sum(r['bytes'] for key in ('runtime','producer') for r in inventory[key])+plan['resources']['all_unique_SourceInputs_recovery_temporaries_bytes']+64*2**20+other+10*2**30
 if shutil.disk_usage(ROOT).free<bound:raise MemoryError('full conservative staging/cold-fallback/concurrent reserve required')
 out=packet/'payload';out.mkdir(exist_ok=False)
 record={'staging_complete':False,'release_sha256':batch.sha(release_path),'required_free_bytes':bound,'initial_free_bytes':shutil.disk_usage(ROOT).free,'source_inputs_sha256':inventory['source_inputs_sha256'],'archive_recovery_sha256':inventory['archive_recovery_sha256'],'verify_warm_archives':True,'conservative_reservations_unchanged':True}
 try:
  for r in inventory['runtime']:
   source=ROOT/r['source'];dest=out/r['destination']
   if not dest.resolve().is_relative_to(out) or source.stat().st_size!=r['bytes'] or batch.sha(source)!=r['sha256']:raise ValueError('runtime changed/unsafe destination')
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
  # Exact vendored helpers; callers never import arbitrary cached implementations.
  if 'source_inputs' in sys.modules or 'git_archive_recovery' in sys.modules:raise RuntimeError('fresh staging helper namespace required')
  sys.path.insert(0,str(ROOT/'reference_inputs'));import source_inputs
  os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='gc.auto',GIT_CONFIG_VALUE_0='0')
  with owned_inputs(source_inputs,reference_repo,inventory) as sources:
   sources.materialize(inventory['producer'],out);record['verified_warm_inputs']=sources.verified_warm_inputs;record['cold_archives_materialized']=list(sources.archives)
  shutil.copyfile(packet/'INPUT-INVENTORY.json',out/'INPUT-INVENTORY.json');shutil.copyfile(release_path,out/'BATCH-RELEASE.json')
  (out/'SOURCE_COMMIT.txt').write_text(release['source_commit']+'\n')
  (out/'SOURCE-SHA256SUMS').write_text(''.join(batch.sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SOURCE-SHA256SUMS'))
  record['staging_complete']=True
 except BaseException as exc:record['exception']=repr(exc);raise
 finally:
  record['actual_staged_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());record['final_free_bytes']=shutil.disk_usage(ROOT).free
  (packet/'STAGING.json').write_text(json.dumps(record,indent=2)+'\n')
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--reference-repo',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.packet,a.release,a.reference_repo),indent=2))
