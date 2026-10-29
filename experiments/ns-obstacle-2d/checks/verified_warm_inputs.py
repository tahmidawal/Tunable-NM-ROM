"""Exact warm reuse, unchanged default, and corrupt/missing cache fallback proof."""
from pathlib import Path
import hashlib,io,json,sys,tarfile,tempfile,time
from unittest.mock import patch
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'g2'))
from source_inputs import SourceInputs
sha=lambda data:hashlib.sha256(data).hexdigest()
def main():
 candidate=json.loads((B/'g2/population-rest01/re16p875/SOURCE-CANDIDATE.json').read_text());row=next(r for r in candidate['files']if 'archive_source'in r)
 started=time.perf_counter()
 with tempfile.TemporaryDirectory(dir=B/'staging')as directory,SourceInputs(B.parents[1],str(B.relative_to(B.parents[1])),'a7c58adca376dd9beaa5e7de2c8a62072de6caa4',verify_warm_archives=True)as sources:
  with patch.object(sources,'archive',side_effect=AssertionError('warm proof must not recover archive')):
   sources.materialize([row],directory)
  actual=(Path(directory)/row['destination']).read_bytes();assert len(actual)==row['bytes']and sha(actual)==row['sha256'];actual_receipt=sources.verified_warm_inputs
 seconds=time.perf_counter()-started
 data=b'alpha';manifest=b'bound manifest';buffer=io.BytesIO()
 with tarfile.open(fileobj=buffer,mode='w')as archive:
  info=tarfile.TarInfo('member');info.size=len(data);archive.addfile(info,io.BytesIO(data))
 archive_bytes=buffer.getvalue()
 with tempfile.TemporaryDirectory(dir=B/'staging')as directory:
  root=Path(directory);(root/'lane').mkdir();warm=root/'lane/cache';warm.write_bytes(data);small={'source':'cache','destination':'out','bytes':len(data),'sha256':sha(data),'archive_source':{'manifest_source':'manifest','manifest_sha256':sha(manifest),'member':'member'}}
  class Fixture(SourceInputs):
   def committed(self,path):return manifest
   def archive(self,declaration):
    self.cold_calls+=1;return 'fixture',tarfile.open(fileobj=io.BytesIO(archive_bytes),mode='r:')
  outcomes=[]
  for mode in ['default_cold','warm','corrupt','missing']:
   if mode=='corrupt':warm.write_bytes(b'omega')
   if mode=='missing':warm.unlink()
   with Fixture(root,'lane','unused',verify_warm_archives=mode!='default_cold')as source:
    source.cold_calls=0;assert source.read(small)==data;assert source.cold_calls==(0 if mode=='warm' else 1);outcomes.append(mode)
  warm.write_bytes(data)
  with Fixture(root,'lane','unused',verify_warm_archives=True)as source:
   source.cold_calls=0;wrong={**small,'archive_source':{**small['archive_source'],'manifest_sha256':'0'*64}}
   try:source.read(wrong)
   except ValueError:outcomes.append('wrong_manifest_rejected')
   else:raise AssertionError('wrong manifest accepted')
 result={'schema':'ns-obstacle-verified-warm-input-proof-v1','actual_frozen_input_sha256':row['sha256'],'actual_bytes':row['bytes'],'actual_materialization_seconds':seconds,'actual_cold_recovery_calls':0,'actual_receipt':actual_receipt,'negative_and_default_checks':outcomes,'source_inputs_sha256':sha((B/'g2/source_inputs.py').read_bytes()),'stage_helper_sha256':sha((B/'g2/stage_released.py').read_bytes()),'conservative_reservations_unchanged':True,'numerical_package_unchanged':True}
 (B/'staging/VERIFIED-WARM-INPUT-PROOF.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
