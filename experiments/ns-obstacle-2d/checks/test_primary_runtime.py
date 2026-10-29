"""Metadata-only dependency closure checks; no native work or scientific admission."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reference_primary as primary
import reference_export as rex
class RuntimeClosure(unittest.TestCase):
 def test_audit_cold_but_required_cache_cannot_be_omitted(self):
  with tempfile.TemporaryDirectory() as t:
   folder=Path(t).resolve()
   for name in primary.RUNTIME_NAMES:(folder/name).write_text('{}')
   cache=folder/'geometry-cache/volume';cache.mkdir(parents=True)
   (cache/'arrays.npz').write_bytes(b'test-only immutable array bytes')
   record={'arrays_file':'arrays.npz','arrays_sha256':rex.sha_file(cache/'arrays.npz'),'support_files':{}}
   rex.dump(cache/'cache.json',record)
   rex.dump(folder/'PRIMARY-REFERENCE.json',{'immutable_geometry_rules':{'volume':{'manifest_path':'geometry-cache/volume/cache.json','manifest_sha256':rex.sha_file(cache/'cache.json')}}})
   files={str(p.relative_to(folder)):rex.sha_file(p) for p in folder.rglob('*') if p.is_file()};files['jacobian_arrays.npz']='a'*64
   runtime,audit=primary.split_runtime_inventory(files);manifest={'files':files,'runtime_files':runtime,'audit_files':audit,'runtime_validation_policy':'explicit_primary_runtime_v1'}
   self.assertFalse(primary.verify_runtime_inventory(folder,manifest))
   bad=copy.deepcopy(manifest);bad['runtime_files'].pop('geometry-cache/volume/arrays.npz')
   with self.assertRaises(ValueError):primary.verify_runtime_inventory(folder,bad)
   bad=copy.deepcopy(manifest)
   for key in ('files','runtime_files'):bad[key].pop('geometry-cache/volume/arrays.npz')
   with self.assertRaisesRegex(ValueError,'transitive member'):primary.verify_runtime_inventory(folder,bad)
   (cache/'arrays.npz').unlink()
   with self.assertRaises(FileNotFoundError):primary.verify_runtime_inventory(folder,manifest)
if __name__=='__main__':unittest.main()
