"""Bind the independently rehashed working inventory to its original tar member."""
from pathlib import Path
import io,tarfile,json,hashlib
R=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-27-ns-obstacle-rom/experiments/ns-obstacle-rom/artifacts/training-materialize01')
class Chunks(io.RawIOBase):
 def __init__(self,paths):self.paths=iter(paths);self.current=None
 def readable(self):return True
 def readinto(self,b):
  while True:
   if self.current is None:
    path=next(self.paths,None)
    if path is None:return 0
    self.current=path.open('rb')
   n=self.current.readinto(b)
   if n:return n
   self.current.close();self.current=None
 def close(self):
  if self.current:self.current.close()
  super().close()
ret=json.loads((R/'RETENTION.json').read_text());found=[]
with io.BufferedReader(Chunks([R/x['path']for x in ret['chunks']]))as stream:
 with tarfile.open(fileobj=stream,mode='r|gz')as archive:
  for member in archive:
   if member.name=='training-materialize01/COLLECTION-SHA256SUMS':
    assert member.isfile() and member.size<2**20
    raw=archive.extractfile(member).read();assert raw==(R/'COLLECTION-SHA256SUMS').read_bytes();found.append(hashlib.sha256(raw).hexdigest())
assert len(found)==1
out=Path(__file__).with_name('ARCHIVE-LEDGER-REVIEW.json');assert not out.exists();out.write_text(json.dumps({'passed':True,'archive_sha256':ret['archive_sha256'],'archived_original_inventory_sha256':found[0],'local_inventory_is_exact_original_archive_member':True,'file_rehash_review_sha256':hashlib.sha256(Path(__file__).with_name('RESULT.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_extraction_performed':True},indent=2)+'\n');print(out.read_text())
