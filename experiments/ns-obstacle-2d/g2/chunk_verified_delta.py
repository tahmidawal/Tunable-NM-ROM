"""After verified reconstruction, split delta and retire only its verified transport copy."""
from pathlib import Path
import argparse,hashlib,json,shutil
RESERVE=10*1024**3;CHUNK=48*1024**2
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=json.loads(a.recovery.read_text());expected=r['archive_sha256']
 if r.get('all_regular_files_verified')is not True or sha(a.archive)!=expected:raise ValueError('verified recovery and exact transport required')
 if shutil.disk_usage(a.out.parent).free<a.archive.stat().st_size+RESERVE:raise MemoryError('chunk copy reserve')
 a.out.mkdir(exist_ok=False);chunks=[]
 with a.archive.open('rb')as source:
  while data:=source.read(CHUNK):
   if shutil.disk_usage(a.out).free<len(data)+RESERVE:raise MemoryError('chunk write reserve')
   path=a.out/('collection.delta.tar.gz.part-'+str(len(chunks)).zfill(3));path.write_bytes(data);chunks.append({'path':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 total=hashlib.sha256()
 for row in chunks:
  path=a.out/row['path']
  if sha(path)!=row['sha256']:raise ValueError('written chunk differs')
  with path.open('rb')as f:
   while data:=f.read(8*1024**2):total.update(data)
 if total.hexdigest()!=expected:raise ValueError('full chunk archive differs')
 record={'archive_sha256':expected,'archive_bytes':a.archive.stat().st_size,'chunks':chunks,'recovery_receipt_sha256':sha(a.recovery),'all_chunk_and_full_hashes_verified':True,'duplicate_transport_retired_before_Git_object_write':True};(a.out/'CHUNKS.json').write_text(json.dumps(record,indent=2)+'\n');a.archive.unlink();print(json.dumps(record))
if __name__=='__main__':main()
