"""Exercise real CLI alias/symlink recovery and verified transport retirement."""
from pathlib import Path
import hashlib,json,subprocess,tempfile
B=Path(__file__).resolve().parents[1];PY='/home/tahmid/Dev/.venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(dir=B/'staging',prefix='delta-edges-')as temp:
 t=Path(temp);source=t/'source';source.mkdir();(source/'a').write_bytes(b'new-identical-content');(source/'b').write_bytes((source/'a').read_bytes());(source/'link').symlink_to('a')
 index=t/'index.json';index.write_text(json.dumps({'archives':{},'entries':{},'recovery_helper_sha256':sha(B/'g2/git_archive_recovery.py')}));archive=t/'delta.tar.gz';written=t/'write.json';recovered=t/'recover.json';target=t/'restored'
 cli=[PY,'-B',str(B/'g2/delta_retention.py')]
 subprocess.run(cli+['write','--root',str(source),'--index',str(index),'--output',str(archive),'--receipt',str(written)],check=True,capture_output=True)
 write=json.loads(written.read_text());assert write['alias_logical_bytes']==len((source/'a').read_bytes())
 args=['recover','--archive',str(archive),'--sha256',sha(archive),'--index',str(index),'--target',str(target),'--repo',str(B.parents[1]),'--receipt',str(recovered)]
 subprocess.run(cli+args,check=True,capture_output=True);assert(target/'a').read_bytes()==(target/'b').read_bytes()==(source/'a').read_bytes();assert(target/'link').readlink()==Path('a')
 assert subprocess.run(cli+args,capture_output=True).returncode!=0
 wrong=t/'wrong.json';wrong.write_text(index.read_text()+' ');bad=args.copy();bad[bad.index('--index')+1]=str(wrong);bad[bad.index('--target')+1]=str(t/'bad');assert subprocess.run(cli+bad,capture_output=True).returncode!=0
 digest=sha(archive);subprocess.run([PY,'-B',str(B/'g2/chunk_verified_delta.py'),'--archive',str(archive),'--recovery',str(recovered),'--out',str(t/'chunks')],check=True,capture_output=True);assert not archive.exists();chunks=json.loads((t/'chunks/CHUNKS.json').read_text());assert hashlib.sha256(b''.join((t/'chunks'/r['path']).read_bytes()for r in chunks['chunks'])).hexdigest()==digest
 record={'schema':'ns-obstacle-delta-edge-proof-v1','new_alias_and_internal_relative_symlink':True,'no_overwrite_rejected':True,'changed_index_rejected':True,'chunk_and_full_hashes_verified_before_transport_retirement':True,'historical_fallback_full_real_inventory_proof':'staging/delta-retention-proof02/ROUNDTRIP.json','delta_source_sha256':sha(B/'g2/delta_retention.py'),'chunk_source_sha256':sha(B/'g2/chunk_verified_delta.py')}
(B/'staging/DELTA-EDGE-PROOF.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
