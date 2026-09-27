"""Checksum collection into bounded-memory uncompressed archive parts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import cluster as transport


def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class PartWriter(io.RawIOBase):
    def __init__(self,record,limit=90*1024**2):
        self.record=record;self.limit=limit;self.part=None;self.size=0;self.total=0;self.whole=hashlib.sha256();self.parts=[];self.current_hash=None
    def writable(self):return True
    def write(self,value):
        original=len(value);view=memoryview(value)
        while len(view):
            if self.part is None:
                self.path=self.record/f'verified-cluster.tar.part-{len(self.parts):03d}'
                self.part=self.path.open('xb');self.size=0;self.current_hash=hashlib.sha256()
            count=min(len(view),self.limit-self.size);chunk=view[:count]
            self.part.write(chunk);self.whole.update(chunk);self.current_hash.update(chunk);self.size+=count;self.total+=count;view=view[count:]
            if self.size==self.limit:self.finish_part()
        return original
    def finish_part(self):
        if self.part is not None:
            self.part.close();self.parts.append((self.path.name,self.current_hash.hexdigest()));self.part=None
    def finish(self):
        self.finish_part()
        for name,expected in self.parts:assert file_sha(self.record/name)==expected
        manifest=dict(original_name='verified-cluster.tar',compression='none',original_sha256=self.whole.hexdigest(),original_bytes=self.total,ordered_parts=[n for n,_ in self.parts],restore='Use restore_archive.py; concatenated parts form an uncompressed tar.')
        (self.record/'ARCHIVE.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (self.record/'ARCHIVE.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in self.parts))
        return manifest


def collect(label):
    assert re.fullmatch(r'iterative[a-z0-9_]{0,15}',label)
    record=transport.CELL/'runs'/label;cfg=json.loads((record/'submission.json').read_text())
    jid=cfg['job_id'];remote=transport.NAMESPACE+'/'+label
    assert cfg['remote']==remote and re.fullmatch(r'[0-9]+',jid)
    queue=transport.ssh('squeue -h -u tawal01 -o "%i %T"')
    assert not any(line.split()[0]==jid for line in queue.splitlines()),'still queued'
    accounting=transport.ssh(f'sacct -j {jid} --format=JobID,JobName,State%30,ExitCode,Elapsed,NodeList,AllocTRES -P')
    (record/'accounting.txt').write_text(accounting)
    assert any(line.startswith(jid+'|') and 'COMPLETED|0:0|' in line for line in accounting.splitlines()),accounting
    quoted=shlex.quote(remote)
    transport.ssh(f'cd {quoted} && test -f EXIT_CODE && find . -type f ! -name PULL.sha256 -print0 | sort -z | xargs -0 sha256sum > PULL.sha256')
    local=record/'cluster';local.mkdir(exist_ok=False)
    subprocess.run(['scp','-q','-r','tufts-login:'+remote+'/.',str(local)+'/'],check=True)
    for name in ('PULL.sha256','MANIFEST.sha256','RESULTS.sha256'):
        if (local/name).read_text().strip():subprocess.run(['sha256sum','-c',name,'--quiet'],cwd=local,check=True)
    assert (local/'EXIT_CODE').read_text().strip()=='0'
    for path,expected in cfg['source_hashes'].items():assert file_sha(local/path)==expected
    writer=PartWriter(record)
    with tarfile.open(fileobj=writer,mode='w|') as tar:tar.add(local,arcname='cluster')
    archive=writer.finish()
    transport.ssh(f'test -f {quoted}/PULL.sha256 && rm -rf -- {quoted} && test ! -e {quoted}')
    transport.write_json(record/'cleanup.json',dict(remote=remote,job_id=jid,all_three_manifests_verified=True,source_hashes_verified=True,archive_sha256=archive['original_sha256'],remote_deleted_and_absence_checked=True,queue_after=transport.ssh('squeue -u tawal01 -o "%.18i %.35j %.8T"')))
    print(json.dumps(archive,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('label');collect(ap.parse_args().label)
