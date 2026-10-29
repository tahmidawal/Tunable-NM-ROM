"""Verify/recover archive chunks from reachable Git history. Never deletes evidence."""
import argparse, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path

SCHEMA='ns-obstacle-git-cold-archive-v1'
MIN_RESERVE=10*2**30

def git(repo,*args):
    return subprocess.check_output(['git','-c','gc.auto=0','-C',str(repo),*args])

def safe_path(value):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts or not value or '\n' in value:
        raise ValueError('repository-relative path required')
    return p

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def blob_stream(repo,blob,expected_size,whole,output=None):
    if int(git(repo,'cat-file','-s',blob))!=expected_size:raise ValueError('Git blob size mismatch')
    h=hashlib.sha256();size=0
    with subprocess.Popen(['git','-c','gc.auto=0','-C',str(repo),'cat-file','blob',blob],stdout=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for data in iter(lambda:proc.stdout.read(2**20),b''):
            size+=len(data)
            if size>expected_size:
                proc.kill();raise ValueError('oversized Git blob')
            h.update(data);whole.update(data)
            if output is not None:output.write(data)
        error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError('Git blob read failed: '+error.decode(errors='replace'))
    if size!=expected_size:raise ValueError('truncated Git blob')
    return h.hexdigest()

def verify(repo,record,output=None):
    if record['schema']!=SCHEMA or not record['chunks']:raise ValueError('invalid cold archive manifest')
    whole=hashlib.sha256();total=0;paths=set()
    for item in record['chunks']:
        path=str(safe_path(item['repository_path']))
        if path in paths or item['bytes']<=0:raise ValueError('duplicate/empty chunk')
        paths.add(path)
        # A fresh code-only clone does not contain the unpushed experiment commit.
        for commit in [record['mirror_commit']]:
            if git(repo,'rev-parse',commit+':'+path).decode().strip()!=item['git_blob']:
                raise ValueError('chunk differs from pinned source/mirror history')
        if blob_stream(repo,item['git_blob'],item['bytes'],whole,output)!=item['sha256']:
            raise ValueError('chunk SHA256 mismatch')
        total+=item['bytes']
    if total!=record['archive_bytes'] or whole.hexdigest()!=record['archive_sha256']:
        raise ValueError('complete ordered archive identity mismatch')
    return {'archive_bytes':total,'archive_sha256':whole.hexdigest(),'chunks':len(paths)}

def prepare(a):
    repo=a.repo.resolve();relative=str(safe_path(a.retention_path))
    source=git(repo,'rev-parse',a.source_commit+'^{commit}').decode().strip()
    mirror=git(repo,'rev-parse',a.mirror_commit+'^{commit}').decode().strip()
    if not a.mirror_ref.startswith('refs/heads/codeonly/exp/'):
        raise ValueError('explicit code-only lane mirror ref required')
    advertised=git(repo,'ls-remote','--exit-code','origin',a.mirror_ref).decode().splitlines()
    if len(advertised)!=1 or advertised[0].split()[1]!=a.mirror_ref:raise ValueError('ambiguous remote ref')
    tip=advertised[0].split()[0]
    git(repo,'merge-base','--is-ancestor',source,'HEAD')
    git(repo,'merge-base','--is-ancestor',mirror,tip)
    raw=git(repo,'show',source+':'+relative);retained=json.loads(raw)
    if retained['status']!='complete_all_regular_files_verified':raise ValueError('verified original retention required')
    rows=[]
    for chunk in retained['chunks']:
        path=safe_path(relative).parent/safe_path(chunk['path'])
        blob=git(repo,'rev-parse',source+':'+str(path)).decode().strip()
        local=repo/path
        if local.is_symlink() or not local.is_file() or local.stat().st_size!=chunk['bytes'] or file_hash(local)!=chunk['sha256']:
            raise ValueError('working chunk is not an exact retained duplicate: '+str(path))
        rows.append({'repository_path':str(path),'git_blob':blob,'bytes':chunk['bytes'],'sha256':chunk['sha256']})
    record={'schema':SCHEMA,'source_commit':source,'mirror_commit':mirror,'mirror_ref':a.mirror_ref,
            'verified_advertised_remote_tip':tip,'source_retention_path':relative,'source_retention_sha256':hashlib.sha256(raw).hexdigest(),
            'archive_bytes':retained['archive_bytes'],'archive_sha256':retained['archive_sha256'],'chunks':rows,
            'working_duplicates_verified':True,'unique_bytes_deleted':False,
            'recovery_needs_only_mirror_history':True,'verification_tool_sha256':file_hash(Path(__file__)),
            'policy':'Only working chunk duplicates may later be retired by their lane owner. Preserve both reachable histories; no pruning or history rewrite. Canonical result arrays remain unchanged.'}
    record['git_only_stream_verification']=verify(repo,record)
    with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record['git_only_stream_verification']))

def recover(a):
    record=json.loads(a.manifest.read_text());output=a.output.resolve()
    if output.exists() or not output.parent.is_dir():raise ValueError('fresh output in existing directory required')
    if a.reserve_bytes<MIN_RESERVE:raise ValueError('minimum10GiB reserve cannot be lowered')
    if shutil.disk_usage(output.parent).free<record['archive_bytes']+a.reserve_bytes:
        raise MemoryError('archive recovery would violate disk reserve')
    fd,name=tempfile.mkstemp(prefix=output.name+'.',suffix='.partial',dir=output.parent);temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            result=verify(a.repo.resolve(),record,stream);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,output)  # Atomic no-overwrite publication, even if another writer appeared.
        print(json.dumps(result))
    finally:temporary.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--repo',required=True,type=Path);q.add_argument('--retention-path',required=True)
    q.add_argument('--source-commit',required=True);q.add_argument('--mirror-commit',required=True);q.add_argument('--mirror-ref',required=True);q.add_argument('--output',required=True,type=Path)
    q=sub.add_parser('recover');q.add_argument('--repo',required=True,type=Path);q.add_argument('--manifest',required=True,type=Path);q.add_argument('--output',required=True,type=Path);q.add_argument('--reserve-bytes',type=int,default=MIN_RESERVE)
    a=p.parse_args();prepare(a) if a.command=='prepare' else recover(a)
