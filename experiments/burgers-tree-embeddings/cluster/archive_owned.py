"""Archive verified raw outputs, remove only owned reproducible staging/cache copies."""
from pathlib import Path
import json,subprocess as sp,shlex,hashlib,datetime
H=Path(__file__).resolve().parents[1]
owner=json.loads((H/'OWNER.json').read_text());ns=owner['namespace']
rows=json.loads((H/'cluster/ledger.json').read_text())
def ssh(s):return sp.check_output(['ssh','tufts-login',s],text=True)
q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"')
assert ns not in q,'Finish all owned jobs before moving registered input caches'
assert all(r.get('state')=='COMPLETED' for r in rows)
proof=[]
for row in rows:
    name=row['name'];assert name.replace('-','').isalnum()
    old=ns+'/'+name;dest=ns+'/archive/'+name
    manifest=H/'manifests'/f'retention-{name}.json';m=json.loads(manifest.read_text())
    assert m['local_and_remote_verified'] and m['job']==row['id']
    loc=Path(m['local']);assert str(loc).startswith(str(H/'results')+'/')
    # The local raw copy and the designated archive must both pass before cleanup.
    sp.run(['sha256sum','-c','OUTPUTS.sha256','--quiet'],cwd=loc,check=True)
    if not row.get('archived_remote'):
        ssh(f'cd {shlex.quote(old)} && sha256sum -c OUTPUTS.sha256 --quiet')
        ssh(f'mkdir -p {shlex.quote(ns+"/archive")} && test ! -e {shlex.quote(dest)} && mv {shlex.quote(old)} {shlex.quote(dest)}')
    ssh(f'cd {shlex.quote(dest)} && sha256sum -c OUTPUTS.sha256 --quiet')
    # Only code already in the committed source snapshot and disposable caches.
    remove=[dest+'/experiments',dest+'/cache',dest+'/tmp']
    assert all(x.startswith(ns+'/archive/'+name+'/') for x in remove)
    ssh('rm -rf -- '+' '.join(shlex.quote(x) for x in remove))
    m.update(archived_remote=dest,original_job_directory_removed=True,raw_outputs_preserved=True,
             archive_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    manifest.write_text(json.dumps(m,indent=2)+'\n')
    row['archived_remote']=dest;proof.append(dict(job=row['id'],name=name,local=str(loc),backup=dest,
                                               files=len(m['files']),raw_bytes=sum(x['bytes'] for x in m['files'])))
    (H/'cluster/ledger.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,'archived and verified',flush=True)
(H/'manifests/archive-closeout.json').write_text(json.dumps(proof,indent=2)+'\n')
