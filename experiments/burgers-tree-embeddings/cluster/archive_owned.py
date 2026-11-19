"""Archive verified raw outputs, remove only owned reproducible staging/cache copies."""
from pathlib import Path
import argparse,json,subprocess as sp,shlex,hashlib,datetime
H=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);args=ap.parse_args()
assert args.label.replace('-','').isalnum()
proof_path=H/f'manifests/archive-closeout-{args.label}.json'
assert not proof_path.exists(),'Retain prior archive evidence; choose a new explicit label'
owner=json.loads((H/'OWNER.json').read_text());ns=owner['namespace']
rows=json.loads((H/'cluster/ledger.json').read_text())
def ssh(s,input=None):return sp.check_output(['ssh','tufts-login',s],input=input,text=True,timeout=180)
def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
q=ssh('squeue -u tawal01 -h -o "%i %j %T %Z"')
assert ns not in q,'Finish all owned jobs before moving registered input caches'
assert all(r.get('state','').split()[0] in {'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'} for r in rows)
proof=[]
for row in rows:
    # Prior archives and their manifests are immutable history. In particular,
    # a later batch must not rewrite B0 retention or its archive proof.
    if row.get('archived_remote') and row.get('archive_label')!=args.label:continue
    name=row['name'];assert name.replace('-','').isalnum()
    old=ns+'/'+name;dest=ns+'/archive/'+name
    manifest=H/'manifests'/f'retention-{name}.json';m=json.loads(manifest.read_text())
    assert m['local_and_remote_verified'] and m['job']==row['id']
    loc=Path(m['local']);assert str(loc).startswith(str(H/'results')+'/')
    # The local raw copy and the designated archive must both pass before cleanup.
    checksum_file=m.get('checksum_file','OUTPUTS.sha256')
    sp.run(['sha256sum','-c',checksum_file,'--quiet'],cwd=loc,check=True)
    # A previous interruption may have completed the move before recording it.
    # Never overwrite a destination or infer validity merely from its existence.
    locations=ssh(f'test ! -d {shlex.quote(old)} || echo source; test ! -d {shlex.quote(dest)} || echo archive').split()
    assert locations in (['source'],['archive']),(name,locations)
    resumed=locations==['archive'] and not row.get('archived_remote')
    if locations==['source']:
        ssh(f'cd {shlex.quote(old)} && sha256sum -c - --quiet',input=(loc/checksum_file).read_text())
        ssh(f'mkdir -p {shlex.quote(ns+"/archive")} && test ! -e {shlex.quote(dest)} && mv {shlex.quote(old)} {shlex.quote(dest)}')
    # Verify against the locally retained checksum list, not a new remote list.
    ssh(f'cd {shlex.quote(dest)} && sha256sum -c - --quiet',input=(loc/checksum_file).read_text())
    # Only code already in the committed source snapshot and disposable caches.
    remove=[dest+'/experiments',dest+'/cache',dest+'/tmp']
    assert all(x.startswith(ns+'/archive/'+name+'/') for x in remove)
    ssh('rm -rf -- '+' '.join(shlex.quote(x) for x in remove))
    m.update(archived_remote=dest,original_job_directory_removed=True,raw_outputs_preserved=True,
             archive_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if resumed:m['recovered_after_interrupted_move']=True
    write_json(manifest,m)
    row['archived_remote']=dest;row['archive_label']=args.label;proof.append(dict(job=row['id'],name=name,local=str(loc),backup=dest,
                                               files=len(m['files']),raw_bytes=sum(x['bytes'] for x in m['files'])))
    write_json(H/'cluster/ledger.json',rows)
    print(name,'archived and verified',flush=True)
write_json(proof_path,proof)
