"""Reclaim approved duplicate final fields and restore them from pinned Git bytes.

This command is intentionally restricted to B007. Models, metadata, offline
assets, individually tracked files, and every archive part are preserved.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
import numpy as np
from scientific_archive import EXP, ROOT, GitReader, digest

COMMIT='3f12a29b0cbcd95a98f5ded9cc065b3d217b9719'
RUN=EXP/'runs/b3d007'
COLLECTED=RUN/'collected'
ARCHIVE=RUN/'scientific-archive'
RECORD=RUN/'FIELD-RECLAMATION.json'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path,value):
    temp=path.with_name(path.name+'.writing')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def manifest():
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':'+str((ARCHIVE/'manifest.json').relative_to(ROOT))])
    assert raw==(ARCHIVE/'manifest.json').read_bytes()
    data=json.loads(raw)
    for part in data['parts']:
        path=ARCHIVE/part['path'];assert path.is_file() and not path.is_symlink() and path.stat().st_size==part['bytes']
    return data,hashlib.sha256(raw).hexdigest()


def tracked():
    raw=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','--',str(COLLECTED.relative_to(ROOT))])
    return {value.decode() for value in raw.split(b'\0') if value}


def field_path(relative):
    parts=PurePosixPath(relative).parts
    return len(parts)==3 and parts[0]=='out' and parts[1] in ('seed0','seed1','physical-reference') and parts[2].endswith('.npz')


def plan():
    assert not RECORD.exists()
    acceptance=json.loads((RUN/'COLLECTED.json').read_text())
    assert acceptance['checksums_passed'] and acceptance['independent_audit_passed'] and acceptance['remote_directory_removed']
    assert acceptance['final_cohort_unopened'] is False
    data,sha=manifest();kept=tracked();selected=[]
    for row in data['files']:
        path=COLLECTED/row['path']
        if not field_path(row['path']) or str(path.relative_to(ROOT)) in kept:continue
        assert path.is_file() and not path.is_symlink()
        with np.load(path,allow_pickle=False) as arrays:
            if 'fields' not in arrays.files:continue
        stat=path.stat();assert stat.st_nlink==1 and stat.st_size==row['bytes'] and digest(path)==row['sha256']
        selected.append(dict(path=row['path'],bytes=row['bytes'],sha256=row['sha256'],
            allocated_bytes=stat.st_blocks*512,inode=stat.st_ino,mtime_ns=stat.st_mtime_ns,removed=False))
    result=dict(state='planned; local field hashes verified; removal awaits fresh actual-Git reconstruction proof',
        created_utc=now(),authorization='Root explicitly authorized bounded removal of duplicate materialized final NPZ fields after archived-byte rechecks; preserve all archive parts/Git blobs, checkpoints, metadata and restoration manifests',
        archive_commit=COMMIT,archive_manifest_sha256=sha,files=selected,
        candidate_bytes=sum(row['bytes'] for row in selected),candidate_allocated_bytes=sum(row['allocated_bytes'] for row in selected),
        preserved='all tracked files, all checkpoints and model/offline assets, all metadata, all non-final/development fields, all archive parts and Git blobs',
        restore_helper='experiments/paper-b3d/materialized_fields.py restore --all-removed --destination <new-directory>',
        scientific_results_changed=False)
    dump(RECORD,result);print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2));print('candidate_files',len(selected))


def reclaim():
    record=json.loads(RECORD.read_text());assert record['state'].startswith('planned')
    data,sha=manifest();assert sha==record['archive_manifest_sha256']
    proof=json.loads((RUN/'GIT-RECHECK-BEFORE-RECLAMATION.json').read_text())
    assert proof['passed'] and proof['commit']==COMMIT and proof['manifest_sha256']==sha
    assert proof['files']==len(data['files']) and proof['archive_bytes']==data['archive_bytes']
    expected={row['path']:row for row in data['files']};kept=tracked()
    for row in record['files']:
        path=COLLECTED/row['path'];assert field_path(row['path']) and str(path.relative_to(ROOT)) not in kept
        assert row['bytes']==expected[row['path']]['bytes'] and row['sha256']==expected[row['path']]['sha256']
        stat=path.stat();assert not path.is_symlink() and stat.st_nlink==1 and stat.st_ino==row['inode'] and stat.st_mtime_ns==row['mtime_ns']
        assert stat.st_size==row['bytes'] and digest(path)==row['sha256']
    record.update(state='removing verified duplicates',git_recheck=proof,removal_started_utc=now());dump(RECORD,record)
    for index,row in enumerate(record['files']):
        (COLLECTED/row['path']).unlink();row['removed']=True
        if (index+1)%100==0:dump(RECORD,record)
    record.update(state='complete; duplicate final fields recoverable from pinned Git archive',completed_utc=now(),
        removed_files=len(record['files']),removed_bytes=sum(row['bytes'] for row in record['files']),
        reclaimed_allocated_bytes=sum(row['allocated_bytes'] for row in record['files']))
    dump(RECORD,record);manifest()
    print(json.dumps({k:v for k,v in record.items() if k not in ('files','git_recheck')},indent=2))


def restore(args):
    data,sha=manifest();record=json.loads(RECORD.read_text());assert sha==record['archive_manifest_sha256']
    available={row['path']:row for row in record['files'] if row['removed']}
    names=sorted(available) if args.all_removed else args.path
    assert names and len(set(names))==len(names) and set(names)<=set(available)
    destination=Path(args.destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    entries={row['path']:row for row in data['files']};sources={};already=[];temporary=[]
    for name in names:
        path=destination/name;path.resolve().relative_to(destination)
        row=entries[name];source=row.get('hardlink_to',name)
        assert 'hardlink_to' not in entries[source] and entries[source]['sha256']==row['sha256']
        if path.exists():
            assert path.is_file() and not path.is_symlink() and digest(path)==row['sha256'];already.append(name);continue
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.restoring');assert not temp.exists()
        sources.setdefault(source,[]).append((name,path,temp))
    reader=GitReader(COMMIT,ARCHIVE,data['parts']);restored=[]
    try:
        with tarfile.open(fileobj=reader,mode='r|') as archive:
            for member in archive:
                if member.name not in sources:continue
                assert member.isfile();targets=sources.pop(member.name);handles=[temp.open('xb') for _,_,temp in targets]
                temporary.extend(targets);value=hashlib.sha256();count=0
                try:
                    stream=archive.extractfile(member)
                    for block in iter(lambda:stream.read(8*1024*1024),b''):
                        value.update(block);count+=len(block)
                        for handle in handles:handle.write(block)
                    for handle in handles:handle.flush();os.fsync(handle.fileno())
                finally:
                    for handle in handles:handle.close()
                for name,path,temp in targets:
                    row=entries[name];assert value.hexdigest()==row['sha256'] and count==row['bytes']
                    restored.append(dict(path=name,sha256=row['sha256'],bytes=count,canonical_archive_entry=member.name))
        while reader.read(8*1024*1024):pass
        assert not sources and reader.total.hexdigest()==data['archive_sha256'] and reader.bytes==data['archive_bytes']
        for name,path,temp in temporary:os.link(temp,path);temp.unlink()
    finally:
        for name,path,temp in temporary:
            if temp.exists():temp.unlink()
    answer=dict(passed=True,completed_utc=now(),archive_commit=COMMIT,manifest_sha256=sha,
        archive_bytes_checked=reader.bytes,destination=str(destination),restored=restored,already_matching=already,
        scope='requested fields individually restored and hashed from actual Git blobs; every archive part and the full concatenated archive checksum also verified')
    dump(destination/'restore-audit.json',answer);print(json.dumps(answer,indent=2))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('plan');sub.add_parser('reclaim');rest=sub.add_parser('restore')
    choice=rest.add_mutually_exclusive_group(required=True);choice.add_argument('--all-removed',action='store_true');choice.add_argument('--path',action='append')
    rest.add_argument('--destination',required=True);args=parser.parse_args()
    if args.mode=='plan':plan()
    elif args.mode=='reclaim':reclaim()
    else:restore(args)


if __name__=='__main__':main()
