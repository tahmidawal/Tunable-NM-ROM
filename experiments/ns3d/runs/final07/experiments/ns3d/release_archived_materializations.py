"""Release approved duplicate raw binaries, preserving verified Git archives."""
import datetime,hashlib,json,subprocess
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def verify_available(commit,archive,manifest):
    blob=subprocess.check_output(['git','show',commit+':'+str(archive/'archive.json')])
    assert json.loads(blob)==manifest
    references=[commit+':'+str(archive/row['path']) for row in manifest['parts']]
    values=subprocess.check_output(['git','cat-file','--batch-check'],input='\n'.join(references)+'\n',text=True).splitlines()
    assert len(values)==len(references)
    for line,row in zip(values,manifest['parts']):
        _,kind,size=line.split();assert kind=='blob' and int(size)==row['bytes']


def main():
    assert Path.cwd().resolve()==ROOT.resolve()
    destination=ROOT/'experiments/ns3d/runs/archived_materialization_release.json'
    assert not destination.exists(),'Do not overwrite an earlier release record'
    record=dict(complete=False,authorization='Parent approved bounded duplicate-materialization relief on 2026-09-20',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=[],logical_bytes_released=0,allocated_bytes_released=0)
    for attempt in ('coverage04','capacity05'):
        run=Path('experiments/ns3d/runs')/attempt;archive=Path('experiments/ns3d/artifacts')/attempt;collected=run/'collected'
        proof=json.loads((run/'git_archive_audit.json').read_text());assert proof['passed']
        commit=proof['archive_commit'];manifest=json.loads((archive/'archive.json').read_text());assert manifest['passed']
        assert json.loads((run/'audit.json').read_text())['passed'] and json.loads((run/'cleanup.json').read_text())['deleted']
        candidates=[]
        for name,digest in manifest['files'].items():
            path=collected/name
            if path.is_file() and path.suffix in ('.npz','.pkl') and path.stat().st_size>=64*1024**2:
                candidates.append((name,digest,path))
        for name,digest,path in candidates:
            verify_available(commit,archive,manifest)
            assert path.resolve().is_relative_to((ROOT/collected).resolve())
            assert sha(path)==digest,'Current materialization differs from the verified archive'
            stat=path.stat();entry=dict(path=str(path),archive_member='./'+name,sha256=digest,bytes=stat.st_size,
                allocated_bytes=stat.st_blocks*512 if stat.st_nlink==1 else 0,archive_commit=commit,archive_directory=str(archive),
                actual_git_byte_proof=str(run/'git_archive_audit.json'),removed=False,
                restore_command=f"cat {archive}/collection.tar.part* | tar -xf - -C {collected} ./{name}")
            record['files'].append(entry);destination.write_text(json.dumps(record,indent=2)+'\n')
            path.unlink();entry['removed']=True;record['logical_bytes_released']+=entry['bytes'];record['allocated_bytes_released']+=entry['allocated_bytes']
            destination.write_text(json.dumps(record,indent=2)+'\n');print('RELEASED',path,entry['bytes'],flush=True)
    record['complete']=True;record['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='files'}))


if __name__=='__main__':main()
