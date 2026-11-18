"""Copy and hash-check owned completed outputs. Never deletes a remote artifact."""
from pathlib import Path
import argparse,hashlib,json,subprocess as sp,datetime

H=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('names',nargs='+');a=ap.parse_args()
ledger=json.loads((H/'cluster/ledger.json').read_text());records=[]
for name in a.names:
    row=next(r for r in ledger if r['name']==name);remote=row['remote']
    state=sp.check_output(['ssh','tufts-login',f'sacct -j {row["id"]} -X -n --format=State -P'],text=True).strip()
    assert state=='COMPLETED',state
    target=H/'results'/name;target.mkdir(parents=True,exist_ok=True)
    sp.run(['rsync','-az','--exclude=cache/','--exclude=tmp/','--exclude=experiments/',
            f'tufts-login:{remote}/',str(target)+'/'],check=True)
    lines=(target/'OUTPUTS.sha256').read_text().splitlines();verified=[]
    for line in lines:
        expected,path=line.split('  ',1);p=target/path
        assert p.is_file() and sha(p)==expected,(name,path)
        verified.append(dict(file=path,sha256=expected,bytes=p.stat().st_size))
    check=sp.check_output(['ssh','tufts-login',f'cd {remote} && sha256sum -c OUTPUTS.sha256 --quiet'],text=True)
    record=dict(name=name,job=row['id'],source_commit=row['commit'],remote=remote,local=str(target),
                checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=verified,
                local_and_remote_verified=True,remote_removed=False)
    (H/'manifests'/f'retention-{name}.json').write_text(json.dumps(record,indent=2)+'\n')
    records.append(record);print(name,'verified',len(verified),'files',sum(f['bytes'] for f in verified),flush=True)
