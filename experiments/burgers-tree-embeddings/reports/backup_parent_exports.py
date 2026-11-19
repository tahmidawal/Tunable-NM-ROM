"""Verify full-resolution exports against the existing owned renderer archive."""
from pathlib import Path
from PIL import Image
import json,hashlib,subprocess,shlex,datetime
H=Path(__file__).resolve().parents[1]
rows=json.loads((H/'manifests/B7-parent-full-resolution-exports.json').read_text());assert len(rows)==72
folder=Path(rows[0]['source']).parent;assert folder.is_relative_to(H/'results/b7render/output')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
for row in rows:
    p=Path(row['source']);assert p.parent==folder and p.suffix=='.png'
    with Image.open(p) as im:assert im.size==(4096,4096)
    assert sha(p)==row['sha256']==sha(Path(row['published']))
m=json.loads((H/'manifests/retention-b7render.json').read_text());assert m['local_and_remote_verified'] and m['raw_outputs_preserved']
dest=m['archived_remote']+'/output/2026-11-19-neural-parent-assets/full-resolution'
checks=''.join(r['sha256']+'  '+Path(r['source']).name+'\n' for r in rows)
subprocess.run(['ssh','tufts-login','cd '+shlex.quote(dest)+' && sha256sum -c - --quiet'],input=checks,text=True,check=True,timeout=120)
result=dict(local_and_remote_verified=True,shape=[4096,4096],count=len(rows),local=str(folder),remote=dest,files=rows,
            checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),classification='Derived visualization exports; raw float64 arrays retained independently')
(H/'manifests/B7-parent-full-resolution-backup.json').write_text(json.dumps(result,indent=2)+'\n');print('Verified72 full-resolution PNGs locally, in published assets, and in the renderer archive')
