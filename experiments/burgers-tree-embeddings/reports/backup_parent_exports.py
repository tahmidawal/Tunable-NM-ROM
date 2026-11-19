"""Verify actual4096 PNG exports and retain a designated owned cluster backup."""
from pathlib import Path
from PIL import Image
import json,hashlib,subprocess,shlex,datetime
H=Path(__file__).resolve().parents[1]
rows=json.loads((H/'manifests/B7-parent-full-resolution-exports.json').read_text());assert len(rows)==72
folder=H/'results/b7-parent-exports';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in rows:
    p=Path(row['source']);assert p.parent==folder and p.suffix=='.png'
    assert Image.open(p).size==(4096,4096)
    assert sha(p)==row['sha256']==sha(Path(row['published']))
(folder/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+Path(r['source']).name+'\n' for r in rows))
ns=json.loads((H/'OWNER.json').read_text())['namespace'];dest=ns+'/publication/B7-parent-full-resolution'
subprocess.run(['ssh','tufts-login','mkdir -p '+shlex.quote(dest)],check=True)
subprocess.run(['rsync','-az','--checksum',str(folder)+'/',f'tufts-login:{dest}/'],check=True)
subprocess.run(['ssh','tufts-login','cd '+shlex.quote(dest)+' && sha256sum -c SHA256SUMS --quiet'],check=True)
result=dict(local_and_remote_verified=True,shape=[4096,4096],count=len(rows),local=str(folder),remote=dest,files=rows,
            checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),classification='Derived visualization exports; raw float64 arrays retained independently')
(H/'manifests/B7-parent-full-resolution-backup.json').write_text(json.dumps(result,indent=2)+'\n');print('Verified72 full-resolution PNGs locally, in published assets, and on the owned cluster backup')
