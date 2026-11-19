"""Publish verified cluster-rendered artifacts without loading large field arrays."""
from pathlib import Path
import json,hashlib,shutil
H=Path(__file__).resolve().parents[1];root=H/'results/b7render/output'
r=json.loads((root/'result.json').read_text());assert r['status']=='complete' and r['full_resolution_png_count']==72
retention=json.loads((H/'manifests/retention-b7render.json').read_text());assert retention['local_and_remote_verified']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
pub=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');records=[]
for row in json.loads((root/'render-manifest.json').read_text()):
    rel=Path(row['file']);assert not rel.is_absolute() and '..' not in rel.parts
    source=root/rel;assert sha(source)==row['sha256'];target=pub/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target);assert sha(target)==row['sha256']
    if 'full-resolution' not in rel.parts:
        lane=H/'reports'/rel;lane.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,lane)
    records.append(dict(source=str(source),published=str(target),sha256=row['sha256']))
exports=json.loads((root/'full-resolution-exports.json').read_text())
for row in exports:row.update(source=str(root/row['file']),published=str(pub/row['file']))
(H/'manifests/B7-parent-publication.json').write_text(json.dumps(records,indent=2)+'\n')
(H/'manifests/B7-parent-full-resolution-exports.json').write_text(json.dumps(exports,indent=2)+'\n')
print('Published',len(records),'verified report/image files without loading field arrays')
