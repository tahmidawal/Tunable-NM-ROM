"""Split a verified compressed raw archive into Git-trackable bounded chunks."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(__file__).resolve().parents[3]
attempt=sys.argv[1];assert attempt.isalnum()
source=root/'experiments/neural-operator-burgers/runs'/attempt/'archive'
out=root/'experiments/neural-operator-burgers/artifacts'/attempt
out.mkdir(parents=True,exist_ok=False)
records=[];whole=hashlib.sha256()
with (source/'collection.tar.gz').open('rb') as stream:
    index=0
    while block:=stream.read(64*1024*1024):
        name=f'collection.tar.gz.part{index:04d}';(out/name).write_bytes(block)
        records.append(dict(path=name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
        whole.update(block);index+=1
expected=(source/'collection.tar.gz.sha256').read_text().split()[0];assert whole.hexdigest()==expected
(out/'archive.json').write_text(json.dumps(dict(archive='collection.tar.gz',sha256=expected,chunks=records),indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
for name in ['COMMIT.txt','PROVENANCE.json','MANIFEST.sha256','OUTPUTS.sha256','run.sbatch','scheduler.txt']:
    if (source/name).exists():shutil.copy2(source/name,out/name)
for directory in ['calibration','output']:
    if (source/directory/'index.json').exists():shutil.copy2(source/directory/'index.json',out/f'{directory}-index.json')
if (source/'output/worker.json').exists():shutil.copy2(source/'output/worker.json',out/'worker.json')
if (source/'output/diagnosis/index.json').exists():shutil.copy2(source/'output/diagnosis/index.json',out/'diagnosis-index.json')
(out/'README.md').write_text('''# Full raw Burgers archive

The ordered chunks preserve the checksum-verified original compressed job archive,
including full numerical arrays, immutable staged source, logs and manifests.

From this directory, verify and restore into a fresh destination:

```bash
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored
tar -xzf collection.tar.gz -C restored
cd restored
sha256sum -c OUTPUTS.sha256
sha256sum -c MANIFEST.sha256
```

The whole-archive checksum is recorded in `archive.json`. Referenced calibration
inputs for a successor job are retained in the separately archived calibration job.
''')
print(out)
