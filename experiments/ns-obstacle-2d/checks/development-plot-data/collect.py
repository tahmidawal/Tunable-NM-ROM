from pathlib import Path
import subprocess,json,hashlib,numpy as np
p=Path(__file__).resolve().parent;B=p.parents[1];R=B.parents[2]/'2026-10-27-ns-obstacle-rom'/'experiments/ns-obstacle-rom'
expected=json.loads((B/'checks/partial-model-development/EXPECTED.json').read_text());manifest=R/'artifacts/fine-preparation01/results/preparation/prepared/volume/manifest.json'
src='EXPECTED='+repr(expected)+'\nVOLUME_MANIFEST_SHA='+repr(hashlib.sha256(manifest.read_bytes()).hexdigest())+'\n'+(p/'export_remote.py').read_text()
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=src.encode(),capture_output=True,timeout=120)
(p/'EXPORT.stderr').write_bytes(result.stderr)
if result.returncode:raise RuntimeError(result.stderr.decode()[:2000])
assert len(result.stdout)<=5000000;(p/'plots.npz').write_bytes(result.stdout)
with np.load(p/'plots.npz',allow_pickle=False) as z:meta=json.loads(z['metadata_json'].tobytes());schema={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files if k!='metadata_json'}
(p/'METADATA.json').write_text(json.dumps(meta,indent=2)+'\n');receipt={'sha256':hashlib.sha256(result.stdout).hexdigest(),'bytes':len(result.stdout),'arrays':schema,'metadata_sha256':hashlib.sha256((p/'METADATA.json').read_bytes()).hexdigest(),'no_new_solve':True};(p/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
