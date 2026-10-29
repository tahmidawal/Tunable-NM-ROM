"""Run only after owner reports completed status and supplies its exact SHA."""
from pathlib import Path
import argparse,subprocess,hashlib,json,numpy as np
p=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--completed-status-sha256',required=True);args=a.parse_args();assert len(args.completed_status_sha256)==64 and all(c in '0123456789abcdef' for c in args.completed_status_sha256)
base=p.parent/'development-plot-data/plots.npz';base_sha=hashlib.sha256(base.read_bytes()).hexdigest();assert base_sha=='2f554ba694f6f2b34e23dc39ca94dd8067c4a8cdbf2ba955dd17ddeff5f611f2'
with np.load(base,allow_pickle=False) as z:arrays={k:z[k] for k in z.files};ids=arrays['indices'];metadata=json.loads(arrays.pop('metadata_json').tobytes())
source='STATUS_SHA='+repr(args.completed_status_sha256)+'\nBASE_SHA='+repr(base_sha)+'\nINDICES='+repr(ids.tolist())+'\nINDICES_SHA='+repr(hashlib.sha256(ids.tobytes()).hexdigest())+'\n'+(p/'export_remote.py').read_text()
for name in ('extension.npz','plots-with-neural.npz','METADATA.json','RECEIPT.json'):assert not (p/name).exists(),name
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=source.encode(),capture_output=True,timeout=90);(p/'EXPORT.stderr').write_bytes(result.stderr)
if result.returncode:raise RuntimeError(result.stderr.decode()[:2000])
assert len(result.stdout)<=1200000;(p/'extension.npz').write_bytes(result.stdout)
with np.load(p/'extension.npz',allow_pickle=False) as z:
 extension=json.loads(z['metadata_json'].tobytes());arrays.update({k:z[k] for k in z.files if k!='metadata_json'})
metadata['actual_neural_diagnostic_extension']=extension;arrays['metadata_json']=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8);np.savez_compressed(p/'plots-with-neural.npz',**arrays);(p/'METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n');receipt={name:{'sha256':hashlib.sha256((p/name).read_bytes()).hexdigest(),'bytes':(p/name).stat().st_size} for name in ('extension.npz','plots-with-neural.npz','METADATA.json')};receipt.update(original_archive_preserved=base_sha,completed_status_sha256=args.completed_status_sha256,new_scientific_computation=False);(p/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
