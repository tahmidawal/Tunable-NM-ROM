"""Run only after owner reports completed status and supplies its exact SHA."""
from pathlib import Path
import argparse,subprocess,hashlib,json,numpy as np
p=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--completed-status-sha256',required=True);a.add_argument('--audit-sha256',required=True);args=a.parse_args();assert len(args.completed_status_sha256)==64 and all(c in '0123456789abcdef' for c in args.completed_status_sha256)
base=p.parent/'neural-guided-diagnostic-plot-data/plots-with-guided-neural.npz';base_sha=hashlib.sha256(base.read_bytes()).hexdigest();assert base_sha=='eec597a616aa145a753b29492c90e7c95f6efc1364a1e8000f1c6f0ac55b52a5'
with np.load(base,allow_pickle=False) as z:arrays={k:z[k] for k in z.files};ids=arrays['indices'];metadata=json.loads(arrays.pop('metadata_json').tobytes())
audit=p.parent/'neural-refined-diagnostic-review/RESULT.json';audit_sha=hashlib.sha256(audit.read_bytes()).hexdigest();assert audit_sha==args.audit_sha256;actual_audit=json.loads(audit.read_text());assert actual_audit['passed'];binding=json.loads((audit.parent/'BINDING.json').read_text());assert binding['status_sha256']==args.completed_status_sha256 and binding['model_sha256']==actual_audit['model_sha256']
source='MODEL_SHA='+repr(binding['model_sha256'])+'\nLATEST_SHA='+repr(binding['latest_record_sha256'])+'\nAUDIT_SHA='+repr(audit_sha)+'\nSTATUS_SHA='+repr(args.completed_status_sha256)+'\nBASE_SHA='+repr(base_sha)+'\nINDICES='+repr(ids.tolist())+'\nINDICES_SHA='+repr(hashlib.sha256(ids.tobytes()).hexdigest())+'\n'+(p/'export_remote.py').read_text()
for name in ('extension.npz','plots-with-refined-neural.npz','METADATA.json','RECEIPT.json'):assert not (p/name).exists(),name
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=source.encode(),capture_output=True,timeout=90);(p/'EXPORT.stderr').write_bytes(result.stderr)
if result.returncode:raise RuntimeError(result.stderr.decode()[:2000])
assert len(result.stdout)<=1200000;(p/'extension.npz').write_bytes(result.stdout)
with np.load(p/'extension.npz',allow_pickle=False) as z:
 extension=json.loads(z['metadata_json'].tobytes());assert not (set(arrays)&(set(z.files)-{'metadata_json'}));arrays.update({k:z[k] for k in z.files if k!='metadata_json'})
metadata['actual_refined_neural_diagnostic_extension']=extension;arrays['metadata_json']=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8);np.savez_compressed(p/'plots-with-refined-neural.npz',**arrays);(p/'METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n');receipt={name:{'sha256':hashlib.sha256((p/name).read_bytes()).hexdigest(),'bytes':(p/name).stat().st_size} for name in ('extension.npz','plots-with-refined-neural.npz','METADATA.json')};receipt.update(original_archive_preserved=base_sha,completed_status_sha256=args.completed_status_sha256,new_scientific_computation=False,source_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ('collect.py','export_remote.py')});(p/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

assert (p/'plots-with-refined-neural.npz').stat().st_size<=16*2**20
