"""Source-only reusable append; invoke only after actual terminal independent review."""
from pathlib import Path
import argparse,hashlib,json,subprocess,numpy as np
p=Path(__file__).resolve().parent;parser=argparse.ArgumentParser()
for name in ('base','audit','output'):parser.add_argument('--'+name,type=Path,required=True)
for name in ('base-sha256','audit-sha256'):parser.add_argument('--'+name,required=True)
parser.add_argument('--variant',choices=('traction',),required=True);args=parser.parse_args()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(args.base)==args.base_sha256 and sha(args.audit)==args.audit_sha256
j=json.loads(args.audit.read_text());assert j['passed'] and j['physical_admission'] is False
binding=json.loads((args.audit.parent/'BINDING.json').read_text());assert binding['model_sha256']==j['model_sha256']
paths=[Path(k) for k,v in j['provenance'].items() if k.endswith('/results/diagnostic/DIAGNOSTIC.json') and v==binding['status_sha256']];assert len(paths)==1;job=paths[0].parents[2]
launchers=[str(Path(k).relative_to(job)) for k in j['provenance'] if k.startswith(str(job)+'/accuracy_v2/diagnostic') and k.endswith('_launch.py')];assert len(launchers)==1
with np.load(args.base,allow_pickle=False) as z:arrays={k:z[k] for k in z.files};metadata=json.loads(arrays.pop('metadata_json').tobytes())
assert all(k in arrays for k in ('indices','points_physical','reference_velocity','reference_pressure','pod_guided_neural_r4_solve_velocity','refined_neural_r4_solve_velocity'))
if args.variant=='trace':assert args.base_sha256=='6f5b96ea4ba8c7f682b23c46dcc4f47948765e64b9b9069f9da521ee9819d07f'
else:assert args.base_sha256=='c69144693b979ca7e2ed3ef070d0a15ba25dca5f307083fd12c6af9b8ee18e2e' and all(k in arrays for k in ('trace_neural_r4_solve_velocity','trace_neural_r4_solve_pressure','energy_neural_r4_solve_velocity','energy_neural_r4_solve_pressure'))
ids=arrays['indices'];assert ids.shape==(18000,) and hashlib.sha256(ids.tobytes()).hexdigest()=='f1a9de8dd9bfb6c32400bb2d9b99f1b54a13185807fbd64aeba29f011df13715'
assert not args.output.exists();args.output.mkdir(parents=True)
context=dict(VARIANT=args.variant,REMOTE_JOB=str(job),LAUNCHER_FILE=launchers[0],MODEL_SHA=binding['model_sha256'],PLAN_SHA=j['provenance'][str(job/'PLAN.json')],LATEST_SHA=binding['latest_record_sha256'],AUDIT_SHA=args.audit_sha256,STATUS_SHA=binding['status_sha256'],BASE_SHA=args.base_sha256,INDICES=ids.tolist(),INDICES_SHA=hashlib.sha256(ids.tobytes()).hexdigest())
source=''.join(k+'='+repr(v)+'\n' for k,v in context.items())+(p/'export_remote.py').read_text()
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login','PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=source.encode(),capture_output=True,timeout=90)
(args.output/'EXPORT.stderr').write_bytes(result.stderr)
assert result.returncode==0 and not result.stderr
assert len(result.stdout)<=1200000;(args.output/'extension.npz').write_bytes(result.stdout)
with np.load(args.output/'extension.npz',allow_pickle=False) as z:
 extension=json.loads(z['metadata_json'].tobytes());new={k:z[k] for k in z.files if k!='metadata_json'}
expected={args.variant+'_neural_r4_solve_velocity':(2,18000,2),args.variant+'_neural_r4_solve_pressure':(2,18000)};assert set(new)==set(expected) and not set(new)&set(arrays)
for k,v in new.items():assert v.shape==expected[k] and v.dtype==np.float64 and np.isfinite(v).all()
assert [r['Re'] for r in extension['cases']]==[r['Re'] for r in j['cases']]==[16.875,23.125]
for r,a in zip(extension['cases'],j['cases']):
 for k,v in a['metrics'].items():assert np.isclose(r['recorded_full_rule_scores'][k],v,rtol=2e-12,atol=2e-14)
arrays.update(new);metadata['actual_'+args.variant+'_neural_diagnostic_extension']=extension;arrays['metadata_json']=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8)
out=args.output/('plots-with-'+args.variant+'-neural.npz');np.savez_compressed(out,**arrays);assert out.stat().st_size<=16*2**20
(args.output/'METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n');receipt=dict(base_sha256=args.base_sha256,audit_sha256=args.audit_sha256,variant=args.variant,remote_job=str(job),output_sha256=sha(out),output_bytes=out.stat().st_size,extension_sha256=sha(args.output/'extension.npz'),source_sha256={n:sha(p/n) for n in ('collect.py','export_remote.py')},all_old_scientific_arrays_preserved=True,new_scientific_computation=False)
(args.output/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
