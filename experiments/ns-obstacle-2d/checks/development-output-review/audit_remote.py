"""Independent development field handoff audit, saved data only."""
from pathlib import Path
import hashlib,json,sys,time,resource
import numpy as np
start=time.monotonic();artifact=Path(sys.argv[1]);manifest_path=artifact/'results/population/development.json';out=manifest_path.parent

def sha(p):
 with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def adigest(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps({'dtype':a.dtype.str,'shape':list(a.shape)},sort_keys=True).encode());h.update(memoryview(a).cast('B'));return h.hexdigest()
def rule_digest(arrays):
 h=hashlib.sha256()
 for k,a in sorted(arrays.items()):
  a=np.ascontiguousarray(a);h.update(k.encode());h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
 return h.hexdigest()
admission_path=artifact/'inputs/FD1-REFERENCE-ADMISSION.json';truth=read(admission_path);m=read(manifest_path);plan=read(artifact/'HANDOFF-PLAN.json');assert sha(manifest_path)==EXPECTED['manifest_sha256']
assert sha(admission_path)=='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f'==m['reference_admission_sha256']
assert m['schema']=='ns-obstacle-pilot-reference-population-v1' and m['split']=='development' and [r['Re'] for r in m['cases']]==[16.875,23.125]
for k in ('gate','passed','qualification_scope'):assert m[k]==truth[k]
assert m['velocity_gradient_units']=='d(u/U)/d(x/D)'
u=EXPECTED['uncertainty'];records=[];rule=None
for row in m['cases']:
 re=row['Re'];a=next(r for r in truth['admitted_references'] if r['Re']==re);original=Path(next(r for r in plan['cases'] if r['Re']==re)['directory']);expected=EXPECTED['cases'][str(re)]
 assert a['split']==row['split']=='development' and row['reference_admitted'] is True
 assert row['original_export_sha256']==a['export_manifest_sha256']==sha(original/'EXPORT-MANIFEST.json')
 assert a['config_sha256']==sha(original/'config.json') and a['solution_sha256']==sha(original/'solution.sol')
 primary=read(original/'PRIMARY-REFERENCE.json');cfg=read(original/'config.json');assert primary==expected['primary'] and cfg==expected['config'];assert sha(original/'PRIMARY-REFERENCE.json')==expected['primary_sha256'];case=out/str(re);replay=read(case/'REPLAY.json')
 assert replay['expected']==replay['actual']==primary['derived_full_rule_arrays']
 assert replay['original_export_sha256']==a['export_manifest_sha256'] and sha(case/'REPLAY.json')==row['replay_sha256']
 assert sha(case/'raw-full-rule.npz')==replay['raw_archive_sha256'] and sha(artifact/'handoff_primary_worker.py')==replay['worker_sha256']
 for name,h in primary['source_sha256'].items():assert sha(artifact/'primary_native'/name)==h
 with np.load(case/'raw-full-rule.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
 assert set(raw)==set(primary['derived_full_rule_arrays'])
 for k,v in raw.items():
  d={'dtype':v.dtype.str,'shape':list(v.shape),'content_sha256':adigest(v)};assert d==primary['derived_full_rule_arrays'][k]
  assert v.dtype==np.float64 and np.isfinite(v).all()
 assert cfg['U']==.2 and cfg['D']==.1 and abs(cfg['nu']-.02/re)<1e-15
 current=rule_digest({'points_dimensionless':raw['points']/cfg['D'],'weights_dimensionless':raw['weights']/cfg['D']**2});assert current==m['volume_rule_sha256']==u['volume_rule_sha256']
 assert rule is None or rule==current;rule=current
 physical=u['physical_rule_identity'];assert adigest(raw['points'])==physical['points_sha256'] and adigest(raw['weights'])==physical['weights_sha256']
 expected={'velocity':raw['velocity']/cfg['U'],'pressure':raw['pressure']/cfg['U']**2,'velocity_gradient':raw['velocity_gradient']*cfg['D']/cfg['U']}
 array_file=out/row['array_file'];assert array_file.resolve().is_relative_to(out.resolve()) and sha(array_file)==row['array_sha256']
 with np.load(array_file,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
 assert set(fields)==set(expected)
 for k,a2 in fields.items():assert a2.dtype==np.float64 and np.isfinite(a2).all() and np.array_equal(a2,expected[k]),k
 force=read(original/'result.json')['metrics']['forces']['physical_symmetric_order16']['coefficient']
 assert row['force_definition']=='physical_symmetric' and row['force_coefficient']==force==EXPECTED['cases'][str(re)]['force']
 assert 'jax_backend=gpu' in (case/'worker.stdout').read_text() and not (case/'worker.stderr').read_text().strip()
 records.append({'Re':re,'split':'development','original_export_sha256':a['export_manifest_sha256'],'primary_record_sha256':sha(original/'PRIMARY-REFERENCE.json'),'replay_sha256':sha(case/'REPLAY.json'),'raw_archive_sha256':replay['raw_archive_sha256'],'fields_archive_sha256':sha(array_file),'field_content_sha256':{k:adigest(v) for k,v in fields.items()},'raw_arrays_verified':len(raw),'dimensionless_fields_exact':True,'force_coefficient':force,'common_rule_sha256':current})
 del raw,fields,expected
result={'schema':'ns-obstacle-independent-development-handoff-review-v1','passed':True,'job_id':'6635386','scope':'saved development-only arrays; no native reconstruction or validation access','manifest_sha256':sha(manifest_path),'reference_admission_sha256':sha(admission_path),'cases':records,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
result['expected_packet_sha256']=EXPECTED_SHA;print(json.dumps(result,indent=2))
