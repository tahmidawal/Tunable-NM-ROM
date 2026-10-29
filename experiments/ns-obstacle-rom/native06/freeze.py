"""Freeze a review inventory only; never stage, submit or admit."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from primary_volume import SOURCES,source_digest
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[(n,n) for n in SOURCES]+[('native_primary_control.py','native_primary_control.py')]
rows += [(f'native06/{name}',dest) for name,dest in [('dependencies.py','dependencies.py'),('DEPENDENCIES.json','DEPENDENCIES.json'),
 ('PROVISION.json','REPAIR.json'),('native.sh','native.sh'),('BASIS-INPUTS.json','basis/manifest.json'),('BASIS-INPUTS.npz','basis/arrays.npz')]]
case=ROOT/'artifacts/native-batch03/inputcase';producer=json.loads((case/'EXPORT-MANIFEST.json').read_text())
inputs=[{'source':str((case/n).resolve()),'destination':'inputcase/'+n,'sha256':sha(case/n),'bytes':(case/n).stat().st_size}
 for n in ('EXPORT-MANIFEST.json','config.json','mesh.vol.bin','fe_layout.json','full_rule.npz')]
for row in inputs:
 if Path(row['source']).name!='EXPORT-MANIFEST.json':assert row['sha256']==producer['files'][Path(row['source']).name]
out={'status':'source_only_unexecuted_not_staged_or_released','physical_admission':False,'training_released':False,
 'primary_source_sha256':source_digest(),'fixture_source_sha256':sha(ROOT/'native_primary_control.py'),
 'producer_manifest_sha256':sha(case/'EXPORT-MANIFEST.json'),'basis_manifest_sha256':sha(ROOT/'native06/BASIS-INPUTS.json'),
 'retained_basis_job_id':6615810,'source_scope':'actual accepted native-batch03 mesh/eigen/test columns, 64/16 test widths; no PDE/factor/eigen/training',
 'fresh_directory':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/native-primary06',
 'release_schema':'ns-obstacle-primary-volume-replay-control-v1',
 'caps':{'gpu':1,'CPUs':16,'memory_GiB':64,'wall_seconds':1200,'unchanged_internal_process_GiB':16,'result_bytes':2**30,
         'local_collection_reservation_bytes':3*2**30,'minimum_local_reserve_bytes':10*2**30},
 'files':[{'source':s,'destination':d,'sha256':sha(ROOT/s),'bytes':(ROOT/s).stat().st_size} for s,d in rows],
 'producer_inputs':inputs,'staging_generated_files':['CONTROL-RELEASE.json','SOURCE_COMMIT.txt','INPUT-MANIFEST.json','SOURCE-SHA256SUMS covering all staged files'],
 'no_automatic_retry':True,'fine_P4_omission_authorized':False}
(ROOT/'native06/SOURCE-CANDIDATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'runtime_files':len(rows),'producer_files':len(inputs),'inventory_sha256':sha(ROOT/'native06/SOURCE-CANDIDATE.json'),'primary_source_sha256':source_digest()},indent=2))
