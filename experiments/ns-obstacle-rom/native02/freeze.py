"""Review inventory only. Does not stage, submit or construct an admission."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fe_adapter import adapter_digest
names=['native_block_control.py','fe_blocks.py','fe_adapter.py','fe_algebra.py','fe_pod_a6.py','fe_controls_a6.py',
       'spaces.py','weak.py','train_bank.py','oriented_normals.py','scalars.py','MODEL-CONTRACT.json']
rows=[(n,n) for n in names]+[(f'native02/{n}',d) for n,d in
 [('dependencies.py','dependencies.py'),('DEPENDENCIES.json','DEPENDENCIES.json'),('PROVISION.json','REPAIR.json'),('native.sh','native.sh')]]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
case=ROOT.parents[2]/'2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d/artifacts/g2-native01/results/dfg'
assert case.is_dir()
m=json.loads((case/'EXPORT-MANIFEST.json').read_text())
inputs=[{'source':str(case/n),'destination':'inputcase/'+n,'sha256':sha(case/n),'bytes':(case/n).stat().st_size}
        for n in sorted([*m['files'],'EXPORT-MANIFEST.json'])]
for n,h in m['files'].items():assert sha(case/n)==h
out={'status':'source_only_unexecuted_no_staging_or_submission','physical_admission':False,'training_released':False,
'adapter_source_sha256':adapter_digest(),'fixture_source_sha256':sha(ROOT/'native_block_control.py'),
'block_source_sha256':sha(ROOT/'fe_blocks.py'),
'producer_manifest_sha256':sha(case/'EXPORT-MANIFEST.json'),
'producer_retention_commit':'13083eb5b','producer_retention_mirror':'2ef3375f2',
'fresh_directory':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/native-block02',
'caps':{'gpu':1,'CPUs':16,'memory_GiB':64,'wall_seconds':1200,'working_set_projection_GiB':8,'result_projection_GiB':1},
'files':[{'source':s,'destination':d,'sha256':sha(ROOT/s),'bytes':(ROOT/s).stat().st_size} for s,d in rows],
'producer_inputs':inputs,
'staging_generated_files':['CONTROL-RELEASE.json with independently reviewed exact-source/producer binding','SOURCE_COMMIT.txt','INPUT-MANIFEST.json','SOURCE-SHA256SUMS covering every staged runtime input'],
'no_automatic_retry':True}
(ROOT/'native02/SOURCE-CANDIDATE.json').write_text(json.dumps(out,indent=2)+'\n')
