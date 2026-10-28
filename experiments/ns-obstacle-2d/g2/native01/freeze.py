"""Content freeze for source review only; never stages or submits."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
names=['fom.py','reference_export.py','shared_lift.py','preflight.py','scalars.py','oriented_normals.py',
       'benchmark.json','manufactured_witness.json','dependencies.py','DEPENDENCIES.json',
       'g2/campaign-A6.json','g2/DESIGN-A6.md','g2/native01/dfg-native-control.json',
       'g2/native01/summary.py','g2/native01/native.sh','g2/native01/PROVISION.json']
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(ROOT/'oriented_normals.py')=='3a13ea7ca6c9110997b1c44fd04006bdf9d5f089121beb91b2988a3ca6d7779b'
assert sha(ROOT/'preflight.py')=='6bd7a26fd9849dd021138471d04a01c52b520382a60e060e73b0e3a30f0e87d3'
base=json.loads((ROOT/'artifacts/g1-smoke07/results/dfg/config.json').read_text())
cfg=json.loads((HERE/'dfg-native-control.json').read_text())
changed={key:{'old':base.get(key),'new':cfg.get(key)} for key in sorted(set(base)|set(cfg)) if base.get(key)!=cfg.get(key)}
allowed={'reference_export_controls','export_common_stokes_lift','campaign_file','campaign_sha256',
         'design_sha256','release','split','purpose'}
assert set(changed)<=allowed
assert cfg['campaign_sha256']==sha(ROOT/'g2/campaign-A6.json')
assert cfg['design_sha256']==sha(ROOT/'g2/DESIGN-A6.md')
record={'schema':'ns-obstacle-native-control-source-candidate-v1','staging_authorized':False,
        'numerical_execution_authorized':False,'physical_admission':False,
        'source_commit_binding':'exact reviewed commit plus these content hashes; filled only in staging receipt after release',
        'G1_A8_reviewed_and_executed_base':'262a0ef0546bdbecb1507fd3a48d22c10b137067',
        'fresh_cluster_attempt':'g2-native01',
        'source_parent_at_freeze':subprocess.check_output(['git','-c','gc.auto=0','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'unchanged_A8_normal_helper_sha256':sha(ROOT/'oriented_normals.py'),
        'configuration_delta_against_executed_DFG':changed,
        'caps':{'gpu_count':1,'CPUs':16,'memory_GiB':64,'wall_seconds':1200},
        'files':[{'source':name,'destination':'REPAIR.json' if name=='g2/native01/PROVISION.json' else name,
                  'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in names]}
(HERE/'SOURCE-CANDIDATE.json').write_text(json.dumps(record,indent=2)+'\n')
# Focus the source review on numerical additions over the executed isolated G1.
parts=[]
for name in ['fom.py','preflight.py']:
    compared=subprocess.run(['diff','-u',str(ROOT/'repairs/normal01'/name),str(ROOT/name)],capture_output=True,text=True)
    if compared.returncode not in (0,1):raise RuntimeError(compared.stderr)
    parts.append(compared.stdout)
(HERE/'DELTA-AGAINST-A8.patch').write_text(''.join(parts))
