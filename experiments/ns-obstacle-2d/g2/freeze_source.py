"""Development-only source inventory: never a submit manifest or release."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
names=['oriented_normals.py','fom.py','reference_export.py','shared_lift.py','scalars.py','preflight.py','freeze_inputs.py',
       'campaign-A2.json','g2/campaign-A6.json','g2/DESIGN-A6.md','benchmark.json','manufactured_witness.json','dependencies.py','DEPENDENCIES.json',
       'g2/plan.py','g2/freeze_source.py','g2/IMPLEMENTATION.md','g2/CASE-MATRIX.DRAFT.json','g2/CASE-MATRIX.DRAFT.md',
       'checks/test_reference_export.py','checks/reference-export-tests.txt']
names+=sorted(str(p.relative_to(root)) for p in (root/'configs').glob('*.json'))
names+=sorted(str(p.relative_to(root)) for p in (root/'g2/native01').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
record={'schema':'ns-obstacle-development-source-v1','stageable':False,'numerical_execution_authorized':False,
        'absolute_mesh_sizes_frozen':False,'status':'prospective source review only, no runtime claims',
        'normal_repair_candidate':'262a0ef0546bdbecb1507fd3a48d22c10b137067', 'normal_propagation_native_execution':False,
        'source_parent':subprocess.check_output(['git','-c','gc.auto=0','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'released_G1_source_unchanged':'262a0ef0546bdbecb1507fd3a48d22c10b137067',
        'configuration_binding':'prospective G2 bound to final A6 campaign/design; A8 helper byte-identical to executed smoke07; G1 artifacts unchanged',
        'files':[{ 'path':name,'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in names]}
(root/'g2/SOURCE-MANIFEST.DRAFT.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
