"""Independent NumPy readback of the first calibration's limiting case."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/runs/calibration01/archive/calibration')
index = json.loads((ROOT/'index.json').read_text())
case = 'burgers-calibration-00001'
hashes = {}


def load(mesh, dt):
    record = next(r for r in index['solves'] if r['case_id']==case and r['intervals']==mesh and r['dt']==dt)
    path = ROOT/record['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == record['sha256']
    hashes[path.name] = digest
    with np.load(path) as data:
        fields = data['fields'].copy()
    assert fields.shape == (6,1025,1025) and fields.dtype==np.float64 and np.isfinite(fields).all()
    return fields[:,::4,::4]


fine = load(4096, .0003125)
space = load(2048, .0003125)
temporal = load(4096, .000625)
scale = np.linalg.norm(fine[0,1:-1,1:-1])
def difference(field):
    return float(np.linalg.norm((field-fine)[:,1:-1,1:-1].reshape(6,-1),axis=1).max()/scale)
space_error,time_error = difference(space),difference(temporal)
declared = next(r for r in index['gate']['by_output']['256']['cases'] if r['case_id']==case)
assert np.isclose(space_error,declared['anchor_space_difference'],rtol=1e-12,atol=1e-14)
assert np.isclose(time_error,declared['anchor_time_difference'],rtol=1e-12,atol=1e-14)
result = dict(passed=True,case_id=case,output_intervals=256,source_index_sha256=hashlib.sha256((ROOT/'index.json').read_bytes()).hexdigest(),
              field_hashes=hashes,space_difference=space_error,time_difference=time_error,
              empirical_margin=space_error+time_error,reference_budget=.001,
              reference_budget_pass=space_error+time_error<=.001,
              interpretation='Agreement with reported refinement differences; not a continuum error certificate.')
out = Path(__file__).parent/'checks/burgers-reference-case-audit.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(out.read_text())
