"""Sub-manifest of the frozen burgers-ops-matched selection: the FNO (F0) and U-Net (U0) 4,608-trajectory runs only.
Paths and checkpoint hashes are copied unchanged; evaluate.py re-verifies every hash before loading."""
import json, sys
m = json.load(open(sys.argv[1]))
runs = {k: v for k, v in m['runs'].items() if v['rung'] == 4608 and v['arm'] in ('F0', 'U0')}
assert len(runs) == 6, sorted(runs)
json.dump(dict(m, runs=runs), open(sys.argv[2], 'w'), indent=1)
print('MANIFEST', sorted(runs))
