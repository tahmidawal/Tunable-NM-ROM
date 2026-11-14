"""Sub-manifest of the operator lane's frozen selection: the runs trained on all 4,608 trajectories, entries unchanged.
ops_manifest.py SRC DST"""
import json, sys
m = json.load(open(sys.argv[1]))
m['runs'] = {k: v for k, v in m['runs'].items() if v['rung'] == 4608}
assert len(m['runs']) == 10, list(m['runs'])
json.dump(m, open(sys.argv[2], 'w'), indent=1)
print('MANIFEST', list(m['runs']))
