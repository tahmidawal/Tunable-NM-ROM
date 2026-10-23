"""A16 wave-5 certificate: time-stage restarts of W40_A3 and W28_A3 at their 6 h wall limit, W08_A2 at the largest
measured production A1/A2 cell (x 1.15), test-once allowance 2.22 GPU-h per retained cell (largest measured
ref + time + controls stage hours, W40_A2), failed-job allowance = the largest exposure (6 h)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
L = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
el = {j['name']: j['gpu_hours'] for j in L['jobs']}
b12 = max(el[c] for c in ('W40_A1', 'W40_A2', 'W28_A1', 'W28_A2', 'W08_A1') if c in el)
retained = ['W40_A1', 'W40_A2', 'W40_A3', 'W28_A1', 'W28_A2', 'W28_A3', 'W08_A1', 'W08_A2']   # tests reserved for all
exposure = {'W40_A3t': 6.0, 'W28_A3t': 6.0}
queued = {}                                      # W08_A2 decided after the restarts (A16)
tests = 2.22 * len(retained)
total = L['total_gpu_hours'] + sum(exposure.values()) + 1.15 * sum(queued.values()) + 6.0 + tests
out = dict(spent=L['total_gpu_hours'], A12_bound=b12, exposure=exposure, queued=queued, retained=retained,
           test_once_allowance=tests, failed_job_allowance=6.0, projected_total=total, cap=87, release=total <= 87)
(HERE / 'runs' / 'certificate-wave5.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))
