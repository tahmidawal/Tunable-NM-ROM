"""A15 wave-4 certificate: A1/A2 cell bound = max measured production A1/A2 cell elapsed (ledger), A3 at the 10 h wall
limit, test-once allowance 0.5 h per cell; 15 % headroom on not-yet-started cells; failed-job allowance = largest
exposure (10 h)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
L = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
el = {j['name']: j['gpu_hours'] for j in L['jobs']}
b12 = max(el[c] for c in ('W40_A1', 'W40_A2', 'W28_A1', 'W28_A2') if c in el)
running = {'W08_A1': b12, 'W40_A3': 10.0}
queued = {'W28_A3': 10.0, 'W08_A2': b12, 'W16_A1': b12}          # W16_A2 deferred (defer order item 3)
tests = 0.5 * 10
total = L['total_gpu_hours'] + sum(running.values()) + 1.15 * sum(queued.values()) + 10.0 + tests
out = dict(spent=L['total_gpu_hours'], A12_bound=b12, running=running, queued=queued, test_once_allowance=tests,
           failed_job_allowance=10.0, headroom=1.15, projected_total=total, cap=87, release=total <= 87)
(HERE / 'runs' / 'certificate-wave4.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))
