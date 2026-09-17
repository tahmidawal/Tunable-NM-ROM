"""Print the DESIGN section-1 motivating table from the archived JSONs (never hand-typed)."""
import json, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'consolidated/evidence/worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs'
for run, arms in (('accel12', [('chol_guard', 0.01), ('trained_nested40', 0.01)]), ('accel07', [('chol_guard', 0.01), ('linear_bank64', 0.0)])):
    raw = (E / run / 'cluster/out/pilot/result.json').read_bytes(); r = json.loads(raw)
    print(f"| {run} (job {r['provenance']['job_id']}, {r['provenance']['device_kind'][0]}, result sha256 {hashlib.sha256(raw).hexdigest()[:12]}), 64 intervals | cases | GPU ms (median, all reps) | worst energy-state % | worst u % |")
    print('|---|---:|---:|---:|---:|')
    for name, dt in arms:
        inv = [i for i in r['invocations'] if i['method'] == name and i['setting'] == dt and i['intervals'] == 64]
        gpu = np.median([i['seconds']['complete_device_query'] * 1e3 for i in inv])
        e = max(i['same_grid_discrepancy']['energy_state']['max_initial_normalized'] for i in inv)
        u = max(i['same_grid_discrepancy']['displacement']['max_initial_normalized'] for i in inv)
        print(f"| `{name}` (dt {dt}) | {len({i['case'] for i in inv})} | {gpu:.4f} | {100*e:.4f} | {100*u:.4f} |")
    print()
