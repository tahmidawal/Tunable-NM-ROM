"""Extract retained-value gates from the archived accel12 (job 3565786) and accel07 (job 3563590) JSONs.

Each row: mesh, case, method-as-named-in-this-lane, metric, archived worst initial-normalised value.
The driver compares its repetition-0 value on the same A100 class to 1e-9 relative.
"""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'consolidated/evidence/worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs'
rename = {'chol_guard': 'head_q0', 'trained_nested40': 'trained_nested40', 'linear_bank64': 'linear_bank64'}
rows, sources = [], {}
for run, keep in (('accel12', ('chol_guard', 'trained_nested40')), ('accel07', ('linear_bank64',))):
    path = E / run / 'cluster/out/pilot/result.json'
    raw = path.read_bytes(); sources[run] = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest())
    r = json.loads(raw); sources[run]['job_id'] = r['provenance']['job_id']; sources[run]['device'] = r['provenance']['device_kind']
    for i in r['invocations']:
        if i['method'] in keep and i['repetition'] == 0 and i.get('setting') in (0.01, 0.0):
            for metric in ('displacement', 'velocity', 'energy_state'):
                rows.append(dict(source=run, job_id=r['provenance']['job_id'], intervals=i['intervals'], case=i['case'], method=rename[i['method']],
                                 archived_method=i['method'], setting=i['setting'], metric=metric,
                                 value=i['same_grid_discrepancy'][metric]['max_initial_normalized']))
out = Path(__file__).parent / 'retained-gates.json'
out.write_text(json.dumps(dict(sources=sources, gates=rows), indent=2) + '\n')
print(len(rows), 'gate rows;', {(x['intervals'], x['method']) for x in rows})
