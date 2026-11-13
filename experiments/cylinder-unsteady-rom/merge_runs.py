"""Second rebuttal push, PROTOCOL.md A9(c): merge a recorded window and its continuation into one run directory.

The merged directory has the layout the exporters read (stages.json with one completed stage `record`, snapshot list
with hashes, history.npz).  Snapshot chunks are referenced by absolute path, not copied; only the first continuation
chunk is rewritten without the duplicate of the junction state.  History: the continuation's rows at or before the
junction time are dropped, the first run's rows before its record stage are dropped (the readers select by the
stage's time window, so their selection for the first window is unchanged), and reaction-force entries that are NaN (start-up rows without a consistent time derivative)
are filled by linear interpolation in time between their neighbours; the count is recorded.
Usage: merge_runs.py --first RUN_DIR --cont RUN_DIR --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ap = argparse.ArgumentParser()
for k in ('--first', '--cont', '--out'):
    ap.add_argument(k, required=True)
a = ap.parse_args()
A, B, out = Path(a.first).resolve(), Path(a.cont).resolve(), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
sa, sb = (json.loads((d / 'stages.json').read_text()) for d in (A, B))
(ra,) = [s for s in sa['stages'] if s['name'] == 'record']
(rb,) = [s for s in sb['stages'] if s['name'] == 'record']
if ra['status'] != 'completed' or rb['status'] != 'completed':
    raise ValueError('incomplete stage')
tj = ra['t_end']
if abs(rb['t_start'] - tj) > 1e-9 or abs(rb['dt'] - ra['dt']) > 1e-15:
    raise ValueError('continuation does not start at the end of the first window with the same step')
ca, cb = (json.loads((d / 'config.json').read_text()) for d in (A, B))
for k in ('nu', 'mesh_sha256', 'velocity_order', 'pressure_order', 'scheme', 'refactor_every'):
    if ca[k] != cb[k]:
        raise ValueError(f'config mismatch: {k}')
if cb['initial']['sha256'] != [s for s in sa['states'] if s['file'] == 'record.npz'][0]['sha256']:
    raise ValueError('continuation did not start from the recorded end state')
snaps = [{'file': str(A / r['file']), 'sha256': r['sha256'], 'count': r['count']} for r in sa['snapshots']]
dropped = 0
for i, r in enumerate(sb['snapshots']):
    p = B / r['file']
    z = np.load(p)
    keep = z['t'] > tj + 1e-9
    if keep.all():
        snaps.append({'file': str(p), 'sha256': r['sha256'], 'count': r['count']})
    elif keep.any():
        q = out / f'cont_{r["file"]}'
        np.savez(q, t=z['t'][keep], x=z['x'][keep])
        dropped += int((~keep).sum())
        snaps.append({'file': q.name, 'sha256': sha(q), 'count': int(keep.sum())})
    else:
        dropped += int((~keep).sum())
ha, hb = np.load(A / 'history.npz'), np.load(B / 'history.npz')
rows_b = hb['rows'][hb['rows'][:, 0] > tj + 1e-9]
rows_a = ha['rows'][ha['rows'][:, 0] >= ra['t_start'] - 1e-9]   # exactly the rows the readers select for the first window
rows_b[:, -1] = rows_a[-1, -1]                                  # same stage label as the first window
rows = np.concatenate([rows_a, rows_b])
d = np.diff(rows[:, 0])
if np.any(d < 0) or np.any(d[len(rows_a) - 2:] <= 0):
    raise ValueError('merged history times are not increasing across the junction')
in_rec = rows[:, 0] > ra['t_start'] + 1e-9          # the first row of the record stage keeps its NaN (as in every run)
filled = {}
for j in range(1, rows.shape[1] - 1):
    bad = ~np.isfinite(rows[:, j]) & in_rec
    if bad.any():
        good = np.isfinite(rows[:, j])
        rows[bad, j] = np.interp(rows[bad, 0], rows[good, 0], rows[good, j])
        filled[str(hb['columns'][j])] = [float(t) for t in rows[bad, 0]]
np.savez(out / 'history.npz', rows=rows, columns=hb['columns'])
merged = {'status': 'completed',
          'stages': [{'name': 'record', 'dt': ra['dt'], 't_start': ra['t_start'], 't_end': rb['t_end'], 'status': 'completed',
                      'steps': ra['steps'] + rb['steps']}],
          'snapshots': snaps, 'states': [],
          'merged_from': {'first': str(A), 'continuation': str(B), 'junction_time': tj, 'duplicate_snapshots_dropped': dropped,
                          'history_rows_dropped': int(len(hb['rows']) - len(rows_b)), 'force_entries_interpolated': filled,
                          'first_stages_sha256': sha(A / 'stages.json'), 'continuation_stages_sha256': sha(B / 'stages.json')}}
(out / 'stages.json').write_text(json.dumps(merged, indent=2) + '\n')
ca['stages'] = [{'name': 'record', 'dt': ra['dt'], 'duration': rb['t_end'] - ra['t_start']}]
(out / 'config.json').write_text(json.dumps(ca, indent=2) + '\n')
n = sum(s['count'] for s in snaps)
print('merged', out, 'snapshots', n, 't', ra['t_start'], rb['t_end'], json.dumps(merged['merged_from']['force_entries_interpolated']))
