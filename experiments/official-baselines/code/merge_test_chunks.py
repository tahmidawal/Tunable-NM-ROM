"""Merge case-chunk test JSONs of one selection row (online_ours.py test --test-cases a:b) into one test JSON.

  merge_test_chunks.py OUT.json CHUNK.json [CHUNK.json ...]
Chunks must cover cases 0..63 exactly once, come from the same selection file (selection_sha256) and the same row.
"""
import json, sys
import numpy as np

out, parts = sys.argv[1], [json.load(open(p)) for p in sys.argv[2:]]
parts.sort(key=lambda d: d['test_cases'][0])
assert [d['test_cases'][0] for d in parts] == [0] + [d['test_cases'][1] for d in parts[:-1]] and parts[-1]['test_cases'][1] == 64
assert len({d['selection_sha256'] for d in parts}) == 1 and len({tuple(d['rows']) for d in parts}) == 1
res = []
for i in range(len(parts[0]['results'])):
    rs = [d['results'][i] for d in parts]
    assert len({json.dumps(r['selection'], sort_keys=True) for r in rs}) == 1
    err = np.concatenate([r['errors'] for r in rs])
    fin = np.isfinite(err)
    # per-case times: each chunk timed its evaluation pass (1 warm-up case per job); median/p5/p95 over all 64 cases
    ms = np.concatenate([r['ms_cases'] for r in rs])
    res.append(dict(selection=rs[0]['selection'], errors=err.tolist(), timing_warmup=1, timing_repetitions=1,
                    test=dict(worst_percent=float(100 * err.max()) if fin.all() else float('inf'), median_percent=float(100 * np.median(err)),
                              mean_percent=float(100 * err.mean()), nonfinite=int((~fin).sum())),
                    ms_full=dict(median=float(np.median(ms)), p5=float(np.percentile(ms, 5)), p95=float(np.percentile(ms, 95)),
                                 min=float(ms.min()), max=float(ms.max()), count=int(ms.size)), decoder=rs[0]['decoder']))
d = dict(parts[0]); d.pop('test_cases')
d.update(job_id=','.join(p['job_id'] for p in parts), node=','.join(p['node'] for p in parts),
         gpu=' / '.join(sorted({p['gpu'] for p in parts})), chunks=[p['test_cases'] for p in parts], results=res)
open(out, 'w').write(json.dumps(d, indent=1) + '\n')
for r in res:
    print(r['selection']['name'], r['selection']['arm'], r['selection']['setting'], r['test'], r['ms_full'])
