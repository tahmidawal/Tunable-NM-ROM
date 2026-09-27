"""g2d: score the operators timed by ops/optime.py against the job's own reference (Burgers cells).

    python opscore.py <driver output dir> <operator name> [<operator name> ...]

Error = the Table 1 Burgers metric of b2speed.py / bankknob.py (`rel_per_time`): per case
max_{k>=1} ||u_op(t_k) - u_ref(t_k)||_2 / ||u_0||_2 with u_ref the in-job fft_tight solve written to opcohort/
(same arrays the NM-ROM/FOM arms were scored against), worst and median over the cases. Timing = optime's pooled
median GPU ms (every case x repetition), the same pooling the source audits use for the JAX arms. Writes
<out>/opscore.json and deletes the full operator fields afterwards (disk).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    out = Path(sys.argv[1])
    index = json.loads((out / 'opcohort' / 'index.json').read_text())
    prev = out / 'opscore.json'
    res = json.loads(prev.read_text()) if prev.exists() else {}   # g2d: merge, one operator at a time
    for name in sys.argv[2:]:
        tj = out / 'optiming' / f'{name}-timing.json'
        if not tj.exists():
            res[name] = dict(ok=False, reason='no timing record (operator failed to load/run; see job log)')
            continue
        t = json.loads(tj.read_text())
        per = []
        for rec in index['records']:
            c = rec['case_index']
            ref = np.load(out / 'opcohort' / rec['path'])['target'][:, 0]
            f = np.load(out / 'fields' / f'full_{name}_case{c}.npy')
            assert f.shape == ref.shape, (f.shape, ref.shape)
            n0 = float(np.linalg.norm(ref[0]))
            assert abs(n0 - rec['n0']) <= 1e-12 * n0, (n0, rec['n0'])
            sg = [float(np.linalg.norm(a - b)) / n0 for a, b in zip(f, ref)]
            per.append(dict(case=c, same_grid_per_time=sg, same_grid_evolved=max(sg[1:]), t0_error=sg[0],
                            finite=bool(np.isfinite(f).all())))
        ev = [p['same_grid_evolved'] for p in per]
        res[name] = dict(ok=all(p['finite'] for p in per), worst_evolved_percent=100 * max(ev),
                         median_evolved_percent=100 * float(np.median(ev)),
                         median_gpu_ms=t['device_query_pooled']['median_ms'],
                         median_of_case_medians_ms=t['device_query_median_of_case_medians_ms'],
                         host_transfer_median_ms=t['host_transfer_pooled']['median_ms'],
                         gpu_name=t['gpu_name'], best_epoch=t['best_epoch'], checkpoint_sha256=t['checkpoint_sha256'],
                         real_parameter_count=t['real_parameter_count'], config=t['config'],
                         t0_returned_exactly=all(c_['t0_returned_exactly'] for c_ in t['cases']),
                         repetitions=[len(c_['device_seconds']) for c_ in t['cases']], per_case=per)
        print('OPSCORE', name, round(res[name]['worst_evolved_percent'], 4), round(res[name]['median_evolved_percent'], 4),
              round(res[name]['median_gpu_ms'], 3), flush=True)
    (out / 'opscore.json').write_text(json.dumps(res, indent=1) + '\n')
    for name in sys.argv[2:]:
        for p in (out / 'fields').glob(f'full_{name}_case*.npy'):
            p.unlink()


if __name__ == '__main__':
    main()
