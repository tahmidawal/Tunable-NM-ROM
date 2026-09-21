"""Independent audit of a Phase-2 Poisson result: NumPy/SciPy only, no JAX, no lane modules.

Rebuilds each dev12 source and the DST truth with SciPy, recomputes every saved field's relative
error, and compares with the reported per-case errors.
usage: audit_solve_poisson_np.py RESULT.json FIELDS_DIR OUT.json   (PYTHONPATH must hold ms_parametric)
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.fft import dstn
import ms_parametric as mp


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    fd = Path(sys.argv[2])
    cfg = res['config']
    N = cfg['intervals']
    draws = []
    for seed, count in cfg['cohorts'][cfg['save_fields_cohort']]:
        cx, cy, w, a, _ = mp.sample_params(seed, count)
        draws += list(np.column_stack((cx, cy, w, a)))
    p = np.arange(1, N)
    l = 4.0 * N ** 2 * np.sin(np.pi * p / (2 * N)) ** 2
    lam = l[:, None] + l[None, :]
    truth = [dstn(dstn(mp.source_interior(N + 1, *q), type=1, norm='ortho') / lam, type=1, norm='ortho')
             for q in draws]
    rows, worst = {}, 0.
    for name, s in res['subjects'].items():
        if not s['results']:
            continue
        rep = s['results'][cfg['save_fields_cohort']]['per_case']
        diffs = []
        for i, t in enumerate(truth):
            f = np.load(fd / f'{name}_case{i}.npy')[1:-1, 1:-1]
            e = np.linalg.norm(f - t) / np.linalg.norm(t)
            diffs.append(abs(e - rep[i]))
        rows[name] = dict(max_abs_error_difference=float(max(diffs)), reported_worst=float(max(rep)))
        worst = max(worst, max(diffs))
    out = dict(result=sys.argv[1], result_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
               tolerance=1e-10, worst_abs_difference=float(worst), subjects=rows,
               passed=bool(worst < 1e-10 and rows))
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + '\n')
    print('worst', worst, 'AUDIT', 'PASS' if out['passed'] else 'FAIL')


if __name__ == '__main__':
    main()
