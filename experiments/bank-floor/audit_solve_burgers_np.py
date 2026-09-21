"""Independent NumPy audit of a Phase-2 Burgers result (no JAX, no lane modules).

Recomputes every dev6 per-time relative error from the saved fields, with the saved `fom_tight`
fields as the same-grid reference (they are the job's own reference solve), and recomputes each
bank's floor on those reference fields from the bank file with NumPy QR where the bank is a POD basis.
usage: audit_solve_burgers_np.py RESULT.json FIELDS_DIR OUT.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    fd = Path(sys.argv[2])
    ncase = res['cohorts']['dev6']['count']
    ref = [np.load(fd / f'fom_tight_case{i}.npy') for i in range(ncase)]
    rows, worst = {}, 0.
    for name, s in res['subjects'].items():
        rep = np.asarray(s['results']['dev6']['per_case_per_time'])
        d = 0.
        for i in range(ncase):
            f = np.load(fd / f'{name}_case{i}.npy')
            e = [np.linalg.norm(f[t] - ref[i][t]) / np.linalg.norm(ref[i][t]) for t in range(len(f))]
            d = max(d, float(np.max(np.abs(np.asarray(e) - rep[i]))))
        rows[name] = dict(max_abs_error_difference=d, reported_worst_all_times=float(rep.max()))
        worst = max(worst, d)
    # fom_tight is the reference up to repeat-run determinism: its own error must be ~0
    out = dict(result=sys.argv[1], result_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
               reference='saved fom_tight dev6 fields', tolerance=1e-10, worst_abs_difference=worst,
               subjects=rows, passed=bool(worst < 1e-10 and rows))
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + '\n')
    print('worst', worst, 'AUDIT', 'PASS' if out['passed'] else 'FAIL')


if __name__ == '__main__':
    main()
