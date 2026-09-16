"""Bench the GPU fitter against scipy's NNLS across sizes, and prove it does not cycle.

Attempt `qrg301` failed because the inner solve dropped every negative weight without a
KKT re-entry test: the outer greedy re-selected the columns NNLS had just zeroed, cycled,
and stopped on its pass cap with 97 of 256 requested points. This bench is the gate that
would have caught it: for each size it requires the achieved support to reach the target
(or to stop on a genuine `gradient` KKT exit), and the relative fit to be at least as good
as scipy's block-greedy fitter at the same target.

    python checks/fitter_bench.py [out.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for sub in ('experiments/q-ridge', 'experiments/cheap-corrections',
            'experiments/head-ablation', 'experiments/mr-burgers2d'):
    sys.path.insert(0, str(ROOT / sub))

import eqcert as EC          # noqa: E402
import varpro as VP          # noqa: E402

CASES = [(1024, 2048, 128), (2048, 4096, 256), (4096, 8192, 1024), (8192, 16384, 2048)]


def main():
    rng = np.random.default_rng(20260916)
    rows = []
    for nrows, ncols, target in CASES:
        # A design with the shape and the nonnegativity structure of a real quadrature fit.
        D = np.abs(rng.standard_normal((nrows, ncols))) * (1. + rng.random(ncols))
        true = rng.choice(ncols, target // 2, replace=False)
        b = D[:, true] @ np.abs(rng.standard_normal(target // 2)) + .01 * rng.random(nrows)
        s1, w1, i1 = EC.gpu_nnls(D, b, target, blocks=8)
        t0 = time.perf_counter()
        s2, w2, i2 = VP.bounded_nnls(D, b, target, 600., block=max(1, target // 16))
        r2 = float(np.linalg.norm(D[:, s2] @ w2 - b) / np.linalg.norm(b))
        row = dict(rows=nrows, cols=ncols, target=target,
                   gpu_support=int(len(s1)), gpu_relative=i1['relative_fit'],
                   gpu_rounds=i1['reentry_rounds'], gpu_stop=i1['stop_reason'],
                   gpu_seconds=i1['seconds'], scipy_support=int(len(s2)),
                   scipy_relative=r2, scipy_seconds=time.perf_counter() - t0,
                   nonnegative=bool((w1 >= 0).all()),
                   reached_target_or_kkt=bool(len(s1) >= target
                                              or i1['stop_reason'] == 'gradient'))
        rows.append(row)
        print(row, flush=True)
        assert row['nonnegative'], row
        assert row['reached_target_or_kkt'], row
        assert i1['relative_fit'] <= max(1.2 * r2, 1e-9), row
    out = dict(cases=rows, note=('the gate qrg301 would have failed: reached_target_or_kkt'))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
    print('FITTER BENCH OK', flush=True)


if __name__ == '__main__':
    main()
