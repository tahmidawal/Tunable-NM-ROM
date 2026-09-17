"""Build `checks/comparators.json`: the cross-job fidelity targets, read from the parents' audits.

Three source jobs, all independently NumPy-audited in their own lanes:

  qtd02   job 3757505  the dense ladder, budget 600, M = 4(K+q)   (q-trajdirs worktree)
  btq201  job 3747245  the same-job envelope, fixed M = 256        (this tree)
  qrg304  job 3768168  the certified EQ rules                      (this tree)

Every comparator number is copied from those audit JSONs; nothing is typed by hand.

    python make_comparators.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDITS = {
    'qtd02': ROOT.parent / '2026-09-16-q-trajdirs/experiments/q-trajdirs/checks/qtd02-audit.json',
    'btq201': ROOT / 'experiments/b-ladder-top/checks/btq201-audit.json',
    'qrg304': ROOT / 'experiments/q-ridge/checks/qrg304-audit.json',
}
KEEP = ('arm', 'q', 'k', 'M', 'm', 'quadrature', 'gtol', 'converged', 'total_budget_exits',
        'max_joint_stationarity', 'median_gpu_ms', 'median_host_ms', 'worst_reference_percent',
        'worst_all_times_percent', 'worst_evolved_percent', 'worst_t0_compression_percent',
        'per_case_evolved_percent', 'per_case_all_times_percent', 'median_iterations')


def main():
    table = {}
    for name, path in AUDITS.items():
        au = json.loads(path.read_text())
        table[name] = dict(job_id=au.get('job_id'), gpu=au.get('gpu'), commit=au.get('commit'),
                           source=str(path),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           arms={x['arm']: {k: x.get(k) for k in KEEP} for x in au['arms']})
        print(name, au.get('job_id'), au.get('gpu'), 'arms', len(table[name]['arms']))
    (HERE / 'checks/comparators.json').write_text(json.dumps(table, indent=2) + '\n')
    print('WROTE', HERE / 'checks/comparators.json')


if __name__ == '__main__':
    main()
