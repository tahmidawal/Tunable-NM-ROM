"""Build `checks/comparators.json`: both same-grid metrics for every arm of every source job.

Nothing is typed by hand. Four sources come from the parent lane's committed comparator
table (`experiments/q-ridge/checks/comparators.json`: `btq101`, `btq102`, `btq201` from
their audit JSONs and `cclad01` recomputed from its retained fields); two more are read from
committed audit JSONs — the `q-trajdirs` lane's `qtd02` (the dense 4(K+q) ladder and the
fixed-M=256 rungs, job 3757505) read-only from its sibling worktree, and this worktree's
own `qrg201` (M in {4, 8, 16}(K+q), job 3757237). Every source file's SHA256 is recorded.

    python make_comparators.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'experiments/q-ridge/checks/comparators.json'
AUDITS = {
    'qtd02': ROOT.parent / '2026-09-16-q-trajdirs/experiments/q-trajdirs/artifacts/qtd02/audit.json',
    'qrg201': ROOT / 'experiments/q-ridge/checks/qrg201-audit.json',
    # This lane's own first round, so the extension jobs can be gated against it.
    'bqx101': HERE / 'checks/bqx101-audit.json',
    'bqx201': HERE / 'checks/bqx201-audit.json',
    'bqx301': HERE / 'checks/bqx301-audit.json',
}
KEEP = ('arm', 'q', 'M', 'm', 'rule', 'quadrature', 'converged', 'total_budget_exits',
        'max_joint_stationarity', 'median_gpu_ms', 'median_iterations',
        'worst_reference_percent', 'worst_all_times_percent', 'worst_evolved_percent',
        'worst_t0_compression_percent')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parent = json.loads(PARENT.read_text())
    table = {}
    for name, v in parent.items():
        table[name] = dict(job_id=v.get('job_id'), gpu=v.get('gpu'),
                           source=str(PARENT), source_sha256=sha(PARENT),
                           via='experiments/q-ridge/checks/comparators.json',
                           arms={k: {kk: x.get(kk) for kk in KEEP} for k, x in v['arms'].items()})
    for name, path in AUDITS.items():
        au = json.loads(path.read_text())
        dirs = au.get('directions_sha256')
        chk = au.get('checks', {})
        if dirs is None:
            for key in ('old_directions_hash_matches_comparator', 'directions_hash_matches_cclad01'):
                if key in chk and isinstance(chk[key].get('detail'), dict):
                    d = chk[key]['detail']
                    dirs = d.get('got') or d.get('ours')
        table[name] = dict(job_id=au.get('job_id'), gpu=au.get('gpu'), commit=au.get('commit'),
                           source=str(path), source_sha256=sha(path), directions_sha256=dirs,
                           arms={x['arm']: {kk: x.get(kk) for kk in KEEP} for x in au['arms']})
    out = HERE / 'checks/comparators.json'
    out.write_text(json.dumps(table, indent=2) + '\n')
    for k, v in table.items():
        print(k, v['job_id'], v.get('gpu'), 'arms', len(v['arms']), 'dirs', (v.get('directions_sha256') or '')[:12])
    print('WROTE', out)


if __name__ == '__main__':
    main()
