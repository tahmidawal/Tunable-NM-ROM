"""jcp-dims slim archives (COMMON-BRIEF 16): before a collected job's raw archive is offloaded to the paralab
burgers-archive, keep locally only what the cached analysis still reads: 2D R_*.npz -> slim copies without the 257^2
fields, internal states and per-step certificate arrays (their offline checks are already cached in checks/cache2d,
keyed by the raw file's sha256, which the slim file records); 3D W_*.npz -> without the 'wall|' internal-state keys
(only used in-job). result.json, logs and manifests are kept. Writes runs/<att>/slim/... and SLIM.sha256.

    python slim.py <attempt>
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEEP2D = ('W', 'W_tight', 'W_tighter', 'valid', 'valid_tight', 'valid_tighter', 'rho', 'case_ids', 'dev_case_ids', 'Rp', 'M',
          'scheme', 'form', 'dt', 'policy', 'mode_hash', 'steps', 'n0', 'rule', 'qp_rule', 'rho_only', 'square', 'kx', 'ky', 'tol',
          'factor', 'mesh', 'time_tag', 'rule_sha256', 'W_qp', 'valid_qp', 'W_qpp', 'valid_qpp', 'q2_case_ids')


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    att = sys.argv[1]
    arch = HERE / 'runs' / att / 'archive'
    slim = HERE / 'runs' / att / 'slim'
    lines = []
    for p in sorted(arch.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(arch)
        dst = slim / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == '.npz' and p.name.startswith('R_'):
            z = np.load(p)
            np.savez_compressed(dst, raw_sha256=sha_file(p), **{k: z[k] for k in KEEP2D if k in z.files})
        elif p.suffix == '.npz' and p.name.startswith('W_'):
            z = np.load(p)
            np.savez_compressed(dst, **{k: z[k] for k in z.files if not k.startswith('wall|')})
        elif p.stat().st_size < 100e6:
            shutil.copy2(p, dst)
        else:
            continue
        lines.append(f'{sha_file(dst)}  {rel}')
    (slim / 'SLIM.sha256').write_text('\n'.join(lines) + '\n')
    print(att, 'slim files', len(lines))


if __name__ == '__main__':
    main()
