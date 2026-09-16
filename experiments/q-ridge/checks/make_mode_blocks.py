"""Record the R3 test-mode blocks: widths, eigenvalue ranges, orthonormality.

Pure NumPy, no GPU. Establishes that the common held-out block is strictly above every
in-space block any arm of either job solves against, so the falsification comparison grades
$q=0$ and $q=16$ on identical modes that neither ever saw.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r3 as R3                                                    # noqa: E402

L = 256
BLOCKS = [(0, 64, 'in-space M=64 (q=0, m4)'), (0, 128, 'in-space M=128 (q=16, m4)'),
          (0, 256, 'in-space M=256'), (0, 320, 'in-space M=320 (q=64, m4)'),
          (0, 512, 'in-space M=512'), (0, 544, 'in-space M=544 (q=256, m2)'),
          (0, 640, 'in-space M=640'), (0, 1088, 'in-space M=1088 (q=256, m4)'),
          (0, 1280, 'in-space M=1280 (largest in either job)'),
          (1536, 512, 'COMMON held-out block')]


def main():
    rows = []
    for skip, count, tag in BLOCKS:
        phi, lam, _ = R3.modes(L, count, skip)
        rows.append(dict(label=tag, skip=skip, count=count, columns=int(phi.shape[1]),
                         eigenvalue_min=float(lam.min()), eigenvalue_max=float(lam.max()),
                         orthonormality_deviation=float(
                             np.abs(phi.T @ phi - np.eye(phi.shape[1])).max())))
    common = rows[-1]
    in_space_max = max(r['eigenvalue_max'] for r in rows[:-1])
    out = dict(intervals=L, blocks=rows,
               common_block_strictly_above_every_in_space_block=bool(
                   common['eigenvalue_min'] > in_space_max),
               largest_in_space_eigenvalue=in_space_max)
    Path(__file__).with_name('mode-blocks.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'blocks'}, indent=2))


if __name__ == '__main__':
    main()
