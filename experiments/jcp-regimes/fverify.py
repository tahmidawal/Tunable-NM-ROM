"""jcp-regimes: on-cluster full-field independent decode check (code audit 1, fix 7). CPU / NumPy only.

    python fverify.py --config configs/<x>.json --out <dir>       config: {"source": "qacc_<stem>"}
For every R_<setting>_<arm>.npz / F_<setting>_<arm>.npz pair in the job's output/<source>: the F file's sha256 equals the
R file's recorded f257_sha256, its case ids equal R's, and the independent NumPy decode (npcheck.decode257) of every
production coefficient vector reproduces the in-job lattice field (max abs diff / max abs field, per case) -- the full
field, not only the probe. Writes <out>/fverify.json (small; pulled).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import npcheck as NC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    src = Path(a.out).parent / cfg['source']
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for rp in sorted(src.glob('R_*.npz')):
        fp = src / ('F_' + rp.name[2:])
        R = np.load(rp)
        Fz = np.load(fp)
        F = Fz['f257']
        Rp = int(R['Rp'])
        sha_ok = hashlib.sha256(np.ascontiguousarray(F).tobytes()).hexdigest() == str(R['f257_sha256'])
        ids_ok = [str(x) for x in Fz['case_ids']] == [str(x) for x in R['case_ids']]
        rel = []
        for j in range(len(R['W'])):
            if not np.isfinite(R['W'][j]).all():
                rel.append(None)
                continue
            g = NC.decode257(R['W'][j], Rp)
            rel.append(float(np.max(np.abs(g - F[j])) / max(float(np.max(np.abs(F[j]))), 1e-300)))
        fin = [x for x in rel if x is not None]
        rows[rp.stem[2:]] = dict(sha_ok=sha_ok, ids_ok=ids_ok, n=len(rel), n_nonfinite=sum(x is None for x in rel),
                                 max_rel=max(fin) if fin else None, passed=bool(sha_ok and ids_ok and fin and max(fin) <= 1e-12))
        print('FVERIFY', rp.stem, rows[rp.stem[2:]], flush=True)
    res = dict(source=cfg['source'], arms=rows, all_pass=bool(rows) and all(v['passed'] for v in rows.values()))
    (out / 'fverify.json').write_text(json.dumps(res, indent=1) + '\n')
    print('FVERIFY ALL', res['all_pass'], flush=True)


if __name__ == '__main__':
    main()
