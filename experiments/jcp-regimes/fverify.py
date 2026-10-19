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

import sys

import numpy as np

import guard as G
import npcheck as NC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    src = Path(a.out).parent / G.check_relative(cfg['source'])
    out = Path(a.out)
    G.check_write(out / 'fverify.json')
    out.mkdir(parents=True, exist_ok=True)
    want = sorted(cfg['arms']) if 'arms' in cfg else None
    rows = {}
    for rp in sorted(src.glob('R_*.npz')):
        fp = src / ('F_' + rp.name[2:])
        R = np.load(rp)
        Fz = np.load(fp)
        F = Fz['f257']
        Rp = int(R['Rp'])
        sha_ok = hashlib.sha256(np.ascontiguousarray(F).tobytes()).hexdigest() == str(R['f257_sha256'])
        ids_ok = [str(x) for x in Fz['case_ids']] == [str(x) for x in R['case_ids']]
        shape_ok = F.shape == (len(R['W']), 6, 257, 257) and R['W'].shape[1:] == (6, Rp)
        rel, invalid_runs, bad = [], [], []
        for j in range(len(R['W'])):
            if not bool(R['valid'][j]):                  # an invalid production run is counted, not compared
                invalid_runs.append(j)
                continue
            if not (np.isfinite(R['W'][j]).all() and np.isfinite(F[j]).all()):
                bad.append(j)                            # a valid run must have finite coefficients and field
                continue
            g = NC.decode257(R['W'][j], Rp)
            r_ = float(np.max(np.abs(g - F[j])) / max(float(np.max(np.abs(F[j]))), 1e-300))
            if not np.isfinite(r_):
                bad.append(j)
            rel.append(r_)
        ok = bool(sha_ok and ids_ok and shape_ok and not bad and rel and np.all(np.isfinite(rel)) and max(rel) <= 1e-12)
        rows[rp.stem[2:]] = dict(R_file_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(), f257_sha256=str(R['f257_sha256']),
                                 sha_ok=sha_ok, ids_ok=ids_ok, shape_ok=shape_ok, n=len(R['W']), n_compared=len(rel),
                                 invalid_runs=invalid_runs, nonfinite_valid=bad,
                                 max_rel=(max(rel) if rel and np.all(np.isfinite(rel)) else None), passed=ok)
        print('FVERIFY', rp.stem, rows[rp.stem[2:]], flush=True)
    inv_ok = want is None or sorted(rows) == want
    res = dict(source=cfg['source'], config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), arms=rows, inventory_ok=inv_ok,
               all_pass=bool(rows) and inv_ok and all(v['passed'] for v in rows.values()))
    G.check_write(out / 'fverify.json').write_text(json.dumps(res, indent=1) + '\n')
    print('FVERIFY ALL', res['all_pass'], flush=True)
    sys.exit(0 if res['all_pass'] else 3)


if __name__ == '__main__':
    main()
