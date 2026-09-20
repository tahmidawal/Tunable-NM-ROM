"""Independent NumPy audit of a Phase-1 result: no JAX, no lane modules.

Rebuilds every saved bank (coordinate network re-implemented here in NumPy; POD modes
loaded) and recomputes the primary held-out cohort's floors from the saved fields with
NumPy QR. usage: audit_rep_np.py RESULT.json CKPT_DIR OUT.json
"""
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''):
            h.update(c)
    return h.hexdigest()


def silu(x):
    return x / (1.0 + np.exp(-x))


def block_features(b, xy):
    ang = 2.0 * np.pi * (xy @ np.asarray(b['B']))
    x = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    g = b['g']
    for w, c in g[:-1]:
        x = silu(x @ np.asarray(w) + np.asarray(c))
    x = x @ np.asarray(g[-1][0]) + np.asarray(g[-1][1])
    bc = 16.0 * xy[:, 0] * (1 - xy[:, 0]) * xy[:, 1] * (1 - xy[:, 1])
    return (float(b['out_scale']) * bc)[:, None] * x


def floors(Q, U):
    E = U - (U @ Q) @ Q.T
    return np.sqrt((E * E).sum(1) / (U * U).sum(1))


def main():
    res = json.loads(Path(sys.argv[1]).read_text())
    ck = Path(sys.argv[2])
    cfg = res['config']
    primary = 'dev6' if cfg['pde'] == 'burgers2d' else 'dev12'
    U = np.load(ck / f'{cfg["pde"]}_{primary}_fields.npz')['U']
    iv = cfg['intervals']
    p = np.arange(1, iv) / iv
    xx, yy = np.meshgrid(p, p, indexing='ij')
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    rows, ok = {}, True
    for tag, arm in res['arms'].items():
        c = arm.get('checkpoint')
        if not c:
            continue
        path = ck / c['file']
        hash_ok = sha_file(path) == c['sha256']
        if path.suffix == '.npy':
            Qall = np.load(path)
            prefix = re.match(r'([a-z_]+)\d+$', tag).group(1)
            todo = {t: Qall[:, :a['basis']['columns']] for t, a in res['arms'].items()
                    if re.fullmatch(prefix + r'\d+', t)}
        else:
            blocks = pickle.load(open(path, 'rb'))['blocks']
            G = np.concatenate([block_features(b, xy) for b in blocks], axis=1)
            Q, _ = np.linalg.qr(G)
            todo = {tag: Q}
        for t, Q in todo.items():
            e = floors(Q, U)
            ref = res['arms'][t]['floors'][primary]
            d = abs(e.max() / ref['worst'] - 1)
            good = bool(hash_ok and d < 1e-6)
            ok &= good
            rows[t] = dict(numpy_worst=float(e.max()), reported_worst=ref['worst'],
                           relative_difference=float(d), file_hash_ok=bool(hash_ok), passed=good)
            print(t, rows[t], flush=True)
    out = dict(result=sys.argv[1], result_sha256=sha_file(sys.argv[1]), cohort=primary,
               tolerance=1e-6, arms=rows, passed=bool(ok and rows))
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + '\n')
    print('AUDIT', 'PASS' if out['passed'] else 'FAIL')


if __name__ == '__main__':
    main()
