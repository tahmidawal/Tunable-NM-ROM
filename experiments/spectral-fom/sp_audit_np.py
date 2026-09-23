"""Independent NumPy audit of a spectral-fom run (runs on the cluster before the fields are deleted).

Recomputes every recorded same-grid error from the saved fields against a truth built HERE by a
different method (dense orthonormal sine matrix in NumPy, not an FFT), re-hashes every field,
checks determinism, and runs two negative controls that MUST be detected on the real data:
a truth taken from a swapped case, and a recorded error perturbed by 1e-4 relative.
Agreement rule: |recomputed - recorded| <= 1e-8 * recorded + 1e-13 (the absolute floor matters only
for the spectral subjects, whose errors are round-off).

    python sp_audit_np.py <output dir> [--delete-fields]
Writes <output>/audit.json and <output>/sub/<name>_case<k>.npy (every field subsampled to <= 257 points
per axis, kept for local re-audit).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sine(n):
    p = np.arange(1, n)
    return np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(p, p) / n), 4.0 * n * n * np.sin(np.pi * p / (2 * n)) ** 2


def apply(S, x):
    if x.ndim == 2:
        return S @ x @ S
    x = np.einsum('ai,ijk->ajk', S, x, optimize=True)
    x = np.einsum('bj,ajk->abk', S, x, optimize=True)
    return np.einsum('ck,abk->abc', S, x, optimize=True)


def poisson_truth(F, n):
    S, l = sine(n)
    lam = l[:, None] + l[None, :] if F.ndim == 2 else l[:, None, None] + l[None, :, None] + l[None, None, :]
    return np.pad(apply(S, apply(S, F[(slice(1, -1),) * F.ndim]) / lam), 1)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def truths_poisson2d(R):
    import core as C          # source formula only (the input), not a solver
    n = R['intervals']
    return [poisson_truth(C.full_source(n, p), n) for p in np.asarray(R['cohort']['parameters'])]


def truths_poisson3d(R):
    import poisson as PP      # source formula only (the input), not a solver
    n = R['intervals']
    S, l = sine(n)
    lam = l[:, None, None] + l[None, :, None] + l[None, None, :]
    return [apply(S, apply(S, np.asarray(PP.source(n, p))) / lam) for p in np.asarray(R['cohort']['parameters'])]


TRUTHS = {'poisson2d': truths_poisson2d, 'poisson3d': truths_poisson3d}


def main(run, delete=False):
    run = Path(run)
    R = json.loads((run / 'result.json').read_text())
    truths = TRUTHS[R['problem']](R)
    inv = R['invocations']
    first = {}
    for x in inv:
        first.setdefault((x['name'], x['case']), x)
    worst, hash_ok, rows = 0.0, True, []
    (run / 'sub').mkdir(exist_ok=True)
    for (name, case), x in sorted(first.items()):
        f = np.load(run / 'fields' / f'{name}_case{case}.npy')
        hash_ok &= hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() == x['field_sha256']
        e = rel(f, truths[case])
        d = abs(e - x['same_grid_error'])
        ok = d <= 1e-8 * x['same_grid_error'] + 1e-13
        worst = max(worst, d / (1e-8 * x['same_grid_error'] + 1e-13))
        rows.append(dict(name=name, case=case, recorded=x['same_grid_error'], recomputed=e, agrees=bool(ok)))
        stride = max(1, (f.shape[0] - 1) // 256)
        np.save(run / 'sub' / f'{name}_case{case}.npy', f[(slice(None, None, stride),) * f.ndim])
    deterministic = all(x['field_sha256'] == first[(x['name'], x['case'])]['field_sha256'] for x in inv)
    # negative controls on the real data (the first ROM subject, case 0)
    name0 = sorted({k[0] for k in first if not k[0].startswith(('dst', 'spec', 'modal'))})[0]
    f0 = np.load(run / 'fields' / f'{name0}_case0.npy')
    rec0 = first[(name0, 0)]['same_grid_error']
    swapped = rel(f0, truths[1])
    ctrl_swap = abs(swapped - rec0) > 1e-8 * rec0 + 1e-13
    pert = rec0 * (1 + 1e-4)
    ctrl_pert = abs(rel(f0, truths[0]) - pert) > 1e-8 * pert + 1e-13
    ok = bool(all(r['agrees'] for r in rows) and hash_ok and deterministic and ctrl_swap and ctrl_pert)
    audit = dict(verdict='PASS' if ok else 'FAIL', recomputed=len(rows), worst_deviation_over_tolerance=worst,
                 field_hashes_match=bool(hash_ok), deterministic=bool(deterministic),
                 control_swapped_truth_detected=bool(ctrl_swap), control_perturbed_error_detected=bool(ctrl_pert),
                 control_subject=name0, truth_method='NumPy dense orthonormal sine matrix (not FFT)', rows=rows)
    (run / 'audit.json').write_text(json.dumps(audit, indent=1) + '\n')
    print('AUDIT', audit['verdict'], len(rows), worst, flush=True)
    if delete:
        shutil.rmtree(run / 'fields')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1], '--delete-fields' in sys.argv)
