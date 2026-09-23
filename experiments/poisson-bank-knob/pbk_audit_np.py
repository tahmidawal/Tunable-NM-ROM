"""Independent NumPy/SciPy check of a poisson-bank-knob run: recompute every recorded error from
the saved fields against a truth built here (own source formula via the parent sampler, own DST-I
solve by a dense sine matrix, not scipy.fft), re-hash the fields, and run two negative controls
that MUST fail on real data (swapped-case truth; a perturbed recorded error).

    python pbk_audit_np.py runs/<attempt> [--delete-fields]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import core as C   # only for the source sampler / source formula (inputs), not the solve


def truth_np(param, n):
    F = C.full_source(n, param)[1:-1, 1:-1]
    p = np.arange(1, n)
    S = np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(p, p) / n)        # orthonormal DST-I matrix
    lam1 = 4.0 * n * n * np.sin(np.pi * p / (2 * n)) ** 2
    lam = lam1[:, None] + lam1[None, :]
    return np.pad(S @ ((S @ F @ S) / lam) @ S, 1)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main(run, delete=False):
    run = Path(run)
    R = json.loads((run / 'result.json').read_text())
    n = R['intervals']
    dev = np.asarray(R['cohort']['parameters'])
    truths = [truth_np(p, n) for p in dev]
    first = {}
    for x in R['invocations']:
        first.setdefault((x['name'], x['case']), x)
    worst_dev, count, hash_ok = 0.0, 0, True
    recomputed = {}
    for (name, case), x in first.items():
        f = np.load(run / 'fields' / f'{name}_case{case}.npy')
        hash_ok &= hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() == x['field_sha256']
        e = rel(f, truths[case])
        recomputed[f'{name}|{case}'] = e
        worst_dev = max(worst_dev, abs(e - x['same_grid_error']) / e)
        count += 1
    # every repetition must have produced the identical field
    deterministic = all(x['field_sha256'] == first[(x['name'], x['case'])]['field_sha256'] for x in R['invocations'])
    # controls that must FAIL
    name0 = R['arms'][0]['name']
    f0 = np.load(run / 'fields' / f'{name0}_case0.npy')
    swapped = rel(f0, truths[1])
    ctrl_swap_detected = abs(swapped - first[(name0, 0)]['same_grid_error']) / swapped > 1e-6
    ctrl_perturb_detected = abs(recomputed[f'{name0}|0'] * (1 + 1e-4) - recomputed[f'{name0}|0']) / recomputed[f'{name0}|0'] > 1e-8
    ok = bool(worst_dev <= 1e-8 and hash_ok and deterministic and ctrl_swap_detected and ctrl_perturb_detected)
    audit = dict(verdict='PASS' if ok else 'FAIL', recomputed_errors=count,
                 worst_relative_deviation_of_recorded_error=worst_dev, field_hashes_match=bool(hash_ok),
                 all_repetitions_identical=bool(deterministic),
                 control_swapped_case_detected=bool(ctrl_swap_detected), control_swapped_case_error=swapped,
                 control_perturbed_error_detected=bool(ctrl_perturb_detected),
                 truth='dense orthonormal DST-I matrix solve in NumPy (independent of scipy.fft and of the JAX path)',
                 recomputed=recomputed)
    audit['summary'] = (f"{count} errors recomputed, worst relative deviation {worst_dev:.1e} (bar 1e-8), hashes "
                        f"{'match' if hash_ok else 'MISMATCH'}, repetitions identical: {deterministic}; controls detected: "
                        f"swapped-case {ctrl_swap_detected}, perturbed-error {ctrl_perturb_detected}")
    (run / 'audit.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(audit['verdict'], audit['summary'])
    if delete and ok:
        for f in (run / 'fields').glob('*.npy'):
            f.unlink()
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], '--delete-fields' in sys.argv) else 1)
