"""Independent NumPy/SciPy audit of one poisson-bank-knob-3d run (runs on the cluster before the
fields are deleted).

    python pbk3_audit_np.py <output dir> [--delete-fields]

For every (subject, case) it re-hashes the saved field, checks every repetition produced the same
field, and recomputes the same-grid error against a truth built HERE by an independent route:
  * L-shape: source from its own NumPy formula; operator assembled as a principal submatrix of the
    square's Kronecker-sum Laplacian (not the driver's stencil loop); SuperLU solve of that matrix.
  * cube:    source from its own NumPy formula; solve by a dense orthonormal sine matrix applied
    along each axis (not scipy.fft, not the JAX DST).
It also recomputes ONE linear-rung field end to end in NumPy (bank from a NumPy MLP, rotation,
weak tests, least squares, decode) on small meshes, and runs controls that MUST fail on real data:
a swapped-case truth, a perturbed recorded error, and the linear-rung recompute at the wrong R'.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def err_ok(recorded, recomputed):
    return abs(recorded - recomputed) <= max(1e-8 * recomputed, 1e-9)


# ------------------------------------------------------------------ L-shape
def lshape_mask(N):
    m = np.zeros((N + 1, N + 1), dtype=bool)
    m[1:N, 1:N] = True
    m[N // 2:, N // 2:] = False
    return m


def lshape_source(N, p):
    cx, cy, w, a = p
    x = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.where(lshape_mask(N), a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)), 0.0)


def lshape_operator(N):
    m = N - 1
    T = sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], format='csr')
    Asq = (N * N) * (sp.kron(sp.identity(m, format='csr'), T) + sp.kron(T, sp.identity(m, format='csr')))
    mask = lshape_mask(N)
    ii, jj = np.nonzero(mask)                     # lexicographic, i-major
    sel = (ii - 1) * m + (jj - 1)
    return Asq.tocsr()[sel][:, sel].tocsc(), mask


def lshape_truths(N, params):
    A, mask = lshape_operator(N)
    lu = spla.splu(A)
    out = []
    for p in params:
        f = lshape_source(N, p)[mask]
        u = lu.solve(f)
        full = np.zeros((N + 1, N + 1))
        full[mask] = u
        out.append(full)
    return out, (A, mask, lu)


def lshape_features_np(params, xy):
    def mlp(layers, x):
        for w, b in layers[:-1]:
            x = x @ np.asarray(w) + np.asarray(b)
            x = x / (1.0 + np.exp(-x))
        w, b = layers[-1]
        return x @ np.asarray(w) + np.asarray(b)
    x, y = xy[:, 0], xy[:, 1]
    right = np.where(y <= 0.5, 1.0 - x, 0.5 - x)
    top = np.where(x <= 0.5, 1.0 - y, 0.5 - y)
    corner = np.sqrt((0.5 - x) ** 2 + (0.5 - y) ** 2)
    d = 4.0 * np.minimum(np.minimum(np.minimum(x, y), np.minimum(right, top)), corner)
    ang = 2.0 * np.pi * (xy @ np.asarray(params['B']))
    ff = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    return float(params['out_scale']) * d[:, None] * mlp(params['g'], ff)


def lshape_linear_np(R, run_dir, N, Rp, case, sys_):
    """NumPy recompute of the R' linear rung on case `case` (eigen tests by SciPy on the audit's
    own operator)."""
    A, mask, lu = sys_
    with open(run_dir.parent / 'code' / 'head_sdf_R512_K16.pkl', 'rb') as fh:
        d = pickle.load(fh)
    params = d['params']
    prep = np.load(run_dir.parent / 'code' / R['config']['prep'])
    T = prep['T']
    ii, jj = np.nonzero(mask)
    G = lshape_features_np(params, np.column_stack((ii / N, jj / N)))
    M = int(R['config']['requested_modes'])
    n = A.shape[0]
    op = spla.LinearOperator((n, n), matvec=lu.solve, dtype=float)
    lam, phi = spla.eigsh(A, k=M, sigma=0.0, which='LM', OPinv=op,
                          v0=np.random.default_rng(1).standard_normal(n), tol=0)
    Bt = phi.T @ (G @ T[:, :Rp])                                  # Lambda^{-1} Phi^T A G T = Phi^T G T
    f = lshape_source(N, R['cohort']['parameters'][case])[mask]
    target = (phi.T @ f) / lam
    a, *_ = np.linalg.lstsq(Bt, target, rcond=None)
    full = np.zeros((N + 1, N + 1))
    full[mask] = G @ T[:, :Rp] @ a
    return full


# ------------------------------------------------------------------ cube
def sine_matrix(n):
    k = np.arange(1, n)
    return np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(k, k) / n)


def apply3(S, u):
    u = np.tensordot(S, u, axes=(1, 0))
    u = np.tensordot(S, u, axes=(1, 1)).transpose(1, 0, 2)
    return np.tensordot(u, S, axes=(2, 1))


def cube_source(n, p):
    a = np.arange(1, n) / n
    X, Y, Z = np.meshgrid(a, a, a, indexing='ij')
    mask = 64 * X * (1 - X) * Y * (1 - Y) * Z * (1 - Z)
    return p[4] * mask * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2) / (2 * p[3] ** 2))


def cube_truths(n, params):
    S = sine_matrix(n)
    l1 = 4.0 * n * n * np.sin(np.pi * np.arange(1, n) / (2 * n)) ** 2
    lam = l1[:, None, None] + l1[None, :, None] + l1[None, None, :]
    return [apply3(S, apply3(S, cube_source(n, p)) / lam) for p in params]


def cube_linear_np(R, run_dir, n, Rp, case):
    code = run_dir.parent / 'code'
    bankck = pickle.loads((code / 'bank.pkl').read_bytes())
    prep = np.load(code / R['config']['prep'])
    T = prep['T']
    p = bankck['params']
    a = np.arange(1, n) / n
    x = np.stack(np.meshgrid(a, a, a, indexing='ij'), -1).reshape(-1, 3)
    ang = 2 * np.pi * (x @ np.asarray(p['freq']))
    h = np.concatenate((np.sin(ang), np.cos(ang)), -1)
    for w, b in p['net'][:-1]:
        h = h @ np.asarray(w) + np.asarray(b)
        h = h / (1.0 + np.exp(-h))
    w, b = p['net'][-1]
    h = h @ np.asarray(w) + np.asarray(b)
    G = (float(p['scale']) * 64 * np.prod(x * (1 - x), axis=1))[:, None] * h
    G = G @ np.asarray(bankck['rotation']) @ T[:, :Rp]
    # the M lowest sine modes in the harness's order (sum of squared indices, stable)
    k = np.arange(1, n)
    tri = np.stack(np.meshgrid(k, k, k, indexing='ij'), -1).reshape(-1, 3)
    tri = tri[np.argsort(np.sum(tri ** 2, axis=1), kind='stable')[:int(R['config']['weak_tests'])]]
    S = sine_matrix(n)                                             # S[k-1, i-1] = sqrt(2/n) sin(pi k i / n)
    lam = np.sum(4 * n * n * np.sin(np.pi * tri / (2 * n)) ** 2, axis=1)
    Phi = S[tri[:, 0] - 1][:, :, None, None] * S[tri[:, 1] - 1][:, None, :, None] * S[tri[:, 2] - 1][:, None, None, :]
    Phi = Phi.reshape(len(tri), -1)                                # orthonormal discrete sine vectors
    Bt = Phi @ G
    f = cube_source(n, R['cohort']['parameters'][case]).ravel()
    target = (Phi @ f) / lam
    coef, *_ = np.linalg.lstsq(Bt, target, rcond=None)
    return (G @ coef).reshape((n - 1,) * 3)


# ------------------------------------------------------------------ main
def main(run, delete=False):
    run = Path(run)
    R = json.loads((run / 'result.json').read_text())
    problem, n = R['problem'], R['intervals']
    params = np.asarray(R['cohort']['parameters'])
    if problem == 'lshape':
        truths, sys_ = lshape_truths(n, params)
    else:
        truths, sys_ = cube_truths(n, params), None
    ref_dev = max(rel(np.load(run / 'fields' / f'reference_same_case{c}.npy'), truths[c]) for c in range(len(truths)))
    allinv = R['invocations'] + R['slow_invocations'] + R['neighbour']
    first = {}
    for x in allinv:
        first.setdefault((x['name'], x['case']), x)
    worst_dev, count, hash_ok, recomputed, bad = 0.0, 0, True, {}, []
    for (name, case), x in first.items():
        f = np.load(run / 'fields' / f'{name}_case{case}.npy')
        hash_ok &= sha(f) == x['field_sha256']
        e = rel(f, truths[case])
        recomputed[f'{name}|{case}'] = e
        worst_dev = max(worst_dev, abs(e - x['same_grid_error']) / max(e, 1e-300))
        if not err_ok(x['same_grid_error'], e):
            bad.append(f'{name}|{case}')
        count += 1
    deterministic = all(x['field_sha256'] == first[(x['name'], x['case'])]['field_sha256'] for x in allinv)
    # every recorded invocation error equals the recomputed error of its (identical) field
    rows_ok = all(err_ok(x['same_grid_error'], recomputed[f"{x['name']}|{x['case']}"]) for x in allinv)
    # controls that must FAIL
    name0 = R['arms'][0]['name']
    f0 = np.load(run / 'fields' / f'{name0}_case0.npy')
    swapped = rel(f0, truths[1])
    ctrl_swap = not err_ok(first[(name0, 0)]['same_grid_error'], swapped)
    ctrl_perturb = not err_ok(first[(name0, 0)]['same_grid_error'] * (1 + 1e-5), recomputed[f'{name0}|0'])
    # independent end-to-end linear rung (small meshes)
    rung = None
    lim = {'lshape': 512, 'cube': 64}[problem]
    lin = sorted([a['Rp'] for a in R['arms'] if a['kind'] == 'linear'])
    if n <= lim and len(lin) >= 2:
        Rp = lin[len(lin) // 2]
        case = 0
        if problem == 'lshape':
            f_np = lshape_linear_np(R, run, n, Rp, case, sys_)
            f_wrong = lshape_linear_np(R, run, n, lin[len(lin) // 2 - 1], case, sys_)
        else:
            f_np = cube_linear_np(R, run, n, Rp, case)
            f_wrong = cube_linear_np(R, run, n, lin[len(lin) // 2 - 1], case)
        saved = np.load(run / 'fields' / f'R{Rp}_linear_case{case}.npy')
        d_ok, d_wrong = rel(saved, f_np), rel(saved, f_wrong)
        limit = 1e-6
        rung = dict(Rp=int(Rp), case=case, relative_difference=d_ok, limit=limit, passed=bool(d_ok <= limit),
                    control_wrong_Rp=int(lin[len(lin) // 2 - 1]), control_relative_difference=d_wrong,
                    control_detected=bool(d_wrong > limit))
    ok = bool(ref_dev <= 1e-8 and not bad and rows_ok and hash_ok and deterministic and ctrl_swap and ctrl_perturb
              and (rung is None or (rung['passed'] and rung['control_detected'])))
    audit = dict(verdict='PASS' if ok else 'FAIL', problem=problem, intervals=n, recomputed_errors=count,
                 driver_reference_vs_audit_truth_worst=ref_dev,
                 worst_relative_deviation_of_recorded_error=worst_dev, mismatched=bad, all_rows_match=bool(rows_ok),
                 field_hashes_match=bool(hash_ok), all_repetitions_identical=bool(deterministic),
                 control_swapped_case_detected=bool(ctrl_swap), control_swapped_case_error=swapped,
                 control_perturbed_error_detected=bool(ctrl_perturb), linear_rung_numpy=rung,
                 truth=('L-shape: own NumPy source, Kronecker principal-submatrix operator, SuperLU'
                        if problem == 'lshape' else 'cube: own NumPy source, dense orthonormal sine-matrix solve'),
                 recomputed=recomputed)
    audit['summary'] = (f"{count} errors recomputed, worst relative deviation {worst_dev:.1e}, reference vs audit "
                        f"truth {ref_dev:.1e}, hashes {'match' if hash_ok else 'MISMATCH'}, repetitions identical: "
                        f"{deterministic}; controls detected: swapped-case {ctrl_swap}, perturbed-error {ctrl_perturb}"
                        + (f"; NumPy linear rung R'={rung['Rp']}: {rung['relative_difference']:.1e} "
                           f"(wrong-R' control {rung['control_relative_difference']:.1e})" if rung else ''))
    (run / 'audit.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(audit['verdict'], audit['summary'], flush=True)
    if delete and ok:
        for f in (run / 'fields').glob('*.npy'):
            f.unlink()
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], '--delete-fields' in sys.argv) else 1)
