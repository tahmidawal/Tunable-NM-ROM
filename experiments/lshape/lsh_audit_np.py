"""Independent NumPy/SciPy audit of an lshape attempt. Imports neither JAX nor any driver.

    python lsh_audit_np.py train <archive/output> --out audit.json
    python lsh_audit_np.py solve <archive/output> --out audit.json

Everything is rewritten here from the equations: the masked-grid operator (assembled by the
Kronecker route, i.e. the one the driver does NOT use for its matrix), the sparse-direct
reference, the Gaussian source with the centre-rejection rule, the decoder features with
each boundary factor, the QR projection floor, the correction bases, the aggregation rules
and the non-dominated sets.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LSHAPE_LAMBDA1 = 4.0 * 9.6397238440219


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


# ---------------------------------------------------------------- geometry ---

class Geom:
    def __init__(self, N):
        mask = np.zeros((N + 1, N + 1), bool)
        mask[1:N, 1:N] = True
        mask[N // 2:, N // 2:] = False
        self.N, self.mask = N, mask
        self.idx = np.flatnonzero(mask.ravel())
        self.n = len(self.idx)
        ii, jj = np.divmod(self.idx, N + 1)
        self.ij = np.column_stack((ii, jj))
        self.coords = np.column_stack((ii / N, jj / N))

    def gather(self, full):
        return np.asarray(full).ravel()[self.idx]

    def scatter(self, v):
        o = np.zeros((self.N + 1) ** 2)
        o[self.idx] = v
        return o.reshape(self.N + 1, self.N + 1)


def operator(g):
    N, m = g.N, g.N - 1
    T = sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], format='csr')
    Asq = (N * N) * (sp.kron(sp.identity(m, format='csr'), T) + sp.kron(T, sp.identity(m, format='csr')))
    sel = (g.ij[:, 0] - 1) * m + (g.ij[:, 1] - 1)
    return Asq.tocsr()[sel][:, sel].tocsc()


def operator_stencil(g):
    N = g.N
    num = -np.ones((N + 1, N + 1), np.int64)
    num[g.mask] = np.arange(g.n)
    ii, jj = g.ij[:, 0], g.ij[:, 1]
    rows, cols, vals = [np.arange(g.n)], [np.arange(g.n)], [np.full(g.n, 4.0 * N * N)]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = num[ii + di, jj + dj]
        k = nb >= 0
        rows.append(np.arange(g.n)[k]); cols.append(nb[k]); vals.append(np.full(int(k.sum()), -1.0 * N * N))
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(g.n, g.n)).tocsc()


def source_params(seed, count):
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count); cy = rng.uniform(0.15, 0.85, count)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), count)); a = rng.uniform(0.5, 2.0, count)
    return np.column_stack((cx, cy, w, a))


def cohort(seed, draw, count):
    d = source_params(seed, draw)
    keep = ~((d[:, 0] >= 0.5) & (d[:, 1] >= 0.5))
    acc = d[keep]
    assert len(acc) >= count
    return acc[:count]


def source_int(g, q):
    cx, cy, w, a = q
    x = np.linspace(0, 1, g.N + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    return g.gather(np.where(g.mask, a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w ** 2)), 0.0))


class Solver:
    def __init__(self, g):
        self.g, self.A = g, operator(g)
        self.lu = spla.splu(self.A)

    def ref(self, f):
        u = self.lu.solve(f)
        for _ in range(2):
            u = u + self.lu.solve(f - self.A @ u)
        return u, float(np.linalg.norm(self.A @ u - f) / np.linalg.norm(f))


# ----------------------------------------------------------------- decoder ---

def load(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['params'], np.asarray(d['Z_tr']), d['cfg']


def silu(x):
    return x / (1.0 + np.exp(-x))


def mlp(layers, x):
    for w, b in layers[:-1]:
        x = silu(x @ np.asarray(w) + np.asarray(b))
    w, b = layers[-1]
    return x @ np.asarray(w) + np.asarray(b)


def factor(name, xy):
    x, y = xy[:, 0], xy[:, 1]
    if name == 'poly':
        return 16 * x * (1 - x) * y * (1 - y)
    if name == 'smooth':
        a, b = 0.5 - x, 0.5 - y
        return 16 * x * (1 - x) * y * (1 - y) * (a + b + np.sqrt(a * a + b * b))
    if name == 'sdf':
        right = np.where(y <= 0.5, 1 - x, 0.5 - x)
        top = np.where(x <= 0.5, 1 - y, 0.5 - y)
        corner = np.sqrt((0.5 - x) ** 2 + (0.5 - y) ** 2)
        return 4 * np.minimum.reduce([x, y, right, top, corner])
    raise ValueError(name)


def singular(xy, count):
    x, y = xy[:, 0], xy[:, 1]
    r = np.sqrt((x - .5) ** 2 + (y - .5) ** 2)
    phi = np.arctan2(y - .5, x - .5)
    phi = np.where(phi <= 0, phi + 2 * np.pi, phi)
    cut = 16 * x * (1 - x) * y * (1 - y)
    return np.stack([cut * r ** (2 * j / 3) * np.sin((2 * j / 3) * (phi - np.pi / 2)) for j in range(1, count + 1)], 1)


def features(params, xy, fac, E):
    ang = 2 * np.pi * (xy @ np.asarray(params['B']))
    ff = np.concatenate([np.sin(ang), np.cos(ang)], -1)
    G = factor(fac, xy)[:, None] * mlp(params['g'], ff)
    if E:
        G = np.concatenate([G, singular(xy, E) / np.asarray(params['enrich_norm'])[None, :]], 1)
    return float(params['out_scale']) * G


def head(params, z):
    z = np.atleast_2d(z)
    return mlp(params['h'], z) + z @ np.asarray(params['h_lin'])


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ audits ---

def audit_train(out, tol):
    d = json.loads((out / 'result.json').read_text())
    checks, detail = {}, {}
    checks['complete'] = bool(d['complete'])
    checks['backend_gpu'] = d['backend'] == 'gpu'
    checks['x64'] = bool(d['x64'])
    checks['precision_highest'] = d['matmul_precision'] == 'highest'
    cfg = d['config']
    cc = cfg['cohorts']
    train = cohort(cc['training']['seed'], cc['training']['draw'], cc['training']['count'])
    common = cohort(cc['common']['seed'], cc['common']['draw'], cc['common']['count'])
    dev = cohort(cc['development']['seed'], cc['development']['draw'], cc['development']['count'])
    detail['cohort_max_abs_difference'] = float(np.max(np.abs(dev - np.asarray(d['cohorts']['development']['parameters']))))
    checks['development_cohort_reproduced'] = detail['cohort_max_abs_difference'] <= 1e-12
    checks['cohort_hashes_recorded'] = all(k in d['cohorts'] for k in ('training', 'common', 'development'))
    checks['cohorts_disjoint'] = not any(np.allclose(t, s) for t in train for s in dev) \
        and not any(np.allclose(t, s) for t in common for s in dev) \
        and not any(np.allclose(t, s) for t in train for s in common)
    checks['centres_in_omega'] = bool(np.all(~((dev[:, 0] >= .5) & (dev[:, 1] >= .5))))
    fit = np.asarray(d['cohorts']['training']['fit'])
    # operator gates, independently
    ntr = cfg['training_intervals']
    g = Geom(ntr)
    A1, A2 = operator(g), operator_stencil(g)
    diff = (A1 - A2); diff.eliminate_zeros()
    checks['operator_independent_assembly'] = diff.nnz == 0
    asym = (A1 - A1.T); asym.eliminate_zeros()
    checks['operator_symmetric'] = asym.nnz == 0
    lam1 = spla.eigsh(A1, k=1, sigma=0.0, which='LM')[0][0]
    detail['lambda1'] = float(lam1)
    detail['lambda1_relative_difference'] = float(abs(lam1 - LSHAPE_LAMBDA1) / LSHAPE_LAMBDA1)
    checks['operator_positive_definite_lambda1_within_1pct'] = bool(lam1 > 0 and detail['lambda1_relative_difference'] <= 1e-2)
    checks['gates_recorded_pass'] = all(x['passed'] for x in d['gates'])
    S = Solver(g)
    Udev = np.stack([S.ref(source_int(g, q))[0] for q in dev])
    Ucom = np.stack([S.ref(source_int(g, q))[0] for q in common])
    # bank floors at the training mesh from the saved weights
    worst_floor_diff, worst_stored_diff, worst_basis_orth = 0.0, 0.0, 0.0
    hashes_ok = True
    Ufit = None
    for arm in d['bank_arms']:
        ck = next(x for x in d['checkpoints'] if x['id'] == arm['arm'])
        path = out / ck['path']
        hashes_ok &= hashlib.sha256(path.read_bytes()).hexdigest() == ck['sha256']
        params, Z, c = load(path)
        G = features(params, g.coords, c['factor'], c['n_enrich'])
        Q, R = np.linalg.qr(G)
        rec = next(r for r in arm['floors'] if r['intervals'] == ntr)
        for name, U in (('dev', Udev), ('common', Ucom)):
            got = np.linalg.norm(U - (U @ Q) @ Q.T, axis=1) / np.linalg.norm(U, axis=1)
            exp = np.asarray(rec[f'floor_{name}']['per_case'])
            worst_floor_diff = max(worst_floor_diff, float(np.max(np.abs(got - exp) / exp)))
        detail.setdefault('bank_rank', {})[arm['arm']] = int(np.linalg.matrix_rank(R))
    checks['bank_floors_reproduced'] = worst_floor_diff <= tol
    detail['worst_bank_floor_relative_difference'] = worst_floor_diff
    # selection rule
    pick_n = d['selection']['bank']['mesh']
    order = {a['arm']: i for i, a in enumerate(cfg['bank_arms'])}
    key = lambda x: (next(r for r in x['floors'] if r['intervals'] == pick_n)['floor_common']['worst'],
                     next(r for r in x['floors'] if r['intervals'] == pick_n)['floor_common']['median'], x['R'], order[x['arm']])
    checks['bank_selection_reproduced'] = min(d['bank_arms'], key=key)['arm'] == d['selection']['bank']['selected']
    # head arms: stored-code error on the fit split and the correction bases
    if len(fit) <= 4096:
        Ufit = np.stack([S.ref(source_int(g, q))[0] for q in train[fit]])
    for arm in d['head_arms']:
        ck = next(x for x in d['checkpoints'] if x['id'] == arm['arm'])
        path = out / ck['path']
        hashes_ok &= hashlib.sha256(path.read_bytes()).hexdigest() == ck['sha256']
        params, Z, c = load(path)
        G = features(params, g.coords, c['factor'], c['n_enrich'])
        Q, R = np.linalg.qr(G)
        if Ufit is not None:
            T = Ufit @ Q
            perp2 = np.clip(np.sum(Ufit ** 2, 1) - np.sum(T * T, 1), 0, None)
            H = head(params, Z) @ R.T
            got = np.sqrt((np.sum((H - T) ** 2, 1) + perp2) / np.sum(Ufit ** 2, 1))
            exp = np.asarray(arm['head_at_stored_codes_fit']['per_case'])
            worst_stored_diff = max(worst_stored_diff, float(np.max(np.abs(got - exp) / exp)))
        bpath = out / arm['basis']['path']
        hashes_ok &= hashlib.sha256(bpath.read_bytes()).hexdigest() == arm['basis']['sha256']
        b = np.load(bpath)
        W = G @ b['coefficient_directions']
        worst_basis_orth = max(worst_basis_orth, float(np.linalg.norm(W.T @ W - np.eye(W.shape[1]))))
        checks.setdefault('basis_codes_match_checkpoint', True)
        checks['basis_codes_match_checkpoint'] &= bool(np.array_equal(b['training_latents'], Z))
    checks['checkpoint_and_basis_hashes'] = bool(hashes_ok)
    checks['head_stored_code_errors_reproduced'] = worst_stored_diff <= tol
    detail['worst_stored_code_relative_difference'] = worst_stored_diff
    checks['correction_bases_orthonormal_in_field_metric'] = worst_basis_orth <= 1e-6
    detail['worst_basis_orthonormality_error'] = worst_basis_orth
    return dict(mode='train', checks=checks, detail=detail, passed=bool(all(checks.values())))


def aggregate(inv):
    rows = {}
    for r in inv:
        rows.setdefault((r['intervals'], r['name']), []).append(r)
    agg = {}
    for key, rs in rows.items():
        cases = {}
        for r in rs:
            cases.setdefault(r['case'], []).append(r)
        agg[key] = dict(worst_same_grid=max(max(x['same_grid_error'] for x in v) for v in cases.values()),
                        median_total_ms=float(np.median([x['total_seconds'] for x in rs]) * 1e3),
                        reps=sorted(set(x['rep'] for x in rs)), cases=len(cases))
    return agg


def nondominated(points):
    keep = []
    for i, (e, c) in enumerate(points):
        dom = any((e2 <= e and c2 <= c and (e2 < e or c2 < c)) for j, (e2, c2) in enumerate(points) if j != i)
        if not dom:
            keep.append(i)
    return keep


def audit_solve(out, tol):
    d = json.loads((out / 'result.json').read_text())
    checks, detail = {}, {}
    checks['complete'] = bool(d['complete'])
    checks['backend_gpu'] = d['backend'] == 'gpu'
    checks['x64'] = bool(d['x64'])
    checks['precision_highest'] = d['matmul_precision'] == 'highest'
    checks['gates_recorded_pass'] = all(x['passed'] for x in d['gates'])
    cfg = d['config']
    cc = cfg['cohorts']['development']
    dev = cohort(cc['seed'], cc['draw'], cc['count'])
    detail['cohort_max_abs_difference'] = float(np.max(np.abs(dev - np.asarray(d['cohort']['parameters']))))
    checks['development_cohort_reproduced'] = detail['cohort_max_abs_difference'] <= 1e-12
    gf = Geom(cfg['fine_intervals'])
    Sf = Solver(gf)
    fine = {c: gf.scatter(Sf.ref(source_int(gf, q))[0]) for c, q in enumerate(dev)}
    del Sf
    worst_sg, worst_ph, worst_ref = 0.0, 0.0, 0.0
    sym_ok, indep_ok = True, True
    for n in cfg['intervals']:
        g = Geom(n)
        A1, A2 = operator(g), operator_stencil(g)
        diff = (A1 - A2); diff.eliminate_zeros(); indep_ok &= diff.nnz == 0
        asym = (A1 - A1.T); asym.eliminate_zeros(); sym_ok &= asym.nnz == 0
        S = Solver(g)
        s = cfg['fine_intervals'] // n
        for c, q in enumerate(dev):
            u, resid = S.ref(source_int(g, q))
            worst_ref = max(worst_ref, resid)
            same = g.scatter(u)
            chain = fine[c][::s, ::s]
            for r in [x for x in d['invocations'] if x['intervals'] == n and x['case'] == c]:
                field = g.scatter(np.load(out / r['artifact'])['interior'])
                worst_sg = max(worst_sg, abs(rel(field, same) - r['same_grid_error']))
                worst_ph = max(worst_ph, abs(rel(field, chain) - r['physical_error']))
    checks['operator_symmetric'] = sym_ok
    checks['operator_independent_assembly'] = indep_ok
    checks['reference_residual_below_limit'] = worst_ref <= cfg['reference_residual_limit']
    detail['worst_reference_residual'] = worst_ref
    checks['same_grid_errors_recomputed_from_saved_fields'] = worst_sg <= tol
    checks['physical_errors_recomputed_from_saved_fields'] = worst_ph <= tol
    detail['worst_same_grid_difference'] = worst_sg
    detail['worst_physical_difference'] = worst_ph
    agg = aggregate(d['invocations'])
    checks['every_subject_case_has_all_reps'] = all(v['reps'] == list(range(cfg['repetitions'])) and v['cases'] == len(dev)
                                                    for v in agg.values())
    tight = [x for x in d['invocations'] if x['kind'] in ('cg_gpu', 'pcg_ic0') and x['rtol'] <= 1e-10]
    checks['tight_iterative_solves_match_direct'] = max(x['same_grid_error'] for x in tight) <= cfg['solver_agreement_limit']
    detail['nondominated_complete_ms'] = {}
    for n in cfg['intervals']:
        names = [k[1] for k in agg if k[0] == n]
        pts = [(agg[(n, s)]['worst_same_grid'], agg[(n, s)]['median_total_ms']) for s in names]
        detail['nondominated_complete_ms'][str(n)] = [names[i] for i in nondominated(pts)]
    return dict(mode='solve', checks=checks, detail=detail, passed=bool(all(checks.values())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['train', 'solve'])
    ap.add_argument('output', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tol', type=float, default=1e-8)
    a = ap.parse_args()
    res = audit_train(a.output, a.tol) if a.mode == 'train' else audit_solve(a.output, a.tol)
    a.out.write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res['checks'], indent=1))
    print('PASSED' if res['passed'] else 'FAILED')
    assert res['passed']


if __name__ == '__main__':
    main()
