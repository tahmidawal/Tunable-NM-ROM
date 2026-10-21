"""imp-fronts Phase B floors (DESIGN B6) and the policy-field sine tail (DESIGN A2 / Amendment 3 A2 definition). CPU, NumPy.

    python phaseb/floors_b.py --bank K0|K1|K2|K3|<path/to/BANK.json> --refs <slim reference root> [--out <json>]

  f_r(u)     = max_{k=1..5} ||(I - P_{G_r}) u(t_k)|| / ||u(t_0)||, G_r = first r ordered bank functions on the 257^2 lattice
               (wall nodes zeroed, orthonormalised by SVD, relative cut 1e-13), r in {384, 512}; on every ELIGIBLE fresh
               test case (reference outcome 'accepted'); medians and worst per cohort; ineligible cases counted.
  tau_4096(v) = max_{k=1..5} ||v(t_k) - S_4096 v(t_k)|| / ||u_ref(t_0)||, from v's own orthonormal DST-I coefficients
               beyond Laplacian rank 4096, clipped below at 1e-14 (A2; the B0 tail endpoint of decoded policy fields).

Banks: K0 = the incumbent through fcommon (dn256b checkpoint + rotation_R512, the jcp-regimes npcheck defaults); K1..K3 =
jcp-robust inputs/banks/s{1,2,3}/ (ckpt.pkl + rotation_R512.npz, sha256 from BANK.json); an arm = its own BANK.json with
ckpt / rotation (paths relative to the BANK.json directory unless absolute) and their sha256. Every file is hashed
before use and npcheck's caches are reset (jcp-robust e2bank.use pattern), so no table of another bank survives.

References: <refs>/2d/<cohort>/ST/c<j>.npz|json (the refslim layout of phaseb/frefs); cohorts FI, FN0s, FC1, FC2 only, test
split only; the descriptor must equal phaseb/fresh_cohorts.json (relative 1e-12) and t0 the generator IC (1e-14).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(LANE))
sys.path.insert(0, str(HERE))
import fcommon as F  # noqa: E402
import fcohorts as FC  # noqa: E402

ROBUST = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/inputs/banks')
RS = (384, 512)
TAIL_K = 4096
CLIP = 1e-14
FORBIDDEN = ('/sealed/', '/custody/')
_INC = (F.NC.CKPT, F.NC.ROT)
_ORTH = {}
CURRENT = {}


class BankError(ValueError):
    pass


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resolve_bank(spec):
    """-> dict(name, ckpt, rotation, ckpt_sha256, rotation_sha256, bank_json)."""
    if spec in (None, 'K0'):
        return dict(name='K0', ckpt=str(_INC[0]), rotation=str(_INC[1]), ckpt_sha256=None, rotation_sha256=None,
                    bank_json=None)
    if spec in ('K1', 'K2', 'K3'):
        bj = ROBUST / f's{spec[1]}' / 'BANK.json'
        meta = json.loads(bj.read_text())
        return dict(name=spec, ckpt=str(bj.parent / 'ckpt.pkl'), rotation=str(bj.parent / 'rotation_R512.npz'),
                    ckpt_sha256=meta['ckpt_sha256'], rotation_sha256=meta['rotation_sha256'], bank_json=str(bj),
                    seed=meta.get('seed'))
    bj = Path(spec)
    meta = json.loads(bj.read_text())
    res_ = lambda p: str(Path(p) if Path(p).is_absolute() else bj.parent / p)
    return dict(name=meta.get('name', bj.parent.name), ckpt=res_(meta['ckpt']), rotation=res_(meta['rotation']),
                ckpt_sha256=meta['ckpt_sha256'], rotation_sha256=meta['rotation_sha256'], bank_json=str(bj),
                seed=meta.get('seed'))


def use(spec):
    """Point fcommon's NumPy evaluator at a bank (hash-checked) and reset every cache."""
    b = resolve_bank(spec)
    got = (sha(b['ckpt']), sha(b['rotation']))
    if b['ckpt_sha256'] is not None and got != (b['ckpt_sha256'], b['rotation_sha256']):
        raise BankError(f"bank {b['name']}: files differ from BANK.json hashes {got} vs "
                        f"{(b['ckpt_sha256'], b['rotation_sha256'])}")
    b['ckpt_sha256'], b['rotation_sha256'] = got
    F.NC.CKPT, F.NC.ROT = Path(b['ckpt']), Path(b['rotation'])
    F.NC._P, F.NC._T = None, None
    F.NC._G257.clear()
    _ORTH.clear()
    CURRENT.clear()
    CURRENT.update(b)
    return b


def Q(r):
    if r not in _ORTH:
        _ORTH[r] = F.orth(F.G(r))
    return _ORTH[r]


def floors(u, n0):
    return {f'f{r}': F.floor_Q(u, Q(r), n0) for r in RS}


def tau(V, n0, K=TAIL_K):
    """V (6, 257, 257) zero-wall fields; t1..t5 used."""
    V = np.asarray(V, float)
    if not np.isfinite(V).all():
        return float('inf')
    C = F.dst_coef(V[1:])
    tail = np.sqrt(np.maximum(np.sum(C[:, K:] ** 2, axis=1), 0.))
    return max(float(np.max(tail)) / n0, CLIP)


def tau_of_coefficients(W, n0, r=384, K=TAIL_K):
    """tau_K of decoded policy fields v = G_r W, W (6, r) coefficient trajectory at t0..t5 (current bank)."""
    W = np.asarray(W, float)
    if W.shape != (6, r):
        raise ValueError(f'coefficients must be (6, {r}), got {W.shape}')
    if not np.isfinite(W).all():
        return float('inf')
    return tau(F.NC.decode257(W, r), n0, K)


# ---------------------------------------------------------------------------------------------- references ---

def load_fresh_ref(root, cohort, j, tab=None, allow_dev=False):
    tab = tab or FC.load()
    if cohort not in FC.COHORTS:
        raise ValueError(f'cohort {cohort} not admissible')
    row = tab['cohorts'][cohort][j]
    if row['split'] != 'test' and not allow_dev:
        raise ValueError(f'{cohort}|{j} is {row["split"]}, not test')
    stem = Path(root) / '2d' / cohort / 'ST' / f'c{j:03d}'
    rp = str(stem.resolve())
    if any(f in rp + '/' for f in FORBIDDEN):
        raise ValueError(f'forbidden path {rp}')
    meta = json.loads(Path(str(stem) + '.json').read_text())
    z = np.load(str(stem) + '.npz')
    if (meta['cohort'], meta['case']) != (cohort, j):
        raise ValueError('reference identity')
    want = {k: row[k] for k in FC.KEYS}
    if any(abs(meta['desc'][k] - want[k]) > 1e-12 * abs(want[k]) for k in want):
        raise ValueError(f'{cohort}|{j}: reference descriptor differs from fresh_cohorts.json')
    t0 = np.asarray(z['t0'], float)
    x = np.arange(257) / 256.
    X, Y = np.meshgrid(x, x, indexing='ij')
    g = row['a'] * np.exp(-((X - row['cx']) ** 2 + (Y - row['cy']) ** 2) / (2 * row['w'] ** 2))
    g[[0, -1], :] = 0.
    g[:, [0, -1]] = 0.
    if t0.shape != g.shape or np.max(np.abs(t0 - g)) > 1e-14 * max(1., float(np.max(np.abs(g)))):
        raise ValueError(f'{cohort}|{j}: reference t0 differs from the generator IC')
    f = np.asarray(z[meta['final'][2]], float)
    return f, dict(n0=float(np.linalg.norm(t0.ravel())), eligible=meta['outcome'] == 'accepted', outcome=meta['outcome'],
                   eps_ref=meta['judge'].get('eps_ref'))


def available_cases(root, cohort, tab):
    return [r['j'] for r in tab['cohorts'][cohort] if r['split'] == 'test'
            and Path(root, '2d', cohort, 'ST', f"c{r['j']:03d}.json").exists()]


def run(spec, root):
    tab = FC.load()
    b = use(spec)
    out = dict(bank={k: v for k, v in b.items()}, fresh_sha256=tab['sha256'], refs=str(root), cohorts={})
    for coh in FC.COHORTS:
        tests = [r['j'] for r in tab['cohorts'][coh] if r['split'] == 'test']
        have = available_cases(root, coh, tab)
        per = {}
        for j in have:
            u, m = load_fresh_ref(root, coh, j, tab)
            per[str(j)] = dict(eligible=m['eligible'], outcome=m['outcome'], eps_ref=m['eps_ref'], n0=m['n0'],
                               **(floors(u, m['n0']) if m['eligible'] else {}))
        el = [v for v in per.values() if v['eligible']]
        summ = {f'f{r}': dict(median=float(np.median([v[f'f{r}'] for v in el])) if el else None,
                              worst=float(np.max([v[f'f{r}'] for v in el])) if el else None) for r in RS}
        out['cohorts'][coh] = dict(per_case=per, n_test=len(tests), n_reference=len(have), n_eligible=len(el),
                                   n_ineligible=len(have) - len(el), n_missing=len(tests) - len(have), summary=summ)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    out = run(a.bank, a.refs)
    s = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(s + '\n')
    for coh, v in out['cohorts'].items():
        print(coh, 'eligible', v['n_eligible'], '/', v['n_test'], {k: x for k, x in v['summary'].items()})


if __name__ == '__main__':
    main()
