"""jcp-wide-bank: independent NumPy acceptance audit of a w3d.py job (DESIGN A0-2, A2-12, A7). No JAX: the bank is
evaluated by the vendored NumPy re-implementation `vendor/quad3d/npcore.py` (bank_np), not by the JAX code.

    python audit_w3.py <result json (as run or *_A7.json)> <output dir with fields/> <refined reference npz>
                       [--out checks/<attempt>-audit.json]

Checks (all must pass for `accepted`):
  B1  job complete, gpu backend, x64, precision highest.
  B2  refined errors of EVERY arm on EVERY validation case recomputed from fields/coefficients.npz x the NumPy bank on
      the 63^3 lattice against the refined reference: relative agreement <= 1e-9 with the recorded per-time errors.
  B3  at 65 nodes (mesh interior = the lattice) the field-metric distance from the converged rollout recomputed the
      same way: agreement <= 1e-9 (relative, floor 1e-15 absolute).
  B4  selection re-implemented independently (pooled eligibility of A7, metric-only controls of A0-2, K-conv/K-target,
      smallest-m per family at both tau, named diagnostics) equals the recorded selection.
  B5  fault injection through B2: a swapped reference between two cases and a 1 % perturbation of one coefficient
      record must each be detected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
import npcore as NP  # noqa: E402  vendored NumPy primitives


def pooled_ok(rs, frac):
    if not rs:
        return False
    rr = np.concatenate([np.asarray(r['reasons']) for r in rs])
    return bool(all(r['finite'] for r in rs) and (rr == 3).sum() == 0 and ((rr == 0) | (rr == 2)).sum() <= frac * len(rr))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('result')
    ap.add_argument('outdir')
    ap.add_argument('ref')
    ap.add_argument('--out')
    a = ap.parse_args()
    rep = json.loads(Path(a.result).read_text())
    cfg = rep['config']
    res = dict(job=rep.get('job_id'), result=a.result, checks={})
    C = res['checks']
    C['B1'] = dict(passed=bool(rep.get('complete') and rep.get('backend') == 'gpu' and rep.get('x64') and
                               rep.get('precision') == 'highest'))
    z = np.load(a.ref)
    coef = np.load(Path(a.outdir) / 'fields' / 'coefficients.npz')
    X = NP.lattice65_coords()
    cases = cfg.get('case_subset') or list(range(cfg['cohort_count']))
    RR = {j: np.asarray(z[f'c{j}']) for j in cases}
    worst2, worst3, n2, n3 = 0., 0., 0, 0
    inj = {}
    for n, m_ in rep['meshes'].items():
        for bn, b in m_['banks'].items():
            bk = next(x for x in cfg['banks'] if x['name'] == bn)
            p, T = NP.load_bank(HERE.parents[1] / bk['model'])
            G = NP.bank_np(p, T, X)                                       # (63^3, R) NumPy bank on the lattice
            for key, S in b['settings'].items():
                Rp = S['Rp']
                Gp = G[:, :Rp]
                cc = {}
                for nm, arm in S['arms'].items():
                    if 'cases' not in arm or arm.get('alias_of'):
                        continue
                    for rec in arm['cases']:
                        j = rec['case']
                        Cv = coef[f'{n}|{bn}|{key}|{nm}|{j}']
                        cc[(nm, j)] = Cv
                        F = Cv @ Gp.T
                        e = np.linalg.norm(F - RR[j], axis=1) / np.linalg.norm(RR[j][0])
                        worst2 = max(worst2, float(np.max(np.abs(e - np.asarray(rec['err_refined'])) /
                                                          np.maximum(np.asarray(rec['err_refined']), 1e-300))))
                        n2 += 1
                        if n == '65' and nm != 'conv':
                            dd = np.linalg.norm((Cv - cc.get(('conv', j), coef[f'{n}|{bn}|{key}|conv|{j}'])) @ Gp.T, axis=1)
                            dd = float(dd[1:].max()) / float(np.linalg.norm(RR[j][0]))
                            # the job normalises by ||u0|| on the mesh; at 65 nodes the mesh interior is the lattice
                            worst3 = max(worst3, abs(dd - rec['dist_conv']) / max(rec['dist_conv'], 1e-15))
                            n3 += 1
                if not inj:                                             # B5 on the first setting, converged arm
                    j0, j1 = cases[0], cases[1]
                    rec0 = S['arms']['conv']['cases'][0]
                    F = coef[f'{n}|{bn}|{key}|conv|{j0}'] @ Gp.T
                    sw = np.linalg.norm(F - RR[j1], axis=1) / np.linalg.norm(RR[j1][0])
                    Cp = coef[f'{n}|{bn}|{key}|conv|{j0}'].copy()
                    Cp[3] *= 1.01
                    pt = np.linalg.norm(Cp @ Gp.T - RR[j0], axis=1) / np.linalg.norm(RR[j0][0])
                    ref_e = np.asarray(rec0['err_refined'])
                    inj = dict(swapped_worst=float(np.max(np.abs(sw - ref_e) / np.maximum(ref_e, 1e-300))),
                               perturbed_worst=float(np.max(np.abs(pt - ref_e) / np.maximum(ref_e, 1e-300))))
    C['B2'] = dict(compared=n2, worst_relative=worst2, passed=bool(n2 > 0 and worst2 <= 1e-9))
    C['B3'] = dict(compared=n3, worst_relative=worst3, passed=bool(n3 == 0 or worst3 <= 1e-9),
                   note='65-node meshes only (mesh interior = lattice)')
    C['B5'] = dict(**inj, passed=bool(inj and inj['swapped_worst'] > 1e-9 and inj['perturbed_worst'] > 1e-9))
    # B4 selection
    taus = dict(primary=cfg['tau'], secondary=cfg['tau_secondary'])
    frac, rho_bar = cfg['nonstat_fraction'], cfg['rho_bar']
    mism = []
    for n, m_ in rep['meshes'].items():
        for bn, b in m_['banks'].items():
            for key, S in b['settings'].items():
                rec = {nm: arm['cases'] for nm, arm in S['arms'].items() if 'cases' in arm}
                crec = S['certification']
                rho = {k: v['worst'] for k, v in S['rho']['rules'].items()}
                conv = (pooled_ok(rec['conv'], frac) and pooled_ok(crec['conv'], frac) and pooled_ok(rec['check'], frac)
                        and pooled_ok(crec['check'], frac) and
                        max([r['dist_conv'] for r in rec['check']] + [r['dist_conv'] for r in crec['check']]) <= cfg['conv_bar'])
                valid = conv and S['rho']['check_worst'] <= cfg['target_bar']
                if valid != S['selection']['valid']:
                    mism.append((n, bn, key, 'valid', valid, S['selection']['valid']))
                ctrls = [nm for nm, arm in S['arms'].items() if arm['control']]
                for t, tv in taus.items():
                    disc = not any(max(r['dist_conv'] for r in rec[c]) <= tv and rho[c] <= rho_bar for c in ctrls)
                    for fam in cfg['families']:
                        cands = sorted([(arm['m'], nm) for nm, arm in S['arms'].items()
                                        if arm['family'] == fam and not arm['control'] and arm.get('m')])
                        pick = None
                        for m, nm in cands:
                            if (pooled_ok(rec[nm], frac) and max(r['dist_conv'] for r in rec[nm]) <= tv and rho[nm] <= rho_bar):
                                pick = nm
                                break
                        e = S['selection']['entries'][f'{t}|{fam}']
                        want = pick if (valid and disc) else None
                        if e['arm'] != want or e['arm_raw'] != pick:
                            mism.append((n, bn, key, t, fam, e['arm'], want, e['arm_raw'], pick))
    C['B4'] = dict(mismatches=mism, passed=not mism)
    res['accepted'] = all(C[k]['passed'] for k in ('B1', 'B2', 'B3', 'B4', 'B5'))
    txt = json.dumps(res, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + '\n')
    raise SystemExit(0 if res['accepted'] else 1)


if __name__ == '__main__':
    main()
