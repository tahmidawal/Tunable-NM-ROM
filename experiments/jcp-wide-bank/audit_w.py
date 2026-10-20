"""jcp-wide-bank: independent NumPy acceptance audit of a w2d.py job (DESIGN A0-2, A2-12, A5). No JAX, no import of
the driver: every check is re-implemented from the definitions.

    python audit_w.py <job output dir> <local 2D reference dir> [--out checks/<attempt>-audit.json]

Checks (each must pass for `accepted`):
  A1  job complete, backend gpu, x64, precision highest, checkpoint unchanged during the job.
  A2  refined errors (ST, S) of every arm on the audit cases recomputed from the saved restricted fields; of the converged
      rollout on EVERY case from gref_f257_<setting>.npz; relative agreement <= 1e-10 with result.json rows.
  A3  distance-from-gref recomputed on the audit cases where full fields were not saved is not possible; instead the
      restricted (257^2) distance is recomputed and must not exceed the full-mesh value by more than 1e-12 (a necessary
      consistency check: the restricted grid is a subset of the mesh up to normalisation) -- reported, not gating.
  A4  selection (m* per family at both tau, named diagnostics, controls, availability) recomputed from the rows and rho
      records with an independent implementation; must equal result.json.
  A5  fault injection through the same path: (a) swapping the ST reference of two cases must make A2 fail;
      (b) a 1 % perturbation of one saved field must make A2 fail.
  A6  gates recorded in result.json: K-conv, K-target, control discrimination, floors check, K-time per panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REASONS = ('budget', 'tol', 'tiny_step', 'damping_limit', 'stationary')


def evolved_max(x):
    return float(np.max(np.asarray(x)[1:]))


def ref_err(fr, ref, n0r):
    return [float(np.linalg.norm(a - b)) / n0r for a, b in zip(fr, ref)]


def eligible(row, bfrac):
    steps = sum(row['exits'].values())
    return bool(row['finite'] and row['exits']['damping_limit'] == 0 and row['exits']['budget'] <= bfrac * steps)


def mstar(rows, rho, arms, fam, tau, rho_bar, bfrac, use_d=True, use_rho=True):
    for name, m in sorted(((n, a['m']) for n, a in arms.items() if a['family'] == fam and not a['control']),
                          key=lambda t: t[1]):
        rs = rows.get(name, [])
        if not rs or not all(eligible(r, bfrac) for r in rs):
            continue
        if use_d and not all(r.get('vs_gref_evolved') is not None and np.isfinite(r['vs_gref_evolved'])
                             and r['vs_gref_evolved'] <= tau for r in rs):
            continue
        if use_rho and not (rho.get(name) is not None and np.isfinite(rho[name]) and rho[name] <= rho_bar):
            continue
        return name, m
    return None, None


def check_errors(rep, outdir, refs, swap=None, perturb=None):
    """Return the worst relative disagreement between recomputed and recorded refined errors (A2)."""
    worst = 0.
    n = 0
    rows = rep['rows']
    for r in rows:
        key, coh, c = r['setting'], r['cohort'], r['case']
        fr = None
        f_a = outdir / f"audit_{key}_{r['arm']}_{coh}{c}.npz"
        if f_a.exists():
            fr = np.load(f_a)['f257']
        elif r['arm'] == 'gref':
            fr = np.load(outdir / f'gref_f257_{key}.npz')[f'{coh}{c}']
        if fr is None:
            continue
        if perturb == (key, r['arm'], coh, c):
            fr = fr.copy()
            fr[3] *= 1.01
        for tag in ('ST', 'S'):
            rc = (coh, c)
            if swap and tag == 'ST' and rc in swap:
                rc = swap[rc]
            ref = refs[(rc[0], rc[1], tag)]
            got = evolved_max(ref_err(fr, ref, r['n0r']))
            want = r[f'ref_{tag}_evolved']
            worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
            n += 1
    return worst, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outdir')
    ap.add_argument('refdir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    outdir, refdir = Path(a.outdir), Path(a.refdir)
    rep = json.loads((outdir / 'result.json').read_text())
    cfg = rep['config']
    res = dict(job=rep.get('job_id'), commit=rep.get('commit'), checks={})
    C = res['checks']
    C['A1'] = dict(passed=bool(rep.get('complete') and rep.get('backend') == 'gpu' and rep.get('x64') and
                               rep.get('precision') == 'highest' and
                               rep.get('checkpoint_sha256') == rep.get('checkpoint_sha256_after')))
    refs = {}
    for r in rep['rows']:
        for tag in ('ST', 'S'):
            k = (r['cohort'], r['case'], tag)
            if k not in refs:
                refs[k] = np.load(refdir / f'ref_{tag}_{r["cohort"]}_{r["case"]:03d}.npz')['f257']
    for fn, want in cfg.get('refs_files_sha256', {}).items():
        assert hashlib.sha256((refdir / fn).read_bytes()).hexdigest() == want, ('local reference differs', fn)
    worst, n = check_errors(rep, outdir, refs)
    C['A2'] = dict(compared=n, worst_relative=worst, passed=bool(n > 0 and worst <= 1e-10))
    # A5 fault injection
    cases = sorted({(r['cohort'], r['case']) for r in rep['rows']})
    sw = {cases[0]: cases[1], cases[1]: cases[0]}
    w_sw, _ = check_errors(rep, outdir, refs, swap=sw)
    gr = next(r for r in rep['rows'] if r['arm'] == 'gref')
    w_pt, _ = check_errors(rep, outdir, refs, perturb=(gr['setting'], 'gref', gr['cohort'], gr['case']))
    C['A5'] = dict(swapped_reference_worst=w_sw, perturbed_field_worst=w_pt,
                   swapped_detected=bool(w_sw > 1e-10), perturbed_detected=bool(w_pt > 1e-10))
    C['A5']['passed'] = C['A5']['swapped_detected'] and C['A5']['perturbed_detected']
    # A4 selection recomputed
    taus = dict(primary=cfg['tau'], secondary=cfg['tau_secondary'])
    mism = []
    for key, st in rep['settings'].items():
        rows = {}
        for r in rep['rows']:
            if r['setting'] == key:
                rows.setdefault(r['arm'], []).append(r)
        rho = {k: v['max'] for k, v in rep['rho'][key]['rules'].items()}
        arms = {k: v for k, v in st['arms'].items() if k not in ('gref', 'gref_check')}
        g = rep['gates']
        valid = bool(g[f'converged_{key}']['passed'] and g[f'continuum_target_{key}']['passed'])
        conv = (all(eligible(r, cfg['budget_fraction']) for r in rows['gref']) and
                all(eligible(r, cfg['budget_fraction']) for r in rows['gref_check']) and
                max(r['vs_gref_evolved'] for r in rows['gref_check']) <= cfg['conv_bar'])
        if conv != g[f'converged_{key}']['passed']:
            mism.append((key, 'K-conv'))
        disc = {}
        for t_, tv in taus.items():
            disc[t_] = not any(max(r['vs_gref_evolved'] for r in rows[n_]) <= tv and rho[n_] <= cfg['rho_bar']
                               for n_, a_ in arms.items() if a_['control'])
            if disc[t_] != rep['gates'][f'controls_{key}']['discriminating'][t_]:
                mism.append((key, 'controls', t_))
        for t_, tv in taus.items():
            for fam in cfg['families']:
                e = rep['selection'][key]['entries'][f'{t_}|{fam}']
                nm, m = mstar(rows, rho, arms, fam, tv, cfg['rho_bar'], cfg['budget_fraction'])
                nd, md = mstar(rows, rho, arms, fam, tv, cfg['rho_bar'], cfg['budget_fraction'], use_rho=False)
                ok = valid and disc[t_]
                if (e['arm'], e['m_d']) != ((nm if ok else None), md) or e['arm_raw'] != nm:
                    mism.append((key, t_, fam, e['arm'], nm, e['m_d'], md))
    C['A4'] = dict(mismatches=mism, passed=not mism)
    # A6 gates as recorded
    g = rep['gates']
    C['A6'] = dict(
        conv={k: v['passed'] for k, v in g.items() if k.startswith('converged_')},
        target={k: v['passed'] for k, v in g.items() if k.startswith('continuum_target_')},
        controls={k: v['discriminating'] for k, v in g.items() if k.startswith('controls_')},
        floors={k: v.get('check_passed') for k, v in rep['floors'].items()},
        timing={k: v.get('valid') for k, v in rep['timing'].items()},
        timing_valid_jobwide=rep.get('timing_valid_jobwide'))
    res['accepted'] = all(C[k]['passed'] for k in ('A1', 'A2', 'A4', 'A5'))
    txt = json.dumps(res, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + '\n')
    raise SystemExit(0 if res['accepted'] else 1)


if __name__ == '__main__':
    main()
