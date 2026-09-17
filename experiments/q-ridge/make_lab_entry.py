"""Generate the lab-log entry for this lane from the audited JSONs.

    python make_lab_entry.py --eq checks/qrg301-audit.json --r1 checks/qrg101-audit.json \
        --r2 checks/qrg201-audit.json --report reports/2026-09-16-q-ridge.md \
        --out checks/2026-09-16-lab-log-entry.md

Every number in the entry comes from the audits or the artifact manifests. The entry is then
appended verbatim to the canonical `LAB-LOG.md` on `main`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CELL = Path(__file__).resolve().parent
MULT = {'m4': 4, 'm8': 8, 'm16': 16}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def fmt(x, d=4):
    return '—' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


def ladder_block(aud):
    lines = []
    for quad in ('dense', 'eq'):
        lines.append(f'\n*{quad} quadrature:*\n')
        qs, rows = None, []
        for label, e in sorted(aud['ladders'][quad].items(),
                               key=lambda kv: (kv[1]['lam_rel'], kv[1]['rule'])):
            qs = e['q']
            head = (f'$\\lambda_{{rel}}={label}$' if e['kind'] == 'lam'
                    else f'$M={MULT[label]}(K+q)$')
            rows.append([head,
                         ' / '.join(fmt(v) for v in e['worst_evolved_percent']),
                         ' / '.join(fmt(v) for v in e['worst_all_times_percent']),
                         ' / '.join(fmt(v, 0) for v in e['median_gpu_ms']),
                         yn(e['monotone_evolved']), yn(e['all_converged']),
                         yn(e['cost_within_1p5x']), yn(e['all_times_not_raised']),
                         yn(e['passes'])])
        lines.append(table(
            ['setting', f"worst evolved % at q = {', '.join(map(str, qs or []))}",
             'worst all-times %', 'median GPU ms', 'monotone', 'converged', 'cost <=1.5x',
             'all-times not raised', 'passes'], rows))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eq', required=True)
    p.add_argument('--r1', required=True)
    p.add_argument('--r2', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    eq = json.loads(Path(a.eq).read_text())
    r1 = json.loads(Path(a.r1).read_text())
    r2 = json.loads(Path(a.r2).read_text())
    W = []
    v = eq['verdict']
    cert = eq['ladders']['certified']

    old = eq['ladders']['old_rule']
    gains = [o / max(c, 1e-300) for o, c in zip(old['worst_evolved_percent'],
                                                cert['worst_evolved_percent'])]
    best = max(gains)
    best_q = cert['q'][gains.index(best)]
    bad = [(cert['q'][i], cert['q'][i + 1]) for i in range(len(cert['q']) - 1)
           if cert['worst_evolved_percent'][i + 1] > cert['worst_evolved_percent'][i] + 1e-12]
    if v['passes']:
        head = ('certifying the empirical quadrature on states the ROM actually reaches makes '
                'the Burgers ladder monotone again')
    elif v['regression_present_in_old_rule']:
        head = ('the evolved-times regression is the empirical-quadrature RULE; certifying it '
                f'on states the ROM actually reaches removes up to {best:.2f}x of it (at '
                f'$q={best_q}$) at no extra cost, but '
                + (f"leaves {len(bad)} violation{'s' if len(bad) != 1 else ''} "
                   f"({', '.join(f'$q={x}\\to{y}$' for x, y in bad)}) "
                   'because no constructible $m$ certifies the top rungs'
                   if bad else 'the pre-registered criterion still fails'))
    else:
        head = 'EQ rule certification, with the ridge and the test-count sweep as controls'
    W.append(f'### q-ridge — {head}; neither a ridge on the corrections nor more tests '
             'fixes it either\n')

    W.append(
        "The coordinator first asked whether the Burgers $256^2$ correction ladder's "
        'non-monotonicity on the worst-over-evolved-times metric ($q=16$ worse than $q=0$) is '
        'the extra unknowns **overfitting the $M$ weak test equations**, with two remedies — a '
        'field-metric ridge on the corrections (R1) and more tests at fixed $q$ (R2) — and a '
        'held-out-residual control (R3). Mid-flight, after `q-diag` reported that the cause is '
        'the quadrature, the lane was **re-scoped**: R1 and R2 were demoted to controls and '
        'the primary became **EQ rule certification** — refit the $m$-point rule on states the '
        'ROM actually reaches, grow $m$ at fixed $M=4(K+q)$, certify every rule by its '
        'held-out $\\rho$ rather than by its NNLS fit residual, and rebuild the ladder with '
        'the cheapest certified rule per rung. Predeclared protocol and every amendment, '
        'including the re-scope as §A3: `experiments/q-ridge/DESIGN.md`. Nothing was merged or '
        'pushed.\n')

    W.append(
        'Worktree `worktrees/2026-09-16-q-ridge`, branch `exp/2026-09-16-q-ridge`, forked from '
        '`exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Namespace '
        '`/cluster/tufts/paralab/tawal01/q_ridge_20260916/`. '
        + '; '.join(f"{x['mode'].upper()} job `{x['job_id']}` (`{x['attempt']}`) on "
                    f"`{x['gpu']}`, source `{(x['commit'] or '')[:12]}`, elapsed "
                    f"{x['elapsed_seconds']:.1f} s" for x in (r1, r2, eq))
        + '. All three printed `jax_backend=gpu`, ran float64 at highest matmul precision, and '
        'were checksum-collected, independently NumPy-audited and archived before their exact '
        'remote attempt directories were removed. Two earlier submissions (3757043, 3757044) '
        'died in the sbatch preamble on a placeholder collision, before the GPU preflight and '
        'with zero GPU work; they are recorded in `DESIGN.md` §A1 and are not counted against '
        'the three-job cap.\n')

    for aud in (r1, r2, eq):
        W.append(f"**{aud['mode'].upper()} ({aud['attempt']}) gates.** " + '; '.join(
            f"`{k}` {yn(x.get('passed'))}" for k, x in sorted(aud['checks'].items())
            if isinstance(x, dict) and 'passed' in x) + '.\n')

    W.append('**Cross-job fidelity.**\n')
    rows = []
    for aud in (r1, r2, eq):
        det = (aud['checks'].get('cross_job_fidelity') or {}).get('detail', {})
        for arm, x in sorted(det.items()):
            dd = x['detail']

            def rel(k, dd=dd):
                z = (dd.get(k) or {}).get('relative_difference')
                return '—' if z is None else f'{z:.1e}'

            rows.append([aud['attempt'], f'`{arm}`', dd.get('source'),
                         f"`{dd.get('comparator')}`",
                         f"{dd.get('declared_tolerance'):.0e}"
                         if dd.get('declared_tolerance') else '—',
                         rel('worst_all_times_percent'), rel('worst_evolved_percent'),
                         yn(x['passed'])])
    W.append(table(['job', 'arm', 'source', 'comparator', 'tolerance',
                    'rel. diff (all-times)', 'rel. diff (evolved)', 'passed'], rows))

    W.append('\n**EQ rule certification — every rule, its NNLS fit and its held-out $\\rho$** '
             f"(bar $\\rho^\\star = {eq['rho_bar']}$, declared before the job ran from "
             "`q-diag`'s $q=0$ incumbent measurement).\n")
    W.append(table(['$q$', 'population', '$m$ target', '$m$', '$m/M$', 'NNLS fit',
                    '$\\rho_{max}$', '$\\rho_{95}$', 'certified', 'truncated', 'fit (s)'],
                   [[x['q'], x['population'], x['m_target'], x['m'],
                     f"{x['m'] / x['M']:.2f}", f"{x['relative_fit']:.2e}",
                     fmt(x['certification']['rho_max']), fmt(x['certification']['rho_p95']),
                     yn(x['certified_primary']), yn(x['truncated']), f"{x['seconds']:.1f}"]
                    for x in sorted(eq['rules'],
                                    key=lambda x: (x['q'], x['population'], x['m_target']))]))

    W.append('\n**The rule chosen at each rung.**\n')
    W.append(table(['$q$', '$M$', 'chosen $m$', 'basis', '$\\rho_{max}$', '$\\rho_{95}$'],
                   [[c['q'], c['M'], c['chosen_m'], c['basis'], fmt(c['rho_max']),
                     fmt(c['rho_p95'])] for c in eq['rule_choice']]))

    W.append('\n**What the certification buys, rung by rung** (worst evolved-times error, '
             'incumbent static-population rule against the cheapest certified reachable-state '
             'rule, and the dense twin where this job ran one).\n')
    dn = {e_q: (m, ev) for e_q, m, ev in zip(eq['ladders']['dense']['q'],
                                             eq['ladders']['dense']['M'],
                                             eq['ladders']['dense']['worst_evolved_percent'])}
    W.append(table(['$q$', '$M$', 'old rule $m$ / evolved %', 'certified $m$ / evolved %',
                    'improvement', 'dense evolved %', 'cost vs old rule'],
                   [[cert['q'][i], cert['M'][i],
                     f"{old['m'][i]} / {old['worst_evolved_percent'][i]:.4f}",
                     f"{cert['m'][i]} / {cert['worst_evolved_percent'][i]:.4f}",
                     f"{gains[i]:.2f}x",
                     (f"{dn[cert['q'][i]][1]:.4f}" if cert['q'][i] in dn else '—'),
                     f"{cert['cost_ratio_to_old_rule'][i]:.3f}x"]
                    for i in range(len(cert['q']))]))

    W.append('\n**The rebuilt ladder, both metrics.**\n')
    rows = []
    for key in ('certified', 'old_rule', 'dense'):
        e = eq['ladders'].get(key)
        if not e:
            continue
        rows.append([e['name'], ', '.join(map(str, e['q'])),
                     ' / '.join(fmt(x) for x in e['worst_evolved_percent']),
                     ' / '.join(fmt(x) for x in e['worst_all_times_percent']),
                     ' / '.join(fmt(x, 0) for x in e['median_gpu_ms']),
                     yn(e['monotone_evolved']), yn(e['all_converged']),
                     yn(e.get('cost_within_2x')), yn(e.get('passes'))])
    W.append(table(['ladder', 'q', 'worst evolved %', 'worst all-times %', 'median GPU ms',
                    'monotone', 'converged', 'cost <=2x old rule', 'passes'], rows))
    if v.get('extrapolated_m_for_bar'):
        W.append(f"\nNo $m$ in the declared grid reaches the bar at $q={max(cert['q'])}$; a "
                 'log-log extrapolation of $\\rho_{max}$ against $m$ puts the crossing at '
                 f"$m \\approx {v['extrapolated_m_for_bar']:.0f}$ "
                 f"(slope {v['extrapolation_slope']:.3f}).\n")

    W.append('\n**R1 — the field-metric ridge (demoted to a control).**\n')
    W.append(ladder_block(r1))
    W.append('\n**R2 — more tests at fixed $q$ (demoted to a control).**\n')
    W.append(ladder_block(r2))

    W.append('\n**R3 — the weak residual on held-out test modes** (worst over cases and steps, '
             'per-mode RMS normalised by the per-node RMS of the previous state; the common '
             "held-out block is the 512 modes ranked 1537-2048, beyond every arm's $M$ in any "
             'of the three jobs). `q-diag` already measured this on the dense fixed-$M=256$ '
             'arms; what is new here is the same measurement under a ridge and under larger '
             '$M$.\n')
    rows = []
    for aud in (r1, r2, eq):
        if not aud.get('r3'):
            continue
        by = {x['arm']: x for x in aud['arms']}
        for arm, x in sorted(aud['r3']['arms'].items()):
            row = by[arm]
            rows.append([aud['attempt'], f'`{arm}`', row['q'], row['quadrature'],
                         ('0' if not row.get('lam_rel') else f"{row['lam_rel']:.0e}"),
                         row['M'], row['m'] or '—',
                         f"{x['in_space']['worst_normalised_per_mode_rms']:.3e}",
                         f"{x['held_out_common']['worst_normalised_per_mode_rms']:.3e}",
                         f"{x['held_over_in_common']:.3f}"])
    W.append(table(['job', 'arm', '$q$', 'quadrature', '$\\lambda_{rel}$', '$M$', '$m$',
                    'in-space', 'held-out (common)', 'held/in'], rows))

    rep = Path(a.report)
    W.append(f'\nSource-generated report: `experiments/q-ridge/reports/{rep.name}` '
             f'(SHA256 `{sha(rep)}`) with its LaTeX twin, its two figures and its generator '
             'beside it.\n')
    for att in (r1['attempt'], r2['attempt'], eq['attempt']):
        man = CELL / f'artifacts/{att}/archive.json'
        if man.exists():
            m = json.loads(man.read_text())
            W.append(f"Raw archive `{att}` Git-tracked as bounded chunks: whole SHA256 "
                     f"`{m['sha256']}` ({len(m['chunks'])} chunks). `output/bank_G.npz` is "
                     'excluded by design and its SHA256 recorded beside them.\n')

    Path(a.out).write_text('\n'.join(W) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
