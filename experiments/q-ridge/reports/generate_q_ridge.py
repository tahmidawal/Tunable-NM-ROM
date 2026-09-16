"""Generate the q-ridge report (Markdown + LaTeX) and its figure.

Every number and every plotted point is read from the audited run JSONs and from the
comparator table. Nothing in the report is typed by hand.

    python reports/generate_q_ridge.py --r1 checks/qrg101-audit.json \
        --r2 checks/qrg201-audit.json --out reports/2026-09-16-q-ridge
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CELL = HERE.parent
QUAD_TITLE = {'dense': 'dense (exact) quadrature', 'eq': 'empirical quadrature, $m = 4M$'}
COLORS = ['#2b2b2b', '#3b6ea5', '#4f8a3d', '#b08a2a', '#b03a3a', '#7a4f9a', '#2a8a8a']


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        return f'{x:.{d}f}'
    return str(x)


def sci(x, d=2):
    return '—' if x is None else f'{x:.{d}e}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Doc:
    """Writes the same tables to Markdown and to LaTeX from one call."""

    def __init__(self):
        self.md, self.tex = [], []

    def h(self, level, text):
        self.md.append('#' * level + ' ' + text + '\n')
        cmd = {1: 'section', 2: 'subsection', 3: 'subsubsection'}.get(level, 'paragraph')
        self.tex.append('\\%s{%s}' % (cmd, _tex_text(text)))

    def p(self, text):
        self.md.append(text + '\n')
        self.tex.append(_tex_text(text) + '\n')

    def table(self, header, rows, caption=None):
        self.md.append('| ' + ' | '.join(header) + ' |')
        self.md.append('|' + '|'.join(['---'] * len(header)) + '|')
        for r in rows:
            self.md.append('| ' + ' | '.join(str(c) for c in r) + ' |')
        self.md.append('')
        self.tex.append('\\begin{table}[htbp]\\centering\\small')
        self.tex.append('\\begin{tabular}{' + 'l' * len(header) + '}\\hline')
        self.tex.append(' & '.join(_tex_text(h) for h in header) + ' \\\\ \\hline')
        for r in rows:
            self.tex.append(' & '.join(_tex_text(str(c)) for c in r) + ' \\\\')
        self.tex.append('\\hline\\end{tabular}')
        if caption:
            self.tex.append('\\caption{%s}' % _tex_text(caption))
        self.tex.append('\\end{table}')

    def raw_md(self, text):
        self.md.append(text + '\n')


def _tex_text(s):
    s = str(s)
    if s.count('$') >= 2:                     # leave inline math alone
        parts = s.split('$')
        return '$'.join(p if i % 2 else _tex_escape(p) for i, p in enumerate(parts))
    return _tex_escape(s)


def _tex_escape(s):
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'),
                 ('#', r'\#'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')):
        s = s.replace(a, b)
    return s.replace('—', '--').replace('`', '')


def ladder_rows(entry):
    out = []
    for i, q in enumerate(entry['q']):
        out.append([q, entry['M'][i], entry['m'][i] or '—',
                    f(entry['worst_evolved_percent'][i]),
                    f(entry['worst_all_times_percent'][i]),
                    f(entry['worst_t0_compression_percent'][i]),
                    f(entry['median_gpu_ms'][i], 1),
                    f(entry['cost_ratio_to_incumbent'][i], 3) + 'x',
                    yn(entry['converged'][i]), yn(entry['eq_rule_valid'][i])])
    return out


def figure(r1, r2, png, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, quad in enumerate(('dense', 'eq')):
        for row, (aud, kind, name) in enumerate(((r1, 'lam', 'R1: ridge $\\lambda_{rel}$'),
                                                 (r2, 'rule', 'R2: test-count rule'))):
            ax = axes[row][col]
            lads = (aud or {}).get('ladders', {}).get(quad, {})
            for i, (label, e) in enumerate(sorted(
                    lads.items(), key=lambda kv: (kv[1]['lam_rel'], kv[1]['rule']))):
                c = COLORS[i % len(COLORS)]
                filled = all(e['converged'])
                ax.plot(e['q'], e['worst_evolved_percent'], 'o-', color=c, lw=1.6,
                        ms=7, markerfacecolor=(c if filled else 'none'),
                        label=(f'$\\lambda_{{rel}}={label}$' if kind == 'lam'
                               else f'$M={ {"m4": 4, "m8": 8, "m16": 16}[label] }(K+q)$'))
            ax.set_xscale('symlog', linthresh=16)
            ax.set_yscale('log')
            ax.set_xlabel('correction directions $q$')
            ax.set_ylabel('worst evolved-times error (%, log)')
            ax.set_title(f'{name} — {QUAD_TITLE[quad]}', fontsize=10)
            ax.grid(alpha=.25, which='both')
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle('One frozen Burgers checkpoint, 256 intervals, six development cases.\n'
                 'Worst-over-evolved-times error against the same-job converged full-order '
                 'solve. Filled marker = every rung converged, open = not.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r1', required=True)
    p.add_argument('--r2', default=None)
    p.add_argument('--comparators', default=str(CELL / 'checks/comparators.json'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r1 = json.loads(Path(a.r1).read_text())
    r2 = json.loads(Path(a.r2).read_text()) if a.r2 and Path(a.r2).exists() else None
    cmp_t = json.loads(Path(a.comparators).read_text())
    out = Path(a.out)
    d = Doc()

    d.h(1, 'q-ridge — the Burgers evolved-times regression is a quadrature fact, not '
           'test-space overfitting')
    jobs = [x for x in (r1, r2) if x]
    d.p('Final numbers. Two cluster jobs on the frozen Burgers checkpoint '
        '`18f0266ae6f0…` at 256 intervals, on the six opened development cases, under the '
        'budget-600 block-damped variable-projection contract inherited from `b-ladder-top`. '
        'The pre-registered design, its gates and its falsification clause are in '
        '[`../DESIGN.md`](../DESIGN.md); nothing below departs from them.')
    d.table(['job', 'attempt', 'question', 'job id', 'GPU', 'source commit', 'elapsed (s)',
             'failed gates'],
            [[i + 1, x['attempt'], x['mode'].upper(), x['job_id'], x['gpu'],
              (x['commit'] or '')[:12], f(x['elapsed_seconds'], 1),
              ', '.join(f'`{g}`' for g in x['failed']) or 'none'] for i, x in enumerate(jobs)])

    # ---------------------------------------------------------------- verdict
    d.h(2, 'Verdict')
    rows = []
    for name, aud in (('R1 (ridge)', r1), ('R2 (more tests)', r2)):
        if not aud:
            continue
        for quad in ('dense', 'eq'):
            v = aud['verdict'][quad]
            rows.append([name, quad, yn(v['regression_present_at_incumbent']),
                         ', '.join(v['passing']) or 'none',
                         ', '.join(v['passing_without_losing_the_gain']) or 'none',
                         yn(v['passes'])])
    d.table(['remedy', 'quadrature', 'regression present at the incumbent',
             'settings that pass', 'of those, keeping the top-rung gain', 'passes'], rows)

    # ------------------------------------------------- the archive diagnosis
    d.h(2, 'Where the regression is, measured before anything was submitted')
    d.p('Recomputed from the retained `cclad01` fields (job '
        f"{cmp_t['cclad01']['job_id']}, archive SHA256 `{cmp_t['cclad01']['archive_sha256'][:16]}…`), "
        'worst over the six cases, against that job\'s own converged full-order solve.')
    lad = {}
    for arm, v in cmp_t['cclad01']['arms'].items():
        for tag, want in (('m4 dense', ('m4', 'dense')), ('M=256 dense', ('m256', 'dense')),
                          ('M=256 EQ', ('m256', 'eq')), ('m4 EQ', ('m4', 'eq'))):
            if v.get('rule') == want[0] and v.get('quadrature') == want[1] and v.get('q') is not None:
                lad.setdefault(tag, {})[v['q']] = v['worst_evolved_percent']
    qs = [0, 16, 32, 64, 128]
    d.table(['configuration'] + [f'$q={q}$' for q in qs] + ['monotone in $q$?'],
            [[tag] + [f(lad[tag].get(q)) for q in qs]
             + [yn(_mono([lad[tag][q] for q in qs if q in lad[tag]]))]
             for tag in ('m4 dense', 'M=256 dense', 'M=256 EQ', 'm4 EQ')])
    d.p('The dense and the EQ arms at $M=256$ solve the **identical** $M$ test equations '
        'against the **identical** $K+q$ unknowns; the only difference is that the advection '
        'integral is replaced by an $m$-point rule. The dense ladder is monotone and the EQ '
        'ladder is not, which is already evidence against test-space overfitting. R1, R2 and '
        'R3 test it directly.')

    # ---------------------------------------------------------------- R1/R2
    for name, aud, kind in (('R1 — the field-metric ridge', r1, 'lam'),
                            ('R2 — more tests at fixed $q$', r2, 'rule')):
        if not aud:
            continue
        d.h(2, name)
        for quad in ('dense', 'eq'):
            d.h(3, QUAD_TITLE[quad].capitalize())
            for label, e in sorted(aud['ladders'][quad].items(),
                                   key=lambda kv: (kv[1]['lam_rel'], kv[1]['rule'])):
                head = (f'$\\lambda_{{rel}} = {label}$' if kind == 'lam'
                        else f'$M = {{"m4": 4, "m8": 8, "m16": 16}}[{label}](K+q)$'.replace(
                            '{"m4": 4, "m8": 8, "m16": 16}[' + label + ']',
                            str({'m4': 4, 'm8': 8, 'm16': 16}[label])))
                d.p(f'**{head}** — monotone in $q$ on evolved times: '
                    f"**{yn(e['monotone_evolved'])}**; every rung converged: "
                    f"{yn(e['all_converged'])}; cost within 1.5x of the incumbent: "
                    f"{yn(e['cost_within_1p5x'])}; all-times metric not raised: "
                    f"{yn(e['all_times_not_raised'])}; **passes: {yn(e['passes'])}**"
                    + ('' if e['passes'] else '.')
                    + (f" Top rung keeps the accuracy gain: {yn(e['top_rung_gain_kept'])}."
                       if e['passes'] else ''))
                d.table(['$q$', '$M$', '$m$', 'worst evolved %', 'worst all-times %',
                         '$t=0$ compression %', 'median GPU ms', 'cost vs incumbent',
                         'converged', 'EQ rule valid'], ladder_rows(e))

    # ---------------------------------------------------------------- R3
    d.h(2, 'R3 — the weak residual on held-out test modes')
    src = r1 if (r1 and r1.get('r3')) else (r2 if (r2 and r2.get('r3')) else None)
    if src is None:
        d.p('Not computed.')
    else:
        cb = src['r3']['common_block']
        d.p('Exactly integrated, in NumPy, from the retained per-step bank coefficients: '
            f"the common held-out block is the {cb['count']} modes ranked "
            f"{cb['skip'] + 1}..{cb['skip'] + cb['count']} in the shared ordering, beyond every "
            "arm's $M$ in either job. Values are the worst over the six cases and over the "
            'fifty solved steps of the per-mode RMS residual, normalised by the per-node RMS '
            'of the previous state.')
        for aud in [x for x in (r1, r2) if x and x.get('r3')]:
            rows = []
            by = {x['arm']: x for x in aud['arms']}
            for arm, v in sorted(aud['r3']['arms'].items()):
                row = by[arm]
                rows.append([f'`{arm}`', row['q'], row['quadrature'],
                             sci(row['lam_rel'], 0) if row['lam_rel'] else '0',
                             row['M'],
                             sci(v['in_space']['worst_normalised_per_mode_rms']),
                             sci(v['held_out_next_4M']['worst_normalised_per_mode_rms']),
                             sci(v['held_out_common']['worst_normalised_per_mode_rms']),
                             f(v['held_over_in_common'], 3)])
            d.h(3, f"{aud['mode'].upper()} ({aud['attempt']})")
            d.table(['arm', '$q$', 'quadrature', '$\\lambda_{rel}$', '$M$',
                     'in-space', 'held-out next $4M$', 'held-out common',
                     'held/in (common)'], rows)
            fp = aud['r3']['falsification_pairs']
            d.table(['quadrature', '$q=0$ arm', '$q=16$ arm', 'held-out at $q=0$',
                     'held-out at $q=16$', 'higher at $q=16$?'],
                    [[q, f"`{v['low']}`", f"`{v['high']}`", sci(v['low_held']),
                      sci(v['high_held']), yn(v['held_out_higher_at_high_q'])]
                     for q, v in sorted(fp.items()) if v],
                    caption='The pre-registered falsification test.')

    # ---------------------------------------------------------------- controls
    d.h(2, 'Same-job full-order controls and every arm')
    for aud in jobs:
        d.h(3, f"{aud['mode'].upper()} ({aud['attempt']}) — full-order controls")
        d.table(['control', 'worst all-times %', 'worst evolved %', 'median GPU ms',
                 'repetitions'],
                [[f"`{x['arm']}`", f(x['worst_all_times_percent']),
                  f(x['worst_evolved_percent']), f(x['median_gpu_ms'], 3),
                  x['gpu_ms_repetitions']]
                 for x in aud['arms'] if x['family'] == 'fom'])
        d.h(3, f"{aud['mode'].upper()} ({aud['attempt']}) — every reduced-order arm")
        d.table(['arm', '$q$', '$M$', '$m$', '$\\lambda_{rel}$', '$\\sigma_q$',
                 'tests/unknown', 'worst all %', 'worst evolved %', 'median GPU ms',
                 'median iters', 'budget exits', 'worst joint gradient', 'converged'],
                [[f"`{x['arm']}`", x['q'], x['M'], x['m'] or '—',
                  ('0' if not x['lam_rel'] else sci(x['lam_rel'], 0)),
                  f(x['sigma_q'], 4), f(x['tests_per_unknown'], 2),
                  f(x['worst_all_times_percent']), f(x['worst_evolved_percent']),
                  f(x['median_gpu_ms'], 1), f(x['median_iterations'], 1),
                  x['total_budget_exits'], sci(x['max_joint_stationarity']),
                  yn(x['converged'])]
                 for x in aud['arms'] if x['family'] == 'rom'])

    # ---------------------------------------------------------------- gates
    d.h(2, 'Gates')
    for aud in jobs:
        d.h(3, f"{aud['mode'].upper()} ({aud['attempt']})")
        d.table(['gate', 'passed', 'kind'],
                [[f'`{k}`', yn(v.get('passed')),
                  'informational' if v.get('blocking') is False else 'blocking']
                 for k, v in sorted(aud['checks'].items())
                 if isinstance(v, dict) and 'passed' in v])
        d.h(3, f"{aud['mode'].upper()} — cross-job fidelity")
        rows = []
        for arm, v in sorted((aud['checks'].get('cross_job_fidelity') or {}).get('detail', {}).items()):
            det = v['detail']
            rows.append([f'`{arm}`', det.get('source', '—'), f"`{det.get('comparator', '—')}`",
                         sci(det.get('declared_tolerance'), 0),
                         sci((det.get('worst_all_times_percent') or {}).get('relative_difference')),
                         sci((det.get('worst_evolved_percent') or {}).get('relative_difference')),
                         yn(v['passed'])])
        d.table(['arm', 'source job', 'comparator arm', 'declared tolerance',
                 'relative difference (all-times)', 'relative difference (evolved)', 'passed'],
                rows)

    # ---------------------------------------------------------------- figure
    png = out.with_name(out.name + '-ladders.png')
    pdf = out.with_name(out.name + '-ladders.pdf')
    figure(r1, r2, png, pdf)
    d.h(2, 'Figure')
    d.raw_md(f'![Evolved-times error against q, per lambda and per test-count rule]'
             f'({png.name})')
    d.tex.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{%s}'
                 '\\caption{Worst evolved-times error against $q$, per $\\lambda_{rel}$ (top) '
                 'and per test-count rule (bottom), dense (left) and empirical (right) '
                 'quadrature.}\\end{figure}' % pdf.name)

    # ---------------------------------------------------------------- glossary
    d.h(2, 'Glossary')
    for term, text in GLOSSARY:
        d.md.append(f'- **{term}** — {text}')
        d.tex.append('\\item \\textbf{%s} --- %s' % (_tex_text(term), _tex_text(text)))
    d.md.append('')

    md = out.with_suffix('.md')
    md.write_text('\n'.join(d.md) + '\n')
    body = '\n'.join(d.tex)
    body = body.replace('\\item \\textbf', '\\item \\textbf')      # no-op, keeps intent clear
    first = body.index('\\item ') if '\\item ' in body else None
    if first is not None:
        body = body[:first] + '\\begin{itemize}\n' + body[first:] + '\n\\end{itemize}'
    tex = out.with_suffix('.tex')
    tex.write_text(TEX_PREAMBLE + body + '\n\\end{document}\n')
    print(md, sha(md))
    print(tex, sha(tex))
    print(png, sha(png))


def _mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


TEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable}
\title{q-ridge: the Burgers evolved-times regression is a quadrature fact,\\
not test-space overfitting}
\date{2026-09-16}
\begin{document}\maketitle
"""

GLOSSARY = [
    ('$q$', 'the number of extra correction directions added to the frozen neural head. '
            'The solved state is $u = G(h_\\theta(z) + C_q y)$ with $z \\in \\mathbb{R}^{16}$ '
            'the latent code and $y \\in \\mathbb{R}^{q}$ the corrections. $q = 0$ is the '
            'incumbent reduced-order model.'),
    ('rung', 'one value of $q$ in the ladder. "The ladder" is the sequence of rungs at a '
             'fixed rule for everything else.'),
    ('$M$', 'the number of weak test equations the per-step solve is graded on: $M$ sine '
            'modes of the square, taken in ascending discrete-Laplacian order. The '
            'incumbent rule is $M = 4(K+q)$, four tests per unknown.'),
    ('$m$', 'the number of grid points the empirical-quadrature rule keeps when it '
            'approximates the advection integral. The dense arms use every interior point '
            '(65\\,025 of them) and have no $m$.'),
    ('dense / EQ quadrature', 'dense evaluates the advection integral exactly on the grid; '
                              'EQ replaces it with a nonnegative $m$-point rule fitted '
                              'offline. Same unknowns, same tests, different integral.'),
    ('$\\lambda_{rel}$', 'the dimensionless ridge strength. The solve minimises '
                         '$\\|r_w\\|^2 + \\lambda\\|y\\|_W^2$ with $W$ the field metric and '
                         '$\\lambda = \\lambda_{rel}\\sigma_q^2$; $\\lambda_{rel} = 0$ is the '
                         'incumbent solve exactly.'),
    ('$\\sigma_q$', 'the largest singular value of $\\Phi^\\top G C_q$, the exact linear '
                    'response of the weak residual to the corrections. It is what makes '
                    '$\\lambda_{rel}$ dimensionless and comparable across $q$, $M$ and '
                    'quadrature.'),
    ('worst all-times error', 'the largest relative error over the six output times, the six '
                              'cases included; $t=0$ is included, where the error is the '
                              "decoder's compression of the supplied initial field."),
    ('worst evolved-times error', 'the same quantity over $t>0$ only — the trajectory error '
                                  'proper, with the compression term removed.'),
    ('$t=0$ compression', 'the relative error at $t=0$: how well the reduced model can '
                          'represent the initial field it was handed. It does not depend on '
                          'the ridge, because the initial fit is deliberately not ridged.'),
    ('same-grid metric', 'error measured against the same job\'s own converged full-order '
                         'solve (`fft_tight`) on the same 256-interval grid, so the '
                         'discretisation error cancels.'),
    ('converged', 'every one of the 900 solved steps exited on the gradient, residual or '
                  'tiny-step criterion — no step ran out of its 600-iteration budget — and '
                  'the worst normalised joint gradient is at or inside $10^{-6}$.'),
    ('budget exit', 'a time step that stopped because it hit the iteration budget rather '
                    'than a convergence criterion. Any budget exit disqualifies a rung.'),
    ('held-out test modes', 'sine modes the solve never saw, used only afterwards to measure '
                            'the weak residual. If the solve were overfitting its $M$ tests, '
                            'the residual on these would rise with $q$.'),
    ('in-space residual', 'the same weak residual on the arm\'s own $M$ modes, integrated '
                          'exactly. For a converged dense arm it is essentially zero; for an '
                          'EQ arm it is not, because the solve drove the *quadrature-'
                          'approximated* residual to zero instead.'),
    ('per-mode RMS', 'a residual norm divided by the square root of the number of modes, so '
                     'blocks of different size are comparable; normalised again by the '
                     'per-node RMS of the previous state to make it dimensionless.'),
    ('`fft_tight` / `fft_loose` / `nt1e-2_dt01`', 'same-job full-order Newton solves at '
                                                  'three tolerance/time-step settings. '
                                                  '`fft_tight` is the reference the same-grid '
                                                  'metric is measured against, so its own '
                                                  'error is zero by construction.'),
    ('cost vs incumbent', 'median complete-query GPU time divided by that rung\'s '
                          '$\\lambda_{rel} = 0$ / $M = 4(K+q)$ time. The pre-registered pass '
                          'requires every rung to stay within 1.5x.'),
    ('cross-job fidelity', 'a named arm of this lane reproducing a named arm of an earlier '
                           'audited job to a declared tolerance. It is how the lane proves it '
                           'changed only what it said it changed.'),
]


if __name__ == '__main__':
    main()
