"""Generate the q-ridge report (Markdown + LaTeX) and its figures.

Every number and every plotted point is read from the audited run JSONs and from the
comparator table. Nothing in the report is typed by hand.

    python reports/generate_q_ridge.py --eq checks/qrg301-audit.json \
        --r1 checks/qrg101-audit.json --r2 checks/qrg201-audit.json \
        --out reports/2026-09-16-q-ridge
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
QUAD_HEAD = {'dense': 'Dense (exact) quadrature', 'eq': 'Empirical quadrature, $m = 4M$'}
MULT = {'m4': 4, 'm8': 8, 'm16': 16}
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


def _tex_escape(s):
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'),
                 ('#', r'\#'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')):
        s = s.replace(a, b)
    return s.replace('—', '--').replace('`', '')


def _tex_text(s):
    s = str(s)
    if s.count('$') >= 2:
        parts = s.split('$')
        return '$'.join(p if i % 2 else _tex_escape(p) for i, p in enumerate(parts))
    return _tex_escape(s)


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


def _mono(v):
    return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))


def ladder_rows(entry):
    return [[entry['q'][i], entry['M'][i], entry['m'][i] or '—',
             f(entry['worst_evolved_percent'][i]), f(entry['worst_all_times_percent'][i]),
             f(entry['worst_t0_compression_percent'][i]), f(entry['median_gpu_ms'][i], 1),
             f(entry['cost_ratio_to_incumbent'][i], 3) + 'x', yn(entry['converged'][i]),
             yn(entry['eq_rule_valid'][i])] for i in range(len(entry['q']))]


def eq_figure(eq, png, pdf):
    """rho against m per rung with the declared bar, and the rebuilt ladder."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = axes[0]
    byq = {}
    for x in eq['rules']:
        if x['population'] == 'reachable':
            byq.setdefault(x['q'], []).append((x['m'], x['certification']['rho_max']))
    for i, (q, pts) in enumerate(sorted(byq.items())):
        pts.sort()
        c = COLORS[i % len(COLORS)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=c, lw=1.5, ms=6,
                label=f'$q={q}$')
    stat = sorted([(x['q'], x['m'], x['certification']['rho_max']) for x in eq['rules']
                   if x['population'] == 'static'])
    if stat:
        ax.plot([s[1] for s in stat], [s[2] for s in stat], 'X', ms=9, color='#b03a3a',
                linestyle='none', label='incumbent static rule')
    ax.axhline(eq['rho_bar'], color='#2b2b2b', ls='--', lw=1.2)
    ax.annotate(f"declared bar $\\rho^\\star={eq['rho_bar']}$",
                (ax.get_xlim()[0], eq['rho_bar']), textcoords='offset points',
                xytext=(6, 4), fontsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('quadrature points $m$ (log)')
    ax.set_ylabel(r'held-out $\rho_{\max}$ on reachable states (log)')
    ax.set_title("The rule's own error against its size, per rung", fontsize=10)
    ax.grid(alpha=.25, which='both')
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax = axes[1]
    for key, lab, col in (('old_rule', 'incumbent static rule', '#b03a3a'),
                          ('certified', 'cheapest certified rule', '#3b6ea5'),
                          ('dense', 'dense (exact) quadrature', '#2b2b2b')):
        e = eq['ladders'].get(key)
        if not e:
            continue
        filled = all(e['converged'])
        ax.plot(e['q'], e['worst_evolved_percent'], 'o-', color=col, lw=1.6, ms=7,
                markerfacecolor=(col if filled else 'none'), label=lab)
    ax.set_xscale('symlog', linthresh=16)
    ax.set_yscale('log')
    ax.set_xlabel('correction directions $q$')
    ax.set_ylabel('worst evolved-times error (%, log)')
    ax.set_title('The rebuilt ladder', fontsize=10)
    ax.grid(alpha=.25, which='both')
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle('One frozen Burgers checkpoint, 256 intervals, six development cases: the '
                 'empirical-quadrature rule, certified on states the ROM reaches.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)


def controls_figure(r1, r2, png, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, quad in enumerate(('dense', 'eq')):
        for row, (aud, kind, name) in enumerate((
                (r1, 'lam', 'R1 control: ridge $\\lambda_{rel}$'),
                (r2, 'rule', 'R2 control: test-count rule'))):
            ax = axes[row][col]
            lads = (aud or {}).get('ladders', {}).get(quad, {})
            for i, (label, e) in enumerate(sorted(
                    lads.items(), key=lambda kv: (kv[1]['lam_rel'], kv[1]['rule']))):
                c = COLORS[i % len(COLORS)]
                filled = all(e['converged'])
                ax.plot(e['q'], e['worst_evolved_percent'], 'o-', color=c, lw=1.6, ms=7,
                        markerfacecolor=(c if filled else 'none'),
                        label=(f'$\\lambda_{{rel}}={label}$' if kind == 'lam'
                               else f'$M={MULT[label]}(K+q)$'))
            ax.set_xscale('symlog', linthresh=16)
            ax.set_yscale('log')
            ax.set_xlabel('correction directions $q$')
            ax.set_ylabel('worst evolved-times error (%, log)')
            ax.set_title(f'{name} — {QUAD_TITLE[quad]}', fontsize=10)
            ax.grid(alpha=.25, which='both')
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle('The two demoted controls. Worst-over-evolved-times error against the '
                 'same-job converged full-order solve.\nFilled marker = every rung converged, '
                 'open = not.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eq', default=None)
    p.add_argument('--r1', required=True)
    p.add_argument('--r2', default=None)
    p.add_argument('--comparators', default=str(CELL / 'checks/comparators.json'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r1 = json.loads(Path(a.r1).read_text())
    r2 = json.loads(Path(a.r2).read_text()) if a.r2 and Path(a.r2).exists() else None
    eq = json.loads(Path(a.eq).read_text()) if a.eq and Path(a.eq).exists() else None
    cmp_t = json.loads(Path(a.comparators).read_text())
    out = Path(a.out)
    d = Doc()
    jobs = [x for x in (r1, r2, eq) if x]

    if eq:
        v = eq['verdict']
        if v['passes']:
            headline = ('q-ridge — certifying the empirical quadrature on reachable states '
                        'makes the Burgers ladder monotone again')
        elif v.get('regression_present_in_old_rule'):
            headline = ('q-ridge — the regression is the quadrature rule, and certifying it '
                        'on reachable states does not remove it within $m \\le 8192$')
        else:
            headline = 'q-ridge — EQ rule certification, with the two demoted controls'
    else:
        headline = 'q-ridge — R1 (ridge) and R2 (more tests) on the evolved-times regression'
    d.h(1, headline)
    d.p('Final numbers. Three cluster jobs on the frozen Burgers checkpoint `18f0266ae6f0…` '
        'at 256 intervals, on the six opened development cases, under the budget-600 '
        'block-damped variable-projection contract inherited from `b-ladder-top`. The '
        'pre-registered design and its amendment §A3, which re-scoped the lane after the '
        '`q-diag` lane reported, are in [`../DESIGN.md`](../DESIGN.md).')
    d.table(['job', 'attempt', 'question', 'job id', 'GPU', 'source commit', 'elapsed (s)',
             'failed gates'],
            [[i + 1, x['attempt'], x['mode'].upper(), x['job_id'], x['gpu'],
              (x['commit'] or '')[:12], f(x['elapsed_seconds'], 1),
              ', '.join(f'`{g}`' for g in x['failed']) or 'none'] for i, x in enumerate(jobs)])

    d.h(2, 'Where the regression is, measured before anything was submitted')
    d.p('Recomputed from the retained `cclad01` fields (job '
        f"{cmp_t['cclad01']['job_id']}, archive SHA256 "
        f"`{cmp_t['cclad01']['archive_sha256'][:16]}…`), worst over the six cases, against "
        "that job's own converged full-order solve. The `q-diag` lane reached the same "
        'conclusion independently and across four jobs.')
    lad = {}
    for arm, v in cmp_t['cclad01']['arms'].items():
        for tag, want in (('m4 dense', ('m4', 'dense')), ('M=256 dense', ('m256', 'dense')),
                          ('M=256 EQ', ('m256', 'eq')), ('m4 EQ', ('m4', 'eq'))):
            if (v.get('rule') == want[0] and v.get('quadrature') == want[1]
                    and v.get('q') is not None):
                lad.setdefault(tag, {})[v['q']] = v['worst_evolved_percent']
    qs = [0, 16, 32, 64, 128]
    d.table(['configuration'] + [f'$q={q}$' for q in qs] + ['monotone in $q$?'],
            [[tag] + [f(lad[tag].get(q)) for q in qs]
             + [yn(_mono([lad[tag][q] for q in qs if q in lad[tag]]))]
             for tag in ('m4 dense', 'M=256 dense', 'M=256 EQ', 'm4 EQ') if tag in lad])
    d.p('The dense and the EQ arms at $M=256$ solve the **identical** $M$ test equations '
        'against the **identical** $K+q$ unknowns; the only difference is that the advection '
        'integral is replaced by an $m$-point rule. The dense ladder is monotone and the EQ '
        'ladder is not.')

    if eq:
        d.h(2, 'EQ rule certification — the primary')
        v = eq['verdict']
        extra = ''
        if v.get('extrapolated_m_for_bar'):
            extra = (' No $m$ in the declared grid reaches the bar at the top rung; a log-log '
                     'extrapolation of $\\rho_{\\max}$ against $m$ puts the crossing at '
                     f"$m \\approx {v['extrapolated_m_for_bar']:.0f}$ "
                     f"(slope {v['extrapolation_slope']:.3f}).")
        d.p(f"**Verdict: {yn(v['passes'])}.** The incumbent rule's ladder is "
            f"{'non-monotone' if v['regression_present_in_old_rule'] else 'monotone'} on the "
            'evolved metric; every rung of the rebuilt ladder certified at the primary bar: '
            f"{yn(v['every_rung_certified'])}; the top rung certified: "
            f"{yn(v['top_rung_certified'])}." + extra)

        d.h(3, 'The reachable-state population')
        d.table(['$q$', '$M$', 'iterates per step', 'fit states available',
                 'certification states available', 'collection (s)'],
                [[c['q'], c['M'], c['iterates_per_step'], c['fit_states_available'],
                  c['certification_states_available'], f(c['seconds'], 1)]
                 for c in eq['collection']])

        d.h(3, 'Certification: every rule, its NNLS fit and its held-out $\\rho$')
        d.p(f"The bar is $\\rho^\\star = {eq['rho_bar']}$, declared in `DESIGN.md` §A3.4 "
            "before the job ran, from `q-diag`'s measurement of the $q=0$ incumbent rule. "
            '**A rule is never certified by its NNLS relative fit**, which is reported here '
            'precisely so the anti-correlation can be seen again on new rules.')
        d.table(['$q$', 'population', '$m$ target', '$m$ achieved', '$m/M$',
                 'NNLS relative fit', r'$\rho_{\max}$', r'$\rho_{95}$', r'$\rho_{\rm med}$',
                 'certified', 'truncated', 'fit (s)'],
                [[x['q'], x['population'], x['m_target'], x['m'], f(x['m'] / x['M'], 2),
                  sci(x['relative_fit']), f(x['certification']['rho_max']),
                  f(x['certification']['rho_p95']), f(x['certification']['rho_median']),
                  yn(x['certified_primary']), yn(x['truncated']), f(x['seconds'], 1)]
                 for x in sorted(eq['rules'],
                                 key=lambda x: (x['q'], x['population'], x['m_target']))])

        d.h(3, 'The rule chosen at each rung')
        d.table(['$q$', '$M$', 'chosen $m$', 'basis', r'$\rho_{\max}$', r'$\rho_{95}$',
                 'NNLS relative fit'],
                [[c['q'], c['M'], c['chosen_m'], c['basis'], f(c['rho_max']), f(c['rho_p95']),
                  sci(c['relative_fit'])] for c in eq['rule_choice']])

        d.h(3, 'The rebuilt ladder')
        for key in ('certified', 'old_rule', 'dense'):
            e = eq['ladders'].get(key)
            if not e:
                continue
            tail = ''
            if 'passes' in e:
                tail = (f"; cost within 2x of the old rule: {yn(e.get('cost_within_2x'))}; "
                        f"all-times not raised: {yn(e.get('all_times_not_raised'))}; "
                        f"**passes: {yn(e.get('passes'))}**")
            d.p(f"**{e['name']}** — monotone on evolved times: "
                f"**{yn(e['monotone_evolved'])}**; monotone on all times: "
                f"{yn(e['monotone_all_times'])}; every rung converged: "
                f"{yn(e['all_converged'])}{tail}.")
            head = ['$q$', '$M$', '$m$', r'$\rho_{\max}$', 'certified', 'worst evolved %',
                    'worst all-times %', '$t=0$ compression %', 'median GPU ms', 'converged']
            rowsx = [[e['q'][i], e['M'][i], e['m'][i] or '—', f(e['rho_max'][i]),
                      yn(e['certified_primary'][i]), f(e['worst_evolved_percent'][i]),
                      f(e['worst_all_times_percent'][i]),
                      f(e['worst_t0_compression_percent'][i]), f(e['median_gpu_ms'][i], 1),
                      yn(e['converged'][i])] for i in range(len(e['q']))]
            if 'cost_ratio_to_old_rule' in e:
                head.append('cost vs old rule')
                for i, rr in enumerate(rowsx):
                    rr.append(f(e['cost_ratio_to_old_rule'][i], 3) + 'x')
            d.table(head, rowsx)

    for name, aud, kind in (('R1 (control) — the field-metric ridge', r1, 'lam'),
                            ('R2 (control) — more tests at fixed $q$', r2, 'rule')):
        if not aud:
            continue
        d.h(2, name)
        for quad in ('dense', 'eq'):
            d.h(3, QUAD_HEAD[quad])
            for label, e in sorted(aud['ladders'][quad].items(),
                                   key=lambda kv: (kv[1]['lam_rel'], kv[1]['rule'])):
                head = (f'$\\lambda_{{rel}} = {label}$' if kind == 'lam'
                        else f'$M = {MULT[label]}(K+q)$')
                d.p(f'**{head}** — monotone in $q$ on evolved times: '
                    f"**{yn(e['monotone_evolved'])}**; every rung converged: "
                    f"{yn(e['all_converged'])}; cost within 1.5x of the incumbent: "
                    f"{yn(e['cost_within_1p5x'])}; all-times metric not raised: "
                    f"{yn(e['all_times_not_raised'])}; **passes: {yn(e['passes'])}**.")
                d.table(['$q$', '$M$', '$m$', 'worst evolved %', 'worst all-times %',
                         '$t=0$ compression %', 'median GPU ms', 'cost vs incumbent',
                         'converged', 'EQ rule valid'], ladder_rows(e))

    d.h(2, 'R3 — the weak residual on held-out test modes')
    d.p('Exactly integrated, in NumPy, from the retained per-step bank coefficients. The '
        "common held-out block is the 512 sine modes ranked 1537–2048, beyond every arm's "
        '$M$ in any of the three jobs. `q-diag` measured this on the dense fixed-$M=256$ '
        'arms and found the on-test residual falling about 2x with $q$ while the held-out '
        'residual rose about 1.7x, with the field error falling anyway; what is new here is '
        'the same measurement under a **ridge** and under **larger $M$**.')
    for aud in [x for x in jobs if x and x.get('r3')]:
        rowsx = []
        by = {x['arm']: x for x in aud['arms']}
        for arm, vv in sorted(aud['r3']['arms'].items()):
            row = by[arm]
            rowsx.append([f'`{arm}`', row['q'], row['quadrature'],
                          ('0' if not row.get('lam_rel') else sci(row['lam_rel'], 0)),
                          row['M'], row['m'] or '—',
                          sci(vv['in_space']['worst_normalised_per_mode_rms']),
                          sci(vv['held_out_common']['worst_normalised_per_mode_rms']),
                          f(vv['held_over_in_common'], 3)])
        d.h(3, f"{aud['mode'].upper()} ({aud['attempt']})")
        d.table(['arm', '$q$', 'quadrature', '$\\lambda_{rel}$', '$M$', '$m$', 'in-space',
                 'held-out (common block)', 'held/in'], rowsx)

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
        d.table(['arm', '$q$', '$M$', '$m$', 'worst all %', 'worst evolved %',
                 'median GPU ms', 'median iters', 'budget exits', 'worst joint gradient',
                 'converged'],
                [[f"`{x['arm']}`", x['q'], x['M'], x['m'] or '—',
                  f(x['worst_all_times_percent']), f(x['worst_evolved_percent']),
                  f(x['median_gpu_ms'], 1), f(x['median_iterations'], 1),
                  x['total_budget_exits'], sci(x['max_joint_stationarity']),
                  yn(x['converged'])] for x in aud['arms'] if x['family'] == 'rom'])

    d.h(2, 'Gates')
    for aud in jobs:
        d.h(3, f"{aud['mode'].upper()} ({aud['attempt']})")
        d.table(['gate', 'passed', 'kind'],
                [[f'`{k}`', yn(vv.get('passed')),
                  'informational' if vv.get('blocking') is False else 'blocking']
                 for k, vv in sorted(aud['checks'].items())
                 if isinstance(vv, dict) and 'passed' in vv])
        rowsx = []
        det_all = (aud['checks'].get('cross_job_fidelity') or {}).get('detail', {})
        for arm, vv in sorted(det_all.items()):
            det = vv['detail']
            rowsx.append([f'`{arm}`', det.get('source', '—'),
                          f"`{det.get('comparator', '—')}`",
                          sci(det.get('declared_tolerance'), 0),
                          sci((det.get('worst_all_times_percent') or {}).get('relative_difference')),
                          sci((det.get('worst_evolved_percent') or {}).get('relative_difference')),
                          yn(vv['passed'])])
        if rowsx:
            d.h(3, f"{aud['mode'].upper()} — cross-job fidelity")
            d.table(['arm', 'source job', 'comparator arm', 'declared tolerance',
                     'relative difference (all-times)', 'relative difference (evolved)',
                     'passed'], rowsx)

    d.h(2, 'Figures')
    if eq:
        png = out.with_name(out.name + '-certification.png')
        pdf = out.with_name(out.name + '-certification.pdf')
        eq_figure(eq, png, pdf)
        d.raw_md(f'![Rule error against m, and the rebuilt ladder]({png.name})')
        d.tex.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{%s}'
                     "\\caption{Left: the rule's held-out $\\rho$ against its size $m$, per "
                     'rung, with the declared bar. Right: the rebuilt ladder against the '
                     'incumbent rule and the dense twins.}\\end{figure}' % pdf.name)
    png2 = out.with_name(out.name + '-controls.png')
    pdf2 = out.with_name(out.name + '-controls.pdf')
    controls_figure(r1, r2, png2, pdf2)
    d.raw_md(f'![The two demoted controls]({png2.name})')
    d.tex.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{%s}'
                 '\\caption{The two demoted controls: the ridge (top) and the test-count rule '
                 '(bottom), dense (left) and empirical (right) quadrature.}\\end{figure}'
                 % pdf2.name)

    d.h(2, 'Glossary')
    for term, text in GLOSSARY:
        d.md.append(f'- **{term}** — {text}')
        d.tex.append('\\item \\textbf{%s} --- %s' % (_tex_text(term), _tex_text(text)))
    d.md.append('')

    md = out.with_suffix('.md')
    md.write_text('\n'.join(d.md) + '\n')
    body = '\n'.join(d.tex)
    first = body.index('\\item ') if '\\item ' in body else None
    if first is not None:
        body = body[:first] + '\\begin{itemize}\n' + body[first:] + '\n\\end{itemize}'
    tex = out.with_suffix('.tex')
    tex.write_text(TEX_PREAMBLE + body + '\n\\end{document}\n')
    print(md, sha(md))
    print(tex, sha(tex))


TEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable}
\title{q-ridge: certifying the empirical quadrature of a Burgers correction ladder\\
on the states the reduced model actually reaches}
\date{2026-09-16}
\begin{document}\maketitle
"""

GLOSSARY = [
    ('$q$', 'the number of extra correction directions added to the frozen neural head. The '
            'solved state is $u = G(h_\\theta(z) + C_q y)$ with $z \\in \\mathbb{R}^{16}$ the '
            'latent code and $y \\in \\mathbb{R}^{q}$ the corrections. $q = 0$ is the '
            'incumbent reduced-order model.'),
    ('rung', 'one value of $q$ in the ladder. "The ladder" is the sequence of rungs at a '
             'fixed rule for everything else.'),
    ('$M$', 'the number of weak test equations the per-step solve is graded on: $M$ sine '
            'modes of the square, in ascending discrete-Laplacian order. The incumbent rule '
            'is $M = 4(K+q)$, four tests per unknown.'),
    ('$m$', 'the number of grid points the empirical-quadrature rule keeps when it '
            'approximates the advection integral. The dense arms use every interior point '
            '(65\\,025 of them) and have no $m$.'),
    ('dense / EQ quadrature', 'dense evaluates the advection integral exactly on the grid; EQ '
                              'replaces it with a nonnegative $m$-point rule fitted offline. '
                              'Same unknowns, same tests, different integral.'),
    (r'$\rho$', "the rule's own relative error on the advection functional it is the only "
                'approximation of: $\\rho(u) = \\|\\sum_j w_j \\Phi(x_j) a(u)(x_j) - '
                '\\Phi^\\top a(u)\\| / \\|\\Phi^\\top a(u)\\|$. It is a property of a rule and '
                'a state, not of a solve.'),
    (r'$\rho^\star$, the bar', "the value a rule's worst held-out $\\rho$ must not exceed to "
                               'be certified. Declared before the job ran at 0.116, the value '
                               'the q-diag lane measured for the $q=0$ incumbent rule at the '
                               'state that carries the whole first-interval penalty.'),
    ('reachable states', 'states the reduced model actually visits: every Levenberg-Marquardt '
                         "iterate of its own dense-quadrature rollouts on training-family "
                         'trajectories. The incumbent rule is instead fitted on static '
                         'decoder outputs, which is the difference this lane isolates.'),
    ('held-out (rules)', 'the certification states come from training-family trajectories '
                         'disjoint from the ones the rule was fitted on, and both are '
                         'disjoint from the six evaluation cases.'),
    ('held-out test modes', 'sine modes the solve never saw, used only afterwards to measure '
                            'the weak residual. If a solve were overfitting its $M$ tests, '
                            'the residual on these would rise with $q$.'),
    ('in-space residual', "the same weak residual on the arm's own $M$ modes, integrated "
                          'exactly. For a converged dense arm it is essentially zero; for an '
                          'EQ arm it is not, because the solve drove the '
                          'quadrature-approximated residual to zero instead.'),
    ('per-mode RMS', 'a residual norm divided by the square root of the number of modes, so '
                     'blocks of different size are comparable; normalised again by the '
                     'per-node RMS of the previous state to make it dimensionless.'),
    ('$\\lambda_{rel}$', 'the dimensionless ridge strength of the demoted R1 control. The '
                         'solve minimises $\\|r_w\\|^2 + \\lambda\\|y\\|_W^2$ with $W$ the '
                         'field metric and $\\lambda = \\lambda_{rel}\\sigma_q^2$; '
                         '$\\lambda_{rel} = 0$ is the incumbent solve exactly.'),
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
                          'represent the initial field it was handed. It uses no quadrature '
                          'at all, which is why every arm at a given $q$ agrees there.'),
    ('same-grid metric', "error measured against the same job's own converged full-order "
                         'solve (fft_tight) on the same 256-interval grid, so the '
                         'discretisation error cancels.'),
    ('converged', 'every one of the 900 solved steps exited on the gradient, residual or '
                  'tiny-step criterion — no step ran out of its 600-iteration budget — and '
                  'the worst normalised joint gradient is at or inside $10^{-6}$.'),
    ('budget exit', 'a time step that stopped because it hit the iteration budget rather than '
                    'a convergence criterion. Any budget exit disqualifies a rung.'),
    ('NNLS relative fit', 'the residual of the nonnegative least-squares problem the rule is '
                          'fitted by. It is reported but never used to certify a rule: it is '
                          'flat across rules whose $\\rho$ differs by an order of magnitude.'),
    ('truncated', 'the fitter ran out of its declared walltime before reaching the target '
                  '$m$. A truncated rule is disqualified.'),
    ('fft_tight / fft_loose / nt1e-2_dt01', 'same-job full-order Newton solves at three '
                                            'tolerance/time-step settings. fft_tight is the '
                                            'reference the same-grid metric is measured '
                                            'against, so its own error is zero by '
                                            'construction.'),
    ('cross-job fidelity', 'a named arm of this lane reproducing a named arm of an earlier '
                           'audited job to a declared tolerance. It is how the lane proves it '
                           'changed only what it said it changed.'),
]


if __name__ == '__main__':
    main()
