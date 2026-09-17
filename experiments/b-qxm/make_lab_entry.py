"""Generate the dated LAB-LOG entry for the b-qxm lane from the audited JSONs.

    python make_lab_entry.py --analysis reports/analysis.json --summary reports/summary.json \
        --audits checks/bqx101-audit.json ... --report reports/2026-09-17-b-qxm.md \
        --out checks/2026-09-17-lab-log-entry.md

Every number is read from the generated files; the prose around them is fixed text with the
retractions section filled from `--retractions <file>` (plain Markdown written by hand, the one
place a human sentence belongs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        return f'{x:.{d}f}'
    return str(x)


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--audits', nargs='+', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--retractions', default=None)
    p.add_argument('--archives', nargs='*', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--date', default='2026-09-17')
    a = p.parse_args()
    an = json.loads(Path(a.analysis).read_text())
    au = [json.loads(Path(x).read_text()) for x in a.audits]
    au.sort(key=lambda x: {'G1': 0, 'G2': 1, 'S1': 2}.get(x['question'], 9))
    v = an['verdict']
    sp = an['spans']['worst_evolved_percent']
    dec = an['decomposition']['worst_evolved_percent']
    sat = an.get('saturation')
    L = []
    L.append(f'## {a.date}')
    L.append(f"### b-qxm — {v['sentence']}")
    L.append('')
    L.append('The coordinator asked whether the passing Burgers $256^2$ dense correction ladder — which raises '
             'the test count $M = 4(K+q)$ together with the rank $q$ — owes its $3.6\\times$ evolved-error span to '
             'the rank, to the test count, or to both. This lane ran the two factors **crossed**: rows '
             '$q \\in \\{0, 16, 32, 64, 128, 256\\}$ against fixed $M \\in \\{256, 1088\\}$, scheduled '
             '$M \\in \\{4, 8, 16\\}(K+q)$, the bridge cells $(q_k, M_{k+1})$, a solver control, and a saturation '
             'sweep in $M$ at $q = 0$ and $q = 64$; dense quadrature, budget 600, the frozen incumbent checkpoint, '
             'the six opened development cases, same-job FOM controls, three timed repetitions. Pre-registered '
             'protocol, decomposition and falsification clause: `experiments/b-qxm/DESIGN.md` (amendments A0–A1 '
             'and later, appended). Nothing merged, nothing pushed.')
    L.append('')
    L.append('Worktree `worktrees/2026-09-17-b-qxm`, branch `exp/2026-09-17-b-qxm`, forked from `exp/2026-09-16-q-ridge` '
             'at `7dc970fc`. Namespace `/cluster/tufts/paralab/tawal01/b_qxm_20260917/`. Jobs: '
             + '; '.join(f"{x['question']} `{x['attempt']}` = {x['job_id']} on `{x['gpu']}`, source `{(x['commit'] or '')[:12]}`, "
                         f"{f(x['elapsed_seconds'], 0)} s, failed gates: {', '.join('`' + g + '`' for g in x['failed']) or 'none'}"
                         for x in au)
             + '. All printed `jax_backend=gpu`, float64, highest matmul precision; checksum-collected, independently '
               'NumPy-audited, archived as Git chunks, remote attempt directories deleted.')
    L.append('')
    L.append('**Verdict against the pre-registered rules (DESIGN §6).**')
    L.append('')
    L.append('| rule | value |')
    L.append('|---|---|')
    L.append(f"| headline the paper should carry | {v['headline']} |")
    L.append(f"| fixed $M=1088$ ladder monotone / every rung converged | {yn(v['fixed1088_monotone'])} / {yn(v['fixed1088_all_converged'])} |")
    L.append(f"| $q$-span at fixed $M=1088$ ($q = 0 \\to 256$) | {f(v['span_q_at_M1088'], 3)}x |")
    L.append(f"| $q$-span at fixed $M=256$ ($q = 0 \\to 128$) | {f(v['span_q_at_M256_to_q128'], 3)}x |")
    L.append(f"| rank claim false ($<1.5\\times$ at every fixed $M$) | {yn(v['rank_claim_false'])} |")
    L.append(f"| H(rank) false (no saturation in $M$) | {yn(v['H_rank_false'])} |")
    L.append(f"| H(tests) false (rank share $\\ge 0.5$) | {yn(v['H_tests_false'])} |")
    L.append(f"| corner-path shares: rank / test count | {f(v['share_q_corner'], 3)} / {f(v['share_M_corner'], 3)} |")
    L.append(f"| rung-path test-count share | {f(v['share_M_rung'], 3)} |")
    L.append(f"| the two paths disagree by $>0.15$ | {yn(v['paths_disagree'])} |")
    L.append('')
    L.append('**Spans (worst evolved %, same-grid, converged cells only).**')
    L.append('')
    L.append('| ladder | rungs | values | monotone | span |')
    L.append('|---|---|---|---|---|')
    for M, x in sorted(sp['fixed_M'].items(), key=lambda kv: int(kv[0])):
        L.append(f"| fixed $M={M}$ | $q$ = {', '.join(map(str, x['q']))} | {' / '.join(f(t) for t in x['values'])} | {yn(x['monotone'])} | "
                 f"{(f(x['span'], 3) + 'x') if x['span'] is not None else 'unavailable'} |")
    for lab, x in sorted(sp['scheduled'].items()):
        L.append(f"| scheduled `{lab}` | {', '.join(f'({q},{M})' for q, M in x['cells'])} | {' / '.join(f(t) for t in x['values'])} | {yn(x['monotone'])} | {f(x['span'], 3)}x |")
    L.append('')
    w = v.get('fixed1088_within_job')
    if w:
        L.append(f"**The pure-rank ladder as an operating-point family, inside one job** (`{w['job']}`, {w['job_id']}, so the "
                 'costs are comparable): rungs $q$ = ' + ', '.join(map(str, w['q'])) + ' at fixed $M = 1088$, worst evolved '
                 + ' / '.join(f(t) for t in w['values']) + ' %, median GPU '
                 + ' / '.join(f(t, 0) for t in w['median_gpu_ms']) + ' ms — error span '
                 f"**{f(w['error_span'], 3)}x**, cost span **{f(w['cost_span'], 3)}x**, {w['non_dominated_points']} non-dominated "
                 f"points, monotone {yn(w['monotone_error'])}. It **{'passes' if w['passes_tunability_bar'] else 'fails'}** the "
                 "project's standing tunability bar (monotone, $\\ge 3$ non-dominated points, $\\ge 2\\times$ in both error "
                 'and cost, nothing early-stopped) **with the test count held fixed**, which is what the audit\'s objection asked for.')
        L.append('')
    L.append('| fixed $q$ | $M$ | values | monotone in $M$ | span (max/min) | span over $M \\ge 2(K+q)$ |')
    L.append('|---|---|---|---|---|---|')
    for q, x in sorted(sp['fixed_q'].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {q} | {', '.join(map(str, x['M']))} | {' / '.join(f(t) for t in x['values'])} | {yn(x['monotone'])} | {f(x['span'], 3)}x | "
                 f"{(f(x['span_M_ge_2x'], 3) + 'x') if x['span_M_ge_2x'] else '—'} |")
    L.append('')
    if dec.get('corner'):
        c = dec['corner']
        L.append(f"**Decomposition of the scheduled $4(K+q)$ ladder.** Corner path {c['path']}: $\\log S = {c['log_span']:.4f}$ "
                 f"($S = {c['span']:.3f}\\times$) $= \\Delta_M + \\Delta_q = {c['delta_M']:.4f} + {c['delta_q']:.4f}$, shares test count "
                 f"{100 * c['share_M']:.1f} %, rank {100 * c['share_q']:.1f} %.")
    r = dec.get('rung') or {}
    if r.get('complete'):
        L.append(f"Rung path through the bridge cells: test count {100 * r['share_M']:.1f} %, rank {100 * r['share_q']:.1f} %; per rung "
                 + ', '.join(f"({x['from_'][0]}→{x['to'][0]}: {100 * (x['share_M'] or 0):.0f} % tests)" for x in r['rungs']) + '.')
    anv = dec.get('anova')
    if anv:
        L.append(f"Variance shares on the balanced $5 \\times 2$ sub-grid: rank {100 * anv['frac_q']:.1f} %, test count "
                 f"{100 * anv['frac_M']:.1f} %, interaction {100 * anv['frac_interaction']:.1f} %.")
    L.append('')
    if sat:
        L.append(f"**Saturation (S1, {sat['job_id']}).** "
                 + ' '.join(f"$q={q}$: $M^\\star = {x['M_star']}$, error {f(x['error_at_M_star'])} %, cost {f(x['cost_ratio_at_M_star'], 3)}x the "
                            f"$M = {x['reference_M']}$ cell (within-job), cost-neutral claim {yn(x['cost_neutral_claim'])}, monotone in $M$: {yn(x['monotone'])}; "
                            f"curve {' / '.join(f(c['value']) for c in x['curve'])} % at $M$ = {', '.join(str(c['M']) for c in x['curve'])}."
                            for q, x in sorted(sat['per_q'].items(), key=lambda kv: int(kv[0]))))
        L.append('')
    anc = an.get('anchors') or []
    if anc:
        worst = max(max(x['rel_evolved'], x['rel_all']) for x in anc)
        L.append(f"**Cross-job anchors.** {len(anc)} cells appear in two jobs; worst relative difference on either metric "
                 f"{worst:.2e}; all within their declared tolerance: {yn(all(x['passed'] for x in anc))}. No cost ratio was formed across jobs.")
        L.append('')
    L.append('**Gates.** ' + ' '.join(f"{x['question']} ({x['attempt']}): " + '; '.join(f"`{k}` {yn(vv.get('passed'))}" for k, vv in sorted(x['checks'].items())
                                                                                        if isinstance(vv, dict) and 'passed' in vv and not k.startswith('reproduces_')) + '.'
                                    for x in au))
    L.append('')
    fid = []
    for x in au:
        for k, vv in sorted(((x['checks'].get('cross_job_fidelity') or {}).get('detail') or {}).items()):
            d = vv['detail']
            fid.append(f"| {x['attempt']} | `{k.split('__')[0]}` | {d.get('source')} `{d.get('comparator')}` | {yn(d.get('unconditional'))} | "
                       f"{d.get('declared_tolerance'):.0e} | {(d.get('achieved_worst_relative_difference') or 0):.2e} | {yn(vv['passed'])} |")
    if fid:
        L.append('**Cross-job fidelity.**')
        L.append('')
        L.append('| job | arm | source / comparator | unconditional | tolerance | achieved | passed |')
        L.append('|---|---|---|---|---|---|---|')
        L += fid
        L.append('')
    L.append('**What was wrong and got retracted.**')
    L.append('')
    if a.retractions and Path(a.retractions).exists():
        L.append(Path(a.retractions).read_text().strip())
    else:
        L.append('(none recorded)')
    L.append('')
    rep = Path(a.report)
    L.append(f"Source-generated report: `experiments/b-qxm/reports/{rep.name}` (SHA256 `{hashlib.sha256(rep.read_bytes()).hexdigest()}`) "
             f"with `summary.json`, `analysis.json`, its two figures and its generator beside it; the design audit is "
             '`experiments/b-qxm/reports/design-audit.md`.')
    for arc in a.archives:
        meta = json.loads(Path(arc).read_text())
        L.append(f"Raw archive `{Path(arc).parent.name}` Git-tracked as bounded chunks: whole SHA256 `{meta['sha256']}` ({len(meta['chunks'])} chunks).")
    L.append('')
    Path(a.out).write_text('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
