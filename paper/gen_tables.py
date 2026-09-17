#!/usr/bin/env python3
"""Generate every table and every prose number of the ICLR 2027 manuscript.

Rule, without exception: no number in the LaTeX is typed by hand.  This script
reads each lane's machine-readable output (summary.json / analysis.json / audit
JSONs, and where a lane wrote only a generated Markdown report, that report's
tables) and emits

    tables/T01_problems.tex ... tables/T18_lshape.tex   booktabs tabulars
    tables/numbers.tex                                   \\newcommand macros for
                                                         every number used in prose
    tables/provenance.json                               every file read, with SHA256

Where a lane is still running or has not written its output, the table cell or
macro is the placeholder  \\gen{pending: <lane>}  and nothing is estimated.

Run:   /home/tahmid/Dev/.venv/bin/python paper/gen_tables.py
       (CPU only; no GPU, no cluster, no JAX)
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'tables'
OUTMD = HERE / 'tables-md'
ROOT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees')

# --------------------------------------------------------------------------- sources
SOURCES = {
    # 2026-09-17 lanes
    'qxm_analysis': '2026-09-17-b-qxm/experiments/b-qxm/reports/analysis.json',
    'qxm_report': '2026-09-17-b-qxm/experiments/b-qxm/reports/2026-09-17-b-qxm.md',
    'panel_summary': '2026-09-17-b-panel/experiments/b-panel/reports/summary.json',
    'panel_audit': '2026-09-17-b-panel/experiments/b-panel/checks/bpn101-audit.json',
    'panel_report': '2026-09-17-b-panel/experiments/b-panel/reports/2026-09-17-b-panel.md',
    'nosecond_summary': '2026-09-17-no-second/experiments/no-second/reports/summary.json',
    'wladder_summary': '2026-09-17-w-ladder/experiments/w-ladder/reports/summary.json',
    'wladder_report': '2026-09-17-w-ladder/experiments/w-ladder/reports/2026-09-17-w-ladder.md',
    'plinear_summary': '2026-09-17-p-linear/experiments/p-linear/reports/summary.json',
    'plinear_report': '2026-09-17-p-linear/experiments/p-linear/reports/2026-09-17-p-linear.md',
    'plinear_verdicts': '2026-09-17-p-linear/experiments/p-linear/reports/verdicts.json',
    'lshape_summary': '2026-09-17-lshape/experiments/lshape/reports/summary.json',
    'lshape_report': '2026-09-17-lshape/experiments/lshape/reports/2026-09-17-lshape.md',
    'eqtop_summary': '2026-09-17-b-eqtop/experiments/b-eqtop/reports/summary.json',
    'eqtop_report': '2026-09-17-b-eqtop/experiments/b-eqtop/reports/2026-09-17-b-eqtop.md',
    'ns2d_summary': '2026-09-17-ns2d/experiments/ns2d/reports/summary.json',
    # pending lanes (absent on disk today; listed so the placeholder names the lane)
    'seeds_summary': '2026-09-17-b-seeds/experiments/b-seeds/reports/summary.json',
    'panel1024_summary': '2026-09-17-b-panel/experiments/b-panel/reports/summary-1024.json',
    'lshape_solve_summary': '2026-09-17-lshape/experiments/lshape/reports/summary-solve.json',
    # inherited cells
    'abl01': '2026-09-14-head-ablation/experiments/head-ablation/checks/abl01-audit.json',
    'pabl01': '2026-09-14-head-ablation/experiments/head-ablation/checks/pabl01-audit.json',
    'mesh_ladder': '2026-09-14-mesh-ladder/experiments/mesh-ladder/reports/2026-09-14-frozen-checkpoint-mesh-ladder.json',
    'tuning02': '2026-09-14-no-burgers/experiments/neural-operator-burgers/artifacts/tuning02/output-index.json',
    'poisson_caps': '2026-09-14-head-ablation/experiments/head-ablation/checks/cold-start-caps/poisson_cap_curve.json',
    'burgers_caps': '2026-09-14-head-ablation/experiments/head-ablation/checks/cold-start-caps/burgers_cap_curve.json',
    'speed_report': '2026-09-16-b-speed/experiments/b-speed/reports/2026-09-16-b-speed.md',
    'speed_audit_spd01': '2026-09-16-b-speed/experiments/b-speed/artifacts/spd01/audit.json',
    'speed_audit_fine01': '2026-09-16-b-speed/experiments/b-speed/artifacts/fine01/audit.json',
    'speed_audit_comp01': '2026-09-16-b-speed/experiments/b-speed/artifacts/comp01/audit.json',
    'headtrain_eval': '2026-09-16-b-head-train/experiments/b-head-train/checks/eval01-audit.json',
    'headtrain_train': '2026-09-16-b-head-train/experiments/b-head-train/checks/train02-audit.json',
    'cclad01': '2026-09-15-cheap-corrections/experiments/cheap-corrections/checks/cclad01-audit.json',
    # earlier heat cell of the same decoder family (main branch reports/, absolute path handled in load)
    'heat_report': '../reports/2026-09-10-heat-linear-bank-comparison.md',
    'heat_manifest': '../reports/2026-09-10-heat-linear-bank-comparison.manifest.json',
}

PROV: dict[str, dict] = {}
MACROS: dict[str, str] = {}
PENDING: list[tuple[str, str]] = []   # (macro or table, lane)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(key: str):
    path = (ROOT / SOURCES[key]).resolve()
    if not path.exists():
        PROV[key] = {'path': str(path), 'reachable': False}
        return None
    PROV[key] = {'path': str(path), 'reachable': True, 'sha256': sha256(path)}
    if path.suffix == '.json':
        return json.loads(path.read_text())
    return path.read_text()


# --------------------------------------------------------------------------- formatting
def tex_escape(s: str) -> str:
    return (str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_')
            .replace('%', r'\%').replace('&', r'\&').replace('#', r'\#'))


def tt(s) -> str:
    return r'\texttt{' + tex_escape(s) + '}'


def f(x, d=4) -> str:
    if x is None:
        return '---'
    return f'{x:.{d}f}'


def pct(x, d=4) -> str:
    """A fraction or a percent, as a percent with d decimals; caller says which."""
    return f(x, d)


def ms(x, d=1) -> str:
    return f(x, d)


def ratio(x, d=2) -> str:
    return '---' if x is None else f'{x:.{d}f}$\\times$'


def yn(b) -> str:
    if b is None:
        return '---'
    return 'yes' if b else 'no'


def gen(lane: str, what: str = '') -> str:
    PENDING.append((what or lane, lane))
    return r'\gen{pending: ' + tex_escape(lane) + '}'


def macro(name: str, value) -> None:
    assert re.fullmatch(r'[A-Za-z]+', name), name
    MACROS[name] = str(value)


LAST_MD: dict[str, str] = {}   # name -> markdown, filled by tabular() and consumed by write()


def tex2md(cell: str) -> str:
    """LaTeX table-cell text -> GitHub-flavoured Markdown (math kept as $...$)."""
    c = str(cell)
    c = re.sub(r'\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}', lambda m: '`' + m.group(1).replace('\\_', '_').replace('\\%', '%').replace('\\&', '&').replace('\\#', '#') + '`', c)
    c = re.sub(r'\\textbf\{([^{}]*)\}', r'**\1**', c)
    c = re.sub(r'\\emph\{([^{}]*)\}', r'*\1*', c)
    c = re.sub(r'\\gen\{([^{}]*)\}', r'**[PENDING: \1]**', c)
    c = c.replace('\\ldots', '…').replace('\\%', '%').replace('\\_', '_').replace('\\&', '&').replace('\\#', '#')
    c = c.replace('\\\\', ' ').replace('\\ ', ' ').replace('~', ' ')
    c = c.replace('---', '—').replace('--', '–')
    c = c.replace('|', '\\|')
    return c.strip()


def tabular(cols, rows, align, size=r'\small'):
    out = [size, r'\begin{tabular}{' + align + '}', r'\toprule',
           ' & '.join(cols) + r' \\', r'\midrule']
    md = ['| ' + ' | '.join(tex2md(c) for c in cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for r in rows:
        if r == 'MIDRULE':
            out.append(r'\midrule')
        else:
            out.append(' & '.join(r) + r' \\')
            md.append('| ' + ' | '.join(tex2md(c) for c in r) + ' |')
    out += [r'\bottomrule', r'\end{tabular}']
    LAST_MD['_pending'] = '\n'.join(md) + '\n'
    return '\n'.join(out) + '\n'


def write(name: str, body: str, header: str = '') -> None:
    OUT.mkdir(exist_ok=True); OUTMD.mkdir(exist_ok=True)
    (OUT / name).write_text('% GENERATED by paper/gen_tables.py -- do not edit.\n'
                            + ('% ' + header + '\n' if header else '') + body)
    md = LAST_MD.pop('_pending', None)
    if md is None:   # a bare placeholder, not a tabular
        md = tex2md(body.strip()) + '\n'
    stem = name.rsplit('.', 1)[0]
    (OUTMD / (stem + '.md')).write_text(f'<!-- GENERATED by paper/gen_tables.py from {header or "run records"} -- do not edit -->\n' + md)


def md_table(text: str, first_header_cell: str):
    """Parse the first Markdown table whose header starts with `first_header_cell`.
    Returns (headers, rows) with cell strings stripped of backticks and bold."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('|') and line.strip('| ').startswith(first_header_cell):
            headers = [c.strip().strip('`*') for c in line.strip().strip('|').split('|')]
            rows = []
            for l in lines[i + 2:]:
                if not l.startswith('|'):
                    break
                cells = [c.strip().replace('**', '').strip('`') for c in l.strip().strip('|').split('|')]
                rows.append(cells)
            return headers, rows
    return None, None


def hexprefix(s, n=12):
    return s[:n] + '\\ldots' if s and len(s) > n else (s or '---')


# --------------------------------------------------------------------------- pivots
def pivot(rows, key='subject', metric='metric', value='value'):
    d = defaultdict(dict)
    for r in rows:
        d[r[key]][r[metric]] = r[value]
    return d


# =========================================================================== T3 / T5 panel
def build_panel():
    summ = load('panel_summary')
    audit = load('panel_audit')
    if summ is None:
        write('T03_tunability.tex', gen('b-panel', 'T3'))
        write('T05_panel_all.tex', gen('b-panel', 'T5'))
        return
    rows = summ['rows']
    job = rows[0]['job_id']
    P = pivot(rows)
    fam = {r['subject']: r['family'] for r in rows}
    qk = {r['subject']: r.get('q_or_k') for r in rows}
    Mof = {r['subject']: r.get('M') for r in rows}
    quad = {r['subject']: r.get('quadrature') for r in rows}
    tol = {r['subject']: r.get('tol') for r in rows}

    gpu = audit.get('gpu') if audit else None
    commit = audit.get('commit') if audit else None
    # checkpoint sha: search the audit text for the frozen Burgers checkpoint hash
    ck = None
    if audit:
        m = re.search(r'\b(18f0266a[0-9a-f]{56})\b', json.dumps(audit))
        ck = m.group(1) if m else None
    macro('provPanelJob', job)
    macro('provPanelGpu', gpu or '---')
    macro('provPanelCommit', hexprefix(commit))
    macro('provPanelCkpt', hexprefix(ck) if ck else 'incumbent (gate checkpoint\\_unchanged; hash in T2, tuning row)')

    # ---- T3: the ladders side by side (dense / EQ 1e-6 / EQ 1e-3) + POD + FOM
    ladders = [('dense', 'dense_g1em06'), ('EQ, tol $10^{-6}$', 'eqcert_g1em06'),
               ('EQ, tol $10^{-3}$', 'eqcert_g0p001')]
    qs = [0, 16, 32, 64, 128, 256]
    Ms = {0: 64, 16: 128, 32: 192, 64: 320, 128: 576, 256: 1088}
    t3rows = []
    for q in qs:
        cells = [str(q), str(Ms[q])]
        for _, suf in ladders:
            s = f'q{q}_M{Ms[q]}_{suf}'
            p = P[s]
            cells += [pct(p['worst_evolved_percent']), pct(p['worst_all_times_percent']),
                      ms(p['median_gpu_ms'])]
        cells.append(pct(P[f'q{q}_M{Ms[q]}_dense_g1em06']['worst_t0_compression_percent']))
        t3rows.append(cells)
    cols = ['$q$', '$M$']
    for lab, _ in ladders:
        cols += [f'{lab}: evolved \\%', 'all \\%', 'ms']
    cols.append('$t{=}0$ \\%')
    write('T03_tunability.tex', tabular(cols, t3rows, 'rr' + 'rrr' * 3 + 'r', r'\scriptsize'),
          f'b-panel job {job}')

    # ladder spans (from the '*' rows if present; else compute)
    def ladder_stats(suf, ql):
        errs = [P[f'q{q}_M{Ms[q]}_{suf}']['worst_evolved_percent'] for q in ql]
        cost = [P[f'q{q}_M{Ms[q]}_{suf}']['median_gpu_ms'] for q in ql]
        mono = all(errs[i] >= errs[i + 1] for i in range(len(errs) - 1))
        conv = all(P[f'q{q}_M{Ms[q]}_{suf}']['converged'] for q in ql)
        return errs, cost, mono, conv
    e, c, mono, conv = ladder_stats('dense_g1em06', qs)
    macro('nPanelDenseErrSpan', f'{e[0]/e[-1]:.2f}')
    macro('nPanelDenseCostSpan', f'{c[-1]/c[0]:.2f}')
    macro('nPanelDenseMonotone', yn(mono)); macro('nPanelDenseConverged', yn(conv))
    e6, c6, mono6, _ = ladder_stats('eqcert_g1em06', qs)
    macro('nPanelEqMonotone', yn(mono6))
    macro('nPanelEqQtwofiftysixEvolved', pct(P['q256_M1088_eqcert_g1em06']['worst_evolved_percent']))
    macro('nPanelEqQonetwentyeightEvolved', pct(P['q128_M576_eqcert_g1em06']['worst_evolved_percent']))
    macro('nPanelEqQtwofiftysixBasis', str(P['q256_M1088_eqcert_g1em06'].get('rule_basis')))
    macro('nPanelEqQonetwentyeightBasis', str(P['q128_M576_eqcert_g1em06'].get('rule_basis')))
    macro('nPanelEqQsixtyfourBasis', str(P['q64_M320_eqcert_g1em06'].get('rule_basis')))

    # EQ vs dense at matched (q,M): cost and error
    for q, nm in [(0, 'Zero'), (128, 'OneTwoEight')]:
        d = P[f'q{q}_M{Ms[q]}_dense_g1em06']; eq6 = P[f'q{q}_M{Ms[q]}_eqcert_g1em06']
        eq3 = P[f'q{q}_M{Ms[q]}_eqcert_g0p001']
        macro(f'nPanelDenseMs{nm}', ms(d['median_gpu_ms']))
        macro(f'nPanelEqMs{nm}', ms(eq6['median_gpu_ms']))
        macro(f'nPanelEqLooseMs{nm}', ms(eq3['median_gpu_ms']))
        macro(f'nPanelDenseErr{nm}', pct(d['worst_evolved_percent']))
        macro(f'nPanelEqErr{nm}', pct(eq6['worst_evolved_percent']))
        macro(f'nPanelEqLooseErr{nm}', pct(eq3['worst_evolved_percent']))
        macro(f'nPanelEqSpeedup{nm}', f'{d["median_gpu_ms"]/eq6["median_gpu_ms"]:.2f}')
        macro(f'nPanelTolSaving{nm}', f'{100*(1-eq3["median_gpu_ms"]/eq6["median_gpu_ms"]):.0f}')

    # FOM controls and the frontier statement
    foms = [s for s in P if fam[s] == 'fom']
    reduced = [s for s in P if fam[s] in ('rom', 'fast', 'pod', 'free')]
    macro('nPanelReducedCount', str(len(reduced)))
    macro('nPanelSubjectCount', str(len([s for s in P if s != '*'])))
    nd = P.get('*', {})
    nd_evolved = nd.get('nondominated_gpu_evolved_admissible')
    nd_all = nd.get('nondominated_gpu_all_admissible')
    def count_reduced(lst):
        return None if lst is None else sum(1 for s in lst if s in reduced)
    macro('nPanelReducedNonDomEvolved', str(count_reduced(nd_evolved)))
    macro('nPanelReducedNonDomAll', str(count_reduced(nd_all)))
    ndr = nd.get('nondominated_gpu_all_reduced_only') or []
    macro('nPanelReducedOnlyFrontierSize', str(len(ndr)))
    macro('nPanelReducedOnlyFrontierEq', str(sum(1 for s in ndr if fam[s] in ('rom', 'fast'))))
    macro('nPanelReducedOnlyFrontierPod', str(sum(1 for s in ndr if fam[s] == 'pod')))
    ndr_e = nd.get('nondominated_gpu_evolved_reduced_only') or []
    macro('nPanelReducedOnlyFrontierEvolvedPod', str(sum(1 for s in ndr_e if fam[s] == 'pod')))
    for s, nm in [('nt1e-3_dt005', 'NtThreeFine'), ('nt1e-4_dt005', 'NtFourFine'),
                  ('nt1e-2_dt01', 'NtTwoCoarse'), ('fft_tight', 'FftTight'), ('dense_tight', 'DenseTight')]:
        macro(f'nPanelFom{nm}Err', pct(P[s]['worst_evolved_percent']))
        macro(f'nPanelFom{nm}Ms', ms(P[s]['median_gpu_ms']))
        macro(f'nPanelFom{nm}Ref', pct(P[s]['worst_reference_percent']))
    best = P['q256_M1088_dense_g1em06']
    macro('nPanelBestRungErr', pct(best['worst_evolved_percent'])); macro('nPanelBestRungMs', ms(best['median_gpu_ms']))
    macro('nPanelBestRungAll', pct(best['worst_all_times_percent']))
    macro('nPanelFomOverBest', f'{best["median_gpu_ms"]/P["nt1e-3_dt005"]["median_gpu_ms"]:.1f}')
    pod = P['pod512_M2048_dense']; free = P['free512_M1024_dense']
    macro('nPanelPodFiveTwelveErr', pct(pod['worst_evolved_percent'])); macro('nPanelPodFiveTwelveMs', ms(pod['median_gpu_ms']))
    macro('nPanelPodFiveTwelveAll', pct(pod['worst_all_times_percent']))
    macro('nPanelFreeErr', pct(free['worst_evolved_percent'])); macro('nPanelFreeMs', ms(free['median_gpu_ms']))
    macro('nPanelFreeAll', pct(free['worst_all_times_percent']))
    macro('nPanelFreeFloor', pct(free['best_found_percent'])); macro('nPanelFreeOverFloor', f'{free["solved_over_best_found"]:.2f}')
    macro('nPanelPodOneTwentyEightErr', pct(P['pod128_M512_dense']['worst_evolved_percent']))
    macro('nPanelPodOneTwentyEightAll', pct(P['pod128_M512_dense']['worst_all_times_percent']))
    macro('nPanelPodOneTwentyEightMs', ms(P['pod128_M512_dense']['median_gpu_ms']))
    macro('nPanelPodTwoFiftySixErr', pct(P['pod256_M1024_dense']['worst_evolved_percent']))
    macro('nPanelPodTwoFiftySixMs', ms(P['pod256_M1024_dense']['median_gpu_ms']))
    fno = P['fno-large']
    macro('nPanelFnoErr', pct(fno['worst_evolved_percent'])); macro('nPanelFnoMs', ms(fno['median_gpu_ms']))
    macro('nPanelFnoRef', pct(fno['worst_reference_percent']))
    macro('nPanelQzeroT', pct(P['q0_M64_dense_g1em06']['worst_t0_compression_percent']))
    macro('nPanelQzeroEvolved', pct(P['q0_M64_dense_g1em06']['worst_evolved_percent']))
    macro('nPanelQzeroAll', pct(P['q0_M64_dense_g1em06']['worst_all_times_percent']))
    macro('nPanelFastMs', ms(P['q0_M64_eqcert_g1em06_fastL4']['median_gpu_ms']))
    macro('nPanelRefDiscretisation', pct(P['fft_tight']['worst_reference_percent']))
    # POD strict-flag arithmetic
    strict_no = [s for s in reduced if P[s].get('converged') and not P[s].get('converged_strict')]
    macro('nPanelStrictNoCount', str(len(strict_no)))
    icres = [P[s].get('max_ic_relative_residual') for s in strict_no if P[s].get('max_ic_relative_residual') is not None]
    macro('nPanelStrictNoIcResidualMax', f'{max(icres):.1e}' if icres else '---')
    sob = [P[s].get('solved_over_best_found') for s in strict_no if fam[s] == 'pod']
    macro('nPanelPodSolvedOverFloorMax', f'{max(sob):.5f}' if sob else '---')

    # ---- T5: every subject
    order = [s for s in P if s != '*']
    t5 = []
    for s in order:
        p = P[s]
        t5.append([tt(s), fam[s], str(qk[s]) if qk[s] is not None else '---',
                   str(Mof[s]) if Mof[s] else '---', str(quad[s] or '---'),
                   pct(p.get('worst_evolved_percent')), pct(p.get('worst_all_times_percent')),
                   pct(p.get('worst_t0_compression_percent')), pct(p.get('worst_reference_percent')),
                   ms(p.get('median_gpu_ms')), ms(p.get('median_host_ms')),
                   yn(p.get('converged')) if fam[s] in ('rom', 'fast', 'pod', 'free') else '---',
                   yn(p.get('converged_strict')) if fam[s] in ('rom', 'fast', 'pod', 'free') else '---'])
    cols = ['subject', 'family', '$q$ / $k^\\prime$', '$M$', 'quad.', 'evolved \\%', 'all \\%',
            '$t{=}0$ \\%', 'vs ref \\%', 'GPU ms', 'host ms', 'conv.', 'strict']
    write('T05_panel_all.tex', tabular(cols, t5, 'lllllrrrrrrcc', r'\tiny'), f'b-panel job {job}')
    # 1024^2 panel: pending
    if load('panel1024_summary') is None:
        macro('nPanelTenTwentyFour', gen('b-panel 1024$^2$ (job 3783817)', 'T5 1024'))


# =========================================================================== T4 rank vs tests
def build_qxm():
    a = load('qxm_analysis')
    rep = load('qxm_report')
    if a is None:
        write('T04_rank_vs_tests.tex', gen('b-qxm', 'T4'))
        return
    ev = a['spans']['worst_evolved_percent']
    al = a['spans']['worst_all_times_percent']
    wj = ev['fixed_M']['1088']['within_job']
    # provenance from the report's job table
    hdr, rows = md_table(rep, 'job') if rep else (None, None)
    jobs = {r[0]: r for r in rows} if rows else {}
    macro('provQxmJobs', ', '.join(f"{r[0]} = {r[2]} ({tex_escape(r[3])})" for r in rows) if rows else '---')
    macro('provQxmCommit', tex_escape(rows[0][4]) if rows else '---')
    macro('provQxmWithinJob', f"{wj['job']} = {wj['job_id']}")
    macro('nQxmFixedM', str(ev['fixed_M']['1088']['M']))
    macro('nQxmErrSpan', f"{wj['error_span']:.2f}")
    macro('nQxmCostSpan', f"{wj['cost_span']:.2f}")
    macro('nQxmNonDom', str(wj['non_dominated_points']))
    macro('nQxmPasses', yn(wj['passes_tunability_bar']))
    macro('nQxmMonotone', yn(wj['monotone_error']))
    macro('nQxmFixedMConverged', yn(ev['fixed_M']['1088']['all_converged']))
    macro('nQxmSpanFixedTwoFiftySix', f"{ev['fixed_M']['256']['span']:.2f}")
    macro('nQxmSpanScheduledFour', f"{ev['scheduled']['4x']['span']:.2f}")
    macro('nQxmSpanScheduledEight', f"{ev['scheduled']['8x']['span']:.2f}")
    macro('nQxmAllTimesSpanFixed', f"{al['fixed_M']['1088']['span']:.2f}")
    macro('nQxmAllTimesSpanScheduledFour', f"{al['scheduled']['4x']['span']:.2f}")
    dec = a['decomposition']['worst_evolved_percent']
    macro('nQxmShareRankCorner', f"{100*dec['corner']['share_q']:.1f}")
    macro('nQxmShareTestsCorner', f"{100*dec['corner']['share_M']:.1f}")
    macro('nQxmShareTestsRung', f"{100*dec['rung']['share_M']:.1f}")
    macro('nQxmAnovaRank', f"{100*dec['anova']['frac_q']:.1f}")
    macro('nQxmAnovaTests', f"{100*dec['anova']['frac_M']:.1f}")
    macro('nQxmAnovaInter', f"{100*dec['anova']['frac_interaction']:.1f}")
    macro('nQxmRungTestShares', ', '.join(f"{100*r['share_M']:.0f}" for r in dec['rung']['rungs']))
    sat = a['saturation']['per_q']
    macro('nQxmMstarZero', str(sat['0']['M_star'])); macro('nQxmMstarZeroTpu', f"{sat['0']['tests_per_unknown_at_M_star']:.0f}")
    macro('nQxmMstarZeroCost', f"{sat['0']['cost_ratio_at_M_star']:.2f}")
    macro('nQxmMstarSixtyFour', str(sat['64']['M_star'])); macro('nQxmMstarSixtyFourTpu', f"{sat['64']['tests_per_unknown_at_M_star']:.1f}")
    macro('nQxmQzeroMlargest', str(sat['0']['largest_M'])); macro('nQxmQzeroErrLargest', pct(sat['0']['value_at_largest_M']))
    macro('nQxmQzeroErrMstar', pct(sat['0']['error_at_M_star']))
    macro('nQxmSpanMatQtwoFiftySix', f"{ev['fixed_q']['256']['span']:.2f}")
    macro('nQxmSpanMatQzero', f"{ev['fixed_q']['0']['span']:.2f}")
    best_m = ev['fixed_q']['256']['M_best']; best_v = min(ev['fixed_q']['256']['values'])
    macro('nQxmBestErr', pct(best_v)); macro('nQxmBestM', str(best_m))
    macro('nQxmFixedQzeroMs', ms(wj['median_gpu_ms'][0])); macro('nQxmFixedQtopMs', ms(wj['median_gpu_ms'][-1]))
    # the scheduled ladder's q=0 cost inside the same job (G2 anchors the (0,256)/(0,1088) cells only);
    # report G1's (0,64) cost separately and never as a ratio against G2.
    hdr, grid = md_table(rep, '$q$') if rep else (None, None)
    sched_q0_ms = None
    if grid:
        for r in grid:
            if r[0] == '0' and r[1] == '64':
                sched_q0_ms = float(r[8]); sched_q0_job = r[4]
    macro('nQxmScheduledQzeroMs', ms(sched_q0_ms) if sched_q0_ms else '---')
    macro('nQxmScheduledQzeroJob', tex_escape(sched_q0_job) if sched_q0_ms else '---')
    # solver control
    hdr, ctl = md_table(rep, 'cell') if rep else (None, None)
    if ctl:
        macro('nQxmSolverControlRelDiff', tex_escape(ctl[0][7])); macro('nQxmSolverControlCost', tex_escape(ctl[0][8]))

    # ---- T4 table: fixed-M within-job ladder; scheduled 4x; fixed-q spans
    rows_t4 = []
    w256 = ev['fixed_M']['256']['within_job']
    for q, v, c in zip(w256['q'], w256['values'], w256['median_gpu_ms']):
        rows_t4.append(['fixed $M=256$ (job ' + w256['job_id'] + ')', str(q), '256', pct(v), ms(c)])
    rows_t4.append('MIDRULE')
    for q, v, c in zip(wj['q'], wj['values'], wj['median_gpu_ms']):
        rows_t4.append(['fixed $M=1088$ (job ' + wj['job_id'] + ')', str(q), '1088', pct(v), ms(c)])
    rows_t4.append('MIDRULE')
    macro('nQxmTwoFiftySixErrSpan', f"{w256['error_span']:.2f}"); macro('nQxmTwoFiftySixCostSpan', f"{w256['cost_span']:.2f}")
    macro('nQxmTwoFiftySixPasses', yn(w256['passes_tunability_bar'])); macro('nQxmTwoFiftySixNonDom', str(w256['non_dominated_points']))
    macro('nQxmTwoFiftySixQtop', str(max(w256['q'])))
    anc = [x for x in a['anchors'] if x['q'] == 0 and x['M'] == 1088]
    if anc:
        x = anc[0]; macro('nQxmAnchorSpreadPct', f"{100*abs(x['b_ms']-x['a_ms'])/min(x['a_ms'],x['b_ms']):.0f}")
        macro('nQxmAnchorMsA', ms(x['a_ms'])); macro('nQxmAnchorMsB', ms(x['b_ms'])); macro('nQxmAnchorJobs', f"{x['a_job']} / {x['b_job']}")
    for (q, M), v in zip(ev['scheduled']['4x']['cells'], ev['scheduled']['4x']['values']):
        rows_t4.append(['scheduled $M=4(K+q)$', str(q), str(M), pct(v), '---'])
    write('T04_rank_vs_tests.tex', tabular(['ladder', '$q$', '$M$', 'worst evolved \\%', 'GPU ms'],
                                           rows_t4, 'lrrrr'), 'b-qxm; costs only within job ' + wj['job_id'])
    rows_fq = []
    for q in ['0', '16', '32', '64', '128', '256']:
        d = ev['fixed_q'][q]
        rows_fq.append([q, ', '.join(str(m) for m in d['M']), ' / '.join(f'{v:.4f}' for v in d['values']),
                        yn(d['monotone']), ratio(d['span']), str(d['M_best']), ratio(d['span_M_ge_2x'])])
    write('T04b_fixed_q.tex', tabular(['$q$', '$M$ sweep', 'worst evolved \\%', 'monotone', 'span', 'best $M$',
                                       'span, $M\\ge 2(K{+}q)$'], rows_fq, 'llp{4.2cm}lrrr', r'\scriptsize'),
          'b-qxm fixed-q sweeps (errors comparable across jobs; costs not shown)')


# =========================================================================== T14 operators
def build_operators():
    s = load('nosecond_summary')
    if s is None:
        write('T14_operators.tex', gen('no-second', 'T14'))
        return
    allrows = s['rows']
    rows = [r for r in allrows if r.get('rung') is None]      # the resolution ladder is handled below
    by = defaultdict(dict)
    meta = {}
    for r in rows:
        by[(r['arm'], r['job_id'], r['cohort'])][r['metric']] = r['value']
        meta[(r['arm'], r['job_id'])] = r          # keyed on (arm, job): the Poisson screen reuses arm names
    CTRL = 'ctrl-'                      # one-variable controls: never eligible for selection (lane DESIGN 3.4/A4)
    burg8 = sorted({(r['arm'], r['job_id']) for r in rows if r['cohort'] == 'diagnosis-8' and not r['arm'].startswith(CTRL)},
                   key=lambda k: by[(k[0], k[1], 'diagnosis-8')].get('worst_fixed_initial_error', 9))
    ctrl8 = sorted({(r['arm'], r['job_id']) for r in rows if r['cohort'] == 'diagnosis-8' and r['arm'].startswith(CTRL)})
    t = []
    for a, jb in burg8:
        m = meta[(a, jb)]; d = by[(a, jb, 'diagnosis-8')]; v = by.get((a, jb, 'validation-32'), {})
        still = yn(m['still_improving']) if m.get('still_improving') is not None else (yn(m['best_epoch'] >= 0.95 * m['epochs']) if m.get('epochs') and m.get('best_epoch') else '---')
        t.append([tt(a), m['operator'], f"{m['params']:,}" if m.get('params') else '---',
                  str(m.get('epochs') or '---'), still,
                  pct(100 * d['worst_fixed_initial_error']),
                  pct(100 * v['worst_fixed_initial_error']) if v else '---',
                  pct(100 * v['median_fixed_initial_error']) if v else '---',
                  tt(jb)])
    write('T14_operators.tex', tabular(['arm', 'family', 'params', 'epochs', 'still improving',
                                        '8-case worst \\%', 'val-32 worst \\%', 'val-32 median \\%', 'job'], t, 'lllrcrrrl', r'\tiny'),
          'no-second + parent FNO lane; accuracy comparable across jobs, timing never; rows keyed on (arm, job)')
    def w8(a):
        k = [k for k in burg8 if k[0] == a][0]
        return 100 * by[(k[0], k[1], 'diagnosis-8')]['worst_fixed_initial_error']
    for a, nm in [('unet-small', 'UnetSmall'), ('unet-medium', 'UnetMedium'), ('tsol-refine', 'TsolRefine'),
                  ('unet-refine', 'UnetRefine'), ('rom', 'Rom'), ('fno-large', 'FnoLarge'),
                  ('same_nt1e-2_dt005', 'FomEfficient'), ('unet-large', 'UnetLarge')]:
        macro(f'nOpEight{nm}', pct(w8(a)))
    better = [k for k in burg8 if meta[k]['operator'] not in ('ROM', 'FOM') and w8(k[0]) < w8('rom')]
    macro('nOpArmsBeatingRom', str(len(better)))
    macro('nOpEveryFnoWorse', yn(all(w8(k[0]) > w8('rom') for k in burg8 if meta[k]['operator'] == 'FNO')))
    def improving(k):
        m = meta[k]
        if m.get('still_improving') is not None:
            return bool(m['still_improving'])
        return bool(m.get('best_epoch') and m.get('epochs') and m['best_epoch'] >= 0.95 * m['epochs'])
    lane_arms = sorted({(r['arm'], r['job_id']) for r in rows if r['job_id'] in ('3780138', '3780139') and r.get('epochs')})
    fno_arms = sorted({(r['arm'], r['job_id']) for r in rows if r['job_id'] == '3710846' and r.get('epochs')})
    macro('nOpLaneImproving', f"{sum(1 for k in lane_arms if improving(k))} of {len(lane_arms)}")
    macro('nOpFnoImproving', f"{sum(1 for k in fno_arms if improving(k))} of {len(fno_arms)}")
    # ---- one-variable controls (precision, seed): each twins a screen arm
    if ctrl8:
        TWIN = {'ctrl-medium-f64': ('unet-medium', '3780138', 'network dtype float64'),
                'ctrl-medium-seed2': ('unet-medium', '3780138', 'training seed'),
                'ctrl-tsol-small-f64': ('tsol-small', '3780139', 'network dtype float64')}
        rom = w8('rom'); t = []
        for a, jb in ctrl8:
            m = meta[(a, jb)]; d = by[(a, jb, 'diagnosis-8')]; v = by.get((a, jb, 'validation-32'), {})
            tw, twjob, what = TWIN[a]
            tm = meta[(tw, twjob)]; td = by[(tw, twjob, 'diagnosis-8')]
            val = 100 * d['worst_fixed_initial_error']; tval = 100 * td['worst_fixed_initial_error']
            t.append([tt(a), tt(tw), what, f"{m['epochs']} vs {tm['epochs']} ({m['epochs']/tm['epochs']:.2f}$\\times$)",
                      pct(val), pct(tval), ('below' if val < rom else 'above') + f" ({abs(val - rom):.4f} pp)",
                      ('below' if tval < rom else 'above')])
            key = ''.join(w.title() for w in a.replace(CTRL, '').split('-'))
            for dg, wd in [('64', 'SixtyFour'), ('2', 'Two'), ('1', 'One'), ('3', 'Three')]:
                key = key.replace(dg, wd)
            macro('nOpCtrl' + key, pct(val)); macro('nOpCtrl' + key + 'Margin', f"{abs(val - rom):.4f}")
            macro('nOpCtrl' + key + 'Side', 'below' if val < rom else 'above')
            macro('nOpCtrl' + key + 'EpochRatio', f"{m['epochs']/tm['epochs']:.2f}")
        write('T14d_controls.tex', tabular(['control', 'twin', 'one variable changed', 'epochs vs twin', 'control worst \\%',
                                            'twin worst \\%', 'control vs ROM', 'twin vs ROM'], t, 'llp{2.4cm}lrrll', r'\scriptsize'),
              f'no-second controls, job {ctrl8[0][1]}; matched eight-case worst, ROM from job 3702709')
        macro('provOpCtrlJob', ctrl8[0][1])
        surv = [a for a, jb in ctrl8 if TWIN[a][0] == 'unet-medium']
        macro('nOpCtrlCount', str(len(ctrl8)))
        macro('nOpCtrlSurviving', str(sum(1 for a in surv if 100 * by[(a, ctrl8[0][1], 'diagnosis-8')]['worst_fixed_initial_error'] < rom)))
    # job list for the provenance row, from the records
    macro('provOpJobs', ', '.join(sorted({r['job_id'] for r in allrows})))
    # Poisson screen
    pv = {(r['arm'], r['job_id']): by[(r['arm'], r['job_id'], 'poisson-validation-32')] for r in rows if r['cohort'] == 'poisson-validation-32'}
    if pv:
        un = [a for a in pv if a[0].startswith('unet')]; fn = [a for a in pv if a[0].startswith('fno')]
        macro('nOpPoissonUnetWorstRange', ' / '.join(pct(100 * pv[a]['worst_physical_candidate_relative_error'], 2) for a in sorted(un)))
        macro('nOpPoissonFnoWorstRange', f"{100*min(pv[a]['worst_physical_candidate_relative_error'] for a in fn):.1f}--{100*max(pv[a]['worst_physical_candidate_relative_error'] for a in fn):.1f}")
        tp = []
        for a in sorted(pv):
            m = meta[a]; d = pv[a]
            tp.append([tt(a[0]), m['operator'], f"{m['params']:,}" if m.get('params') else '---',
                       pct(100 * d['median_physical_candidate_relative_error'], 2),
                       pct(100 * d['worst_physical_candidate_relative_error'], 2), tt(a[1])])
        write('T14b_operators_poisson.tex', tabular(['arm', 'family', 'params', 'median \\%', 'worst \\%', 'job'],
                                                    tp, 'lllrrl', r'\scriptsize'), 'no-second Poisson screen')
    # ---- the operators' own inference-time knob: evaluation resolution (pre-registered R-USABLE / R-DEGENERATE)
    res = [r for r in allrows if r.get('rung') is not None and r.get('cohort') == 'validation-32']
    if res:
        Rv = defaultdict(dict)
        for r in res:
            Rv[(r['operator'], int(r['rung']))][r['metric']] = r['value']
        ops = sorted({k[0] for k in Rv}); job = res[0]['job_id']; gpu = res[0].get('gpu', '---')
        macro('provResJob', job); macro('provResGpu', gpu)
        t = []; verdicts = {}
        for op in ops:
            usable = False; first_fail = None
            for rung in sorted({k[1] for k in Rv if k[0] == op}, reverse=True):
                d = Rv[(op, rung)]
                sp = d.get('same_job_speedup_vs_own_256'); er = d.get('worst_evolved_error_ratio_vs_own_256')
                ok = bool(d.get('meets_error_gate_2x')) and bool(d.get('meets_speed_gate_1p5x'))
                if rung < 256 and ok: usable = True
                t.append([tt(op), str(rung), pct(100 * d['worst_evolved_fixed_initial_error'], 2), pct(100 * d.get('interpolation_floor_worst_evolved', 0), 2),
                          ms(d['same_job_device_query_pooled_median_ms'], 2), f"{sp:.2f}" if sp else '---', f"{er:.2f}" if er else '---',
                          ('yes' if ok else 'no') if rung < 256 else 'ref.'])
            verdicts[op] = 'R-USABLE' if usable else 'R-DEGENERATE'
            macro('nRes' + re.sub(r'[^A-Za-z]', '', op.title()), verdicts[op])
        write('T14c_resolution.tex', tabular(['operator', 'rung', 'worst evolved \\%', 'interp.\\ floor \\%', 'device ms', 'speed vs 256', 'error vs 256', 'both gates'],
                                             t, 'lrrrrrrc', r'\scriptsize'), f'no-second resolution ladder, job {job}')
        macro('nResAnyUsable', yn(any(v == 'R-USABLE' for v in verdicts.values())))
        macro('nResUsableList', ', '.join(tt(o) for o, v in verdicts.items() if v == 'R-USABLE') or 'none')
        macro('nResDegenerateList', ', '.join(tt(o) for o, v in verdicts.items() if v == 'R-DEGENERATE') or 'none')
        f128 = Rv.get(('fno-large', 128), {})
        if f128:
            macro('nResFnoSpeedup', f"{f128['same_job_speedup_vs_own_256']:.2f}"); macro('nResFnoErrRatio', f"{f128['worst_evolved_error_ratio_vs_own_256']:.2f}")
        for op, nm in [('fno-large', 'Fno'), ('unet-refine', 'Unet'), ('tsol-refine', 'Tsol')]:
            for rung, rn in [(128, 'OneTwentyEight'), (64, 'SixtyFour'), (32, 'ThirtyTwo')]:
                d = Rv.get((op, rung), {})
                if d:
                    macro(f'nRes{nm}Ms{rn}', ms(d['same_job_device_query_pooled_median_ms'], 2))
                    macro(f'nRes{nm}ErrRatio{rn}', f"{d['worst_evolved_error_ratio_vs_own_256']:.2f}")
                    macro(f'nRes{nm}Speed{rn}', f"{d['same_job_speedup_vs_own_256']:.2f}")
                    if d.get('interpolation_floor_worst_evolved'):
                        macro(f'nRes{nm}FloorRatio{rn}', f"{d['worst_evolved_fixed_initial_error']/d['interpolation_floor_worst_evolved']:.0f}")
        fr = [Rv[(op, 128)]['worst_evolved_fixed_initial_error'] / Rv[(op, 128)]['interpolation_floor_worst_evolved'] for op in ops if (op, 128) in Rv and Rv[(op, 128)].get('interpolation_floor_worst_evolved')]
        if fr:
            macro('nResMinFloorRatioOneTwentyEight', f"{min(fr):.0f}")
        macro('nOpResolutionKnob', 'landed (Table~\\ref{tab:resolution})')
    else:
        macro('nOpResolutionKnob', gen('no-second res01, operator resolution knob', 'operator resolution knob'))
    macro('nOpSeedControl', gen('no-second ctrl01 (job 3783831)', 'operator seed/precision control'))


# =========================================================================== T11 linear PDEs
def build_linear():
    # ---- waves
    w = load('wladder_summary'); wrep = load('wladder_report')
    if w is None:
        write('T11a_waves.tex', gen('w-ladder', 'T11 waves'))
    else:
        by = defaultdict(dict); jobs = {}
        for r in w:
            by[(r['mesh'], r['subject'])][r['metric']] = r['value']; jobs[r['mesh']] = (r['job_id'], r.get('attempt'))
        arms = ['head_q0', 'trained_nested40', 'nested_q8', 'nested_q16', 'nested_q32', 'linear_bank64',
                'pod_k40', 'pod_k64', 'pod_k128', 'dst', 'rk4_fom', 'cg_1e-06']
        qk = {'head_q0': '0', 'trained_nested40': '8', 'nested_q8': '8', 'nested_q16': '16', 'nested_q32': '32',
              'linear_bank64': '64 ($=R$)', 'pod_k40': "$k'{=}40$", 'pod_k64': "$k'{=}64$", 'pod_k128': "$k'{=}128$"}
        rows = []
        for mesh in (64, 256, 1024):
            for a in arms:
                d = by.get((mesh, a))
                if not d:
                    continue
                rows.append([f'${mesh}^2$', tt(a), qk.get(a, '---'), pct(100 * d['worst_energy_state']),
                             pct(100 * d['worst_t0_energy_state']) if 'worst_t0_energy_state' in d else '---',
                             ms(d['median_gpu_ms'], 3), ms(d['median_complete_ms'], 3)])
            rows.append('MIDRULE')
        rows.pop()
        write('T11a_waves.tex', tabular(['mesh', 'arm', '$q$ / $k^\\prime$', 'worst energy-state \\%',
                                         '$t{=}0$ \\%', 'GPU ms', 'complete ms'], rows, 'llrrrrr', r'\scriptsize'),
              'w-ladder; one job per mesh, costs comparable only within a mesh')
        gpus = {}
        if wrep:
            for m in re.finditer(r'(\d+)² ran on (NVIDIA [^ ]+(?: [^ ]+)*?) at commit `([0-9a-f]+)` \(job (\d+)\)', wrep):
                gpus[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
        macro('provWaveJobs', '; '.join(f"${k}^2$: job {v[0]}" + (f" ({tex_escape(gpus[k][0])}, commit {gpus[k][1]})" if k in gpus else '')
                                       for k, v in sorted(jobs.items())))
        for mesh, nm in [(64, 'SixtyFour'), (256, 'TwoFiftySix'), (1024, 'TenTwentyFour')]:
            h = by[(mesh, 'head_q0')]; b = by[(mesh, 'linear_bank64')]; p = by[(mesh, 'pod_k64')]; d = by[(mesh, 'dst')]
            macro(f'nWaveHeadErr{nm}', pct(100 * h['worst_energy_state'], 3)); macro(f'nWaveHeadMs{nm}', ms(h['median_gpu_ms']))
            macro(f'nWaveBankErr{nm}', pct(100 * b['worst_energy_state'], 3)); macro(f'nWaveBankMs{nm}', ms(b['median_gpu_ms'], 2))
            macro(f'nWavePodErr{nm}', pct(100 * p['worst_energy_state'], 3)); macro(f'nWavePodMs{nm}', ms(p['median_gpu_ms'], 2))
            macro(f'nWaveDstMs{nm}', ms(d['median_gpu_ms'], 2))
            macro(f'nWaveBankOverHeadCost{nm}', f"{h['median_gpu_ms']/b['median_gpu_ms']:.0f}")
            macro(f'nWaveHeadOverBankErr{nm}', f"{h['worst_energy_state']/b['worst_energy_state']:.2f}")
            v = by[(mesh, 'verdict')]
            macro(f'nWaveDegenerate{nm}', yn(v.get('all')))
            macro(f'nWaveMidRungMs{nm}', ms(by[(mesh, 'nested_q32')]['median_gpu_ms']))
            macro(f'nWaveTieBandPp{nm}', f"{100*v['tie_band_delta']:.3f}" if v.get('tie_band_delta') is not None else '---')
            macro(f'nWaveTopWithinBand{nm}', yn(v.get('D1_accuracy'))); macro(f'nWaveTopStrictBest{nm}', yn(v.get('D1_strict')))
            macro(f'nWaveHeadLadderMonotone{nm}', yn(v.get('H_mono_ladder')))
            q32 = by[(mesh, 'nested_q32')]
            macro(f'nWaveQthirtytwoErr{nm}', pct(100 * q32['worst_energy_state'], 3))
            macro(f'nWaveBankMinusQthirtytwoPp{nm}', f"{100*(b['worst_energy_state']-q32['worst_energy_state']):.3f}")
        # three layers at 256
        pf = by[(256, 'linear_bank64@projection_floor')]; bf = by[(256, 'nested_q32@best_found')]
        macro('nWaveFloorTwoFiftySix', pct(100 * pf['worst_energy_state'], 3))
        macro('nWaveBestFoundQthirtytwoTwoFiftySix', pct(100 * bf['worst_energy_state'], 3))
        macro('nWaveBestFoundHeadTwoFiftySix', pct(100 * by[(256, 'head_q0@best_found')]['worst_energy_state'], 3))
        macro('nWavePodFloorTwoFiftySix', pct(100 * by[(256, 'pod_k64@projection_floor')]['worst_energy_state'], 3))

    # ---- Poisson (p-linear, both meshes; one job per mesh; costs never compared across meshes)
    p = load('plinear_summary'); prep = load('plinear_report'); verd = load('plinear_verdicts')
    if p is None:
        write('T11b_poisson.tex', gen('p-linear', 'T11 Poisson'))
    else:
        p = [r for r in p if not r.get('retracted')]
        jobs = {}
        for m in re.finditer(r'## (\d+) intervals — `(\w+)`, job `(\d+)`, `(NVIDIA [^`]+)`, source `([0-9a-f]+)`', prep or ''):
            jobs[int(m.group(1))] = {'attempt': m.group(2), 'job': m.group(3), 'gpu': m.group(4), 'commit': m.group(5)}
        macro('provPlinJobs', '; '.join(f"${k}^2$: job {v['job']} ({tex_escape(v['gpu'])}, commit {v['commit']})" for k, v in sorted(jobs.items())))
        mh = re.search(r'`plhead1`, job `(\d+)`, `(NVIDIA [^`]+)`', prep or '')
        macro('provPlinHeadJob', mh.group(1) if mh else '---'); macro('provPlinHeadGpu', mh.group(2) if mh else '---')
        order = ['q0_m4@new_K32', 'q32_m4@new_K32', 'q64_m4@new_K32', 'q128_m4@new_K32', 'q256_m4@new_K32',
                 'q512_m4@new_K32', 'd_linear_qr_m4@new_K32', 'd_freebank_m4@new_K32',
                 'e_pod32_m4@trainset', 'e_pod64_m4@trainset', 'e_pod128_m4@trainset', 'e_pod256_m4@trainset',
                 'e_pod512_m4@trainset', 'dst_direct', 'cg_0.01', 'cg_0.0001', 'cg_1e-06']
        lab = {'d_linear_qr_m4@new_K32': 'linear top rung ($q{=}R$, QR)', 'd_freebank_m4@new_K32': 'free bank (LM)',
               'dst_direct': 'direct DST', 'cg_0.01': 'CG $10^{-2}$', 'cg_0.0001': 'CG $10^{-4}$', 'cg_1e-06': 'CG $10^{-6}$'}
        rows = []
        for mesh, nm in [(256, 'TwoFiftySix'), (1024, 'TenTwentyFour')]:
            job = jobs.get(mesh, {}).get('job')
            by = defaultdict(dict); meta = {}
            for r in p:
                if r['mesh'] == mesh and (job is None or r['job_id'] == job):
                    by[r['subject']][r['metric']] = r['value']; meta[r['subject']] = r
            for s_ in order:
                if s_ not in by: continue
                d = by[s_]; mt = meta[s_]
                name = lab.get(s_) or (f"$q{{=}}{mt['q_or_k']}$" if mt['family'] == 'neural+linear' else f"POD $k'{{=}}{mt['q_or_k']}$")
                rows.append([f'${mesh}^2$', name, str(mt.get('M') or '---'), pct(100 * d['worst_same_grid']), pct(100 * d['median_same_grid']),
                             ms(d['median_total_ms'], 3), ms(d['median_device_ms'], 3),
                             str(int(d['valid_count'])) + '/36' if d.get('valid_count') is not None else '---',
                             yn(mt.get('non_dominated_all')), yn(mt.get('non_dominated_reduced'))])
            rows.append('MIDRULE')
            macro(f'nPlin{nm}QzeroErr', pct(100 * by['q0_m4@new_K32']['worst_same_grid'])); macro(f'nPlin{nm}QzeroMs', ms(by['q0_m4@new_K32']['median_total_ms'], 2))
            macro(f'nPlin{nm}TopErr', pct(100 * by['d_linear_qr_m4@new_K32']['worst_same_grid'])); macro(f'nPlin{nm}TopMs', ms(by['d_linear_qr_m4@new_K32']['median_total_ms'], 2))
            macro(f'nPlin{nm}Floor', pct(100 * by['bank_floor@new_K32']['worst_same_grid']))
            macro(f'nPlin{nm}QtwoFiftySixErr', pct(100 * by['q256_m4@new_K32']['worst_same_grid'])); macro(f'nPlin{nm}QtwoFiftySixMs', ms(by['q256_m4@new_K32']['median_total_ms'], 2))
            macro(f'nPlin{nm}DstMs', ms(by['dst_direct']['median_total_ms'], 3))
            macro(f'nPlin{nm}PodFiveTwelveErr', pct(100 * by['e_pod512_m4@trainset']['worst_same_grid'])); macro(f'nPlin{nm}PodFiveTwelveMs', ms(by['e_pod512_m4@trainset']['median_total_ms'], 2))
            macro(f'nPlin{nm}CgLooseMs', ms(by['cg_0.01']['median_total_ms'], 1)); macro(f'nPlin{nm}CgTightMs', ms(by['cg_1e-06']['median_total_ms'], 1))
            macro(f'nPlin{nm}BestFoundQzero', pct(100 * by['augmented_best_found_q0@new_K32']['worst_same_grid']))
            mid = [by[k]['median_total_ms'] for k in order[:5]]
            macro(f'nPlin{nm}MidMsMin', ms(min(mid), 1)); macro(f'nPlin{nm}MidMsMax', ms(max(mid), 1))
            macro(f'nPlin{nm}TopOverMidCost', f"{max(mid)/by['d_linear_qr_m4@new_K32']['median_total_ms']:.2f}")
            if verd and str(mesh) in verd:
                v = verd[str(mesh)]
                macro(f'nPlin{nm}Dspan', f"{v['D1_span']:.2f}"); macro(f'nPlin{nm}Degenerate', yn(v['D1'] and v['D2_lowest'] and v['D3_fom']))
                macro(f'nPlin{nm}Monotone', yn(v['monotone'])); macro(f'nPlin{nm}FalsifiedIntent', yn(v['falsified_intent']))
        rows.pop()
        write('T11b_poisson.tex', tabular(['mesh', 'subject', '$M$', 'worst \\%', 'median \\%', 'total ms', 'device ms', 'valid',
                                           'non-dom.\\ (all)', 'non-dom.\\ (reduced)'], rows, 'llrrrrrrcc', r'\scriptsize'),
              'p-linear; one job per mesh, never compare costs across meshes')
        # head-capacity appendix table (job plhead1)
        hc = defaultdict(dict); solved = defaultdict(dict)
        for r in p:
            if r.get('family') == 'head-capacity': hc[r['subject']][r['metric']] = r['value']
            if r.get('family') == 'rom' and r.get('job_id') == (mh.group(1) if mh else None): solved[r['subject']][r['metric']] = r['value']
        if hc:
            t = []
            for k in ['K32_w128_L2', 'K32_w256_L2', 'K32_w128_L3', 'K32_w256_L3', 'K64_w128_L2', 'K64_w256_L3', 'K32_w128_L2_x3']:
                if k not in hc: continue
                sv = solved.get('a_neural@' + k, {})
                t.append([tt(k), pct(100 * hc[k]['dev_best_found_worst']), f"{hc[k]['dev_best_found_over_floor']:.2f}",
                          pct(100 * sv['worst_same_grid']) if sv else '---', ms(sv['median_total_ms'], 2) if sv else '---'])
            write('T11c_head_capacity.tex', tabular(['head', 'best-found dev.\\ \\%', 'best-found / floor', 'solved $1024^2$ \\%', 'total ms'], t, 'lrrrr', r'\scriptsize'),
                  f"p-linear head capacity, job {mh.group(1) if mh else '---'}")
            macro('nPlinHeadRatioPrimary', f"{hc['K32_w128_L2']['dev_best_found_over_floor']:.2f}")
            best = min(hc, key=lambda k: hc[k]['dev_best_found_over_floor'])
            macro('nPlinHeadRatioBest', f"{hc[best]['dev_best_found_over_floor']:.2f}"); macro('nPlinHeadBestArm', tt(best))
            if verd and 'head' in verd: macro('nPlinHeadMaxDrop', f"{100*verd['head']['H2_max_drop']:.1f}")


# =========================================================================== T11d heat (earlier cell)
def build_heat():
    rep = load('heat_report'); man = load('heat_manifest')
    if rep is None:
        write('T11d_heat.tex', gen('heat linear-bank comparison (2026-09-10)', 'T11 heat')); return
    m = re.search(r'GPU job `(\d+)` on `(NVIDIA [^`]+)`, node `[^`]+`, scientific source `([0-9a-f]+)`', rep)
    macro('provHeatJob', m.group(1) if m else '---'); macro('provHeatGpu', m.group(2) if m else '---'); macro('provHeatCommit', hexprefix(m.group(3)) if m else '---')
    hdr, rows = md_table(rep, 'Intervals')
    lab = {'Linear bank, exact reduced time evolution': 'linear bank, exact modal evolution ($q{=}R$, no head)',
           'Current nonlinear ROM': 'nonlinear head ($k{=}8$)', 'Same-grid direct FOM': 'same-grid direct solve',
           'Coarse direct FOM, interpolated': 'coarse ($16^2$) direct solve, interpolated'}
    t = []; by = {}
    for r in rows:
        if r[1] in lab and r[0] in ('64', '256', '1024'):
            t.append([f'${r[0]}^2$', lab[r[1]], r[4], r[5], r[2], r[3]]); by[(r[0], r[1])] = r
    write('T11d_heat.tex', tabular(['mesh', 'arm', 'worst physical \\%', 'worst same-grid \\%', 'GPU ms', 'host ms'], t, 'llrrrr', r'\scriptsize'),
          f"heat 2D, earlier cell of the same decoder family, job {m.group(1) if m else '---'}")
    for mesh, nm in [('1024', 'TenTwentyFour'), ('256', 'TwoFiftySix')]:
        lb = by[(mesh, 'Linear bank, exact reduced time evolution')]; nl = by[(mesh, 'Current nonlinear ROM')]; fo = by[(mesh, 'Same-grid direct FOM')]
        macro(f'nHeatBankErr{nm}', f"{float(lb[4]):.2f}"); macro(f'nHeatBankMs{nm}', f"{float(lb[2]):.2f}")
        macro(f'nHeatHeadErr{nm}', f"{float(nl[4]):.2f}"); macro(f'nHeatHeadMs{nm}', f"{float(nl[2]):.1f}")
        macro(f'nHeatFomMs{nm}', f"{float(fo[2]):.2f}"); macro(f'nHeatBankOverHeadCost{nm}', f"{float(nl[2])/float(lb[2]):.0f}")


# =========================================================================== T6 / T7 head ablation + layers
def build_head_ablation():
    a = load('abl01'); p = load('pabl01')
    if a is None or p is None:
        write('T06_head_ablation.tex', gen('head-ablation', 'T6')); return
    macro('provAblBurgersJob', a['job_id']); macro('provAblBurgersGpu', a['gpu']); macro('provAblBurgersCommit', hexprefix(a['commit']))
    macro('provAblPoissonJob', p['job_id']); macro('provAblPoissonGpu', p['gpu']); macro('provAblPoissonCommit', hexprefix(p['commit']))
    A = {r['arm']: r for r in a['checks']['arm_table'] if r['intervals'] == 256}
    Pp = {r['arm']: r for r in p['checks']['arm_table'] if r['intervals'] == 1024}
    lab = {'a_neural_eq': '(a) neural head, EQ', 'a_neural_dense': '(a) neural head, dense',
           'b_linear_dec_eq': '(b) linear map (fit to head outputs)', 'b_linear_truth_eq': '(b) linear map (fit to truth)',
           'c_quad_dec_eq': '(c) quadratic map', 'e_pod16_dense': "(e) POD-LSPG $k'{=}16$", 'e_pod32_dense': "(e) POD-LSPG $k'{=}32$",
           'e_pod64_dense': "(e) POD-LSPG $k'{=}64$", 'e_pod128_dense': "(e) POD-LSPG $k'{=}128$",
           'd_freebank_dense': '(d) unrestricted bank ($R{=}512$)', 'fft_loose': 'FOM Newton, loose', 'fft_tight': 'FOM Newton, tight (reference)'}
    rows = []
    for k in ['a_neural_eq', 'a_neural_dense', 'b_linear_dec_eq', 'b_linear_truth_eq', 'c_quad_dec_eq', 'e_pod16_dense',
              'e_pod32_dense', 'e_pod64_dense', 'e_pod128_dense', 'd_freebank_dense', 'fft_loose', 'fft_tight']:
        r = A[k]
        rows.append([lab[k], str(r.get('k') or '---'), pct(r.get('worst_bank_projection_percent')),
                     pct(r.get('worst_best_found_percent')), pct(r['worst_same_grid_percent']), pct(r['worst_rollout_percent']),
                     ms(r['median_gpu_ms']), yn(r.get('all_stationary')) if r['kind'] == 'rom' else '---'])
    write('T06a_head_burgers.tex', tabular(['arm', 'dim.', 'bank floor \\%', 'best-found \\%', 'solved same-grid \\%',
                                            'vs ref \\%', 'GPU ms', 'stationary'], rows, 'lrrrrrrc', r'\scriptsize'),
          f"head-ablation Burgers job {a['job_id']}")
    labp = {'a_neural': '(a) neural head', 'a_neural_q32': '(a$+$) head $+$ 32 eliminated corrections', 'b_linear_dec': '(b) linear map (head outputs)',
            'b_linear_truth': '(b) linear map (truth)', 'c_quad_dec': '(c) quadratic map', 'e_pod8': "(e) POD-LSPG $k'{=}8$", 'e_pod16': "(e) POD-LSPG $k'{=}16$",
            'e_pod32': "(e) POD-LSPG $k'{=}32$", 'e_pod64': "(e) POD-LSPG $k'{=}64$", 'e_pod128': "(e) POD-LSPG $k'{=}128$",
            'd_freebank': '(d) unrestricted bank ($R{=}128$)', 'dst_direct': 'direct DST'}
    rows = []
    for k in ['a_neural', 'a_neural_q32', 'b_linear_dec', 'b_linear_truth', 'c_quad_dec', 'e_pod8', 'e_pod16', 'e_pod32', 'e_pod64',
              'e_pod128', 'd_freebank', 'dst_direct']:
        r = Pp[k]
        rows.append([labp[k], str(r.get('k') or '---'), pct(r.get('worst_bank_projection_percent')), pct(r.get('worst_best_found_percent')),
                     pct(r['worst_error_percent']), ms(r['median_query_ms'], 3)])
    write('T06b_head_poisson.tex', tabular(['arm', 'dim.', 'bank floor \\%', 'best-found \\%', 'solved \\%', 'query ms'],
                                           rows, 'lrrrrr', r'\scriptsize'), f"head-ablation Poisson job {p['job_id']}, 1024 intervals")
    n = A['a_neural_eq']
    macro('nAblBurgersNeural', pct(n['worst_same_grid_percent'])); macro('nAblBurgersNeuralMs', ms(n['median_gpu_ms']))
    macro('nAblBurgersLinear', pct(A['b_linear_dec_eq']['worst_same_grid_percent'])); macro('nAblBurgersQuad', pct(A['c_quad_dec_eq']['worst_same_grid_percent']))
    macro('nAblBurgersPodSixteen', pct(A['e_pod16_dense']['worst_same_grid_percent']))
    macro('nAblBurgersPodOneTwentyEight', pct(A['e_pod128_dense']['worst_same_grid_percent'])); macro('nAblBurgersPodOneTwentyEightMs', ms(A['e_pod128_dense']['median_gpu_ms']))
    macro('nAblBurgersFree', pct(A['d_freebank_dense']['worst_same_grid_percent'])); macro('nAblBurgersFreeMs', ms(A['d_freebank_dense']['median_gpu_ms']))
    macro('nAblBurgersFloor', pct(n['worst_bank_projection_percent'])); macro('nAblBurgersBestFound', pct(n['worst_best_found_percent']))
    macro('nAblBurgersSolverGapPp', f"{n['worst_same_grid_percent']-n['worst_best_found_percent']:.3f}")
    macro('nAblBurgersReductionOverFloor', f"{n['worst_best_found_percent']/n['worst_bank_projection_percent']:.1f}")
    q = Pp['a_neural']
    macro('nAblPoissonNeural', pct(q['worst_error_percent'])); macro('nAblPoissonNeuralMs', ms(q['median_query_ms'], 3))
    macro('nAblPoissonLinear', pct(Pp['b_linear_dec']['worst_error_percent'])); macro('nAblPoissonQuad', pct(Pp['c_quad_dec']['worst_error_percent']))
    macro('nAblPoissonPodSixteen', pct(Pp['e_pod16']['worst_error_percent']))
    macro('nAblPoissonPodOneTwentyEight', pct(Pp['e_pod128']['worst_error_percent'])); macro('nAblPoissonPodOneTwentyEightMs', ms(Pp['e_pod128']['median_query_ms'], 3))
    macro('nAblPoissonFree', pct(Pp['d_freebank']['worst_error_percent'])); macro('nAblPoissonDst', pct(Pp['dst_direct']['worst_error_percent'])); macro('nAblPoissonDstMs', ms(Pp['dst_direct']['median_query_ms'], 3))
    macro('nAblPoissonFloor', pct(q['worst_bank_projection_percent'])); macro('nAblPoissonBestFound', pct(q['worst_best_found_percent']))
    macro('nAblPoissonQthirtytwo', pct(Pp['a_neural_q32']['worst_error_percent'])); macro('nAblPoissonQthirtytwoMs', ms(Pp['a_neural_q32']['median_query_ms'], 3))
    return A, Pp


def build_three_layers(A, Pp):
    rows = []
    def row(cell, arm, floor, best, solved, note=''):
        rows.append([cell, arm, pct(floor), pct(best), pct(solved), f"{best/floor:.2f}" if floor else '---',
                     f"{(solved-best):.3f}" if best is not None else '---', note])
    if A:
        n = A['a_neural_eq']
        row('Burgers $256^2$', 'neural head, $q{=}0$', n['worst_bank_projection_percent'], n['worst_best_found_percent'], n['worst_same_grid_percent'], 'head-ablation')
    if Pp:
        q = Pp['a_neural']
        row('Poisson $1024^2$, $R{=}128$', 'neural head', q['worst_bank_projection_percent'], q['worst_best_found_percent'], q['worst_error_percent'], 'head-ablation')
    p = load('plinear_summary')
    if p:
        for mesh, job in [(256, '3780692'), (1024, '3783813')]:
            by = defaultdict(dict)
            for r in p:
                if r['mesh'] == mesh and r['job_id'] == job and not r.get('retracted'):
                    by[r['subject']][r['metric']] = r['value']
            if 'q0_m256@new_K32' in by:
                row(f'Poisson ${mesh}^2$, $R{{=}}512$', 'neural head $K{=}32$, $q{=}0$', 100 * by['bank_floor@new_K32']['worst_same_grid'],
                    100 * by['augmented_best_found_q0@new_K32']['worst_same_grid'], 100 * by['q0_m256@new_K32']['worst_same_grid'], 'p-linear')
    w = load('wladder_summary')
    if w:
        by = defaultdict(dict)
        for r in w:
            by[(r['mesh'], r['subject'])][r['metric']] = r['value']
        row('wave $256^2$', 'head, $q{=}0$', 100 * by[(256, 'linear_bank64@projection_floor')]['worst_energy_state'],
            100 * by[(256, 'head_q0@best_found')]['worst_energy_state'], 100 * by[(256, 'head_q0')]['worst_energy_state'], 'w-ladder')
        row('wave $256^2$', 'head, $q{=}32$', 100 * by[(256, 'linear_bank64@projection_floor')]['worst_energy_state'],
            100 * by[(256, 'nested_q32@best_found')]['worst_energy_state'], 100 * by[(256, 'nested_q32')]['worst_energy_state'], 'w-ladder')
    ls = load('lshape_summary')
    if ls:
        by = defaultdict(dict)
        for r in ls:
            by[(r['mesh'], r['subject'])][r['metric']] = r['value']
        for h, bank in [('head_sdf_R512_K32', 'sdf_R512'), ('head_sdf_R512_K16', 'sdf_R512')]:
            d = by[(256, h)]
            row('L-shape $256^2$', tt(h), 100 * by[(256, bank)]['bank_floor_dev_worst'], 100 * d['best_found_dev_worst'],
                100 * d['weak_solved_dev_worst'], 'lshape (untimed)')
    write('T07_three_layers.tex', tabular(['cell', 'arm', 'bank floor \\%', 'best-found \\%', 'solved \\%', 'best/floor',
                                           'solved $-$ best (pp)', 'source'], rows, 'llrrrrrl', r'\scriptsize'),
          'three-layer decomposition; worst over each cell\'s development cases')


# =========================================================================== T8 solver knobs
def build_knobs():
    t = load('tuning02')
    if t is None:
        write('T08_solver_knobs.tex', gen('fixed-checkpoint tuning', 'T8')); return
    S = t['summaries']
    macro('provTuneJob', t['provenance']['job_id']); macro('provTuneGpu', t['provenance']['gpu'])
    macro('provTuneCommit', hexprefix(t['provenance']['source_commit'])); macro('provTuneCkpt', hexprefix(t['checkpoint_sha256']))
    macro('provBurgersCkpt', hexprefix(t['checkpoint_sha256'], 16))
    macro('provBurgersCkptFull', t['checkpoint_sha256'])
    lab = {'m256_native': 'EQ $m{=}256$, tol $10^{-6}$, cap 180', 'm512_converged': 'EQ $m{=}512$, tol $10^{-8}$',
           'm512_gtol0.001': 'EQ $m{=}512$, tol $10^{-3}$', 'm512_cap2': 'EQ $m{=}512$, cap 2 (early-stopped)',
           'same_nt1e-2_dt005': 'FOM Newton $10^{-2}$, $\\Delta t{=}0.005$', 'same_nt1e-4_dt005': 'FOM Newton $10^{-4}$, $\\Delta t{=}0.005$',
           'same_nt1e-6_dt005': 'FOM Newton $10^{-6}$, $\\Delta t{=}0.005$', 'same_nt1e-2_dt01': 'FOM Newton $10^{-2}$, $\\Delta t{=}0.01$',
           'coarse_half_dt005': 'FOM $128^2$, $\\Delta t{=}0.005$', 'coarse_quarter_dt01': 'FOM $64^2$, $\\Delta t{=}0.01$'}
    rows = []
    for k in ['m256_native', 'm512_converged', 'm512_gtol0.001', 'm512_cap2', 'same_nt1e-2_dt01', 'same_nt1e-2_dt005',
              'same_nt1e-4_dt005', 'same_nt1e-6_dt005', 'coarse_half_dt005', 'coarse_quarter_dt01']:
        s = S['validation/' + k]
        rows.append([lab[k], pct(100 * s['worst_fixed_initial_error'], 3), ms(1000 * s['median_gpu_seconds']),
                     str(s['early_stopped_invocations']) + '/' + str(s['invocations'])])
    write('T08_solver_knobs.tex', tabular(['setting', 'worst \\% (32 held-out)', 'GPU ms', 'early-stopped'], rows, 'lrrr', r'\scriptsize'),
          f"fixed-checkpoint tuning, validation pass, job {t['provenance']['job_id']}")
    c = S['validation/m512_converged']; g = S['validation/m512_gtol0.001']; cap = S['validation/m512_cap2']
    macro('nTuneTolSavingPct', f"{100*(1-g['median_gpu_seconds']/c['median_gpu_seconds']):.1f}")
    macro('nTuneTolErrConverged', pct(100 * c['worst_fixed_initial_error'], 3)); macro('nTuneTolErrLoose', pct(100 * g['worst_fixed_initial_error'], 3))
    macro('nTuneCapTwoErr', pct(100 * cap['worst_fixed_initial_error'], 1)); macro('nTuneCapTwoMs', ms(1000 * cap['median_gpu_seconds']))
    f4 = S['validation/same_nt1e-4_dt005']
    macro('nTuneFomErr', pct(100 * f4['worst_fixed_initial_error'], 3)); macro('nTuneFomMs', ms(1000 * f4['median_gpu_seconds']))
    macro('nTuneBestRomErr', pct(100 * g['worst_fixed_initial_error'], 3)); macro('nTuneBestRomMs', ms(1000 * g['median_gpu_seconds']))
    f2 = S['validation/same_nt1e-2_dt005']
    macro('nTuneFomLooseWorst', pct(100 * f2['worst_fixed_initial_error'], 1))
    # cold-start caps (local GB10 diagnostics; ratios only)
    pc = load('poisson_caps')
    if pc:
        rows = pc['rows']; meshes = sorted({r['mesh'] for r in rows}); mesh = max(meshes)
        starts = sorted({r['start'] for r in rows})
        agg = defaultdict(list)
        for r in rows:
            if r['mesh'] == mesh:
                agg[(r['start'], r['cap'])].append(r)
        t8 = []
        caps = sorted({c for (_, c) in agg})
        for st in starts:
            for c in caps:
                L = agg.get((st, c))
                if not L or c not in (1, 2, 4, 6, 8, 10, 12, 20, max(caps)):
                    continue
                t8.append([tex_escape(st), str(c), pct(100 * max(x['error'] for x in L), 2), pct(100 * statistics.median(x['error'] for x in L), 2),
                           f"{statistics.median(x['iterations'] for x in L):.0f}", ms(statistics.median(x['ms'] for x in L), 2)])
            t8.append('MIDRULE')
        t8.pop()
        write('T08b_cold_start.tex', tabular(['start', 'LM cap', 'worst \\%', 'median \\%', 'median iters', 'ms (GB10, ratios only)'],
                                             t8, 'lrrrrr', r'\scriptsize'), f'Poisson cold-start cap sweep at {mesh} intervals, local GB10')
        macro('nColdMesh', str(mesh)); macro('nColdStarts', ', '.join(tex_escape(s) for s in starts))
        def worst(st, c):
            L = agg.get((st, c)); return 100 * max(x['error'] for x in L) if L else None
        # converged value = largest cap for each start
        for st in starts:
            key = re.sub(r'[^A-Za-z]', '', st.title())
            macro(f'nColdConv{key}', pct(worst(st, max(caps)), 2))
            macro(f'nColdCapTwo{key}', pct(worst(st, 2), 2) if worst(st, 2) is not None else '---')
            macro(f'nColdCapEight{key}', pct(worst(st, 8), 2) if worst(st, 8) is not None else '---')
            macro(f'nColdCapTwelve{key}', pct(worst(st, 12), 2) if worst(st, 12) is not None else '---')


# =========================================================================== T9 EQ certification
def build_eqtop():
    e = load('eqtop_summary')
    if e is None:
        write('T09_eq_ladder.tex', gen('b-eqtop', 'T9')); write('T09b_eq_certification.tex', gen('b-eqtop', 'T9')); return
    rows = e['rows']
    pend = e.get('pending') or []
    pend_job = ', '.join(x['job_id'] for x in pend) or 'none'
    macro('nEqtopPendingJob', pend_job)
    macro('nEqtopStatus', tex_escape(e.get('status', 'provisional')))
    macro('provEqtopJobs', '; '.join(f"{j['attempt']} = {j['job_id']} ({tex_escape(j['gpu'])}, commit {hexprefix(j['commit'])})" for j in e.get('jobs', [])))
    macro('provEqtopJob', ', '.join(j['job_id'] for j in e.get('jobs', [])))
    # ---- T9a: the primary EQ ladder and its dense twins, one allocation
    L = defaultdict(dict); Lmeta = {}
    ladder_rows = [r for r in rows if r['table'] == 'ladder']
    if not ladder_rows:
        raise SystemExit('b-eqtop final summary carries no timed-ladder rows')
    macro('provEqtopLadderSource', 'lane summary.json (final)')
    for r in ladder_rows:
        L[(r['ladder'], r['arm'])][r['metric']] = r['value']; Lmeta[(r['ladder'], r['arm'])] = r
    prim = sorted([k for k in L if k[0] == 'primary'], key=lambda k: Lmeta[k]['q'])
    dense = {Lmeta[k]['q']: L[k] for k in L if k[0] == 'dense'}
    dense_meta = {Lmeta[k]['q']: Lmeta[k] for k in L if k[0] == 'dense'}
    V = {v['q']: v for v in e.get('verdict_per_rung', [])}
    t = []; ratios = []
    for k in prim:
        m = Lmeta[k]; d = L[k]; q = m['q']
        dn = dense.get(q)
        ratio_cd = d['median_gpu_ms'] / dn['median_gpu_ms'] if dn else None
        if ratio_cd: ratios.append(ratio_cd)
        v = V.get(q, {})
        st = v.get('ladder_rule_status', 'one draw')
        t.append([str(q), str(m['m']), str(v.get('ladder_rule', {}).get('fit_states', '---')), f"{m['rho_max']:.4f}", tex_escape(st),
                  pct(d['worst_evolved_percent']), pct(d['worst_all_times_percent']), ms(d['median_gpu_ms']),
                  pct(dn['worst_evolved_percent']) if dn else '---', ms(dn['median_gpu_ms']) if dn else '---',
                  f"{ratio_cd:.3f}" if ratio_cd else '---'])
        macro('nEqtopStatusQ' + {0:'Zero',16:'Sixteen',32:'ThirtyTwo',64:'SixtyFour',128:'OneTwentyEight',256:'TwoFiftySix'}[q], tex_escape(st))
    job = Lmeta[prim[0]]['job_id']
    write('T09_eq_ladder.tex', tabular(['$q$', '$m$', 'fit states', '$\\rho_{\\max}$ (this draw)', 'construction status', 'EQ evolved \\%', 'EQ all \\%', 'EQ ms',
                                        'dense evolved \\%', 'dense ms', 'EQ/dense cost'], t, 'rrrrlrrrrrr', r'\scriptsize'),
          f'b-eqtop job {job}; construction status from the draw replication (job 3783811)')
    conf = [str(q) for q in sorted(V) if V[q]['ladder_rule_status'].startswith('confirmed')]
    marg = [f"{q} ({V[q]['draws_certifying_primary']}/{V[q]['draws']})" for q in sorted(V) if not V[q]['ladder_rule_status'].startswith('confirmed')]
    macro('nEqtopConfirmedRungs', ', '.join(conf)); macro('nEqtopMarginalRungs', ', '.join(marg))
    macro('nEqtopConfirmedCount', str(len(conf))); macro('nEqtopMarginalCount', str(len(marg)))
    if 256 in V:
        macro('nEqtopTopDraws', f"{V[256]['draws_certifying_primary']} of {V[256]['draws']}")
        macro('nEqtopTopRedrawMin', f"{sorted(x for x in [V[256]['rho_median'], V[256]['rho_max_of_draws']])[0]:.3f}")
    # replication table and spread
    rep = defaultdict(dict)
    for r in rows:
        if r['table'] == 'replication':
            rep[r['arm']][r['metric']] = r['value']; rep[r['arm']]['_q'] = r['q']; rep[r['arm']]['_m'] = r['m']; rep[r['arm']]['_status'] = r.get('construction_status')
    draws = defaultdict(list)
    for r in rows:
        if r['table'] == 'rules' and r['job_id'] == '3783811' and r['metric'] == 'rho_max' and r['arm'].startswith('reprow'):
            key = (r['q'], r['m'], '64' if 'w64' in r['arm'] else 'incumbent'); draws[key].append((r['arm'][-2:], r['value']))
    t = []; spreads = []
    for arm in sorted(rep, key=lambda a: (rep[a]['_q'], rep[a]['_m'])):
        d = rep[arm]; st = re.search(r'_(\d+)states', arm); fs = st.group(1) if st else '---'
        key = (d['_q'], d['_m'], '64' if fs == '64' else 'incumbent')
        vals = ', '.join(f"{v:.4f}" for _, v in sorted(draws.get(key, [])))
        spreads.append(d['spread_ratio'])
        t.append([str(d['_q']), str(d['_m']), fs, vals, f"{d['rho_min']:.4f} / {d['rho_median']:.4f} / {d['rho_max_of_draws']:.4f}",
                  f"{d['spread_ratio']:.2f}", f"{round(4*d['certified_primary_fraction']):.0f}/4", f"{round(4*d['certified_tight_fraction']):.0f}/4", tex_escape(d['_status'] or '---')])
    write('T09d_replication.tex', tabular(['$q$', '$m$', 'fit states', 'four draws: $\\rho_{\\max}$', 'min / median / max', 'spread', 'primary', 'tight', 'construction status (all draws)'],
                                          t, 'rrrp{3.6cm}p{2.9cm}rccp{2.6cm}', r'\scriptsize'), 'b-eqtop draw replication, job 3783811, four independent draws per construction')
    if spreads:
        macro('nEqtopSpreadMin', f"{min(spreads):.1f}"); macro('nEqtopSpreadMax', f"{max(spreads):.1f}")
    q256 = sorted(v for _, v in draws.get((256, 2048, '64'), []))
    if q256: macro('nEqtopTopRedrawRange', f"{q256[0]:.3f}--{q256[-1]:.3f}")
    macro('provEqtopRepJob', '3783811')
    macro('provEqtopLadderJob', job)
    errs = [L[k]['worst_evolved_percent'] for k in prim]
    macro('nEqtopLadderMonotone', yn(all(errs[i] >= errs[i + 1] for i in range(len(errs) - 1))))
    macro('nEqtopLadderConverged', yn(all(L[k].get('converged', True) for k in prim)))
    macro('nEqtopCostRatioMin', f"{min(ratios):.2f}"); macro('nEqtopCostRatioMax', f"{max(ratios):.2f}")
    macro('nEqtopLadderFirstErr', pct(errs[0])); macro('nEqtopLadderLastErr', pct(errs[-1]))
    macro('nEqtopLadderFirstMs', ms(L[prim[0]]['median_gpu_ms'])); macro('nEqtopLadderLastMs', ms(L[prim[-1]]['median_gpu_ms']))
    macro('nEqtopLadderErrSpan', f"{errs[0]/errs[-1]:.2f}"); macro('nEqtopLadderCostSpan', f"{L[prim[-1]]['median_gpu_ms']/L[prim[0]]['median_gpu_ms']:.2f}")
    top = [k for k in prim if Lmeta[k]['q'] == 256][0]
    macro('nEqtopTopEqErr', pct(L[top]['worst_evolved_percent'])); macro('nEqtopTopDenseErr', pct(dense[256]['worst_evolved_percent']))
    macro('nEqtopTopRho', f"{Lmeta[top]['rho_max']:.4f}"); macro('nEqtopTopM', str(Lmeta[top]['m']))
    macro('nEqtopTopEqMs', ms(L[top]['median_gpu_ms'])); macro('nEqtopTopDenseMs', ms(dense[256]['median_gpu_ms']))
    # ---- T9b: cheapest certified rule per rung, with fit-state count, and the parent lane's rule
    C = defaultdict(dict)
    for r in rows:
        if r['table'] == 'T9_cheapest_certified':
            C[r['q']][r['metric']] = r
    R = defaultdict(list)
    for r in rows:
        if r['table'] == 'rules' and r['metric'] == 'rho_max':
            R[r['q']].append(r)
    t = []
    for q in sorted(C):
        pr = C[q].get('cheapest_certified_primary_fit_states'); ti = C[q].get('cheapest_certified_tight_fit_states')
        parent = [r for r in R[q] if r['population'] == 'qrg304:reachable']
        parent_best = min(parent, key=lambda r: r['value']) if parent else None
        def cell(r):
            return '---' if r is None else f"{tex_escape(r['arm'])}, $m{{=}}{r['m']}$, {int(r['value'])} states, $\\rho_{{\\max}}{{=}}{r['rho_max']:.4f}$"
        t.append([str(q), cell(pr), cell(ti),
                  f"$m{{=}}{parent_best['m']}$, $\\rho_{{\\max}}{{=}}{parent_best['value']:.4f}$, {yn(parent_best['certified_primary'])}" if parent_best else '---'])
    write('T09b_eq_certification.tex', tabular(['$q$', 'cheapest rule passing the primary bar (its draw)', 'cheapest rule passing the tight bar (its draw)',
                                                'best parent-lane rule, re-scored: primary?'], t, r'rp{4.3cm}p{4.3cm}p{3.2cm}', r'\scriptsize'),
          'b-eqtop; single-draw passes, see the replication table for construction status')
    # the fit-state story at the top rungs, and the draw sensitivity at q=64
    for q, nm in [(128, 'OneTwentyEight'), (256, 'TwoFiftySix')]:
        pr = C[q].get('cheapest_certified_primary_fit_states')
        macro(f'nEqtopPrimary{nm}', yn(pr is not None))
        if pr: macro(f'nEqtopPrimary{nm}States', str(int(pr['value']))); macro(f'nEqtopPrimary{nm}M', str(pr['m'])); macro(f'nEqtopPrimary{nm}Rho', f"{pr['rho_max']:.4f}")
        parent = [r for r in R[q] if r['population'] == 'qrg304:reachable' and r['m'] == 2048]
        if parent: macro(f'nEqtopParent{nm}Rho', f"{parent[0]['value']:.4f}")
    rep = load('eqtop_report')
    hdr, arms = md_table(rep, 'arm') if rep else (None, None)
    old = [r for r in (arms or []) if r[0] == 'q256_eq_qrg304_m2048']
    if old:
        macro('nEqtopTopOldRuleErr', old[0][8]); macro('nEqtopTopOldRuleRho', old[0][5])
    else:
        macro('nEqtopTopOldRuleErr', gen('b-eqtop timed-arm table', 'old q=256 rule')); macro('nEqtopTopOldRuleRho', '---')
    q64 = {r['arm']: r['value'] for r in R[64] if r['m'] == 1024}
    macro('nEqtopQsixtyFourArchivedRho', f"{q64.get('reachable', float('nan')):.4f}"); macro('nEqtopQsixtyFourRedrawRho', f"{q64.get('std', float('nan')):.4f}")
    # the NNLS-fit-is-not-a-certificate pair (static rules with a small fit and rho above the bar)
    F = defaultdict(dict)
    for r in rows:
        if r['table'] == 'rules':
            F[(r['q'], r['arm'], r['m'], r['population'], r['job_id'])][r['metric']] = r['value']
            F[(r['q'], r['arm'], r['m'], r['population'], r['job_id'])]['_cert'] = r['certified_primary']
    stat = [(k, v) for k, v in F.items() if k[1] == 'static' and v.get('rho_max') is not None]
    if stat:
        worst = max(stat, key=lambda kv: kv[1]['rho_max'])
        macro('nEqtopStaticWorstRho', f"{worst[1]['rho_max']:.3f}"); macro('nEqtopStaticWorstQ', str(worst[0][0]))
        macro('nEqtopStaticWorstFit', f"{worst[1].get('relative_fit', float('nan')):.1e}")
    # anti-correlation example at q=128: parent static m=2048 has the smaller fit and the larger rho than fs64 m=2048
    macro('nEqtopBar', '0.116'); macro('nEqtopTightBar', '0.06')
    macro('nEqtopBarOrigin', 'the held-out $\\rho$ of the incumbent $q{=}0$ rule at the state carrying the first-interval penalty, measured in the q-diag cell and adopted as the primary bar in the q-ridge design before any certification job ran; the tight bar was declared before b-eqtop ran')
    # full rules table (appendix)
    t = []
    for k in sorted(F, key=lambda k: (k[0], k[3], k[1], k[2] or 0)):
        q, arm, m, pop, jb = k; d = F[k]
        if d.get('rho_max') is None: continue
        t.append([str(q), tex_escape(arm), str(m), tex_escape(pop), f"{d['relative_fit']:.2e}" if d.get('relative_fit') is not None else '---',
                  f"{d['rho_max']:.4f}", f"{d['rho_p95']:.4f}" if d.get('rho_p95') is not None else '---', yn(d['_cert']), tt(jb)])
    write('T09c_eq_rules_full.tex', tabular(['$q$', 'fit arm', '$m$', 'population', 'NNLS rel.\\ fit', '$\\rho_{\\max}$', '$\\rho_{95}$', 'primary', 'job'],
                                             t, 'rlrlrrrcl', r'\tiny'), f'b-eqtop every rule; PROVISIONAL, {pend_job} pending')


# =========================================================================== T10 mesh ladder
def build_mesh():
    m = load('mesh_ladder')
    if m is None:
        write('T10_mesh_ladder.tex', gen('mesh-ladder', 'T10'))
        return None
    gf = m['generated_from']
    for pde, nm in [('Burgers 2D', 'Burgers'), ('Poisson 2D', 'Poisson')]:
        macro(f'provMesh{nm}Job', gf[pde]['job_id']); macro(f'provMesh{nm}Gpu', gf[pde]['gpu']); macro(f'provMesh{nm}Commit', hexprefix(gf[pde]['commit']))
        macro(f'provMesh{nm}Ckpt', hexprefix(gf[pde]['checkpoint_sha256']))
    macro('provPoissonCkptFull', gf['Poisson 2D']['checkpoint_sha256'])
    rows = []
    for pde, nm in [('Burgers 2D', 'Burgers'), ('Poisson 2D', 'Poisson')]:
        s = m['summaries'][pde]; cr = {r['intervals']: r for r in s['crossover']['rows']}
        for mesh in s['meshes']:
            t = s['table'][f"{mesh}|{s['rom']}"]; cached = s['cached'][str(mesh)]; c = cr[mesh]
            rows.append([nm, str(mesh), f"{(mesh-1)**2:,}", ms(1000 * cached['seconds']['median']),
                         ms(1000 * t['device_seconds']['median']), ms(1000 * t['host_to_host_seconds']['median']),
                         pct(100 * t['worst_error_requested_grid']), pct(100 * t['worst_same_grid_discrepancy']),
                         tt(c['fom_subject']), ms(c['fom_device_ms']), f"{c['speedup_fom_over_rom']:.3f}", yn(t['meets_target'])])
        rows.append('MIDRULE')
    rows.pop()
    write('T10_mesh_ladder.tex', tabular(['PDE', 'intervals', 'unknowns', 'ROM cached ms', 'ROM device ms', 'ROM host ms', 'ROM worst vs ref \\%',
                                          'ROM vs same-grid FOM \\%', 'efficient FOM', 'FOM ms', 'FOM/ROM', 'ROM meets 5\\%'], rows, 'lrrrrrrrlrrc', r'\tiny'),
          'frozen-checkpoint mesh ladder; one job per PDE')
    for pde, nm in [('Burgers 2D', 'Burgers'), ('Poisson 2D', 'Poisson')]:
        s = m['summaries'][pde]
        macro(f'nMesh{nm}CachedRatio', f"{s['flatness_cached']['ratio_finest_over_coarsest']:.3f}")
        macro(f'nMesh{nm}CompleteRatio', f"{s['flatness_complete']['ratio_finest_over_coarsest']:.3f}")
        macro(f'nMesh{nm}UnknownGrowth', f"{s['flatness_cached']['unknown_growth']:.0f}")
        macro(f'nMesh{nm}EverFaster', yn(s['crossover']['rom_ever_faster']))
        vals = s['flatness_cached']['values_ms']
        macro(f'nMesh{nm}CachedFirst', ms(vals[0], 2)); macro(f'nMesh{nm}CachedLast', ms(vals[-1], 2))
        rat = [r['speedup_fom_over_rom'] for r in s['crossover']['rows']]
        macro(f'nMesh{nm}FomOverRomRange', f"{min(rat):.3f}--{max(rat):.3f}")
        tb = s['table'][f"{s['meshes'][0]}|{s['rom']}"]; tl = s['table'][f"{s['meshes'][-1]}|{s['rom']}"]
        macro(f'nMesh{nm}ErrFirst', pct(100 * tb['worst_error_requested_grid'])); macro(f'nMesh{nm}ErrLast', pct(100 * tl['worst_error_requested_grid']))
    return m


# =========================================================================== T15 speed
def build_speed():
    rep = load('speed_report')
    for k in ('speed_audit_spd01', 'speed_audit_fine01', 'speed_audit_comp01'):
        load(k)
    if rep is None:
        write('T15_speed.tex', gen('b-speed', 'T15')); return
    hdr, rows = md_table(rep, 'attempt')
    t = [[tex_escape(r[0]), r[1], tt(r[2]), r[3], r[4], tex_escape(r[5]), tex_escape(r[6]), tex_escape(r[7]), r[8]] for r in rows]
    write('T15_speed.tex', tabular(['attempt', 'intervals', 'arm', 'incumbent ms', 'arm ms', 'GPU speedup', 'incl.\\ host', 'parity', '2$\\times$ target'],
                                   t, 'llllllllc', r'\scriptsize'), 'b-speed; parsed from the lane\'s generated report table (its generator reads result.json + audit.json)')
    r0 = rows[0]
    macro('nSpeedGpuTwoFiftySix', tex_escape(r0[5])); macro('nSpeedIncumbentMs', r0[3]); macro('nSpeedArmMs', r0[4])
    r1024 = [r for r in rows if r[1] == '1024'][0]
    macro('nSpeedGpuTenTwentyFour', tex_escape(r1024[5]))
    m = re.search(r'\*\*(\d+\.\d+) us fixed per time step \+ (\d+\.\d+) us per LM iteration\*\*', rep) or \
        re.search(r'\*\*(\d+\.\d+) µs fixed per time step \+ (\d+\.\d+) µs per LM iteration\*\*', rep)
    if m:
        macro('nSpeedFixedUs', m.group(1)); macro('nSpeedPerIterUs', m.group(2))
    m = re.search(r'\*\*(\d+\.\d+) us per fusion \+ (\d+\.\d+) us\*\* \(R-squared = (\d+\.\d+)\)', rep)
    if m:
        macro('nSpeedUsPerFusion', m.group(1)); macro('nSpeedFusionRsq', m.group(3))
    macro('provSpeedJobs', '3745655 (spd01), 3745656 (fine01), 3745913 (comp01)')  # from the lane's lab-log entry; see PROV for audit SHAs


# =========================================================================== T16 training study
def build_training():
    e = load('headtrain_eval')
    if e is None:
        write('T16_training.tex', gen('b-head-train', 'T16')); return
    arms = {r['arm']: r for r in e['arm_table']}
    keep = ['incumbent_eq', 'd128k16rec_eq', 'd512k16rec_eq', 'd2048k16rec_eq', 'd4608k16rec_eq', 'd128k32rec_eq', 'd2048k32rec_eq',
            'd128k16w_eq', 'd128k16t_eq', 'd128k16z_eq', 'best_d2048k32w_eq', 'joint_d2048k32r512_eq']
    lab = {'incumbent_eq': 'incumbent (4608 traj., $K{=}16$)', 'd128k16rec_eq': '128 traj., $K{=}16$', 'd512k16rec_eq': '512 traj.',
           'd2048k16rec_eq': '2048 traj.', 'd4608k16rec_eq': '4608 traj.\\ (like-for-like retrain)', 'd128k32rec_eq': '128 traj., $K{=}32$',
           'd2048k32rec_eq': '2048 traj., $K{=}32$', 'd128k16w_eq': '128 traj., weak-residual term', 'd128k16t_eq': '128 traj., trajectory term',
           'd128k16z_eq': '128 traj., code-smoothness term', 'best_d2048k32w_eq': 'selected: 2048 traj., $K{=}32$, weak term',
           'joint_d2048k32r512_eq': 'joint bank$+$head, $R{=}512$'}
    t = []
    for k in keep:
        r = arms[k]
        t.append([lab[k], str(r['K']), pct(r['worst_bank_projection_percent']), pct(r['worst_best_found_percent']), pct(r['worst_same_grid_percent']),
                  pct(r['worst_same_grid_evolved_percent']), ms(r['median_gpu_ms']), yn(r['all_stationary'] and r['all_completed'])])
    write('T16_training.tex', tabular(['arm (EQ query)', '$K$', 'bank floor \\%', 'best-found \\%', 'solved all \\%', 'solved evolved \\%', 'GPU ms', 'conv.'],
                                      t, 'lrrrrrrc', r'\scriptsize'), 'b-head-train evaluation job 3749074')
    macro('nTrainIncumbentBest', pct(arms['incumbent_eq']['worst_best_found_percent'])); macro('nTrainBestRetrained', pct(arms['best_d2048k32w_eq']['worst_best_found_percent']))
    macro('nTrainLikeForLike', pct(arms['d4608k16rec_eq']['worst_best_found_percent']))
    macro('nTrainLikeForLikeRatio', f"{arms['d4608k16rec_eq']['worst_best_found_percent']/arms['incumbent_eq']['worst_best_found_percent']:.2f}")
    v = e['checks']['success_criteria_evaluated']['detail']
    macro('nTrainArmsEvaluated', str(len(e['verdicts']))); macro('nTrainArmsPassing', str(sum(1 for x in e['verdicts'] if x['success'])))
    macro('provTrainJobs', '3745912 (training), 3749074 (evaluation)')


# =========================================================================== T18 L-shape
def build_lshape():
    s = load('lshape_summary'); rep = load('lshape_report')
    if s is None:
        write('T18_lshape.tex', gen('lshape', 'T18')); return
    by = defaultdict(dict)
    for r in s:
        by[(r['mesh'], r['subject'])][r['metric']] = r['value']
    job = s[0]['job_id']
    m = re.search(r'Training job `(\d+)` on `(NVIDIA [^`]+)`, source commit `([0-9a-f]+)`', rep or '')
    macro('provLshapeJob', job); macro('provLshapeGpu', m.group(2) if m else '---'); macro('provLshapeCommit', hexprefix(m.group(3)) if m else '---')
    banks = ['smooth_R256', 'smooth_R512', 'sdf_R256', 'sdf_R512', 'enrich_R512', 'smooth_ff128s2_R512']
    t = []
    for b in banks:
        t.append([tt(b), pct(100 * by[(256, b)]['bank_floor_dev_worst']), pct(100 * by[(512, b)]['bank_floor_dev_worst']),
                  pct(100 * by[(256, b)]['bank_floor_common_worst'])])
    write('T18a_lshape_bank.tex', tabular(['bank', 'floor, dev.\\ $256^2$ \\%', 'floor, dev.\\ $512^2$ \\%', 'floor, common $256^2$ \\%'], t, 'lrrr', r'\scriptsize'),
          f'lshape job {job}')
    heads = [k[1] for k in by if k[0] == 256 and k[1].startswith('head_')]
    t = []
    for h in sorted(heads):
        d = by[(256, h)]
        t.append([tt(h), pct(100 * d['best_found_dev_worst']), pct(100 * d['weak_solved_dev_worst']), f"{d['head_floor_over_bank_floor']:.2f}"])
    write('T18b_lshape_head.tex', tabular(['head arm', 'best-found \\%', 'weak solve \\% (untimed)', 'best-found / bank floor'], t, 'lrrr', r'\scriptsize'),
          f'lshape job {job}')
    macro('nLshapeBestFloor', pct(100 * min(by[(512, b)]['bank_floor_dev_worst'] for b in banks)))
    macro('nLshapeBestFloorCommon', pct(100 * min(by[(256, b)]['bank_floor_common_worst'] for b in banks)))
    macro('nLshapeSelectedFloorCommon', pct(100 * by[(256, 'sdf_R512')]['bank_floor_common_worst']))
    macro('nLshapeSmoothFloor', pct(100 * by[(512, 'smooth_R512')]['bank_floor_dev_worst'])); macro('nLshapeEnrichFloor', pct(100 * by[(512, 'enrich_R512')]['bank_floor_dev_worst']))
    macro('nLshapeEnrichGain', f"{by[(512, 'smooth_R512')]['bank_floor_dev_worst']/by[(512, 'enrich_R512')]['bank_floor_dev_worst']:.3f}")
    macro('nLshapeSdfFloor', pct(100 * by[(512, 'sdf_R512')]['bank_floor_dev_worst']))
    ratios = [by[(256, h)]['head_floor_over_bank_floor'] for h in heads]
    macro('nLshapeHeadRatioMin', f"{min(ratios):.2f}"); macro('nLshapeHeadRatioMax', f"{max(ratios):.2f}")
    macro('nLshapeKthirtytwoBest', pct(100 * by[(256, 'head_sdf_R512_K32')]['best_found_dev_worst']))
    macro('nLshapeKsixteenBest', pct(100 * by[(256, 'head_sdf_R512_K16')]['best_found_dev_worst']))
    if load('lshape_solve_summary') is None:
        macro('nLshapeSolve', gen('lshape solve jobs 3784662/3/4', 'T18 solve layer'))


# =========================================================================== T17 offline cost, T1b spec, T19 solver variants
def build_offline_and_spec():
    rows = []
    tr = load('headtrain_train')
    if tr:
        arms = {r['arm']: r for r in tr['arm_table']}
        for k, lab, mac in [('d4608k16rec', 'head training, like-for-like retrain of the incumbent recipe (4608 traj., 200k steps)', 'nOfflineTrainSecondsLikeForLike'),
                            ('best_d2048k32w', 'head training, selected retrained arm', 'nOfflineTrainSecondsBest')]:
            if k in arms and arms[k].get('train_gpu_hours') is not None:
                rows.append([lab, f"{3600*arms[k]['train_gpu_hours']:.0f}", 'once per checkpoint', 'b-head-train job 3745912 (A100-PCIE-40GB)'])
                macro(mac, f"{3600*arms[k]['train_gpu_hours']:.0f}")
    cc = load('cclad01')
    if cc:
        d = cc['checks'].get('flattened_direction_fit', {}).get('detail', {})
        if d.get('seconds'):
            rows.append(['correction directions $C_q$ (PCA of the head residual; flattened LM fit of the best-found codes)', f"{d['seconds']:.0f}", 'once per checkpoint, all $q$', f"cheap-corrections job {cc['job_id']}"])
            macro('nOfflineDirectionSeconds', f"{d['seconds']:.0f}")
    e = load('eqtop_summary')
    if e:
        R = defaultdict(dict)
        for r in e['rows']:
            if r['table'] == 'rules':
                R[(r['q'], r['arm'], r['m'], r['population'], r['job_id'])][r['metric']] = r['value']
        Lm = {}
        for r in e['rows']:
            if r['table'] == 'ladder' and r['ladder'] == 'primary' and r['metric'] == 'worst_evolved_percent':
                Lm[r['q']] = r
        tot = 0.0; names = {0: 'Zero', 16: 'Sixteen', 32: 'ThirtyTwo', 64: 'SixtyFour', 128: 'OneTwentyEight', 256: 'TwoFiftySix'}
        for q in sorted(Lm):
            m = Lm[q]
            cand = [(k, v) for k, v in R.items() if k[0] == q and k[2] == m['m'] and v.get('rho_max') is not None and abs(v['rho_max'] - m['rho_max']) < 1e-9]
            if cand:
                k, v = cand[0]; fs = v.get('fit_seconds') or 0; tot += fs
                rows.append([f"certified EQ rule, $q={q}$, $m={m['m']}$ ({tex_escape(k[1])}, {tex_escape(k[3])})", f"{fs:.0f}", 'once per rung and rule', f"b-eqtop job {k[4]}"])
                macro('nOfflineFitSecondsQ' + names[q], f"{fs:.0f}")
        macro('nOfflineFitSecondsLadderTotal', f"{tot:.0f}"); macro('nOfflineFitHoursLadderTotal', f"{tot/3600:.1f}")
    write('T17_offline_cost.tex', tabular(['offline stage', 'seconds', 'amortised over', 'source'], rows, r'p{6.2cm}rp{2.4cm}p{3.6cm}', r'\scriptsize'),
          'offline cost per stage; NNLS fit seconds are those of the rule each primary-ladder rung ran')
    if cc:
        A = {r['arm']: r for r in cc['checks']['arm_table']}
        t = []
        for arm, lab in [('q64_m4_dense_joint', '$q{=}64$, joint LM on $(z,y)$'), ('q64_m4_dense_block', '$q{=}64$, block-damped'),
                         ('q64_m4_dense_varpro', '$q{=}64$, plain variable projection'),
                         ('q128_m256_dense_varpro', '$q{=}128$, plain variable projection'), ('q128_m256_dense_block', '$q{=}128$, block-damped')]:
            r = A.get(arm)
            if r:
                t.append([lab, str(r['M']), pct(r['worst_same_grid_percent']), str(r['total_budget_exits']), yn(r['all_stationary']), ms(r['median_gpu_ms'])])
        write('T19_solver_variants.tex', tabular(['arm', '$M$', 'worst all-times \\%', 'budget exits', 'all stationary', 'GPU ms'], t, 'lrrrcr', r'\scriptsize'),
              f"cheap-corrections job {cc['job_id']} ({cc['gpu']})")
        macro('provCcladJob', cc['job_id']); macro('provCcladGpu', cc['gpu'])
        j = A.get('q64_m4_dense_joint'); bl = A.get('q64_m4_dense_block')
        if j and bl:
            macro('nBlockJointBudgetExits', str(j['total_budget_exits'])); macro('nBlockBlockBudgetExits', str(bl['total_budget_exits']))
            macro('nBlockJointMs', ms(j['median_gpu_ms'])); macro('nBlockBlockMs', ms(bl['median_gpu_ms']))
            macro('nBlockJointErr', pct(j['worst_same_grid_percent'])); macro('nBlockBlockErr', pct(bl['worst_same_grid_percent']))
    spec = [
        ['Burgers 2D', '$u_t+u(u_x+u_y)=\\nu\\Delta u$ on $(0,1)^2$, $u=0$ on $\\partial\\Omega$',
         '$u_0=a\\exp(-\\lvert x-c\\rvert^2/2w^2)$, $c_i\\sim U(0.15,0.85)$, $w\\sim U(0.05,0.20)$, $a\\sim U(0.5,2)$; $\\nu\\sim\\log U(0.01,0.1)$; outputs $t\\in\\{0.05,\\dots,0.25\\}$',
         tt('experiments/mr-burgers2d/engines.py:params_draw')],
        ['Poisson 2D', '$-\\Delta u=f$ on $(0,1)^2$, $u=0$ on $\\partial\\Omega$',
         '$f=a\\exp(-\\lvert x-c\\rvert^2/2w^2)$, $c_i\\sim U(0.15,0.85)$, $w\\sim\\log U(0.02,0.1)$, $a\\sim U(0.5,2)$',
         tt('multistage-precision/ms_parametric.py:sample_params')],
        ['Poisson, L-shape', 'same source family on $(0,1)^2\\setminus[\\tfrac12,1)^2$', 'as above; sources centred in the removed quadrant rejected', tt('experiments/lshape')],
        ['Heat 2D', '$u_t=\\kappa\\Delta u$ on $(0,1)^2$, $\\kappa=0.02$ fixed', 'polynomial-boundary Gaussian family (single\\_bc\\_poly\\_gaussian\\_v1); Crank--Nicolson $\\Delta t=0.025$', 'heat linear-bank report, job 3511417'],
        ['Wave 2D (reflective)', '$u_{tt}=c^2\\Delta u$ on $(0,1)^2$, $u=0$ on $\\partial\\Omega$',
         'compact bump $\\times$ Gaussian: half-widths $s_i\\sim U(0.36,0.42)$, centre $c_i\\sim U(s_i{+}0.025,\\,1{-}s_i{-}0.025)$, amplitude $\\sim U(0.7,1.3)$, $\\sigma_i\\sim U(0.12,0.16)$, advective velocity $v_i\\sim U(-0.5,0.5)$ (zero every fourth case); speed $c\\sim U(0.85,1.15)$',
         tt('experiments/multiresolution-wave/audit_dynamics.py:parameter_rows')],
    ]
    write('T01b_spec.tex', tabular(['PDE', 'equation and boundary', 'sampled family (transcribed from the generator source)', 'source'], spec, r'p{1.7cm}p{3.4cm}p{6.2cm}p{2.6cm}', r'\tiny'),
          'sampling families transcribed from the generator sources named in the last column')


# =========================================================================== T12/T13 seeds, ns2d
def build_pending():
    if load('seeds_summary') is None:
        write('T12_seeds.tex', gen('b-seeds', 'T12') + '\n')
        write('T13_sealed.tex', gen('b-seeds sealed cohort', 'T13') + '\n')
        macro('nSeedsStatus', gen('b-seeds', 'seeds'))
    n = load('ns2d_summary')
    if n:
        gates = [r for r in n if r.get('gate')]
        passed = sum(1 for r in gates if r['passed']); macro('nNsGatesPassed', f'{passed} of {len(gates)}')
        fails = sorted({r['gate'] for r in gates if not r['passed']})
        macro('nNsGatesFailed', tex_escape(', '.join(fails)) if fails else 'none')
        macro('provNsJob', n[0]['job_id'])
    macro('nNsRom', gen('ns2d phases 2--3', 'NS ROM'))


# =========================================================================== T1 / T2
def build_problems_and_provenance(mesh):
    t = load('tuning02')
    fx = t['config']['fixed'] if t else {}
    ra = t['config']['reference_anchor'] if t else {}
    rows = [
        ['Burgers 2D', '$u_t+u(u_x+u_y)=\\nu\\Delta u$, $(0,1)^2$, $u|_{\\partial\\Omega}=0$',
         f"$256^2$ (ladder 64--1024)", f"$\\Delta t={fx.get('dt','---')}$, backward Euler, sign-upwind",
         f"$K={fx.get('latent_dimension','---')}$, $R={fx.get('bank_rank','---')}$",
         f"refined $ {ra.get('intervals','---')}^2$, $\\Delta t={ra.get('dt','---')}$" if ra else '---',
         '6 development cases; 32 held-out (tuning); sealed cohort unopened'],
        ['Poisson 2D', '$-\\Delta u=f$, $(0,1)^2$, $u|_{\\partial\\Omega}=0$', '$256^2$, $1024^2$', 'none (elliptic)',
         '$K=16$, $R=128$ (incumbent); $K=32$, $R=512$', 'exact discrete (DST); 2048$^2$ refinement', '12 development sources'],
        ['Heat 2D', '$u_t=\\kappa\\Delta u$, $(0,1)^2$', '$64^2$--$1024^2$', 'Crank--Nicolson', '$k=8$, $R=32$', 'exact modal', '12 development cases (earlier cell, job 3511417)'],
        ['Wave 2D (reflective)', '$u_{tt}=c^2\\Delta u$, $(0,1)^2$, $u|_{\\partial\\Omega}=0$', '$64^2$, $256^2$, $1024^2$', 'RK4 on the manifold; exact modal propagation for the bank',
         '$K=32$, $R=64$', 'direct DST', '8 development cases'],
        ['Poisson, L-shape', '$-\\Delta u=f$, $(0,1)^2\\setminus[\\tfrac12,1)^2$', '$256^2$, $512^2$', 'none', '$K\\in\\{16,32\\}$, $R\\in\\{256,512,514\\}$',
         'sparse direct (SuperLU)', '3072 / 256 / 32 sources (train / selection / development)'],
    ]
    write('T01_problems.tex', tabular(['PDE', 'equation, domain, boundary', 'meshes', 'time stepping', 'reduced sizes', 'reference', 'cohorts'],
                                      rows, r'p{1.5cm}p{3.0cm}p{1.4cm}p{2.0cm}p{1.9cm}p{1.9cm}p{2.3cm}', r'\tiny'),
          'problem specification; numeric fields read from tuning02 config where recorded')
    # T2 provenance: one row per result table
    P = []
    def prov(table, lane, job, gpu, commit, ckpt):
        P.append([table, lane, job, gpu, commit, ckpt])
    prov('T3, T5', 'b-panel', MACROS.get('provPanelJob', '---'), MACROS.get('provPanelGpu', '---'), MACROS.get('provPanelCommit', '---'), MACROS.get('provPanelCkpt', '---'))
    prov('T4', 'b-qxm', MACROS.get('provQxmJobs', '---'), 'per job', MACROS.get('provQxmCommit', '---'), MACROS.get('provBurgersCkpt', '---'))
    prov('T6a, T7', 'head-ablation (Burgers)', MACROS.get('provAblBurgersJob', '---'), MACROS.get('provAblBurgersGpu', '---'), MACROS.get('provAblBurgersCommit', '---'), MACROS.get('provBurgersCkpt', '---'))
    prov('T6b, T7', 'head-ablation (Poisson)', MACROS.get('provAblPoissonJob', '---'), MACROS.get('provAblPoissonGpu', '---'), MACROS.get('provAblPoissonCommit', '---'), MACROS.get('provMeshPoissonCkpt', '---'))
    prov('T8', 'fixed-checkpoint tuning', MACROS.get('provTuneJob', '---'), MACROS.get('provTuneGpu', '---'), MACROS.get('provTuneCommit', '---'), MACROS.get('provTuneCkpt', '---'))
    prov('T9', 'b-eqtop', MACROS.get('provEqtopJobs', '---'), 'see job list', 'see job list', MACROS.get('provBurgersCkpt', '---'))
    prov('T10', 'mesh-ladder (Burgers)', MACROS.get('provMeshBurgersJob', '---'), MACROS.get('provMeshBurgersGpu', '---'), MACROS.get('provMeshBurgersCommit', '---'), MACROS.get('provMeshBurgersCkpt', '---'))
    prov('T10', 'mesh-ladder (Poisson)', MACROS.get('provMeshPoissonJob', '---'), MACROS.get('provMeshPoissonGpu', '---'), MACROS.get('provMeshPoissonCommit', '---'), MACROS.get('provMeshPoissonCkpt', '---'))
    prov('T11a', 'w-ladder', MACROS.get('provWaveJobs', '---'), 'per job', 'per job', 'frozen-math SHA asserted in job')
    prov('T11d', 'heat linear bank (2026-09-10)', MACROS.get('provHeatJob', '---'), MACROS.get('provHeatGpu', '---'), MACROS.get('provHeatCommit', '---'), 'expanded\\_seed790715 (frozen)')
    prov('T11b, T11c', 'p-linear', MACROS.get('provPlinJobs', '---') + '; head capacity job ' + MACROS.get('provPlinHeadJob', '---'), 'per job', 'per job', 'R=512/K=32 checkpoint (pbh02 primary)')
    prov('T14, T14c, T14d', 'no-second (5 of 8 jobs counted; two preamble deaths uncounted)', MACROS.get('provOpJobs', '---'), 'A100 (per job)', 'per job', 'operator checkpoints hash-verified in job')
    prov('T15', 'b-speed', MACROS.get('provSpeedJobs', '---'), 'A100 80GB PCIe', '8fdfbb08 / 94399dd6', MACROS.get('provBurgersCkpt', '---'))
    prov('T16', 'b-head-train', MACROS.get('provTrainJobs', '---'), 'A100-PCIE-40GB', '0f0c56f7 / 2b9e7ee7', 'trained checkpoints hashed in archive')
    prov('T18', 'lshape', MACROS.get('provLshapeJob', '---'), MACROS.get('provLshapeGpu', '---'), MACROS.get('provLshapeCommit', '---'), '7 heads + bases Git-tracked')
    prov('T12, T13', 'b-seeds', gen('b-seeds', 'T2 seeds row'), '---', '---', '---')
    write('T02_provenance.tex', tabular(['table', 'lane', 'job id(s)', 'GPU', 'commit', 'checkpoint'], P, r'lp{2.3cm}p{3.6cm}p{2.2cm}p{2cm}p{2.6cm}', r'\tiny'),
          'provenance registry; SHA256 of every file read is in tables/provenance.json')


# =========================================================================== main
def main():
    OUT.mkdir(exist_ok=True)
    build_panel()
    build_qxm()
    build_operators()
    build_linear()
    build_heat()
    A, Pp = build_head_ablation() or (None, None)
    build_three_layers(A, Pp)
    build_knobs()
    build_eqtop()
    mesh = build_mesh()
    build_speed()
    build_training()
    build_lshape()
    build_pending()
    build_offline_and_spec()
    build_problems_and_provenance(mesh)
    try:
        v = [float(MACROS['nEqtopTopDenseMs']), float(MACROS['nPanelBestRungMs']), float(MACROS['nQxmFixedQtopMs'])]
        macro('nSpreadQtwoFiftySixPct', f"{100*(max(v)-min(v))/min(v):.0f}")
    except (KeyError, ValueError):
        pass
    # numbers.tex
    lines = ['% GENERATED by paper/gen_tables.py -- do not edit.',
             '% Every prose number in the manuscript is one of these macros.']
    for k in sorted(MACROS):
        lines.append(f'\\newcommand{{\\{k}}}{{{MACROS[k]}}}')
    (OUT / 'numbers.tex').write_text('\n'.join(lines) + '\n')
    (OUT / 'provenance.json').write_text(json.dumps({'sources': PROV, 'pending': PENDING}, indent=2))
    (OUTMD / 'numbers.json').write_text(json.dumps({k: tex2md(v) for k, v in sorted(MACROS.items())}, indent=1))
    (OUT / 'PENDING.md').write_text('# Placeholders emitted by gen_tables.py\n\n' + '\n'.join(f'- `{w}` waits on **{l}**' for w, l in PENDING) + '\n')
    n_missing = sum(1 for v in PROV.values() if not v['reachable'])
    print(f'wrote {len(MACROS)} macros, {len(list(OUT.glob("T*.tex")))} tables, {len(PENDING)} pending placeholders, '
          f'{n_missing} unreachable sources: {[k for k, v in PROV.items() if not v["reachable"]]}')


if __name__ == '__main__':
    sys.exit(main())
