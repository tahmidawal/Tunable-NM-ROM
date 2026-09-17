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
    'panel_bpn301_recheck': '2026-09-17-b-panel/experiments/b-panel/checks/bpn301-recheck.json',
    'nosecond_summary': '2026-09-17-no-second/experiments/no-second/reports/summary.json',
    'wladder_summary': '2026-09-17-w-ladder/experiments/w-ladder/reports/summary.json',
    'wladder_report': '2026-09-17-w-ladder/experiments/w-ladder/reports/2026-09-17-w-ladder.md',
    'plinear_summary': '2026-09-17-p-linear/experiments/p-linear/reports/summary.json',
    'plinear_report': '2026-09-17-p-linear/experiments/p-linear/reports/2026-09-17-p-linear.md',
    'plinear_verdicts': '2026-09-17-p-linear/experiments/p-linear/reports/verdicts.json',
    'lshape_summary': '2026-09-17-lshape/experiments/lshape/reports/summary.json',
    'lshape_report': '2026-09-17-lshape/experiments/lshape/reports/2026-09-17-lshape.md',
    'lshape_verify': '2026-09-17-lshape/experiments/lshape/checks/verify_report_2026-09-17.json',
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

# Attempts that produced no reported number.  Recorded so a retracted attempt appears as one
# rather than vanishing; job ids and causes are the lane's, relayed by the campaign coordinator.
RETRACTED_ATTEMPTS = [
    ('b-panel', 'bpn201', '3783817', '$1024^2$ panel', 'config-parsing bug; no timed number produced'),
    ('b-panel', 'bpn202', '3787247', '$1024^2$ panel', 'out of memory during autotuning inside an untimed diagnostic'),
    ('lshape', 'lsh01', '3780148', 'bank and head training', 'mis-scaled basis-orthonormality gate; rerun in full as lsh02'),
    ('p-linear', 'plin1024', '3780691', 'Poisson $1024^2$ ladder', 'GPU out of memory in the untimed best-found oracle; rerun on an H200'),
    ('w-ladder', 'wl256b', '3780448', 'wave $256^2$ ladder', 'retained-value gate failed on a tie-breaking difference; rerun as wl256c'),
    ('no-second', 'pois01', '3780224', 'Poisson U-Net screen', 'stager omitted a config directory; no training ran; rerun as pois02'),
    ('ns2d', 'ns202', '3783797', 'Navier--Stokes $K=32$ head', 'pre-\\S A4 attempt on a rank-capped bank; superseded by ns204'),
]
IN_FLIGHT = [
    ('b-panel', 'bpn203', '3789572', '$1024^2$ same-allocation panel, H200 (landed; Tables~\\ref{tab:tunability-tentwentyfour}, \\ref{tab:panel-all-tentwentyfour})'),
    ('b-panel', 'bpn301', '3789570', '$256^2$ re-run carrying both quadrature rule sets (landed; replaces bpn101 wholesale, which is archived, not withdrawn; Tables~\\ref{tab:tunability}, \\ref{tab:panel-all})'),
    ('b-panel', 'bpn401', '3805065', '$512^2$ panel, same GPU model as $256^2$, brackets the frontier crossover (pre-registered in the lane design before submission)'),
    ('b-seeds', 'sealed', '3804465', 'sealed-cohort evaluation of the three seeds (Table~\\ref{tab:sealed})'),
    ('lshape', 'lsh07', '3789568', 'L-shape solve at $512^2$ (landed; Table~\\ref{tab:lshape-solve})'),
    ('b-eqtop', 'bet301', '3783811', 'draw replication (landed; Table~\\ref{tab:replication})'),
    ('b-lowvisc', 'lvt01', '3804337', 'low-viscosity Burgers; gate passed, mesh under-resolved (F4)'),
]

PROV: dict[str, dict] = {}
MACROS: dict[str, str] = {}
PENDING: list[tuple[str, str]] = []   # (macro or table, lane)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


GIT_PINS: dict[str, tuple[str, str]] = {
    # lane file -> (lane worktree, commit): read the COMMITTED blob, not the working tree.
    # Empty since b-qxm committed its round-2 regeneration (b4e38103, 18:15): the pin at
    # 4b9723e8 that held §5.2 while that lane's working tree was dirty is no longer needed
    # (its regenerated span_q_at_M1088 is bit-identical to the pinned value).
    # b-seeds: its working tree is dirty (the lane is still writing up); read the committed
    # three-seed development summary (e533b48e) and nothing newer.
    'seeds_summary': ('2026-09-17-b-seeds', 'e533b48e'),
    # b-panel closed at 13ddecac (bpn301 = 256^2 with both rule sets, bpn203 = 1024^2)
    'panel_summary': ('2026-09-17-b-panel', '13ddecac'),
    'panel_report': ('2026-09-17-b-panel', '13ddecac'),
    # ns2d closed at 50bf36da (ns204 = K=32 also fails H-ORACLE; oracle budget-exit counts carried)
    'ns2d_summary': ('2026-09-17-ns2d', '50bf36da'),
}


def load(key: str):
    if key in GIT_PINS:
        import subprocess
        lane, rev = GIT_PINS[key]
        rel = SOURCES[key].split('/', 1)[1]
        out = subprocess.run(['git', '-C', str(ROOT / lane), 'show', f'{rev}:{rel}'],
                             capture_output=True, text=True)
        if out.returncode == 0:
            PROV[key] = {'path': f'{lane}:{rel}', 'reachable': True, 'pinned_commit': rev,
                         'sha256': hashlib.sha256(out.stdout.encode()).hexdigest(),
                         'note': 'read from the lane commit, not the working tree'}
            _digits = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
            macro('prov' + ''.join(_digits.get(c, c) for c in ''.join(w.capitalize() for w in key.split('_'))) + 'Pin', rev)
            return json.loads(out.stdout) if rel.endswith('.json') else out.stdout
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
    return r'\gen{' + tex_escape(lane) + '}'


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


# --------------------------------------------------------------------------- quadrature status labels
# Every rule row prints exactly one of: confirmed (n of n re-draws) / single-draw /
# marginal (k of n draws pass) / none.  The k-of-n verdicts come from the b-eqtop draw
# replication (job 3783811), keyed by (q, m); the panel's own strings are mapped onto them.
_EQ_REP: dict | None = None
def eq_replication_status():
    global _EQ_REP
    if _EQ_REP is None:
        _EQ_REP = {}
        e = load('eqtop_summary')
        for r in (e['rows'] if e else []):
            if r.get('table') == 'replication' and r.get('construction_status'):
                _EQ_REP[(int(r['q']), int(r['m']))] = r['construction_status']
    return _EQ_REP

def norm_status(raw, q=None, m=None, rule_set=None):
    raw = (raw or '').strip()
    mm = re.match(r'confirmed \((\d+)/(\d+)\)', raw)
    if mm:
        return f'confirmed ({mm.group(1)} of {mm.group(2)} re-draws)'
    if raw.startswith('certified in one draw') or raw == 'one draw':
        return 'single-draw'
    src = raw
    if (not raw) or raw.startswith('marginal (b-eqtop') or raw == 'None':
        src = eq_replication_status().get((int(q), int(m)), '') if (q is not None and m is not None) else ''
    mm = re.search(r'\((\d+)/(\d+)\)', src)
    if mm:
        return f'marginal ({mm.group(1)} of {mm.group(2)} draws pass)'
    if raw.startswith('marginal') or raw.startswith('not certified'):
        return 'marginal (count unavailable)'
    return 'none' if rule_set in (None, '', 'dense') else 'none'


# =========================================================================== T3 / T5 panel
def build_panel():
    summ = load('panel_summary'); rep = load('panel_report')
    if summ is None:
        write('T03_tunability.tex', gen('b-panel', 'T3')); write('T05_panel_all.tex', gen('b-panel', 'T5'))
        return
    rows = summ['rows']
    bymesh = defaultdict(list)
    for r in rows:
        bymesh[int(r['mesh'])].append(r)
    gpu_of = dict(re.findall(r"job `(\d+)` on `(NVIDIA [^`]+)`", rep or ''))
    commits = list(dict.fromkeys(re.findall(r"Source commit `([0-9a-f]+)`", rep or '')))
    macro('provPanelCommit', ' / '.join(hexprefix(c) for c in commits) if commits else '---')
    macro('provPanelCkpt', 'incumbent (gate checkpoint\\_unchanged; hash in T2, tuning row)')
    macro('provPanelJobs', ', '.join(sorted({r['job_id'] for r in rows})))
    SFX = {256: '', 1024: 'TenTwentyFour'}
    RST256 = {}
    Ms = {0: 64, 16: 128, 32: 192, 64: 320, 128: 576, 256: 1088}
    qs = [0, 16, 32, 64, 128, 256]
    REDUCED = ('rom', 'fast', 'pod', 'free')
    for mesh in sorted(bymesh):
        R = bymesh[mesh]; P = pivot(R); job = R[0]['job_id']; sfx = SFX[mesh]
        fam = {r['subject']: r['family'] for r in R}
        qk = {r['subject']: r.get('q_or_k') for r in R}
        Mof = {r['subject']: r.get('M') for r in R}
        quad = {r['subject']: r.get('quadrature') for r in R}
        rset = {r['subject']: r.get('rule_set') for r in R}
        rstat = {r['subject']: r.get('rule_status') for r in R}
        rm = {r['subject']: r.get('rule_m') for r in R}
        adm = {r['subject']: r.get('admissible') for r in R}
        rst = {sub: norm_status(rstat.get(sub), qk.get(sub), rm.get(sub), rset.get(sub)) for sub in rset}
        # transferred rules (eqxfer, 1024^2) carry their 256^2 source rule's verdict; the 256^2 pass runs first
        if mesh == 256:
            RST256 = {int(qk[x]): rst[x] for x in rst if rset.get(x) == 'eqcert' and x.endswith('_g1em06') and qk.get(x) is not None}
        else:
            for x in list(rst):
                if rset.get(x) == 'eqxfer' and rst[x] in ('none', 'marginal (count unavailable)') and qk.get(x) is not None:
                    rst[x] = RST256.get(int(qk[x]), rst[x])
        macro(f'provPanel{sfx}Job', job); macro(f'provPanel{sfx}Gpu', gpu_of.get(job, '---'))
        macro(f'nPanel{sfx}Mesh', str(mesh))
        sets = ['eqcert', 'eqtop'] if mesh == 256 else ['eqxfer']
        SETNAME = {'eqcert': 'EQ, b-eqtop ladder rules', 'eqtop': 'EQ, replication-selected rules', 'eqxfer': 'EQ, rules transferred from $256^2$'}
        # ---- T3: the ladders side by side at tolerance 1e-6 (+ the loose tolerance for the last set)
        t3 = []
        for q in qs:
            cells = [str(q), str(Ms[q])]
            d = P[f'q{q}_M{Ms[q]}_dense_g1em06']
            cells += [pct(d['worst_evolved_percent']), pct(d['worst_all_times_percent']), ms(d['median_gpu_ms'])]
            for st_ in sets:
                e = P[f'q{q}_M{Ms[q]}_{st_}_g1em06']
                cells += [pct(e['worst_evolved_percent']), pct(e['worst_all_times_percent']), ms(e['median_gpu_ms']),
                          tex_escape(rst.get(f'q{q}_M{Ms[q]}_{st_}_g1em06', 'none'))]
            cells.append(pct(d['worst_t0_compression_percent']))
            t3.append(cells)
        cols = ['$q$', '$M$', 'dense: evolved \\%', 'all \\%', 'ms']
        for st_ in sets:
            cols += [SETNAME[st_] + ': evolved \\%', 'all \\%', 'ms', 'rule status']
        cols.append('$t{=}0$ \\%')
        write(f'T03{"b" if sfx else ""}_tunability.tex', tabular(cols, t3, 'rr' + 'rrr' + 'rrrl' * len(sets) + 'r', r'\tiny'), f'b-panel job {job}, {mesh}^2, tolerance 1e-6')
        # ---- ladder statistics
        def ladder_stats(suf, ql):
            errs = [P[f'q{q}_M{Ms[q]}_{suf}']['worst_evolved_percent'] for q in ql]
            cost = [P[f'q{q}_M{Ms[q]}_{suf}']['median_gpu_ms'] for q in ql]
            mono = all(errs[i] >= errs[i + 1] for i in range(len(errs) - 1))
            conv = all(P[f'q{q}_M{Ms[q]}_{suf}']['converged'] for q in ql)
            return errs, cost, mono, conv
        e, c, mono, conv = ladder_stats('dense_g1em06', qs)
        macro(f'nPanel{sfx}DenseErrSpan', f'{e[0]/e[-1]:.2f}'); macro(f'nPanel{sfx}DenseCostSpan', f'{c[-1]/c[0]:.2f}')
        macro(f'nPanel{sfx}DenseMonotone', yn(mono)); macro(f'nPanel{sfx}DenseConverged', yn(conv))
        SN = {'eqcert': 'Eq', 'eqtop': 'Eqtop', 'eqxfer': 'Eqxfer'}
        for st_ in sets:
            for tl, tn in [('g1em06', ''), ('g0p001', 'Loose')]:
                _, _, m_, cv_ = ladder_stats(f'{st_}_{tl}', qs)
                macro(f'nPanel{sfx}{SN[st_]}{tn}Monotone', yn(m_)); macro(f'nPanel{sfx}{SN[st_]}{tn}Converged', yn(cv_))
            for q, qn in [(64, 'SixtyFour'), (128, 'OneTwentyEight'), (256, 'TwoFiftySix')]:
                sub = f'q{q}_M{Ms[q]}_{st_}_g1em06'
                macro(f'nPanel{sfx}{SN[st_]}Q{qn}Err', pct(P[sub]['worst_evolved_percent'])); macro(f'nPanel{sfx}{SN[st_]}Q{qn}Ms', ms(P[sub]['median_gpu_ms']))
                macro(f'nPanel{sfx}{SN[st_]}Q{qn}All', pct(P[sub]['worst_all_times_percent']))
                macro(f'nPanel{sfx}{SN[st_]}Q{qn}Status', tex_escape(rst.get(sub, 'none'))); macro(f'nPanel{sfx}{SN[st_]}Q{qn}M', str(rm.get(sub) or '---'))
                macro(f'nPanel{sfx}{SN[st_]}Q{qn}Rho', f"{P[sub]['rho_max']:.4f}" if P[sub].get('rho_max') is not None else '---')
                macro(f'nPanel{sfx}{SN[st_]}Q{qn}FitStates', str(P[sub].get('rule_fit_states') or '---'))
                macro(f'nPanel{sfx}{SN[st_]}Q{qn}Basis', str(P[sub].get('rule_basis') or '---'))
            # dense-twin speedups over the ladder at tolerance 1e-6
            sp = [P[f'q{q}_M{Ms[q]}_dense_g1em06']['median_gpu_ms'] / P[f'q{q}_M{Ms[q]}_{st_}_g1em06']['median_gpu_ms'] for q in qs]
            macro(f'nPanel{sfx}{SN[st_]}TwinSpeedupMin', f'{min(sp):.1f}'); macro(f'nPanel{sfx}{SN[st_]}TwinSpeedupMax', f'{max(sp):.1f}')
            macro(f'nPanel{sfx}{SN[st_]}PassingQs', ', '.join(str(q) for q in qs if P[f'q{q}_M{Ms[q]}_{st_}_g1em06'].get('rule_basis') == 'primary'))
            sd = [str(q) for q in qs if rst.get(f'q{q}_M{Ms[q]}_{st_}_g1em06') == 'single-draw']
            macro(f'nPanel{sfx}{SN[st_]}SingleDrawQs', ', '.join(sd) if sd else 'none'); macro(f'nPanel{sfx}{SN[st_]}SingleDrawCount', str(len(sd)))
        if mesh == 256:
            macro('nPanelEqQtwofiftysixEvolved', pct(P['q256_M1088_eqcert_g1em06']['worst_evolved_percent']))
            macro('nPanelEqQonetwentyeightEvolved', pct(P['q128_M576_eqcert_g1em06']['worst_evolved_percent']))
            macro('nPanelEqQtwofiftysixBasis', str(P['q256_M1088_eqcert_g1em06'].get('rule_basis')))
            macro('nPanelEqQonetwentyeightBasis', str(P['q128_M576_eqcert_g1em06'].get('rule_basis')))
            macro('nPanelEqQsixtyfourBasis', str(P['q64_M320_eqcert_g1em06'].get('rule_basis')))
            # same rule files at q <= 32 in both sets?
            same = all(P[f'q{q}_M{Ms[q]}_eqcert_g1em06'].get('rule_file_sha256') == P[f'q{q}_M{Ms[q]}_eqtop_g1em06'].get('rule_file_sha256') for q in (0, 16, 32))
            macro('nPanelEqSetsSameRulesLowQ', yn(same))
            sp = [P[f'q{q}_M{Ms[q]}_dense_g1em06']['median_gpu_ms'] / P[f'q{q}_M{Ms[q]}_{st_}_g1em06']['median_gpu_ms'] for st_ in sets for q in qs]
            macro('nPanelTwinSpeedupMin', f'{min(sp):.1f}'); macro('nPanelTwinSpeedupMax', f'{max(sp):.1f}')
            macro('nPanelDenseQtwoFiftySixAll', pct(P['q256_M1088_dense_g1em06']['worst_all_times_percent']))
            macro('nPanelEqtopQtwoFiftySixTimeSaving', f"{P['q256_M1088_dense_g1em06']['median_gpu_ms'] / P['q256_M1088_eqtop_g1em06']['median_gpu_ms']:.1f}")
        # EQ vs dense at matched (q, M): cost and error (the first EQ set of the mesh)
        st0 = sets[0]
        for q, nm in [(0, 'Zero'), (128, 'OneTwoEight')]:
            d = P[f'q{q}_M{Ms[q]}_dense_g1em06']; eq6 = P[f'q{q}_M{Ms[q]}_{st0}_g1em06']; eq3 = P[f'q{q}_M{Ms[q]}_{st0}_g0p001']
            macro(f'nPanel{sfx}DenseMs{nm}', ms(d['median_gpu_ms'])); macro(f'nPanel{sfx}EqMs{nm}', ms(eq6['median_gpu_ms']))
            macro(f'nPanel{sfx}EqLooseMs{nm}', ms(eq3['median_gpu_ms'])); macro(f'nPanel{sfx}DenseErr{nm}', pct(d['worst_evolved_percent']))
            macro(f'nPanel{sfx}EqErr{nm}', pct(eq6['worst_evolved_percent'])); macro(f'nPanel{sfx}EqLooseErr{nm}', pct(eq3['worst_evolved_percent']))
            macro(f'nPanel{sfx}EqSpeedup{nm}', f'{d["median_gpu_ms"]/eq6["median_gpu_ms"]:.2f}')
            macro(f'nPanel{sfx}TolSaving{nm}', f'{100*(1-eq3["median_gpu_ms"]/eq6["median_gpu_ms"]):.0f}')
        # ---- frontier statements
        subjects = [x for x in P if x != '*']
        reduced = [x for x in subjects if fam[x] in REDUCED]
        adm_red = [x for x in reduced if adm.get(x)]
        foms = [x for x in subjects if fam[x] == 'fom']
        macro(f'nPanel{sfx}SubjectCount', str(len(subjects))); macro(f'nPanel{sfx}ReducedAllCount', str(len(reduced)))
        macro(f'nPanel{sfx}ReducedCount', str(len(adm_red)))
        nd = P.get('*', {})
        nd_e = nd.get('nondominated_gpu_evolved_admissible') or []; nd_a = nd.get('nondominated_gpu_all_admissible') or []
        macro(f'nPanel{sfx}ReducedNonDomEvolved', str(sum(1 for x in nd_e if x in reduced)))
        macro(f'nPanel{sfx}ReducedNonDomAll', str(sum(1 for x in nd_a if x in reduced)))
        macro(f'nPanel{sfx}NonDomReducedList', ', '.join(tt(x) for x in nd_e if x in reduced) or 'none')
        macro(f'nPanel{sfx}NonDomFomList', ', '.join(tt(x) for x in nd_e if x not in reduced) or 'none')
        ndr = nd.get('nondominated_gpu_all_reduced_only') or []
        macro(f'nPanel{sfx}ReducedOnlyFrontierSize', str(len(ndr)))
        macro(f'nPanel{sfx}ReducedOnlyFrontierEq', str(sum(1 for x in ndr if fam[x] in ('rom', 'fast'))))
        macro(f'nPanel{sfx}ReducedOnlyFrontierPod', str(sum(1 for x in ndr if fam[x] == 'pod')))
        ndr_e = nd.get('nondominated_gpu_evolved_reduced_only') or []
        macro(f'nPanel{sfx}ReducedOnlyFrontierEvolvedPod', str(sum(1 for x in ndr_e if fam[x] == 'pod')))
        # cheapest admissible reduced against the cheapest full-order setting of the same job, and against fft_tight
        cheap_r = min(adm_red, key=lambda x: P[x]['median_gpu_ms']); cheap_f = min(foms, key=lambda x: P[x]['median_gpu_ms'])
        macro(f'nPanel{sfx}CheapestReduced', tt(cheap_r)); macro(f'nPanel{sfx}CheapestReducedMs', ms(P[cheap_r]['median_gpu_ms'])); macro(f'nPanel{sfx}CheapestReducedErr', pct(P[cheap_r]['worst_evolved_percent']))
        macro(f'nPanel{sfx}CheapestFom', tt(cheap_f)); macro(f'nPanel{sfx}CheapestFomMs', ms(P[cheap_f]['median_gpu_ms'])); macro(f'nPanel{sfx}CheapestFomErr', pct(P[cheap_f]['worst_evolved_percent']))
        macro(f'nPanel{sfx}CheapestRatio', f"{P[cheap_r]['median_gpu_ms'] / P[cheap_f]['median_gpu_ms']:.2f}")
        macro(f'nPanel{sfx}CheapestOverFft', f"{P[cheap_r]['median_gpu_ms'] / P['fft_tight']['median_gpu_ms']:.3f}")
        # the most accurate reduced subject and the same-job FOM settings that beat it on both axes
        acc_r = min(adm_red, key=lambda x: P[x]['worst_evolved_percent'])
        beat = [x for x in foms if P[x]['median_gpu_ms'] <= P[acc_r]['median_gpu_ms'] and P[x]['worst_evolved_percent'] <= P[acc_r]['worst_evolved_percent']]
        macro(f'nPanel{sfx}MostAccurateReduced', tt(acc_r)); macro(f'nPanel{sfx}MostAccurateReducedErr', pct(P[acc_r]['worst_evolved_percent'])); macro(f'nPanel{sfx}MostAccurateReducedMs', ms(P[acc_r]['median_gpu_ms']))
        macro(f'nPanel{sfx}FomBeatingMostAccurate', str(len(beat)))
        # t = 0 compression on the frontier's reduced members (bounds the all-times metric)
        tz = [P[x]['worst_t0_compression_percent'] for x in nd_e if x in reduced and P[x].get('worst_t0_compression_percent') is not None]
        if tz: macro(f'nPanel{sfx}FrontierTzeroMin', pct(min(tz), 2)); macro(f'nPanel{sfx}FrontierTzeroMax', pct(max(tz), 2))
        for x, nm in [('nt1e-3_dt005', 'NtThreeFine'), ('nt1e-4_dt005', 'NtFourFine'), ('nt1e-2_dt01', 'NtTwoCoarse'), ('fft_tight', 'FftTight'), ('dense_tight', 'DenseTight')]:
            if x in P:
                macro(f'nPanel{sfx}Fom{nm}Err', pct(P[x]['worst_evolved_percent'])); macro(f'nPanel{sfx}Fom{nm}Ms', ms(P[x]['median_gpu_ms']))
                macro(f'nPanel{sfx}Fom{nm}Ref', pct(P[x]['worst_reference_percent']))
        best = P['q256_M1088_dense_g1em06']
        macro(f'nPanel{sfx}BestRungErr', pct(best['worst_evolved_percent'])); macro(f'nPanel{sfx}BestRungMs', ms(best['median_gpu_ms'])); macro(f'nPanel{sfx}BestRungAll', pct(best['worst_all_times_percent']))
        # error against the fine reference: what the knob does to the physical error (review r2, R2)
        q0d = P['q0_M64_dense_g1em06']
        macro(f'nPanel{sfx}QzeroRef', pct(q0d['worst_reference_percent'], 2)); macro(f'nPanel{sfx}BestRungRef', pct(best['worst_reference_percent'], 2))
        macro(f'nPanel{sfx}RefSpan', f"{q0d['worst_reference_percent'] / best['worst_reference_percent']:.2f}")
        macro(f'nPanel{sfx}RefDiscretisationTwo', pct(P['fft_tight']['worst_reference_percent'], 2))
        dense_rungs = [f'q{q}_M{Ms[q]}_dense_g1em06' for q in qs if f'q{q}_M{Ms[q]}_dense_g1em06' in P]
        macro(f'nPanel{sfx}RungsBelowDiscretisation', str(sum(1 for x in dense_rungs if P[x]['worst_reference_percent'] <= P['fft_tight']['worst_reference_percent'])))
        macro(f'nPanel{sfx}RungCount', str(len(dense_rungs)))
        rr = [P[x]['worst_reference_percent'] / P['fft_tight']['worst_reference_percent'] for x in dense_rungs]
        macro(f'nPanel{sfx}RungRefOverDiscMin', f"{min(rr):.2f}"); macro(f'nPanel{sfx}RungRefOverDiscMax', f"{max(rr):.2f}")
        bref = min(foms, key=lambda x: P[x]['worst_reference_percent'])
        macro(f'nPanel{sfx}FomBestRefArm', tt(bref)); macro(f'nPanel{sfx}FomBestRef', pct(P[bref]['worst_reference_percent'], 2)); macro(f'nPanel{sfx}FomBestRefMs', ms(P[bref]['median_gpu_ms']))
        macro(f'nPanel{sfx}FomBestRefEvolved', pct(P[bref]['worst_evolved_percent']))
        macro(f'nPanel{sfx}FomBestRefCostShare', f"{best['median_gpu_ms'] / P[bref]['median_gpu_ms']:.0f}")
        # compact main-text ladder: dense and the mesh's first EQ set at tolerance 1e-6, with the vs-reference column
        tm = []
        for q in qs:
            d = P[f'q{q}_M{Ms[q]}_dense_g1em06']; e = P.get(f'q{q}_M{Ms[q]}_{sets[-1]}_g1em06', {})
            tm.append([str(q), str(Ms[q]), pct(d['worst_evolved_percent']), pct(d['worst_all_times_percent']), pct(d['worst_reference_percent'], 2), ms(d['median_gpu_ms']),
                       pct(e.get('worst_evolved_percent')) if e else '---', ms(e.get('median_gpu_ms')) if e else '---',
                       tex_escape(rst.get(f'q{q}_M{Ms[q]}_{sets[-1]}_g1em06', 'none'))])
        write(f'T03m{"b" if sfx else ""}_ladder_main.tex', tabular(['$q$', '$M$', 'dense evolved \\%', 'all \\%', 'vs ref \\%', 'GPU ms', 'EQ evolved \\%', 'EQ ms', 'EQ rule'], tm, 'rrrrrrrrl', r'\scriptsize'),
              f'b-panel job {job}, {mesh}^2, tolerance 1e-6, scheduled M=4(K+q); EQ set = {sets[-1]}')
        if 'pod512_M2048_dense' in P:
            pod = P['pod512_M2048_dense']
            macro(f'nPanel{sfx}PodFiveTwelveErr', pct(pod['worst_evolved_percent'])); macro(f'nPanel{sfx}PodFiveTwelveMs', ms(pod['median_gpu_ms'])); macro(f'nPanel{sfx}PodFiveTwelveAll', pct(pod['worst_all_times_percent']))
        free = P['free512_M1024_dense']
        macro(f'nPanel{sfx}FreeErr', pct(free['worst_evolved_percent'])); macro(f'nPanel{sfx}FreeMs', ms(free['median_gpu_ms'])); macro(f'nPanel{sfx}FreeAll', pct(free['worst_all_times_percent']))
        macro(f'nPanel{sfx}FreeFloor', pct(free['best_found_percent'])); macro(f'nPanel{sfx}FreeOverFloor', f'{free["solved_over_best_found"]:.2f}')
        macro(f'nPanel{sfx}PodRanks', ', '.join(str(qk[x]) for x in sorted((x for x in subjects if fam[x] == 'pod'), key=lambda x: int(qk[x]))))
        macro(f'nPanel{sfx}PodOneTwentyEightErr', pct(P['pod128_M512_dense']['worst_evolved_percent'])); macro(f'nPanel{sfx}PodOneTwentyEightAll', pct(P['pod128_M512_dense']['worst_all_times_percent']))
        macro(f'nPanel{sfx}PodOneTwentyEightMs', ms(P['pod128_M512_dense']['median_gpu_ms']))
        macro(f'nPanel{sfx}PodTwoFiftySixErr', pct(P['pod256_M1024_dense']['worst_evolved_percent'])); macro(f'nPanel{sfx}PodTwoFiftySixMs', ms(P['pod256_M1024_dense']['median_gpu_ms']))
        fno = P['fno-large']
        macro(f'nPanel{sfx}FnoErr', pct(fno['worst_evolved_percent'])); macro(f'nPanel{sfx}FnoMs', ms(fno['median_gpu_ms'])); macro(f'nPanel{sfx}FnoRef', pct(fno['worst_reference_percent']))
        macro(f'nPanel{sfx}QzeroT', pct(P['q0_M64_dense_g1em06']['worst_t0_compression_percent'])); macro(f'nPanel{sfx}QzeroEvolved', pct(P['q0_M64_dense_g1em06']['worst_evolved_percent']))
        macro(f'nPanel{sfx}QzeroAll', pct(P['q0_M64_dense_g1em06']['worst_all_times_percent']))
        if 'q0_M64_eqcert_g1em06_fastL4' in P: macro(f'nPanel{sfx}FastMs', ms(P['q0_M64_eqcert_g1em06_fastL4']['median_gpu_ms']))
        macro(f'nPanel{sfx}RefDiscretisation', pct(P['fft_tight']['worst_reference_percent']))
        strict_no = [x for x in reduced if P[x].get('converged') and not P[x].get('converged_strict')]
        macro(f'nPanel{sfx}StrictNoCount', str(len(strict_no)))
        icres = [P[x].get('max_ic_relative_residual') for x in strict_no if P[x].get('max_ic_relative_residual') is not None]
        macro(f'nPanel{sfx}StrictNoIcResidualMax', f'{max(icres):.1e}' if icres else '---')
        sob = [P[x].get('solved_over_best_found') for x in strict_no if fam[x] == 'pod']
        macro(f'nPanel{sfx}PodSolvedOverFloorMax', f'{max(sob):.5f}' if sob else '---')
        # ---- T5: every subject
        t5 = []
        for x in subjects:
            d = P[x]; isred = fam[x] in REDUCED
            rule = (f"{rset[x]} $m{{=}}{rm[x]}$, {rst.get(x, 'none')}" if rset.get(x) else ('none' if isred else '---'))
            t5.append([tt(x), fam[x], str(qk[x]) if qk[x] is not None else '---', str(Mof[x]) if Mof[x] else '---', str(quad[x] or '---'),
                       tex_escape(rule), pct(d.get('worst_evolved_percent')), pct(d.get('worst_all_times_percent')),
                       pct(d.get('worst_t0_compression_percent')), pct(d.get('worst_reference_percent')),
                       ms(d.get('median_gpu_ms')), ms(d.get('median_host_ms')),
                       yn(d.get('converged')) if isred else '---', yn(d.get('converged_strict')) if isred else '---', yn(adm.get(x)) if isred else '---'])
        cols = ['subject', 'family', '$q$ / $k^\\prime$', '$M$', 'quad.', 'rule', 'evolved \\%', 'all \\%', '$t{=}0$ \\%', 'vs ref \\%', 'GPU ms', 'host ms', 'conv.', 'strict', 'adm.']
        write(f'T05{"b" if sfx else ""}_panel_all.tex', tabular(cols, t5, 'lllllp{3.2cm}rrrrrrccc', r'\tiny'), f'b-panel job {job}, {mesh}^2')
    # the crossover ratios, both meshes, same-job only
    macro('nPanelCheapestRatioDrop', f"{float(MACROS['nPanelCheapestRatio']) / float(MACROS['nPanelTenTwentyFourCheapestRatio']):.2f}")


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
    macro('provQxmCommit', ' / '.join(dict.fromkeys(tex_escape(r[4]) for r in rows)) if rows else '---')
    macro('provQxmWithinJob', f"{wj['job']} = {wj['job_id']}")
    macro('nQxmFixedM', str(ev['fixed_M']['1088']['M']))
    macro('nQxmErrSpan', f"{wj['error_span']:.2f}")
    macro('nQxmCostSpan', f"{wj['cost_span']:.2f}")
    macro('nQxmNonDom', str(wj['non_dominated_points']))
    macro('nQxmPasses', yn(wj['passes_tunability_bar']))
    macro('nQxmMonotone', yn(wj['monotone_error']))
    fm = ev['fixed_M']['1088']
    # the declared column also attempted q = 512 (round 2), which did not converge; the lane keeps
    # `span` null for the declared range and reports the converged prefix as `certified_span`
    macro('nQxmFixedMConverged', yn(fm['certified_all_converged']))
    macro('nQxmCertifiedQtop', str(fm['certified_q_range'][1]))
    macro('nQxmCertifiedSpan', f"{fm['certified_span']:.2f}")
    la = fm.get('within_job_longest_attempted')
    if la:
        macro('nQxmExtJob', la['job_id']); macro('nQxmExtQtop', str(la['q'][-1]))
        macro('nQxmExtTopConverged', yn(la['converged'][-1]))
        macro('nQxmExtRoundOneRungsReproduced', str(sum(1 for q in la['q'] if q in wj['q'])))
    # round-2 anchors: every shared cell re-run in E1/E2 against its round-1 twin
    macro('nQxmAnchorWorstAll', f"{max(max(x['rel_evolved'], x['rel_all']) for x in a['anchors']):.1e}")
    macro('nQxmAnchorsAllPass', yn(all(x['passed'] for x in a['anchors'])))
    macro('nQxmJobCount', str(len({x for x in [y['a_job'] for y in a['anchors']] + [y['b_job'] for y in a['anchors']]})))
    r2 = [x for x in a['anchors'] if x['b_job'] in ('3783898', '3783899')]
    if r2:
        macro('nQxmRoundTwoAnchorWorst', f"{max(max(x['rel_evolved'], x['rel_all']) for x in r2):.1e}")
        macro('nQxmRoundTwoAnchorCount', str(len(r2)))
    macro('nQxmSpanFixedTwoFiftySix', f"{ev['fixed_M']['256']['span']:.2f}")
    macro('nQxmSpanScheduledFour', f"{ev['scheduled']['4x']['span']:.2f}")
    macro('nQxmSpanScheduledEight', f"{ev['scheduled']['8x']['span']:.2f}")
    # all-times fixed-M column: the declared range includes the unconverged q = 512 attempt, so the
    # lane nulls `span`; the converged prefix (q <= 256) is `certified_span`
    macro('nQxmAllTimesSpanFixed', f"{al['fixed_M']['1088']['certified_span']:.2f}")
    macro('nQxmAllTimesCertifiedQtop', str(al['fixed_M']['1088']['certified_q_range'][1]))
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
    s256 = sat['256']
    macro('nQxmMstarTwoFiftySix', str(s256['M_star'])); macro('nQxmMstarTwoFiftySixTpu', f"{s256['tests_per_unknown_at_M_star']:.0f}")
    macro('nQxmMstarTwoFiftySixErr', pct(s256['error_at_M_star'])); macro('nQxmMstarTwoFiftySixCost', f"{s256['cost_ratio_at_M_star']:.2f}")
    macro('nQxmQtwoFiftySixMlargest', str(s256['largest_M'])); macro('nQxmQtwoFiftySixErrLargest', pct(s256['value_at_largest_M']))
    c256 = {c['M']: c for c in s256['curve']}
    macro('nQxmQtwoFiftySixCostLargest', f"{c256[s256['largest_M']]['cost_ratio_to_4x']:.2f}")
    macro('nQxmQtwoFiftySixMaxGainPastMstar', f"{100*max(c['improvement_to_next'] for c in s256['curve'] if c['M'] >= s256['M_star'] and c['improvement_to_next'] is not None):.1f}")
    macro('provQxmSaturationTwoFiftySixJob', a['saturation']['sources']['256']['job_id'])
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
    # compact main-text block: the fixed-M = 1088 ladder inside one job (no vs-reference column exists in this lane)
    write('T04m_fixedM_main.tex', tabular(['$q$', '$M$', 'worst evolved \\%', 'GPU ms', 'converged'],
                                          [[str(q), '1088', pct(v), ms(c), yn(cv)] for q, v, c, cv in zip(wj['q'], wj['values'], wj['median_gpu_ms'], wj['converged'])], 'rrrrc', r'\scriptsize'),
          'b-qxm fixed-M ladder inside job ' + wj['job_id'] + ' (same-grid evolved error; this job carries no fine-reference column)')
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
        st = norm_status(v.get('ladder_rule_status', 'one draw'), q, m['m'])
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
                  f"{d['spread_ratio']:.2f}", f"{round(4*d['certified_primary_fraction']):.0f}/4", f"{round(4*d['certified_tight_fraction']):.0f}/4", tex_escape(norm_status(d['_status'], d['_q'], d['_m']))])
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
    s_rows = s['rows'] if isinstance(s, dict) else s
    by = defaultdict(dict)
    for r in s_rows:
        if r['family'] in ('bank', 'head'):
            by[(r['mesh'], r['subject'])][r['metric']] = r['value']
    job = s_rows[0]['job_id']
    m = re.search(r'Training job `(\d+)` on `(NVIDIA [^`]+)`, source commit `([0-9a-f]+)`', rep or '')
    macro('provLshapeJob', job); macro('provLshapeGpu', m.group(2) if m else '---'); macro('provLshapeCommit', hexprefix(m.group(3)) if m else '---')
    banks = ['smooth_R256', 'smooth_R512', 'sdf_R256', 'sdf_R512', 'enrich_R512', 'smooth_ff128s2_R512']
    t = []
    for b in banks:
        t.append([tt(b), pct(100 * by[(256, b)]['bank_floor_dev_worst']), pct(100 * by[(512, b)]['bank_floor_dev_worst']),
                  pct(100 * by[(256, b)]['bank_floor_common_worst'])])
    write('T18a_lshape_bank.tex', tabular(['bank', 'floor, dev.\\ $256^2$ \\%', 'floor, dev.\\ $512^2$ \\%', 'floor, common $256^2$ \\%'], t, 'lrrr', r'\scriptsize'),
          f'lshape job {job}')
    heads = [k[1] for k in by if k[0] == 256 and k[1].startswith('head_') and 'best_found_dev_worst' in by[k]]
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
    # ---- solve layer: the non-dominated sets per mesh (M = 257) and the free rung (M = 1024)
    V = defaultdict(dict); fam = {}; jobs = {}
    for r in s_rows:
        if r.get('test_modes') and r['metric'] in ('worst_same_grid', 'median_total_ms', 'nondominated_complete_ms',
                                                   'nondominated_reduced_only', 'median_iterations'):
            V[(r['mesh'], r['test_modes'], r['subject'])][r['metric']] = r['value']
            fam[r['subject']] = r['family']; jobs[(r['mesh'], r['test_modes'])] = r['job_id']
    lab = {'fom_splu': 'sparse direct (SuperLU)', 'fom_cg_gpu_r0.0001': 'CG $10^{-4}$', 'fom_cg_gpu_r0.01': 'CG $10^{-2}$',
           'fom_cg_gpu_r1e-10': 'CG $10^{-10}$', 'fom_pcg_ic0_cpu_r0.01': 'IC(0)-PCG $10^{-2}$', 'fom_pcg_ic0_cpu_r1e-10': 'IC(0)-PCG $10^{-10}$'}
    def name(sub):
        if sub in lab: return lab[sub]
        if sub.startswith('pod'): return "POD $k'{=}" + sub[3:] + "$"
        if sub.startswith('freebank@'): return 'free rung $q{=}R$ (' + tex_escape(sub.split('@')[1]) + ')'
        if sub.startswith('neural_q'):
            q, h = sub[8:].split('@'); return f'head $q{{=}}{q}$ (' + tex_escape(h) + ')'
        return tt(sub)
    t = []
    for mesh in (64, 128, 256, 512):
        nd = [sub for (m, tm, sub) in V if m == mesh and tm == 257 and V[(m, tm, sub)].get('nondominated_complete_ms')]
        nd = sorted(nd, key=lambda sub: V[(mesh, 257, sub)]['median_total_ms'])
        for sub in nd:
            d = V[(mesh, 257, sub)]
            t.append([f'${mesh}^2$', name(sub), fam[sub], pct(100 * d['worst_same_grid']), ms(d['median_total_ms'], 3)])
        t.append('MIDRULE')
    t.pop()
    write('T18c_lshape_solve.tex', tabular(['mesh', 'subject', 'family', 'worst same-grid \\%', 'complete-query ms'], t, 'lllrr', r'\scriptsize'),
          'lshape solve layer, M = 257; the non-dominated set per mesh, one job per mesh')
    macro('provLshapeSolveJobs', ', '.join(sorted({jobs[k] for k in jobs if k[1] == 257})))
    NM = [(64, 'SixtyFour'), (128, 'OneTwentyEight'), (256, 'TwoFiftySix'), (512, 'FiveTwelve')]
    trend = []
    for mesh, nm in NM:
        d = V.get((mesh, 257, 'fom_splu'), {})
        if d: macro(f'nLshapeSpluMs{nm}', ms(d['median_total_ms'], 2))
        b = V.get((mesh, 257, 'neural_q64@head_sdf_R512_K16'), {})
        if b:
            macro(f'nLshapeNeuralMs{nm}', ms(b['median_total_ms'], 3)); trend.append(ms(b['median_total_ms'], 2))
            macro(f'nLshapeNeuralErr{nm}', pct(100 * b['worst_same_grid'], 3))
            if d: macro(f'nLshapeNeuralCheaper{nm}', f"{d['median_total_ms'] / b['median_total_ms']:.2f}")
        ndm = [sub for (m, tm, sub) in V if m == mesh and tm == 257 and V[(m, tm, sub)].get('nondominated_complete_ms')]
        macro(f'nLshapeNonDomCount{nm}', str(len(ndm))); macro(f'nLshapeNonDomReduced{nm}', str(sum(1 for sub in ndm if fam[sub] != 'fom')))
        # is the head at q=64 on the non-dominated set at this mesh?
        macro(f'nLshapeNeuralNonDom{nm}', yn(bool(V.get((mesh, 257, 'neural_q64@head_sdf_R512_K16'), {}).get('nondominated_complete_ms'))))
    macro('nLshapeNeuralMsTrend', ' $\\to$ '.join(trend))
    # review r2 (B4/M2): the cheapest same-job full-order arm, POD-128, and the head, per mesh, dominated or not
    HEAD = 'neural_q64@head_sdf_R512_K16'; POD = 'pod128'
    tm = []; margins_cheapest = {}
    for mesh, nm in NM:
        foms = [sub for (m, tm_, sub) in V if m == mesh and tm_ == 257 and fam[sub] == 'fom']
        if not foms: continue
        cf = min(foms, key=lambda sub: V[(mesh, 257, sub)]['median_total_ms']); c = V[(mesh, 257, cf)]
        sp = V[(mesh, 257, 'fom_splu')]; hd = V.get((mesh, 257, HEAD), {}); pd = V.get((mesh, 257, POD), {})
        macro(f'nLshapeCheapestFom{nm}', name(cf)); macro(f'nLshapeCheapestFomMs{nm}', ms(c['median_total_ms'], 2)); macro(f'nLshapeCheapestFomErr{nm}', pct(100 * c['worst_same_grid'], 3))
        if hd:
            macro(f'nLshapeNeuralCheaperVsCheapest{nm}', f"{c['median_total_ms'] / hd['median_total_ms']:.2f}")
            macro(f'nLshapeCheapestFomMoreAccurate{nm}', f"{hd['worst_same_grid'] / c['worst_same_grid']:.1f}" if c['worst_same_grid'] > 0 else 'exact')
        if pd:
            macro(f'nLshapePodErr{nm}', pct(100 * pd['worst_same_grid'], 3)); macro(f'nLshapePodMs{nm}', ms(pd['median_total_ms'], 3))
            macro(f'nLshapePodCheaper{nm}', f"{sp['median_total_ms'] / pd['median_total_ms']:.2f}"); macro(f'nLshapePodCheaperVsCheapest{nm}', f"{c['median_total_ms'] / pd['median_total_ms']:.2f}")
        if hd and pd:
            macro(f'nLshapeHeadOverPodCost{nm}', f"{100 * (hd['median_total_ms'] / pd['median_total_ms'] - 1):.0f}")
            macro(f'nLshapePodOverHeadErr{nm}', f"{100 * (pd['worst_same_grid'] / hd['worst_same_grid'] - 1):.0f}")
        tm.append([f'${mesh}^2$', tt(jobs.get((mesh, 257), '---')), ms(sp['median_total_ms'], 2), name(cf), ms(c['median_total_ms'], 2), pct(100 * c['worst_same_grid'], 3),
                   pct(100 * pd['worst_same_grid'], 3) if pd else '---', ms(pd['median_total_ms'], 3) if pd else '---', f"{sp['median_total_ms'] / pd['median_total_ms']:.2f} / {c['median_total_ms'] / pd['median_total_ms']:.2f}" if pd else '---',
                   pct(100 * hd['worst_same_grid'], 3) if hd else '---', ms(hd['median_total_ms'], 3) if hd else '---', f"{sp['median_total_ms'] / hd['median_total_ms']:.2f} / {c['median_total_ms'] / hd['median_total_ms']:.2f}" if hd else '---',
                   yn(bool(hd.get('nondominated_complete_ms'))) if hd else '---'])
    write('T18m_lshape_main.tex', tabular(['mesh', 'job', 'sparse direct ms', 'cheapest FOM', 'ms', 'err \\%', 'POD-128 err \\%', 'ms', '$\\times$ vs direct / cheapest', 'head $q{=}64$ err \\%', 'ms', '$\\times$ vs direct / cheapest', 'head on set'],
                                         tm, 'llrlrrrrrrrrc', r'\scriptsize'), 'lshape solve layer, M = 257, one job per mesh; complete-query ms; every ratio inside its job; head rows printed even where dominated')
    # the 256^2 crossover cell keeps its short names (used in the intro and §5.6)
    best = V.get((256, 257, 'neural_q64@head_sdf_R512_K16'), {}); splu = V.get((256, 257, 'fom_splu'), {})
    if best and splu:
        macro('nLshapeNeuralErr', pct(100 * best['worst_same_grid'], 3)); macro('nLshapeNeuralMs', ms(best['median_total_ms'], 3))
        macro('nLshapeNeuralCheaper', f"{splu['median_total_ms'] / best['median_total_ms']:.2f}")
    macro('nLshapeNonDomCount', MACROS['nLshapeNonDomCountTwoFiftySix']); macro('nLshapeNonDomReduced', MACROS['nLshapeNonDomReducedTwoFiftySix'])
    # 512^2: the most accurate reduced subject on the set, and the cheapest full-order subject
    nd512 = [sub for (m, tm, sub) in V if m == 512 and tm == 257 and V[(m, tm, sub)].get('nondominated_complete_ms')]
    if nd512:
        red = [sub for sub in nd512 if fam[sub] != 'fom']
        bestred = min(red, key=lambda sub: V[(512, 257, sub)]['worst_same_grid'])
        macro('nLshapeFiveTwelveBestReduced', name(bestred))
        macro('nLshapeFiveTwelveBestReducedErr', pct(100 * V[(512, 257, bestred)]['worst_same_grid'], 3))
        pods = [sub for sub in red if sub.startswith('pod')]
        if pods:
            bp = min(pods, key=lambda sub: V[(512, 257, sub)]['worst_same_grid'])
            macro('nLshapeFiveTwelveBestPod', name(bp)); macro('nLshapeFiveTwelveBestPodErr', pct(100 * V[(512, 257, bp)]['worst_same_grid'], 3))
            macro('nLshapeFiveTwelveBestPodMs', ms(V[(512, 257, bp)]['median_total_ms'], 3))
        cg = V.get((512, 257, 'fom_cg_gpu_r0.01'), {})
        if cg:
            macro('nLshapeFiveTwelveCgErr', pct(100 * cg['worst_same_grid'], 3)); macro('nLshapeFiveTwelveCgMs', ms(cg['median_total_ms'], 2))
    nd128 = [sub for (m, tm, sub) in V if m == 128 and tm == 257 and V[(m, tm, sub)].get('nondominated_complete_ms') and fam[sub] != 'fom']
    if nd128:
        bp128 = min(nd128, key=lambda sub: V[(128, 257, sub)]['worst_same_grid'])
        macro('nLshapeOneTwentyEightBestReduced', name(bp128))
        macro('nLshapeOneTwentyEightBestReducedErr', pct(100 * V[(128, 257, bp128)]['worst_same_grid'], 3))
        macro('nLshapeOneTwentyEightBestReducedCheaper', f"{V[(128, 257, 'fom_splu')]['median_total_ms'] / V[(128, 257, bp128)]['median_total_ms']:.2f}")
        errs = [100 * V[(128, 257, sub)]['worst_same_grid'] for sub in nd128]
        macro('nLshapeOneTwentyEightReducedErrRange', f"{min(errs):.1f}--{max(errs):.1f}")
        cheap = [V[(128, 257, 'fom_splu')]['median_total_ms'] / V[(128, 257, sub)]['median_total_ms'] for sub in nd128]
        macro('nLshapeOneTwentyEightCheapRange', f"{min(cheap):.2f}--{max(cheap):.2f}")
    # free rung, M = 1024: its own job, costs not comparable with the M = 257 tables
    t = []
    nd1024 = [sub for (m, tm, sub) in V if m == 256 and tm == 1024 and V[(m, tm, sub)].get('nondominated_complete_ms')]
    for sub in sorted(nd1024, key=lambda sub: V[(256, 1024, sub)]['median_total_ms']):
        d = V[(256, 1024, sub)]
        t.append([name(sub), fam[sub], pct(100 * d['worst_same_grid']), ms(d['median_total_ms'], 3)])
    write('T18d_lshape_free.tex', tabular(['subject', 'family', 'worst same-grid \\%', 'complete-query ms'], t, 'llrr', r'\scriptsize'),
          f"lshape free rung at M = 1024, job {jobs.get((256, 1024), '---')}; costs NOT comparable with the M = 257 tables")
    fr = V.get((256, 1024, 'freebank@head_sdf_R512_K16'), {}); p256 = V.get((256, 1024, 'pod256'), {})
    if fr and p256:
        macro('nLshapeFreeErr', pct(100 * fr['worst_same_grid'], 4)); macro('nLshapeFreeMs', ms(fr['median_total_ms'], 3))
        macro('nLshapePodTwoFiftySixErr', pct(100 * p256['worst_same_grid'], 4)); macro('nLshapePodTwoFiftySixMs', ms(p256['median_total_ms'], 3))
        macro('nLshapeFreeMoreAccurate', f"{p256['worst_same_grid'] / fr['worst_same_grid']:.2f}")
        macro('nLshapeFreeFaster', f"{p256['median_total_ms'] / fr['median_total_ms']:.2f}")
        macro('provLshapeFreeJob', jobs.get((256, 1024), '---'))
    for r in s_rows:
        if r['metric'] == 'free_rung_worst_over_bank_floor' and r['subject'].endswith('K16'):
            macro('nLshapeFreeOverFloor', f"{r['value']:.3f}")
        if r['metric'] == 'best_found_validation_worst_max':
            macro('nLshapeValidationWorst', pct(100 * r['value'], 1))
        if r['metric'] == 'best_found_development_worst_max':
            macro('nLshapeDevelopmentWorst', pct(100 * r['value'], 1))
    try:
        macro('nLshapeValidationGap', f"{float(MACROS['nLshapeValidationWorst']) / float(MACROS['nLshapeDevelopmentWorst']):.1f}")
    except (KeyError, ValueError, ZeroDivisionError):
        pass
    vr = load('lshape_verify')
    if vr:
        hi = vr['checks']['free_rung_is_head_independent_to_roundoff']['detail']
        macro('nLshapeHeadIndepRel', f"{hi['worst_relative_field_difference']:.2e}")
        macro('nLshapeHeadIndepBitwise', yn(hi['bitwise_identical']))
    macro('nLshapeSolve', 'in Table~\\ref{tab:lshape-solve} ($64^2$--$512^2$, one job per mesh)')
    macro('nLshapeJobCount', str(len({r['job_id'] for r in s_rows})))


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
def build_seeds():
    rows = load('seeds_summary')
    if rows is None:
        write('T12_seeds.tex', gen('b-seeds', 'T12') + '\n')
        write('T13_sealed.tex', gen('b-seeds sealed cohort', 'T13') + '\n')
        macro('nSeedsStatus', gen('b-seeds', 'seeds'))
        return
    rows = rows['rows'] if isinstance(rows, dict) else rows
    import statistics as st
    dev = [r for r in rows if r['cohort'] == 'dev' and r['ladder'] == 'dense_m4']
    V = defaultdict(dict)      # (checkpoint, attempt, q) -> metric -> value
    for r in dev:
        V[(r['checkpoint'], r['attempt'], r['q'])][r['metric']] = r['value']
    seeds = sorted({r['checkpoint'] for r in dev if r['checkpoint'].startswith('seed')})
    attempts = sorted({r['attempt'] for r in dev if r['attempt']})
    jobs = {(r['checkpoint'], r['attempt']): (r['job_id'], r['gpu']) for r in dev if r['job_id']}
    qs = sorted({int(r['q']) for r in dev if r['q'] is not None})
    Mof = {int(r['q']): r.get('M') for r in dev if r['q'] is not None and r.get('M') is not None}
    def seedvals(q, metric):
        out = []
        for sd in seeds:
            for k in V:
                if k[0] == sd and k[2] is not None and int(k[2]) == q and metric in V[k]:
                    out.append(V[k][metric])
        return out
    def inc(q, metric):
        vals = {round(V[k][metric], 6) for k in V if k[0] == 'incumbent' and k[2] is not None and int(k[2]) == q and metric in V[k]}
        return vals
    def ms_(vals, d=4):
        return f"{st.mean(vals):.{d}f} $\\pm$ {st.stdev(vals):.{d}f}" if len(vals) > 1 else (f"{vals[0]:.{d}f}" if vals else '---')
    t = []
    for q in qs:
        ev = seedvals(q, 'evolved'); al = seedvals(q, 'all_times'); bf = seedvals(q, 'best_found'); cv = seedvals(q, 'converged')
        ie = inc(q, 'evolved'); ia = inc(q, 'all_times')
        assert len(ie) == 1 and len(ia) == 1, ('incumbent re-runs disagree across jobs', q, ie, ia)
        t.append([str(q), str(Mof.get(q, '---')), ms_(ev), f"{ie.pop():.4f}", ms_(al), f"{ia.pop():.4f}", ms_(bf),
                  f"{sum(1 for c in cv if c)} of {len(cv)}"])
    write('T12_seeds.tex', tabular(['$q$', '$M$', 'evolved \\%, seeds', 'evolved \\%, incumbent', 'all-times \\%, seeds',
                                    'all-times \\%, incumbent', 'best-found \\%, seeds', 'converged'], t, 'rrllllll', r'\scriptsize'),
          'b-seeds development cohort, dense M = 4(K+q), three training seeds (one job each) beside the incumbent re-run in the same job; mean +- sample std over seeds; costs are not averaged across jobs')
    # per-seed ladder verdicts
    per = []
    for sd in seeds:
        k = [k for k in V if k[0] == sd and k[2] is None]
        d = V[k[0]] if k else {}
        jid, gpu = jobs.get((sd, k[0][1]), ('---', '---')) if k else ('---', '---')
        per.append([tt(sd), tt(jid), tex_escape(gpu), yn(d.get('monotone_evolved')), yn(d.get('monotone_all_times')), yn(d.get('all_converged')),
                    ratio(d.get('error_span_evolved')), ratio(d.get('cost_span')), yn(d.get('knob_bar_passes'))])
    write('T12b_seeds_verdicts.tex', tabular(['seed', 'job', 'GPU', 'monotone (evolved)', 'monotone (all-times)', 'every rung converged',
                                              'error span', 'cost span', 'knob bar'], per, 'lllllllll', r'\scriptsize'),
          'b-seeds per-seed ladder verdicts, development cohort, dense M = 4(K+q)')
    cnt = lambda metric: sum(1 for sd in seeds for k in V if k[0] == sd and k[2] is None and V[k].get(metric))
    macro('nSeedsCount', str(len(seeds)))
    macro('nSeedsMonotoneEvolved', str(cnt('monotone_evolved'))); macro('nSeedsMonotoneAll', str(cnt('monotone_all_times')))
    macro('nSeedsAllConverged', str(cnt('all_converged'))); macro('nSeedsKnobBar', str(cnt('knob_bar_passes')))
    spans = [V[k]['error_span_evolved'] for sd in seeds for k in V if k[0] == sd and k[2] is None and 'error_span_evolved' in V[k]]
    macro('nSeedsErrSpanMin', ratio(min(spans))); macro('nSeedsErrSpanMax', ratio(max(spans)))
    cs = [V[k]['cost_span'] for sd in seeds for k in V if k[0] == sd and k[2] is None and 'cost_span' in V[k]]
    macro('nSeedsCostSpanMin', ratio(min(cs))); macro('nSeedsCostSpanMax', ratio(max(cs)))
    # which rung fails to converge on the non-converged seed(s)
    bad = sorted({(k[0], int(k[2])) for k in V if k[0] in seeds and k[2] is not None and V[k].get('converged') is False})
    macro('nSeedsUnconvergedCells', ', '.join(f"{tt(sd)} at $q={q}$" for sd, q in bad) if bad else 'none')
    # review r2 (B2/R5): the unconverged seed's own spans and budget exits; where the incumbent beats every fresh seed
    if bad:
        sd, q = bad[0]
        kk = [k for k in V if k[0] == sd and k[2] is None]
        if kk:
            macro('nSeedsUnconvergedSeedErrSpan', ratio(V[kk[0]].get('error_span_evolved'))); macro('nSeedsUnconvergedSeedCostSpan', ratio(V[kk[0]].get('cost_span')))
        bx = [V[k].get('budget_exits') for k in V if k[0] == sd and k[2] is not None and int(k[2]) == q and V[k].get('budget_exits') is not None]
        if bx: macro('nSeedsUnconvergedSeedBudgetExits', str(int(bx[0])))
    better = []
    for q in qs:
        ev = seedvals(q, 'evolved'); ie = inc(q, 'evolved')
        if ev and len(ie) == 1 and list(ie)[0] < min(ev): better.append(q)
    macro('nSeedsIncumbentBetterQs', ', '.join(str(q) for q in better) if better else 'none'); macro('nSeedsIncumbentBetterCount', str(len(better)))
    macro('nSeedsRungCount', str(len(qs)))
    # three layers at q = 0 per seed
    fl = [V[k]['three_layer_bank_floor_percent'] for sd in seeds for k in V if k[0] == sd and 'three_layer_bank_floor_percent' in V[k]]
    bf = [V[k]['three_layer_best_found_percent'] for sd in seeds for k in V if k[0] == sd and 'three_layer_best_found_percent' in V[k]]
    macro('nSeedsFloorMin', pct(min(fl))); macro('nSeedsFloorMax', pct(max(fl)))
    macro('nSeedsBestFoundMin', pct(min(bf))); macro('nSeedsBestFoundMax', pct(max(bf)))
    # the pre-registered verdict rows
    ver = {r['metric']: r['value'] for r in rows if r['cohort'] == 'verdict'}
    macro('nSeedsCone', yn(ver.get('C1_monotone_converged_on_at_least_2_of_3'))); macro('nSeedsConeCount', str(ver.get('C1_count')))
    macro('nSeedsCfour', yn(ver.get('C4_knob_bar_on_at_least_2_of_3')))
    macro('nSeedsFthree', yn(ver.get('F3_recipe_not_reproduced')))
    macro('nSeedsSealedPresent', yn(ver.get('sealed_present')))
    macro('provSeedsJobs', '; '.join(f"{tt(sd)} = {jobs[(sd, a)][0]} ({tex_escape(jobs[(sd, a)][1])})" for sd in seeds for a in attempts if (sd, a) in jobs))
    # T13: the sealed cohort has not been opened
    if not ver.get('sealed_present'):
        write('T13_sealed.tex', gen('b-seeds sealed cohort', 'T13') + '\n')
        macro('nSeedsStatus', 'three seeds landed on the development cohort (Table~\\ref{tab:seeds}); the sealed cohort is ' + gen('b-seeds sealed cohort', 'seeds sealed'))
    else:
        raise SystemExit('b-seeds sealed rows present: extend build_seeds before reading them')


def build_pending():
    build_seeds()
    n = load('ns2d_summary')
    if n:
        n = n['rows'] if isinstance(n, dict) else n
        macro('provNsFomJob', n[0]['job_id'])
        NS_JOB = '3787319'                       # ns203: K=16, R=256, full-rank bank; the phase-2 verdict job
        P2 = defaultdict(dict)
        for r in n:
            if str(r.get('job_id')) == NS_JOB:
                P2[(r['subject'], r['gate'], r['mesh'])][r['metric']] = (r['value'], r['passed'])
        def v(sub, gate, mesh, metric):
            return P2.get((sub, gate, mesh), {}).get(metric, (None, None))
        m256 = 256
        macro('provNsJob', NS_JOB)
        # gates at 256^2 (the same at 64^2 and 128^2; printed per mesh in the table)
        orc, orc_ok = v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle.oracle_median')
        pod, _ = v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle.podK_median')
        single, _ = v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle.single_start_median')
        bfl, _ = v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle.bank_floor_median')
        macro('nNsOracleMedian', pct(100 * orc)); macro('nNsPodSixteenMedian', pct(100 * pod))
        macro('nNsOracleRatio', f"{pod / orc:.2f}"); macro('nNsOracleBar', '2.0')
        macro('nNsOraclePass', yn(orc_ok))
        macro('nNsSingleStartMedian', pct(100 * single)); macro('nNsSingleOverOracle', f"{single / orc:.2f}")
        macro('nNsBankFloorMedian', pct(100 * bfl))
        t0, _ = v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle_by_time.podK_over_oracle.t0')
        ev, _ = v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle_by_time.podK_over_oracle.evolved')
        macro('nNsPodOverOracleTzero', f"{t0:.2f}"); macro('nNsPodOverOracleEvolved', f"{ev:.2f}")
        macro('nNsOracleTzero', pct(100 * v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle_by_time.oracle.median_t0')[0]))
        macro('nNsPodTzero', pct(100 * v('head_K16_R256', 'H-ORACLE_N256', m256, 'oracle_by_time.POD-16.median_t0')[0]))
        bw, _ = v('bank_256', 'B-FLOOR_N256', m256, 'floor.worst_evolved_fixed'); pw, _ = v('pod_256', 'B-FLOOR_N256', m256, 'floor.worst_evolved_fixed')
        macro('nNsBankWorst', pct(100 * bw)); macro('nNsPodTwoFiftySixWorst', pct(100 * pw)); macro('nNsBankOverPod', f"{bw / pw:.2f}")
        bm, _ = v('bank_256', 'B-FLOOR_N256', m256, 'floor.median_case_worst_fixed'); pm, _ = v('pod_256', 'B-FLOOR_N256', m256, 'floor.median_case_worst_fixed')
        macro('nNsBankMedian', pct(100 * bm)); macro('nNsPodTwoFiftySixMedian', pct(100 * pm))
        rank, _ = v('bank_R256', 'B-ORTH_N256', m256, 'rank'); macro('nNsBankRank', str(int(rank)) if rank is not None else '---')
        tr, _ = v('head_K16_R256', 'H-TRAIN', m256, 'training.recon_rel_l2_median'); macro('nNsTrainRecon', pct(100 * tr))
        gap, _ = v('head_K16_R256', 'H-ORACLE', m256, 'heldout_oracle_over_training_recon_median'); macro('nNsHeldoutOverTrain', f"{gap:.1f}")
        # the rank-capped predecessor ns201 is job 3783796; same metric names, keyed by job id
        PREV = {}
        for r in n:
            if str(r.get('job_id')) == '3783796' and r.get('mesh') == m256:
                PREV[(r['subject'], r['gate'], r['metric'])] = r['value']
        ptr = PREV.get(('head_K16_R256', 'H-TRAIN', 'training.recon_rel_l2_median'))
        por = PREV.get(('head_K16_R256', 'H-ORACLE_N256', 'oracle.oracle_median'))
        prk = PREV.get(('bank_R256', 'B-ORTH_N256', 'rank'))
        macro('provNsPrevJob', '3783796')
        macro('nNsPrevTrainRecon', pct(100 * ptr) if ptr is not None else '---')
        macro('nNsPrevOracleMedian', pct(100 * por) if por is not None else '---')
        macro('nNsPrevRank', str(int(prk)) if prk is not None else '---')
        macro('nNsTrainImprovement', f"{ptr / tr:.1f}" if (ptr and tr) else '---')
        macro('nNsOracleChange', f"{100 * (orc - por):+.2f}" if (por is not None and orc is not None) else '---')
        # the gate table: phase-2 gates per mesh
        gt = []
        for mesh in (64, 128, 256):
            o, ok = v('head_K16_R256', f'H-ORACLE_N{mesh}', mesh, 'oracle.oracle_median'); p, _ = v('head_K16_R256', f'H-ORACLE_N{mesh}', mesh, 'oracle.podK_median')
            b, bok = v('bank_256', f'B-FLOOR_N{mesh}', mesh, 'floor.worst_evolved_fixed'); pp, _ = v('pod_256', f'B-FLOOR_N{mesh}', mesh, 'floor.worst_evolved_fixed')
            rk, rok = v('bank_R256', f'B-ORTH_N{mesh}', mesh, 'rank')
            gt.append([f'${mesh}^2$', str(int(rk)) if rk is not None else '---', yn(rok), pct(100 * b), pct(100 * pp), yn(bok),
                       pct(100 * o), pct(100 * p), f"{p / o:.2f}", yn(ok)])
        # ns204: K=32 on the R=512 full-rank bank (job 3787320), same gates
        NS2 = '3787320'
        P3 = defaultdict(dict)
        for r in n:
            if str(r.get('job_id')) == NS2:
                P3[(r['subject'], r['gate'], r['mesh'])][r['metric']] = (r['value'], r['passed'])
        def v2(sub, gate, mesh, metric):
            return P3.get((sub, gate, mesh), {}).get(metric, (None, None))
        if P3:
            gt.append('MIDRULE')
            for mesh in (64, 128, 256):
                o, ok = v2('head_K32_R512', f'H-ORACLE_N{mesh}', mesh, 'oracle.oracle_median'); p, _ = v2('head_K32_R512', f'H-ORACLE_N{mesh}', mesh, 'oracle.podK_median')
                b, bok = v2('bank_512', f'B-FLOOR_N{mesh}', mesh, 'floor.worst_evolved_fixed'); pp, _ = v2('pod_512', f'B-FLOOR_N{mesh}', mesh, 'floor.worst_evolved_fixed')
                rk, rok = v2('bank_R512', f'B-ORTH_N{mesh}', mesh, 'rank')
                gt.append([f'${mesh}^2$ ($K{{=}}32$)', str(int(rk)) if rk is not None else '---', yn(rok), pct(100 * b) if b is not None else '---', pct(100 * pp) if pp is not None else '---', yn(bok),
                           pct(100 * o), pct(100 * p), f"{p / o:.2f}", yn(ok)])
            o2, ok2 = v2('head_K32_R512', 'H-ORACLE_N256', 256, 'oracle.oracle_median'); p2, _ = v2('head_K32_R512', 'H-ORACLE_N256', 256, 'oracle.podK_median')
            macro('nNsKthirtyTwoOracleMedian', pct(100 * o2)); macro('nNsKthirtyTwoPodMedian', pct(100 * p2)); macro('nNsKthirtyTwoOracleRatio', f"{p2 / o2:.2f}")
            macro('nNsKthirtyTwoOraclePass', yn(ok2))
            ratios = [v2('head_K32_R512', f'H-ORACLE_N{m}', m, 'oracle.podK_median')[0] / v2('head_K32_R512', f'H-ORACLE_N{m}', m, 'oracle.oracle_median')[0] for m in (64, 128, 256)]
            macro('nNsKthirtyTwoOracleRatioMin', f"{min(ratios):.2f}"); macro('nNsKthirtyTwoOracleRatioMax', f"{max(ratios):.2f}")
            t02, _ = v2('head_K32_R512', 'H-ORACLE_N256', 256, 'oracle_by_time.podK_over_oracle.t0'); ev2, _ = v2('head_K32_R512', 'H-ORACLE_N256', 256, 'oracle_by_time.podK_over_oracle.evolved')
            if t02 is not None: macro('nNsKthirtyTwoPodOverOracleTzero', f"{t02:.2f}")
            if ev2 is not None: macro('nNsKthirtyTwoPodOverOracleEvolved', f"{ev2:.2f}")
            gap2, _ = v2('head_K32_R512', 'H-ORACLE', 256, 'heldout_oracle_over_training_recon_median')
            if gap2 is not None: macro('nNsKthirtyTwoHeldoutOverTrain', f"{gap2:.1f}")
            bx1, _ = v('head_K16_R256', 'H-ORACLE_N256', 256, 'oracle.budget_exits_of_states'); bx2, _ = v2('head_K32_R512', 'H-ORACLE_N256', 256, 'oracle.budget_exits_of_states')
            macro('nNsBudgetExitsKsixteen', str(int(bx1)) if bx1 is not None else '---'); macro('nNsBudgetExitsKthirtyTwo', str(int(bx2)) if bx2 is not None else '---')
            macro('provNsJobs', f'{NS_JOB} ($K{{=}}16$, $R{{=}}256$), {NS2} ($K{{=}}32$, $R{{=}}512$)')
        write('T11e_ns.tex', tabular(['mesh', 'bank rank', 'B-ORTH', 'bank worst \\%', 'POD-$R$ worst \\%', 'B-FLOOR',
                                      'oracle median \\%', 'POD-$K$ median \\%', 'POD-$K$ / oracle', 'H-ORACLE ($\\ge$2.0)'],
                                     gt, 'lrcrrcrrrc', r'\scriptsize'), f'ns2d phase 2, jobs {NS_JOB} (K=16, R=256) and {NS2} (K=32, R=512), full-rank banks; every gate passes except H-ORACLE; oracle values are upper bounds (some held-out fits hit the LM budget)')
        macro('nNsRom', 'not run: the pre-registered gate to it failed (\\S\\ref{sec:exp:linear})')
        macro('nNsKthirtyTwo', 'landed (job ' + NS2 + '): fails the same bar (Table~\\ref{tab:ns})')
    if 'nNsKthirtyTwo' not in MACROS:
        macro('nNsKthirtyTwo', gen('ns2d ns204 (job 3787320), K=32 with the full-rank bank', 'NS K=32 arm'))


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
    prov('T3, T5', 'b-panel ($256^2$, bpn301)', MACROS.get('provPanelJob', '---'), MACROS.get('provPanelGpu', '---'), MACROS.get('provPanelCommit', '---'), MACROS.get('provPanelCkpt', '---'))
    prov('T3b, T5b', 'b-panel ($1024^2$, bpn203)', MACROS.get('provPanelTenTwentyFourJob', '---'), MACROS.get('provPanelTenTwentyFourGpu', '---'), MACROS.get('provPanelCommit', '---'), MACROS.get('provPanelCkpt', '---'))
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
    prov('T18, T18c, T18d', 'lshape', 'training ' + MACROS.get('provLshapeJob', '---') + '; solves ' + MACROS.get('provLshapeSolveJobs', '---') + '; free rung ' + MACROS.get('provLshapeFreeJob', '---'), MACROS.get('provLshapeGpu', '---'), MACROS.get('provLshapeCommit', '---'), '7 heads + bases Git-tracked')
    prov('T12', 'b-seeds (development cohort)', MACROS['provSeedsJobs'] if 'provSeedsJobs' in MACROS else gen('b-seeds', 'T2 seeds row'), 'per job', 'per job', 'three seed checkpoints hashed in summary')
    prov('T13', 'b-seeds (sealed cohort)', gen('b-seeds sealed cohort', 'T2 sealed row'), '---', '---', '---')
    write('T02b_retracted.tex', tabular(['lane', 'attempt', 'job', 'what it would have produced', 'why nothing is reported'],
                                        [[l, tt(a), tt(j), w, why] for l, a, j, w, why in RETRACTED_ATTEMPTS],
                                        r'llp{1.6cm}p{3.2cm}p{6.0cm}', r'\scriptsize'),
          'attempts that produced no reported number')
    def fill(w):   # in-flight notes may name generated macros; substitute their values so the Markdown twin reads them too
        return re.sub(r'\\(n[A-Za-z]+)\{\}', lambda m: MACROS.get(m.group(1), m.group(0)), w)
    write('T02c_inflight.tex', tabular(['lane', 'attempt', 'job', 'what it will add'],
                                       [[l, tt(a), tt(j), fill(w)] for l, a, j, w in IN_FLIGHT],
                                       r'llp{1.6cm}p{7.6cm}', r'\scriptsize'),
          'attempts in flight at the time of writing')
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
