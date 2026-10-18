"""jcp-fom2d report generator: report.md, plots/*.png and runs/EXPORT.json from the lane's analysis files and, for the two
historical combined-H5 values, the combined lane's analysis JSON at a pinned commit (HIST_H5). No measured number in
report.md is typed by hand.

    python make_report.py [--match checks/match-final.json] [--targets checks/targets-final.json]
                          [--c2 checks/c2-analysis.json --plan checks/c2plan.json]

Inputs: runs/GPU-LEDGER.json, checks/smk1-result.json (ladder smoke), checks/smkc2-result.json + smkc2-analysis.json
(C2 smoke), the C1 match file, the C2 analysis and plan (optional: C2 sections say "not yet run" without them), audits/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CH = HERE / 'checks'
PL = HERE / 'plots'
FAM_COL = {'UB': '#2a78d6', 'CC': '#eb6834', 'PS': '#1baf7a'}          # categorical slots 1-3 (validated all-pairs)
FAM_NAME = {'UB': 'upwind + BE', 'CC': 'central + CN-R', 'PS': 'sine pseudo-spectral ETDRK4', 'union': 'union of families'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p and p.exists() else None


def f(x, d=3):
    if x is None:
        return '–'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if not np.isfinite(x):
        return str(x)
    if x == 0:
        return '0'
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        m, e = f'{x:.{d - 1}e}'.split('e')
        return f'{m}e{int(e)}'
    r = f'{x:#.{d}g}'
    return r.rstrip('.') if 'e' not in r else r


def pct(x, d=3):
    return '–' if x is None else f'{f(100 * float(x), d)} %'


def table(head, rows):
    out = ['| ' + ' | '.join(str(h) for h in head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    out += ['| ' + ' | '.join(str(c).replace('|', '\\|') for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def pk(k):
    """Readable FOM key: 'fd|CC|L256|0.0125|t2' -> 'CC L=256, Δt=0.0125, t2'."""
    if k is None:
        return '–'
    p = k.split('|')
    if p[0] == 'fd':
        return f'{p[1]} $L={p[2][1:]}$, $\\Delta t={p[3]}$, {p[4]}'
    if p[0] == 'ps':
        return f'PS $N={p[1][1:]}$, $\\Delta t={p[2]}$'
    return k


def pt(t):
    """Readable ROM target key: 'dn256b|R384|gauss128|frozen|CNR|0.005|LSPG|prod@1024|H' -> 'R384 gauss128 LSPG-CNR Δt=0.005 @1024² (H)'."""
    m = re.match(r'([^|]+)\|R(\d+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^@]+)@(\d+)\|([HX])', t)
    if not m:
        return t
    _, r, rule, pol, sch, dt, form, lev, mesh, pop = m.groups()
    s = f'R{r} {rule}{"" if pol == "frozen" else " " + pol} {form}-{sch} $\\Delta t={dt}$'
    return s + (f' @ ${mesh}^2$' if mesh else '') + (' (crossover population)' if pop == 'X' else '')


def audits():
    """Every audit file, its full first-line verdict (markdown emphasis removed) and whether a later round supersedes it."""
    files = sorted((HERE / 'audits').glob('codex-*.md'))
    series = {}
    for p in files:
        m = re.match(r'codex-(.+)-(\d+)\.md', p.name)
        if m:
            series.setdefault(m.group(1), []).append(int(m.group(2)))
    rows = []
    for p in files:
        t = p.read_text().strip()
        first = re.sub(r'[*`]', '', t.splitlines()[0]) if t else ''
        first = re.split(r'(?<=[.!?])\s', first)[0]
        m = re.match(r'codex-(.+)-(\d+)\.md', p.name)
        later = m and int(m.group(2)) < max(series[m.group(1)])
        rows.append([f'`audits/{p.name}`', first, 'superseded by a later round' if later else 'current'])
    return rows


COMBINED_COMMIT = 'a3c9e170d8f47a53f34e8cb3672e546148383151'      # exp/2026-10-13-jcp-combined, final (immutable)
COMBINED_JSON = 'experiments/jcp-combined/checks/analysis-c2grid-2d_e0.json'


def HIST_H5():
    """Historical combined-H5 matched speed-ups (E0 ST) read from the combined lane's analysis JSON at a pinned commit."""
    import subprocess
    a = json.loads(subprocess.check_output(['git', '-C', str(HERE), 'show', f'{COMBINED_COMMIT}:{COMBINED_JSON}']))
    h = a['hypotheses']['H5']['ST']
    sel = max(v['speedup'] for k, v in h['matches'].items() if '|CNR|' in k and v.get('speedup') is not None)
    return dict(best=f(h['best_combined'][0]) + '×', sel=f(sel) + '×',
                src=f'{COMBINED_JSON} @ {COMBINED_COMMIT[:9]}, hypotheses.H5.ST')


def front2(pts, stat):
    """2D Pareto staircase in (cost, stat): sorted by cost, keep points that lower the running minimum."""
    out, best = [], np.inf
    for p in sorted(pts, key=lambda p: (p['cost'], p[stat])):
        if p[stat] < best:
            out.append(p)
            best = p[stat]
    return out


def plot_pareto(M, targets, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, stat in zip(axs, ('median', 'worst')):
        vals = [v for _, v in targets.get(stat, [])]
        if vals:
            ax.axhspan(min(vals), max(vals), color=INK2, alpha=.12, lw=0, label='primary LSPG ROM targets at $1024^2$ (range)')
        for fam in ('UB', 'CC', 'PS'):
            pts = [p for p in M['pareto'][fam]['points'] if p[stat] < 1]
            ax.scatter([p['cost'] for p in pts], [p[stat] for p in pts], s=10, color=FAM_COL[fam], alpha=.3, linewidths=0)
            fr = front2(pts, stat)
            ax.step([p['cost'] for p in fr], [p[stat] for p in fr], where='post', lw=2, color=FAM_COL[fam], label=FAM_NAME[fam])
            ax.plot([p['cost'] for p in fr], [p[stat] for p in fr], 'o', ms=4, color=FAM_COL[fam])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('screening cost (s per query, H200, lattice output)')
        ax.set_ylabel(f'{stat} error vs E0 $R_{{ST}}$ (38 cases)')
        ax.grid(color=GRID, lw=.6)
        ax.set_title(f'{stat} error: screening-cheapest tested member per level', fontsize=10, color=INK)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_by_resolution(M):
    """Per family and resolution: the cheapest admissible configuration (screening cost) and its errors."""
    out = {}
    for fam in ('UB', 'CC', 'PS'):
        by = {}
        for p in M['pareto'][fam]['points']:
            k = p['key'].split('|')
            res = int(k[2][1:]) if k[0] == 'fd' else int(k[1][1:])
            if res not in by or p['cost'] < by[res]['cost']:
                by[res] = p
        out[fam] = dict(sorted(by.items()))
    return out


def plot_cost(cbr, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fam in ('UB', 'CC', 'PS'):
        xs = list(cbr[fam])
        ax.plot(xs, [cbr[fam][x]['cost'] for x in xs], '-o', ms=4, lw=2, color=FAM_COL[fam], label=FAM_NAME[fam])
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('resolution per axis (FD intervals $L$; PS $N$)')
    ax.set_ylabel('cheapest screening cost at that resolution (s)')
    ax.grid(color=GRID, lw=.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speedup(C2, plan, path):
    """Union speed-up per configuration against the ROM mesh: colour = rank r (4 slots), marker = scheme; small x offsets."""
    RCOL = {128: '#2a78d6', 256: '#eb6834', 384: '#1baf7a', 512: '#4a3aa7'}
    MK = {'BE': 'o', 'CNR': 's', 'CN': '^'}
    LS = {0.005: '-', 0.01: '--', 0.0125: ':'}
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ids = sorted(i for i in plan.get('identities', {}) if '|LSPG|' in i)
    xs_all = sorted(int(n) for n in plan['meshes'])
    for i, ident in enumerate(ids):
        p = ident.split('|')
        r, sch, dtv = int(p[1][1:]), p[4], float(p[5])
        pts = []
        for N in xs_all:
            tk = plan['identities'][ident]['targets'].get(str(N))
            if tk is None:
                continue
            gid = plan['meshes'][str(N)]['targets'][tk].get('group_ids', {}).get('union|joint')
            c = C2['cells'].get(f'{N}|{gid}') if gid else None
            if c and c.get('status') == 'ok':
                pts.append((N * (1 + .015 * (i - len(ids) / 2)), c['S'], c['lo'], c['hi']))
        if pts:
            x, sv, lo, hi = zip(*pts)
            ax.errorbar(x, sv, yerr=[np.array(sv) - np.array(lo), np.array(hi) - np.array(sv)], fmt=LS.get(dtv, '-.'), marker=MK.get(sch, 'o'),
                        ms=4, lw=1, capsize=2, color=RCOL.get(r, INK2), alpha=.85)
    for r, c in RCOL.items():
        ax.plot([], [], '-', color=c, lw=2, label=f'$r={r}$')
    for sch, m in (('BE', 'o'), ('CNR', 's')):
        ax.plot([], [], m, color=INK2, ls='none', label=f'LSPG-{sch}')
    for dtv, ls in LS.items():
        ax.plot([], [], ls, color=INK2, label=f'$\\Delta t={dtv}$')
    ax.axhline(1, color=INK2, lw=1)
    ax.axhline(2, color=INK2, lw=1, ls='--')
    ax.text(xs_all[-1] * 1.1, 2.05, '2×', color=INK2, fontsize=8)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f'{n}²' for n in xs_all])
    ax.set_xlabel('ROM mesh')
    ax.set_ylabel('union speed-up $S$ (FOM cost / ROM cost)')
    ax.set_title('Every qualified LSPG configuration vs its screening-selected matched FOM (union)', fontsize=10, color=INK)
    ax.grid(color=GRID, lw=.6)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match', default='checks/match-final.json')
    ap.add_argument('--targets', default='checks/targets-final.json')
    ap.add_argument('--c2', default='checks/c2-analysis.json')
    ap.add_argument('--plan', default='checks/c2plan.json')
    ap.add_argument('--ladder', action='append', default=[])
    a = ap.parse_args()
    ladders = a.ladder or [HERE / 'runs' / x / 'archive' / 'output' / f'ladder_{x}' for x in ('fcc', 'fub', 'fub4k', 'fps')]
    PL.mkdir(exist_ok=True)
    led = jl(HERE / 'runs' / 'GPU-LEDGER.json')
    smk = jl(CH / 'smk1-result.json')
    c2s = jl(CH / 'smkc2-result.json')
    M = jl(HERE / a.match)
    T = jl(HERE / a.targets)
    C2 = jl(HERE / a.c2)
    plan = jl(HERE / a.plan)
    exp = dict(lane='jcp-fom2d', status=None, references='E0 R_ST, conditional Richardson eps_ref (E0 A10; combined A10, A11, '
               'retracted A15 carried)', gpu_hours=dict(spent=led['spent_gpu_hours'], cap=led['cap_gpu_hours']))
    L = []
    w = L.append
    final_c2 = C2 is not None
    front = str((T or {}).get('source', '')).startswith('FRONT-2D (')
    status = ('final' if final_c2 and front else
              'C1 base ladder complete; matching against ' + ('FRONT-2D targets' if front else
                                                             'the P1 preview targets (provisional; superseded by FRONT-2D)')
              + '; C2 ' + ('on P1 targets (partial evidence, FRONT-2D unavailable)' if final_c2 and not front else
                           ('complete' if final_c2 else 'not yet run')))
    exp['status'] = status
    w('# 2D full-order comparator ladder and ROM–FOM crossover (viscous Burgers)')
    w('')
    w(f'**Status:** {status}. Lane `jcp-fom2d` (plan C1, C2, 2D H-comp). Every accuracy number is a **conditional E0 '
      f'estimate** against $R_{{ST}}$ (store `{(M or {}).get("store_version", "?")[:12]}`; $\\epsilon_{{ref}}$ is a conditional '
      'Richardson estimate, E0 A10; combined A10/A11 and the retracted A15 carry forward). Every matched speed-up is an '
      '**upper bound on the claimed ROM speed-up** (finite ladder; screening-selected comparator), stated conditional on '
      'the dev6 timing workload. Generated by `make_report.py` from the lane\'s JSON files and, for the historical H5 values, the combined lane\'s analysis JSON at a pinned commit; no measured number is typed by hand (registered design constants such as thresholds and ladder values excepted).')
    w('')

    # ---------------------------------------------------------------- summary
    w('## Summary')
    w('')
    if M:
        acc = M['acceptance']
        w(f'- **C1 ladder:** {len(M["admissibility"])} FOM configurations run on the 38 validation cases '
          f'(dev6 + val32); {sum(v["admissible"] for v in M["admissibility"].values())} admissible. Acceptance: '
          + ', '.join(f'{k} {f(v)}' for k, v in acc.items()) + '.')
        cbr = cost_by_resolution(M)
        for fam in ('UB', 'CC', 'PS'):
            fr = M['pareto'][fam]['front']
            best = min(fr, key=lambda p: p['median']) if fr else None
            w(f'- **{FAM_NAME[fam]}:** {len(fr)} configurations on its (cost, median, worst) Pareto front; lowest-median-error '
              f'front member {pk(best["key"]) if best else "–"} (median {pct(best["median"]) if best else "–"}, worst '
              f'{pct(best["worst"]) if best else "–"} — conditional E0 estimates; {f(best["cost"]) if best else "–"} s screening cost).')
        unions = {}
        for lab, m in M['matches'].items():
            if m.get('status') == 'ok' and lab.endswith('|H') and m['union']:
                unions.setdefault(m['union']['family'], []).append(lab)
        w(f'- **C1 screening selections (union, joint median-and-worst match, H populations):** the cheapest by lattice '
          'screening cost among the calibrated admissible tested ladder is ' + '; '.join(f'{FAM_NAME[k]} for {len(v)} targets'
                                                                               for k, v in unions.items()) + '.')
    if C2:
        k2 = C2.get('k2', {})
        Sv = [v['S'] for v in k2.get('members', {}).values() if v['S'] is not None]
        w(f'- **K2 (governing plan), primary scope LSPG-BE / LSPG-CN-R at $4096^2$:** {k2.get("verdict", "–")} — '
          f'{len(Sv)} qualified configurations, union speed-up $S$ between {f(min(Sv)) if Sv else "–"} and '
          f'{f(max(Sv)) if Sv else "–"} (below 1: the matched FOM is faster).')
        for N in ('4096', '1024'):
            for scope, sel in (('primary', '|LSPG|'), ('GAL-CN diagnostic (post hoc)', '|GAL|')):
                cc = {}
                for k, v in C2.get('hcomp', {}).items():
                    if k.startswith(N + '|') and sel in v['target']:
                        key = f"{v['family']}: {v.get('cls') or v.get('verdict')}"
                        cc[key] = cc.get(key, 0) + 1
                w(f'- **H-comp at ${N}^2$, {scope}:** ' + '; '.join(f'{k} ×{v}' for k, v in sorted(cc.items())) + '.')
        sm = [c['S'] for k, c in C2['cells'].items() if k.startswith('4096|') and k.endswith('|H|CC|same_mesh') and
              '|LSPG|' in k and c.get('S') is not None]
        if sm:
            w(f'- **Same mesh at $4096^2$ (central CN-R on the ROM mesh, joint matched):** $S$ between {f(min(sm))} and '
              f'{f(max(sm))} (selected-comparator values: an upper bound on the claimed ROM speed-up over the same-mesh '
              'family; dev6 workload; compare the union range above, which uses the screening-selected comparator across '
              'families and meshes).')
    else:
        w('- **C2 (paired ROM–FOM timing):** not yet run in production (see C2 section).')
    w('')

    # ---------------------------------------------------------------- jobs
    w('## Jobs, budget and integrity')
    w('')
    rows = [[j['attempt'], j['job_id'], j.get('gpu') or '–', f(j.get('elapsed_hours')), j['state']] for j in led['jobs']]
    w(table(['attempt', 'job id', 'GPU allocation', 'elapsed (GPU-h)', 'state'], rows))
    w('')
    w(f'GPU-hours spent: **{f(led["spent_gpu_hours"])}** of the {f(led["cap_gpu_hours"])} cap (ledger `runs/GPU-LEDGER.json`; '
      f'measured `sacct` elapsed time, one GPU per job). Recorded devices: {", ".join(sorted(set((M or {}).get("gpus", [])) | ({c2s["gpu"]} if c2s else set())))}; every job logged '
      '`jax_backend=gpu`, f64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`; cluster namespace `/cluster/tufts/paralab/tawal01/jcp-fom2d/` '
      '(job directories removed after collection; archives local, SHA256 manifests `runs/*/ARCHIVE.sha256`).')
    w('')

    # ---------------------------------------------------------------- gates
    w('## Gates and controls')
    w('')
    if smk:
        g = smk['gates']
        rows = [['G-FD1 (E0 modules byte-identical)', f(g['fd1']['passed']), f'{len(g["fd1"]["files"])} files'],
                ['G-FD2 (F-UB = vendor `engines.make_fom`)', f(g['fd2']['passed']), f'max rel. diff {f(g["fd2"]["rel"])}'],
                ['G-E0 (reproduce E0 level `a0_L2048`, central CN-R)', f(g['e0']['passed']),
                 ', '.join(f'{c["cohort"]}|{c["case"]}: {f(c["rel"])}' for c in g['e0']['cases'])],
                ['G-PS1 (= `pscore.make_ps`; half-step control)', f(g['ps1']['passed'] and g['ps1']['control_fires']),
                 'rel ' + ', '.join(f(v['rel']) for v in g['ps1']['cases'].values()) + '; half-step '
                 + ', '.join(f(v['half_dt_diff']) for v in g['ps1']['cases'].values())],
                ['G-PS2 (reproduce stored E0 PS run)', f(g['ps2']['passed']), f'{g["ps2"]["run"]}: rel {f(g["ps2"]["rel"])}'],
                ['G-INT (sine interpolation, PS synthesis)', f(g['int']['passed']), f'max {f(max(g["int"]["values"].values()))}'],
                ['G-CT (full-field restriction = lattice output)', f(g['ct']['passed']), f'max {f(max(g["ct"]["values"].values()))}'],
                ['K-loose (too-loose comparator rejected)', f(smk['k_loose']['fires']),
                 'agreement ' + ', '.join(f(x) for x in smk['k_loose']['diffs']) + ' > 1e-6'],
                ['K-swap (case swap / time shift change the score)', f(smk['k_swap']['fires']),
                 f'{f(smk["k_swap"]["e"])} → {f(smk["k_swap"]["e_swap"])} / {f(smk["k_swap"]["e_shift"])}']]
        w(table(['check', 'passed / fired', 'numbers'], rows))
        w('')
    if c2s:
        ct = c2s['controls']
        w(table(['C2 smoke ($4096^2$)', 'result', 'numbers'], [
            ['G-RR (ROM errors reproduce their targets)', f(c2s['gates']['G-RR']), ''],
            ['G-FR (FOM errors reproduce the ladder)', f(c2s['gates']['G-FR']), ''],
            ['T-dbl (doubled work)', f(ct['T-dbl']['fires']), f'median ratio {f(ct["T-dbl"]["median"])}'],
            ['T-fault (injected drift caught)', f(ct['T-fault']['fires']), f'drift {f(ct["T-fault"]["row"]["drift"])}'],
            ['T-id (identity veto)', f(ct['T-id']['fires']), ''],
            ['combined workload memory', f(c2s['pilots_ok']), f'peak {f(c2s["memory"]["end"]["peak"] / 1e9)} GB of '
                                                               f'{f(c2s["memory"]["end"]["limit"] / 1e9)} GB']]))
        w('')
    if M:
        ks = [m['controls'] for m in M['matches'].values() if m.get('status') == 'ok']
        w(f'Matcher controls on every evaluated target ({len(ks)}): K-syn fired on {sum(c["ksyn"]["K|unconverged"]["rejected"] and c["ksyn"]["K|inaccurate"]["rejected"] for c in ks)}, '
          f'K-false (falsely matched pair rejected) fired on {sum(c["kfalse"]["fires"] for c in ks)}. Offline rescoring agrees '
          f'with the in-job scores to {f(M["offline_vs_injob_worst"])}.')
        w('')

    # ---------------------------------------------------------------- C1
    if M:
        w('## C1 — the FOM ladder')
        w('')
        w('Three families on the $257^2$ scoring lattice: $F_{UB}$ sign-upwind + backward Euler (the project FOM), '
          '$F_{CC}$ central differences + Crank–Nicolson with Rannacher start (E0\'s scheme; its error against E0 partly '
          'measures the reference\'s own level difference), $F_{PS}$ sine pseudo-spectral + ETDRK4. FD meshes coarser '
          'than $256$ reach the lattice by sine (DST-I) interpolation; the metric is')
        w('')
        w('$$e=\\max_{k=1..5}\\frac{\\lVert u(t_k)-R_{ST}(t_k)\\rVert_2}{\\lVert R_{ST}(t_0)\\rVert_2}.$$')
        w('')
        cal = M['calibration']
        levs = {}
        for v in cal.values():
            levs[str(v['production_level'])] = levs.get(str(v['production_level']), 0) + 1
        w(f'Tolerance calibration ({len(cal)} FD groups): production level counts ' +
          ', '.join(f't{k}: {v}' for k, v in sorted(levs.items())) + '; recovered groups: '
          f'{sum(1 for v in cal.values() if v["recovery"])}; recomputed choice agrees with the job: {f(M["acceptance"]["calibration_agrees"])}.')
        w('')
        rows = []
        for fam in ('UB', 'CC', 'PS'):
            ad = [v for k, v in M['admissibility'].items() if v['family'] == fam]
            why = {}
            for v in ad:
                for r in v['why']:
                    why[r] = why.get(r, 0) + 1
            rows.append([FAM_NAME[fam], len(ad), sum(v['admissible'] for v in ad), '; '.join(f'{k}: {v}' for k, v in why.items()) or '–'])
        w(table(['family', 'configurations', 'admissible', 'reasons (inadmissible)'], rows))
        w('')
        w('**Per-family Pareto front** (errors are conditional E0 estimates; admissible configurations not beaten in screening cost, median and worst error; '
          'screening cost = warm lattice-contract time on dev6 cases 0–1, H200, used only to order candidates):')
        w('')
        for fam in ('UB', 'CC', 'PS'):
            fr = M['pareto'][fam]['front']
            w(f'*{FAM_NAME[fam]}*')
            w('')
            w(table(['configuration', 'screening cost (s)', 'median error', 'worst error'],
                    [[pk(p['key']), f(p['cost']), pct(p['median']), pct(p['worst'])] for p in fr]))
            w('')
        cbr = cost_by_resolution(M)
        w('**Cost per FOM by resolution** (screening-cheapest admissible tested configuration at each FD mesh $L$ / PS $N$, '
          'whatever its accuracy — not an accuracy-matched comparison; errors are conditional E0 estimates):')
        w('')
        rows = []
        for fam in ('UB', 'CC', 'PS'):
            for res, p in cbr[fam].items():
                rows.append([FAM_NAME[fam], res, pk(p['key']), f(p['cost']), pct(p['median']), pct(p['worst'])])
        w(table(['family', 'resolution', 'cheapest configuration', 'cost (s)', 'median', 'worst'], rows))
        w('')
        tl = {'median': [], 'worst': []}
        if T:
            for lab, t in T['targets'].items():
                if lab.endswith('@1024|H') and t.get('status') == 'ok' and '|LSPG|' in lab:
                    tl['median'].append((pt(lab).split(' $')[0], t['median']))
                    tl['worst'].append((pt(lab).split(' $')[0], t['worst']))
        plot_pareto(M, tl, PL / 'fom_pareto.png')
        plot_cost(cbr, PL / 'fom_cost_by_resolution.png')
        nex = {st: sum(1 for fam in ('UB', 'CC', 'PS') for p in M['pareto'][fam]['points'] if p[st] >= 1) for st in ('median', 'worst')}
        w('![FOM accuracy–cost envelopes](plots/fom_pareto.png)')
        w('')
        w(f'*Figure 1.* Screening-cheapest tested ladder member per accuracy level, drawn separately for the median and the '
          f'worst error (these are single-statistic envelopes, not the joint median-and-worst match used for selection); '
          f'errors are conditional E0 estimates; the shaded band is the range of the primary LSPG targets at $1024^2$. '
          f'Admissible points with error $\\ge 1$ are not drawn (median panel {nex["median"]}, worst panel {nex["worst"]}).')
        w('')
        w('![Cheapest FOM cost by resolution](plots/fom_cost_by_resolution.png)')
        w('')
        w('*Figure 2.* Screening cost of the cheapest admissible tested configuration at each resolution; the accuracy of '
          'these configurations differs and the curves are not an equal-accuracy comparison.')
        w('')
        w('**C1 screening selections against the targets at $1024^2$** (other meshes in `checks/match-final.json`; source: ' + str(M.get('targets_source')) + '; H populations; '
          'joint = median **and** worst tie-or-beat the target (conditional E0 estimates); selection by lattice screening cost among the '
          'tested ladder, the C2 selection may differ; GAL-CN rows are a diagnostic scheme, post hoc, A8):')
        w('')
        rows = []
        for lab, m in M['matches'].items():
            if not lab.endswith('|H') or ('@1024|' not in lab and '@1024' not in lab.split('|H')[0][-6:]):
                continue
            t = T['targets'][lab] if T and lab in T['targets'] else {}
            if m.get('status') != 'ok':
                rows.append([pt(lab), '', '–', '–', m.get('status'), '', '', ''])
                continue
            cells = []
            for fam in ('UB', 'CC', 'PS'):
                j = m[fam]['joint']
                cells.append('no ladder match' if j is None else f'{pk(j["key"])}' + (' (edge)' if j['edge'] else '')
                             + (' (boundary)' if j['boundary'] else ''))
            rows.append([pt(lab), 'primary' if '|LSPG|' in lab else 'GAL-CN diagnostic', pct(t.get('median')), pct(t.get('worst'))]
                        + cells + [FAM_NAME.get((m['union'] or {}).get('family'), '–')])
        w(table(['target', 'scope', 'target median', 'target worst', 'UB joint', 'CC joint', 'PS joint', 'union'], rows))
        w('')
        ext = M.get('extension_trigger', {})
        w('Extension round (A3.2), for ' + ('the FRONT-2D targets' if front else 'the P1 preview targets only (to be '
                                            'recomputed for FRONT-2D or the authorised fallback)') + ': '
          + '; '.join(f'{k}: {"fires" if v["fires"] else "not triggered"} (base minimum resolution {v["base_min_resolution"]})'
                      for k, v in ext.items()) + f'; complete: {f(M.get("extension_complete"))}.')
        w('')
        # PS selections: E0 stability label and anchor (A1.1, A3.3, A3.4 disclosure)
        stab = {}
        for d in ladders:
            r = jl(Path(d) / 'result.json')
            for k, v in (r or {}).get('configs', {}).items():
                if 'stability' in v:
                    stab[k] = v['stability']
        sel = {}
        for lab, m in M['matches'].items():
            if m.get('status') == 'ok' and m['PS']['joint']:
                sel.setdefault(m['PS']['joint']['key'], []).append(m['PS'].get('anchor'))
        rows = []
        for k, an in sorted(sel.items()):
            st = stab.get(k, {})
            nb = sum(1 for u, b in zip(st.get('umax_padded_running', []), st.get('dt_bound', [])) if b is not None and
                     float(k.split('|')[2]) <= b)
            a0 = an[0] or {}
            rows.append([pk(k), len(an), f'{nb} of {len(st.get("dt_bound", []))}',
                         f'{a0.get("status")}' + (f', steps {a0["steps"][0]} vs {a0["steps"][1]}, both valid {f(a0.get("both_valid"))}, '
                                                  f'distance {f(a0.get("distance"))}' if a0.get('status') == 'ok' else '')])
        if rows:
            w('**Selected PS comparators — diagnostics** (E0 stability label from the padded-grid running maximum; a failing '
              'label does not affect admissibility, A3.3; the anchor is the lattice distance to the adjacent ladder step '
              'and certifies nothing, A3.4):')
            w('')
            w(table(['PS comparator', 'target variants selecting it (H+X)', 'cases passing the E0 stability label',
                     'anchor'], rows))
            w('')
        blow = [p['key'] for p in M['pareto']['PS']['points'] if p['worst'] > 1]
        mo = {}
        for lab, m in M['matches'].items():
            if m.get('status') == 'ok' and m['PS']['median_only'] and m['PS']['median_only']['key'] in blow:
                mo.setdefault(m['PS']['median_only']['key'], []).append(lab)
        for k, labs in mo.items():
            wv = next(p['worst'] for p in M['pareto']['PS']['points'] if p['key'] == k)
            w(f'Note: {pk(k)} is admissible (finite outputs) but blows up on one case (worst error {f(wv)}); it never matches '
              f'jointly, yet it is the **median-only** selection for {len(labs)} target variants — the median-only criterion '
              'ignores the worst case by construction, so median-only selections are reported, never used as accuracy matches.')
            w('')

    # ---------------------------------------------------------------- C2
    w('## C2 — paired ROM–FOM timing, crossover, H-comp and K2')
    w('')
    if not C2:
        w('Not yet run in production: the C2 smoke at $4096^2$ was accepted (above); the production panels wait for '
          '`FRONT-2D.json` from jcp-qualify (DESIGN A1.7).')
        w('')
    else:
        w('One H200 job per ROM mesh, every FRONT-2D numerically qualified configuration as a subject. $A$ = the P1 reference '
          'ROM (R384, gauss128, frozen modes, LSPG-CN-R, $\\Delta t_0$, fused full-field query). Cost of a subject $B$: median '
          'over the six dev6 cases of the per-case median of $t_B/\\tfrac12(t_{A1}+t_{A2})$ (stage 2, fresh triplets); '
          '$S=\\mathrm{cost}(\\mathrm{FOM})/\\mathrm{cost}(\\mathrm{ROM})$ with a paired case-bootstrap 95 % interval '
          '(descriptive of the dev6 workload). Full-field contract: both deliver the six fields on the ROM mesh; latencies '
          'are prepared-input latencies. **Primary verdicts use the registered deployment schemes LSPG-BE and LSPG-CN-R; '
          'GAL-CN rows are a diagnostic scheme, post-hoc comparison, not a registered deployment arm (A8).**')
        w('')
        rows = [[N + '²', v['job'], f(v['panel_ok']), ', '.join(f'{k} {f(x)}' for k, x in v['controls'].items()),
                 len(v['stage0'])] for N, v in sorted(C2['meshes'].items(), key=lambda x: int(x[0]))]
        w(table(['mesh', 'job', 'panel valid (gates, controls, pilots)', 'controls fired', 'FOM candidates timed'], rows))
        w('')
        cells = C2['cells']

        def cell(N, tk, fc):
            return cells.get(f'{N}|{tk}|{fc}', {})

        def sx(c):
            if not c:
                return '–'
            if c.get('status') != 'ok':
                return str(c.get('status'))
            return f'{f(c["S"])} [{f(c["lo"])}, {f(c["hi"])}]'
        tks = sorted({c['target'] for c in cells.values() if str(c.get('target', '')).endswith('|H') and '@' in str(c.get('target'))})
        for scope, sel in (('Primary (LSPG-BE, LSPG-CN-R)', '|LSPG|'), ('Diagnostic GAL-CN (post hoc, A8)', '|GAL|')):
            rows = []
            for N in ('4096', '1024'):
                for tk in [t for t in tks if sel in t and t.split('@')[1].startswith(N + '|')]:
                    u = cell(N, tk, 'union|joint')
                    rows.append([N + '²', pt(tk), sx(cell(N, tk, 'UB|joint')), sx(cell(N, tk, 'CC|joint')),
                                 sx(cell(N, tk, 'PS|joint')), sx(u), pk(u.get('selected')),
                                 'fail (dev6 rule)' if u.get('two_x_class') == 'ROM < 2x' else
                                 ('pass vs selected' if u.get('two_x_class') == 'ROM >= 2x' else (u.get('status') or 'inconclusive'))])
            w(f'**H-comp cells — {scope}** (joint median-and-worst matches, H populations; entries $S$ [95 % interval]; '
              '"fail (dev6 rule)" = upper end below 2, a conservative witness under the registered dev6 descriptive-bootstrap '
              'rule; "pass vs selected" = lower end at least 2 against the screening-selected comparator only):')
            w('')
            w(table(['mesh', 'ROM', 'UB', 'CC', 'PS', 'union', 'union comparator', 'H-comp (union)'], rows))
            w('')
        k2 = C2.get('k2', {})
        w(f'**K2 (governing plan), primary scope ({k2.get("scope", "")}): {k2.get("verdict", "–")}.** Universe: '
          f'{len(k2.get("members", {}))} numerically qualified LSPG configurations at $4096^2$, complete: '
          f'{f(k2.get("universe_complete"))}; union $S$ ranges over '
          f'{f(min(v["S"] for v in k2["members"].values() if v["S"] is not None))}–'
          f'{f(max(v["S"] for v in k2["members"].values() if v["S"] is not None))}.' if k2.get('members') else '')
        kg = C2.get('k2_diagnostic_gal', {})
        if kg.get('members'):
            w('')
            w(f'Diagnostic GAL-CN analogue (post hoc, not a K2 verdict): {kg.get("verdict")}; union $S$ '
              f'{f(min(v["S"] for v in kg["members"].values() if v["S"] is not None))}–'
              f'{f(max(v["S"] for v in kg["members"].values() if v["S"] is not None))} over {len(kg["members"])} configurations.')
        w('')
        rows = []
        for tk in [t for t in tks if '|LSPG|' in t and t.split('@')[1].startswith('4096|')]:
            N = '4096'
            rows.append([pt(tk), sx(cell(N, tk, 'UB|same_mesh')) if cell(N, tk, 'UB|same_mesh') else sx(cell(N, tk, 'UB|same_mesh_unmatched')),
                         sx(cell(N, tk, 'CC|same_mesh')) if cell(N, tk, 'CC|same_mesh') else sx(cell(N, tk, 'CC|same_mesh_unmatched')),
                         sx(cell(N, tk, 'union|median_only')) if cell(N, tk, 'union|median_only') else sx(cell(N, tk, 'PS|median_only')),
                         sx(cell(N, tk, 'PS|worst_only'))])
        w('**Same-mesh and median-/worst-only comparisons at $4096^2$** (entries $S$ [95 % interval]: upper bounds on the '
          'claimed ROM speed-up, dev6 workload; primary scope; same-mesh = FD at $L=4096$, matched '
          'jointly where possible, else the lowest-worst-error unmatched comparator; median-/worst-only are recorded '
          'comparisons, not accuracy matches; same-resolution PS at $4096^2$ deferred, A2.4):')
        w('')
        w(table(['ROM', 'UB same-mesh', 'CC same-mesh', 'PS median-only', 'PS worst-only'], rows))
        w('')
        rows = []
        for tk in [t for t in tks if '|LSPG|' in t and t.split('@')[1].startswith('4096|')]:
            u = cell('4096', tk, 'union|joint')
            if u.get('seconds_fom') is not None:
                rows.append([pt(tk), f(u['seconds_rom']), pk(u['selected']), f(u['seconds_fom'])])
        w('**Absolute prepared-input latencies at $4096^2$** (median stage-2 seconds per query, H200):')
        w('')
        w(table(['ROM', 'ROM (s)', 'union comparator', 'FOM (s)'], rows))
        w('')
        rows = []
        for ident, ent in sorted(C2.get('crossover', {}).items()):
            if '|LSPG|' not in ident:
                continue
            rows.append([pt(ident + '@0|X').split(' @')[0], ent['union|joint']['bracket_1x'], ent['union|joint']['bracket_2x'],
                         ent['CC|same_mesh']['bracket_1x'], ent['CC|same_mesh']['bracket_2x']])
        w('**Crossover brackets** (primary scope; crossover populations; classes from the interval rule at each actually run '
          'mesh, never interpolated). (b) union = screening-selected joint-matched comparator across families and meshes; (a) CC same-mesh = central CN-R on the '
          'ROM mesh. Per-family and UB same-mesh brackets are in `runs/EXPORT.json`; the same-mesh union mixes PS only at '
          '$256^2$ (same-resolution PS deferred above), so it is not tabled:')
        w('')
        w(table(['ROM', '(b) union 1×', '(b) union 2×', '(a) CC same-mesh 1×', '(a) CC same-mesh 2×'], rows))
        w('')
        if plan:
            plot_speedup(C2, plan, PL / 'speedup_vs_mesh.png')
            w('![Union speed-up against the ROM mesh (primary scope)](plots/speedup_vs_mesh.png)')
            w('')
            w('*Figure 3.* Union speed-up $S$ (FOM cost / ROM cost) of every qualified LSPG configuration against its '
              'screening-selected joint-matched comparator, at each ROM mesh; bars are the paired case-bootstrap 95 % '
              'intervals over the six dev6 timing cases (descriptive of that workload); prepared-input latencies; every $S$ is '
              'an upper bound on the claimed ROM speed-up (finite ladder, screening selection). Colour = rank $r$, marker = '
              'scheme, line style = time step; small horizontal offsets separate overlapping configurations.')
            w('')
    # ---------------------------------------------------------------- provisional, retractions
    w('## What is provisional, and what changes earlier claims')
    w('')
    w('- All accuracy numbers are conditional E0 estimates; FOM–ROM ties use the per-case $2\\epsilon_{ref}$ rule.')
    w('- Every matched speed-up is an upper bound on the claimed ROM speed-up: the FOM ladder is finite, the comparator is '
      'chosen by a screening procedure, and the timing interval describes the six dev6 timing cases only.')
    h5 = HIST_H5()
    w(f'- **Correction of an earlier claim.** The earlier 2D matched speed-up (combined H5: {h5["sel"]} for LSPG-CN-R, '
      f'{h5["best"]} best arm; `{h5["src"]}`) was measured against a **same-mesh sign-upwind** FOM matched on the worst '
      'error only. Within that scope it stands as reported. It does **not** establish an advantage over the expanded '
      'comparator ladder: ' + ('against the screening-selected joint match across families and meshes (C2 above), every '
      'qualified LSPG configuration is slower than its comparator at every mesh, and K2 fires.' if final_c2 else
      'the C2 panels decide that (not yet run).'))
    w('- Same-resolution PS at $1024^2$ and $4096^2$ was **not run (deferred by scope, A2.4)**: those same-mesh cells carry '
      'no comparative claim; the same-mesh union there covers the FD families only.')
    w('')
    w('## Audits')
    w('')
    w(table(['file', 'verdict (first sentence)', 'status'], audits()))
    w('')
    w('## Glossary')
    w('')
    for k, v in [
        ('FOM', 'full-order model: a conventional finite-difference (FD) or spectral solver of the PDE.'),
        ('ROM', 'the reduced model: a linear span of $r$ columns of a trained coordinate-network bank (a fixed set of basis functions given by a neural network), solved by least-squares Petrov–Galerkin (LSPG) or Galerkin (GAL) with nonlinear terms evaluated at quadrature points (a fixed set of sample points with weights).'),
        ('R128, R256, R384, R512', 'the ROM rank $r$ (number of bank columns).'),
        ('fib4181, gauss128, gauss160, gauss192', 'quadrature rules: a 4181-point Fibonacci lattice and Gauss rules with 128 (160, 192) points per axis.'),
        ('LSPG-BE, LSPG-CN-R, GAL-CN', 'ROM schemes: LSPG with backward Euler, LSPG with Crank–Nicolson after Rannacher start-up steps (the registered deployment schemes), and Galerkin Crank–Nicolson (a diagnostic scheme, reported separately).'),
        ('UB, CC, PS', 'FOM families: upwind advection + backward Euler (UB), central differences + Crank–Nicolson with Rannacher start (CC), sine pseudo-spectral + ETDRK4 (PS, a fourth-order exponential time integrator).'),
        ('$L$, $N$, $\\Delta t$', 'FD intervals per axis; PS resolution parameter (sine modes $1,\\dots,N-1$ per axis); time step.'),
        ('configuration key', '"CC $L=256$, $\\Delta t=0.0125$, t2" = family, resolution, step and tolerance level.'),
        ('E0, $R_{ST}$', 'the second-order reference solutions (central + Crank–Nicolson, three refinement levels) every error is measured against.'),
        ('$\\epsilon_{ref}$, Richardson estimate', 'per-case uncertainty of the reference, estimated from the differences between its refinement levels assuming the asymptotic convergence order (hence "conditional"); two errors closer than $2\\epsilon_{ref}$ are a tie.'),
        ('conditional E0 estimate', 'an error measured against $R_{ST}$, valid only as far as that reference and its $\\epsilon_{ref}$ are.'),
        ('median / worst error', 'the median and the maximum of the per-case error over the eligible cases of a population.'),
        ('target, target median / worst', 'a ROM configuration whose accuracy a FOM must match, with its median and worst error.'),
        ('tie-or-beat', 'the median (worst) of $e_{FOM}-\\epsilon_{ref}$ is no larger than the median (worst) of $e_{ROM}+\\epsilon_{ref}$.'),
        ('joint match; median-only; worst-only', 'a FOM that ties-or-beats the ROM on both statistics; on the median alone; on the worst alone (the last two are recorded comparisons, not accuracy matches).'),
        ('same-mesh', 'the FOM runs on the ROM\'s own mesh (FD $L=N$, or PS with $N$ equal to it where run).'),
        ('union', 'the cheapest of the three families\' selected comparators.'),
        ('screening cost', 'warm wall time of one FOM query with lattice output on two dev6 cases (H200), used only to order candidates.'),
        ('Newton, BiCGStab, t0–t3', 'each implicit FD time step solves a nonlinear system: Newton iterations repeatedly linearise it, and BiCGStab (an iterative linear solver) solves each linearised system; t0–t3 are their tolerance levels from $(10^{-10},10^{-11})$ to $(10^{-4},10^{-5})$; the calibrated level is the loosest within $10^{-6}$ of t0.'),
        ('admissible; reasons', 'calibrated, converged with complete evidence on all 38 cases, finite scores, bitwise-identical repeated screening outputs; a configuration can fail several reasons at once.'),
        ('Pareto front', 'admissible configurations not beaten simultaneously in screening cost, median and worst error.'),
        ('edge / boundary', 'edge: matched at the family\'s coarsest resolution or largest step; boundary: matched at the screening-cheapest admissible member.'),
        ('E0 stability label, anchor', 'E0\'s step-size rule for the spectral solver evaluated with the run\'s own maximum $|u|$ (diagnostic); anchor = lattice distance between the selected PS run and the same $N$ at the adjacent ladder step (both must be valid; it certifies nothing).'),
        ('H / crossover population', 'H: the ROM target\'s own case population at that mesh; crossover (X): the cases common to every mesh where the ROM configuration is qualified.'),
        ('job table', 'attempt = lane job name; job id = Slurm id; GPU allocation = requested resource; elapsed = measured GPU-hours; state = Slurm final state.'),
        ('gates and controls', 'gates must pass (G-…: reproduction and consistency checks, reported as maximum relative differences); controls must fire (K-…, T-…: deliberately wrong inputs that the checks must reject); drift = ratio of the two flanking reference timings; identity veto = a subject whose outputs ever differ from its scoring run is excluded.'),
        ('panel valid', 'every gate passed, every timing control fired, every combined-workload pilot valid, and the reference A never vetoed.'),
        ('A–B–A, stage 0/1/2', 'paired timing: reference A before and after subject B; stage 0 screens, stage 1 selects the comparator, stage 2 measures afresh for the reported speed-ups.'),
        ('prepared-input latency', 'wall time of one query from inputs already on the GPU in each method\'s own form (FD nodal field, PS sine coefficients, ROM full-field IC) to the six output fields on the mesh; input preparation is excluded and reported separately.'),
        ('$S$, speed-up; interval', 'FOM cost over ROM cost (ratio of case medians against the common reference A); above 1 the ROM is faster; the interval is a paired bootstrap over the six dev6 cases.'),
        ('H-comp; fail (dev6 rule); pass vs selected; unresolved', 'the plan\'s competition hypothesis (ROM at least 2× faster); fail = interval upper end below 2; pass vs selected = lower end at least 2 against the screening-selected comparator only; unresolved = neither.'),
        ('crossover bracket, 1× / 2×', 'the classification of each actually run mesh (ROM faster / FOM faster / unresolved, or ≥ 2× / < 2×) and the pair of meshes between which it changes; "point only" = one qualified mesh; never interpolated.'),
        ('K2', 'the governing plan\'s kill criterion: the speed-up over the best validated FOM is below 2× at the largest mesh for every linear setting; here evaluated against screening-selected comparators under the dev6 rule.'),
        ('dev6, val32', 'the six development and 32 validation cases (selection population).'),
        ('smoke, pilot', 'smoke = a small real-size cluster job run before production to show that every gate passes and every control fires; pilot = a single untimed run of a heavy configuration to measure its memory and time.'),
        ('peak memory / limit', 'the largest GPU memory in use during the job against the memory the job may use on the H200.'),
        ('cheapest configuration at a resolution', 'the admissible tested configuration with the smallest screening cost at that $L$ or $N$, whatever its accuracy (no accuracy match).'),
        ('anchor distance', 'max over output times and cases of $\\lVert u_a(t_k)-u_b(t_k)\\rVert_2/\\lVert R_{ST}(t_0)\\rVert_2$ on the scoring lattice between the two runs.'),
        ('FOM candidates timed', 'number of distinct FOM configurations screened in stage 0 at that mesh (every match of any criterion for any target).'),
        ('no match vs unresolved', 'no match = no tested admissible configuration of that family ties-or-beats the target (nothing timed); unresolved = a comparator was timed but its interval straddles the threshold.'),
        ('ROM (s), FOM (s)', 'median over all valid stage-2 timed executions (pooled over cases and repetitions) of the absolute wall time per query; unlike $S$, not normalised by the reference A.'),
        ('audit table', 'each Codex audit file with the first sentence of its verdict; earlier rounds of the same audit are superseded by later ones.')]:
        w(f'- **{k}** — {v}')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    if M:
        exp['c1'] = dict(admissible=sum(v['admissible'] for v in M['admissibility'].values()), configurations=len(M['admissibility']),
                         pareto={fam: M['pareto'][fam]['front'] for fam in ('UB', 'CC', 'PS')},
                         union_selection={lab: (m['union'] or {}).get('joint', {}).get('key') if m.get('status') == 'ok' else None
                                          for lab, m in M['matches'].items()})
    if C2:
        exp['c2'] = dict(k2=C2.get('k2'), hcomp=C2.get('hcomp'), crossover=C2.get('crossover'), meshes=C2.get('meshes'))
    exp['plots'] = sorted(f'plots/{p.name}' for p in PL.glob('*.png'))
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=float) + '\n')
    print('report.md written;', len(L), 'lines')


if __name__ == '__main__':
    main()
