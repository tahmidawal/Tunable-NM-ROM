"""jcp-fom2d report generator: report.md, plots/*.png and runs/EXPORT.json from the lane's analysis files only. No number
in report.md is typed by hand: every value is read from the JSON files named below.

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
    rows = []
    for p in sorted((HERE / 'audits').glob('codex-*.md')):
        t = p.read_text()
        first = t.strip().splitlines()[0] if t.strip() else ''
        v = re.findall(r'\*\*([A-Z][A-Z \-]+[A-Z])[\.\*]', first)
        rows.append([f'`audits/{p.name}`', v[0] if v else first[:60]])
    return rows


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
            ax.axhspan(min(vals), max(vals), color=INK2, alpha=.12, lw=0, label='ROM targets at $1024^2$ (range)')
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
        ax.set_title(f'{stat} error: cheapest FOM per accuracy level', fontsize=10, color=INK)
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
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ids = sorted(plan.get('identities', {}))
    xs_all = sorted(int(n) for n in plan['meshes'])
    for i, ident in enumerate(ids):
        pts = []
        for N in xs_all:
            tk = plan['identities'][ident]['targets'].get(str(N))
            if tk is None:
                continue
            gid = plan['meshes'][str(N)]['targets'][tk].get('group_ids', {}).get('union|joint')
            c = C2['cells'].get(f'{N}|{gid}') if gid else None
            if c and c.get('status') == 'ok':
                pts.append((N * (1 + .04 * (i - len(ids) / 2)), c['S'], c['lo'], c['hi']))
        if pts:
            x, s, lo, hi = zip(*pts)
            ax.errorbar(x, s, yerr=[np.array(s) - np.array(lo), np.array(hi) - np.array(s)], fmt='-o', ms=4, lw=1.5,
                        capsize=2, label=pt(ident + '@0|H').split(' @')[0].replace('$', ''))
    ax.axhline(1, color=INK2, lw=1)
    ax.axhline(2, color=INK2, lw=1, ls='--')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f'{n}²' for n in xs_all])
    ax.set_xlabel('ROM mesh')
    ax.set_ylabel('speed-up $S$ vs best selected FOM (union)')
    ax.grid(color=GRID, lw=.6)
    ax.legend(frameon=False, fontsize=6, ncol=1)
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
      'the dev6 timing workload. Generated by `make_report.py` from the lane\'s JSON files; no number is typed by hand.')
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
              f'{pct(best["worst"]) if best else "–"}, {f(best["cost"]) if best else "–"} s).')
        unions = {}
        for lab, m in M['matches'].items():
            if m.get('status') == 'ok' and lab.endswith('|H') and m['union']:
                unions.setdefault(m['union']['family'], []).append(lab)
        w(f'- **C1 screening selections (union, joint median-and-worst match, H populations):** the cheapest by lattice '
          'screening cost among the calibrated admissible tested ladder is ' + '; '.join(f'{FAM_NAME[k]} for {len(v)} targets'
                                                                               for k, v in unions.items()) + '.')
    if C2:
        w(f'- **K2 (governing plan) at $4096^2$:** {C2.get("k2", {}).get("verdict", "–")}.')
        hc = C2.get('hcomp', {})
        cls = {}
        for v in hc.values():
            cls[str(v.get('verdict'))] = cls.get(str(v.get('verdict')), 0) + 1
        w('- **H-comp cells (joint matches, H populations, $1024^2$ and $4096^2$):** ' +
          '; '.join(f'{k}: {v}' for k, v in sorted(cls.items())) + '.')
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
        w('**Per-family Pareto front** (admissible configurations not beaten in screening cost, median and worst error; '
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
        w('**Cost per FOM by resolution** (cheapest admissible configuration at each FD mesh $L$ / PS $N$):')
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
                if lab.endswith('@1024|H') and t.get('status') == 'ok':
                    tl['median'].append((pt(lab).split(' $')[0], t['median']))
                    tl['worst'].append((pt(lab).split(' $')[0], t['worst']))
        plot_pareto(M, tl, PL / 'fom_pareto.png')
        plot_cost(cbr, PL / 'fom_cost_by_resolution.png')
        w('![FOM Pareto fronts with ROM target errors at 1024² (dashed)](plots/fom_pareto.png)')
        w('')
        w('![Cheapest FOM cost by resolution](plots/fom_cost_by_resolution.png)')
        w('')
        w('**C1 screening selections against the targets** (source: ' + str(M.get('targets_source')) + '; H populations; '
          'joint = median **and** worst tie-or-beat the target; selection by lattice screening cost, the C2 selection may differ):')
        w('')
        rows = []
        for lab, m in M['matches'].items():
            if not lab.endswith('|H'):
                continue
            t = T['targets'][lab] if T and lab in T['targets'] else {}
            if m.get('status') != 'ok':
                rows.append([pt(lab), '–', '–', m.get('status'), '', '', ''])
                continue
            cells = []
            for fam in ('UB', 'CC', 'PS'):
                j = m[fam]['joint']
                cells.append('no ladder match' if j is None else f'{pk(j["key"])}' + (' (edge)' if j['edge'] else '')
                             + (' (boundary)' if j['boundary'] else ''))
            rows.append([pt(lab), pct(t.get('median')), pct(t.get('worst'))] + cells +
                        [FAM_NAME.get((m['union'] or {}).get('family'), '–')])
        w(table(['target', 'target median', 'target worst', 'UB joint', 'CC joint', 'PS joint', 'union'], rows))
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
        w('One H200 job per ROM mesh. $A$ = the P1 reference ROM (R384, gauss128, frozen modes, LSPG-CN-R, $\\Delta t_0$, '
          'fused full-field query). Cost of a subject $B$: median over dev6 cases of the per-case median of '
          '$t_B/\\tfrac12(t_{A1}+t_{A2})$ (stage 2, fresh triplets); speed-up $S=\\mathrm{cost}(\\mathrm{FOM})/\\mathrm{cost}(\\mathrm{ROM})$ '
          'with a paired case-bootstrap 95 % interval (descriptive of the dev6 workload). Full-field contract: both '
          'deliver the six fields on the ROM mesh.')
        w('')
        rows = [[N, v['job'], f(v['panel_ok']), ', '.join(f'{k} {f(x)}' for k, x in v['controls'].items()),
                 ', '.join(v['veto']) or '–'] for N, v in sorted(C2['meshes'].items(), key=lambda x: int(x[0]))]
        w(table(['mesh', 'job', 'panel valid', 'controls fired', 'vetoed subjects'], rows))
        w('')
        rows = []
        for k, v in sorted(C2.get('hcomp', {}).items()):
            c = C2['cells'].get(k, {})
            rows.append([k.split('|')[0] + '²', pt(v['target']), FAM_NAME.get(v['family'], v['family']), pk(v.get('matched')),
                         f(v.get('S')), f'[{f(v.get("lo"))}, {f(v.get("hi"))}]' if v.get('S') is not None else '–',
                         v.get('verdict')])
        w('**H-comp cells** (joint matches, H populations):')
        w('')
        w(table(['mesh', 'ROM', 'FOM family', 'selected comparator', '$S$', '95 % interval', 'verdict'], rows))
        w('')
        rows = []
        for k, c in sorted(C2['cells'].items()):
            N = k.split('|')[0]
            if c.get('criterion') in ('median_only', 'worst_only', 'same_mesh', 'same_mesh_unmatched') and \
                    str(c.get('target', '')).endswith('|H') and N in ('1024', '4096'):
                rows.append([N + '²', pt(c['target']), FAM_NAME.get(c['family'], c['family']), c['criterion'].replace('_', '-'),
                             pk(c.get('selected')), f(c.get('S')),
                             f'[{f(c.get("lo"))}, {f(c.get("hi"))}]' if c.get('S') is not None else c.get('status', '–')])
        if rows:
            w('**Median-only, worst-only and same-mesh comparisons** (H populations, $1024^2$ and $4096^2$; median-only and '
              'worst-only are recorded comparisons, not accuracy matches; same-mesh = FD on the ROM mesh, or PS with $N$ equal '
              'to the mesh where run):')
            w('')
            w(table(['mesh', 'ROM', 'family', 'criterion', 'comparator', '$S$', '95 % interval / status'], rows))
            w('')
        rows = []
        for k, c in sorted(C2['cells'].items()):
            N = k.split('|')[0]
            if c.get('criterion') == 'joint' and c.get('family') == 'union' and str(c.get('target', '')).endswith('|H') \
                    and c.get('seconds_fom') is not None:
                rows.append([N + '²', pt(c['target']), pk(c['selected']), f(c['seconds_rom']), f(c['seconds_fom'])])
        if rows:
            w('**Absolute prepared-input latencies** (median stage-2 seconds per query on the H200; union joint comparator):')
            w('')
            w(table(['mesh', 'ROM', 'comparator', 'ROM (s)', 'FOM (s)'], rows))
            w('')
        k2 = C2.get('k2', {})
        w(f'**K2 (governing plan): {k2.get("verdict", "–")}.** Universe complete: {f(k2.get("universe_complete"))}.')
        w('')
        rows = []
        for ident, ent in C2.get('crossover', {}).items():
            for fc in ('union|joint', 'CC|joint', 'UB|joint', 'PS|joint', 'union|same_mesh', 'CC|same_mesh', 'UB|same_mesh'):
                e = ent.get(fc)
                if e:
                    rows.append([pt(ident + '@0|X').split(' @')[0], fc.replace('|', ' / '), e['bracket_1x'], e['bracket_2x']])
        w('**Crossover brackets** (crossover populations; between actually run meshes, never interpolated):')
        w('')
        w(table(['ROM', 'family / criterion', '1×', '2×'], rows))
        w('')
        if plan:
            plot_speedup(C2, plan, PL / 'speedup_vs_mesh.png')
            w('![Union speed-up against the ROM mesh](plots/speedup_vs_mesh.png)')
            w('')
    # ---------------------------------------------------------------- provisional, retractions
    w('## What is provisional, and what changes earlier claims')
    w('')
    w('- All accuracy numbers are conditional E0 estimates; FOM–ROM ties use the per-case $2\\epsilon_{ref}$ rule.')
    w('- Every matched speed-up is an upper bound on the claimed ROM speed-up: the FOM ladder is finite, the comparator is '
      'chosen by a screening procedure, and the timing interval describes the six dev6 timing cases only.')
    w('- The earlier 2D matched speed-up (combined H5: 13.6× for LSPG-CN-R, 17.7× best arm) was measured against a '
      '**same-mesh sign-upwind** FOM matched on the worst error only. This lane\'s ladder adds the central and spectral '
      'families, coarser FOM meshes and joint median-and-worst matching; ' + ('the C2 panels below supersede that '
      'comparison where they differ.' if final_c2 else 'whether it changes that comparison is decided by the C2 panels '
      '(not yet run).'))
    w('')
    w('## Audits')
    w('')
    w(table(['file', 'verdict'], audits()))
    w('')
    w('## Glossary')
    w('')
    for k, v in [
        ('FOM', 'full-order model: a conventional finite-difference or spectral solver of the PDE.'),
        ('ROM', 'the reduced model (coordinate-network bank span, LSPG solve, off-mesh quadrature).'),
        ('E0, $R_{ST}$', 'the second-order reference solutions (central + Crank–Nicolson, three refinement levels) every error is measured against.'),
        ('$\\epsilon_{ref}$', 'per-case uncertainty of the reference (a conditional Richardson estimate); two errors closer than $2\\epsilon_{ref}$ are a tie.'),
        ('median / worst error', 'the median and the maximum of the per-case error over the eligible cases of a population.'),
        ('tie-or-beat', 'the median (worst) of $e_{FOM}-\\epsilon_{ref}$ is no larger than the median (worst) of $e_{ROM}+\\epsilon_{ref}$.'),
        ('joint match', 'a FOM configuration that ties-or-beats the ROM on both the median and the worst error.'),
        ('screening cost', 'warm wall time of one FOM query with lattice output on two dev6 cases, used only to order candidates.'),
        ('t0–t3', 'Newton/BiCGStab tolerance levels from $(10^{-10},10^{-11})$ to $(10^{-4},10^{-5})$; the calibrated level is the loosest within $10^{-6}$ of t0.'),
        ('admissible', 'calibrated, converged with complete evidence on all 38 cases, finite scores, bitwise-reproducible screening timings.'),
        ('edge / boundary', 'edge: matched at the family\'s coarsest resolution or largest step; boundary: matched at the screening-cheapest admissible member.'),
        ('H / crossover population', 'H: the ROM target\'s own case population at that mesh; crossover (X): the cases common to every mesh where the ROM configuration is qualified.'),
        ('A–B–A, stage 0/1/2', 'paired timing: reference A before and after subject B; stage 0 screens, stage 1 selects the comparator, stage 2 measures afresh for the reported speed-ups.'),
        ('$S$, speed-up', 'cost of the FOM over cost of the ROM in the same job; above 1 the ROM is faster.'),
        ('H-comp', 'the plan\'s competition hypothesis: is the ROM at least 2× faster than the matched FOM.'),
        ('K2', 'the governing plan\'s kill criterion: the speed-up over the best validated FOM is below 2× at the largest mesh for every linear setting.'),
        ('dev6, val32', 'the six development and 32 validation cases (selection population).')]:
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
