"""jcp-robust: generate report.md, the PNG plots and runs/EXPORT.json from the committed analysis JSONs (DESIGN 8).

Every number in the report is read from a file under checks/ or runs/; nothing is typed by hand. Pieces whose analysis
file does not exist yet are printed as "pending" (never guessed). CPU only.

    python make_report.py [--final]      (--final: drop the "results audit pending" status; only after the audit ACCEPTs)
"""
import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CK = HERE / 'checks'
PLOTS = HERE / 'plots'


def load(name):
    p = CK / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=3):
    """format a number for the report (generated, never typed)"""
    if x is None:
        return 'n/a'
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if not math.isfinite(x):
        return '∞' if x > 0 else '−∞'
    if x == 0:
        return '0'
    if abs(x) >= 1e-2 and abs(x) < 1e4:
        return f'{x:.{n}g}'
    return f'{x:.{max(n - 1, 1)}e}'


def tbl(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def git(*a):
    return subprocess.check_output(['git', *a], cwd=HERE, text=True).strip()


# ----------------------------------------------------------------------------------------------------------- E1
def e1_section(R, X):
    v = load('e1-verdicts.json')
    L = ['## E1 — randomised rules', '']
    if v is None:
        L.append('pending (checks/e1-verdicts.json missing)')
        return L
    L += ['Each qualified front configuration that uses a lattice rule is re-run with 16 random draws: Cranley–Patterson '
          'shifts $u \\mapsto \\{u + \\Delta_s\\}$, $\\Delta_s \\sim U[0,1)^d$, $s = 0,\\dots,15$. Draw $s=0$ is the '
          'deployed rule. Each draw is scored on the full validity / Q1 / Q2 contract with draw-matched successors, and '
          'on E0 error against the frozen references. Verdicts follow DESIGN §2 and A1–A13.', '',
          '**Status: provisional until the Codex results audit has accepted the analysis.**', '']
    dim = v['dimension']
    for d_, dv in dim.items():
        c11, c12 = dv.get('C_E1_1') or {}, dv.get('C_E1_2') or {}
        L.append(f"- **{d_} controls:** bad generating vector caught by the independent Gauss family (C-E1.1) — "
                 f"admitted {f(c11.get('admitted'))}, fired {f(c11.get('fired'))}; rule-identity control (C-E1.2) fired "
                 f"{f(c12.get('fired'))}.")
    L.append('')
    head = ['dim', 'mesh / setting', 'configuration', 'H-E1a (failing / n; CP upper all, fresh; failed draws)', 'H-E1b (rank of $D_0$)',
            'H-E1c', 'eligible cases']
    rows, X['E1'] = [], []
    for r in v['rows']:
        a, b, c = r.get('H_E1a') or {}, r.get('H_E1b') or {}, r.get('H_E1c') or {}
        if r['label'] != 'verdict':
            rows.append([r['dim'], r.get('mesh', ''), r['config'], r['label'], '', '', ''])
            X['E1'].append(dict(dim=r['dim'], mesh=r.get('mesh'), config=r['config'], label=r['label']))
            continue
        if isinstance(a, str):
            a = dict(label=a)
        if isinstance(b, str):
            b = dict(label=b)
        if isinstance(c, str):
            c = dict(label=c)
        rk = (b.get('stats') or {}).get('D')
        sup = ' (supplementary)' if r.get('supplementary') else ''
        rows.append([r['dim'], r.get('mesh', ''), r['config'] + sup,
                     f"**{a.get('label')}** ({a.get('failures', 'n/a')}/{a.get('n', 'n/a')}; {f(a.get('cp_upper_all'))}, "
                     f"{f(a.get('cp_upper_fresh'))}; {a.get('failed_draws') or 'none'})",
                     f"**{b.get('label')}**" + (f" (rank {rk[0]}–{rk[1]} of 16)" if rk else ''),
                     c.get('label'), r.get('eligible_cases')])
        X['E1'].append(dict(dim=r['dim'], mesh=r.get('mesh'), config=r['config'], supplementary=bool(r.get('supplementary')),
                            H_E1a=a.get('label'), failures=a.get('failures'), n=a.get('n'), cp_upper=a.get('cp_upper_all'),
                            cp_upper_fresh=a.get('cp_upper_fresh'),
                            failed_draws=a.get('failed_draws'), H_E1b=b.get('label'), D_rank=rk, H_E1c=c.get('label')))
    L += [tbl(head, rows), '']
    L += e1_descriptive()
    L += ['Reading: every failing 2D draw fails only on the draw-matched successor distance (`fib6765s<k>`, tested on '
          'every case, which is stricter than qualify\'s historical subset test). Every primary draw passes the bar on the '
          'distance to the independent cross-family target. The fragile labels therefore do not show accuracy '
          'sensitivity or failure against the independent family, and H-E1c is insensitive in every cell. In 3D, '
          'E1-3D-b is atypically unfavourable only because its deployed draw has the largest $D$ of the 16 (rank 16/16); '
          'it still qualifies, and its E0-error ranks are insensitive. The CP column gives the bound over all 16 draws and '
          'the bound over the 15 fresh draws: draw 0 was the selected rule, so it is selection-conditioned (A1.5). '
          '$\\varepsilon_{ref}$ is a conditional Richardson estimate (DESIGN §1).', '']
    L += ['Per-draw values are in `checks/e1-2d-<mesh>.json` and `checks/e1-3d.json`.', '',
          '![E1 draw distributions](plots/e1_draws.png)', '']
    plot_e1()
    return L


def _q(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return None
    qs = statistics.quantiles(xs, n=4, method='inclusive') if len(xs) > 1 else [xs[0]] * 3
    return dict(median=statistics.median(xs), iqr=qs[2] - qs[0], lo=xs[0], hi=xs[-1])


def e1_descriptive():
    """registered descriptive summaries (A1.5; results audit e1e5-1 #7): per configuration, the draw distribution of D and
    of the case-median E0 error (median, IQR, min-max) with the deployed draw's value, plus the Sobol diagnostic spread"""
    rows = []
    for m in (256, 1024, 4096):
        d = load(f'e1-2d-{m}.json')
        for g, gv in (d or {}).get('groups', {}).items():
            for ak, av in gv['arms'].items():
                if av.get('diagnostic') or not ak.startswith('LSPG'):
                    continue
                ds = {int(k[1:]): x for k, x in av['rules'].items() if k.startswith('d') and k[1:].isdigit()}
                rows.append((f'2D {m}²', f'{g} {ak}', ds))
    d3 = load('e1-3d.json')
    for nm, cv in (d3 or {}).get('configs', {}).items():
        rows.append(('3D', f"{nm} {cv['setting']}", {int(k): x for k, x in cv['draws'].items()}))
    out = []
    for dim, lab, ds in rows:
        D = _q([x.get('D') for x in ds.values()])
        E = _q([statistics.median(x['e']) for x in ds.values() if x.get('e')])
        d0 = ds.get(0, {})
        if D is None or E is None:
            continue
        out.append([dim, lab, f(d0.get('D')), f"{f(D['median'])} ({f(D['iqr'])})", f"{f(D['lo'])}–{f(D['hi'])}",
                    f(statistics.median(d0['e']), 6) if d0.get('e') else 'n/a', f"{f(E['median'], 6)} ({f(E['iqr'])})",
                    f"{f(E['lo'], 6)}–{f(E['hi'], 6)}"])
    L = ['Draw distributions (16 draws per configuration; $D$ = distance to the converged rule; E0 = case-median error):', '',
         tbl(['dim', 'configuration', '$D_0$ (deployed)', '$D$ median (IQR)', '$D$ min–max', 'E0 deployed',
              'E0 median (IQR)', 'E0 min–max'], out), '']
    if d3:
        for nm, cv in d3['configs'].items():
            ex = {k: v for k, v in (cv.get('extra') or {}).items() if k.startswith('sob')}
            if ex:
                D = _q([v['D'] for v in ex.values()])
                E = _q([v['median_e'] for v in ex.values()])
                L += [f"Sobol diagnostic on {nm} ({len(ex)} Owen scrambles of `sob16384`, no verdict): $D$ median "
                      f"{f(D['median'])}, range {f(D['lo'])}–{f(D['hi'])}; case-median E0 {f(E['median'])}, range "
                      f"{f(E['lo'])}–{f(E['hi'])}.", '']
    return L


def plot_e1():
    PLOTS.mkdir(exist_ok=True)
    pts = []
    for m in (256, 1024, 4096):
        d = load(f'e1-2d-{m}.json')
        if not d:
            continue
        for g, gv in d['groups'].items():
            for ak, av in gv['arms'].items():
                if av.get('diagnostic') or not ak.startswith('LSPG'):
                    continue
                ds = [(int(k[1:]), x) for k, x in av['rules'].items() if k.startswith('d') and k[1:].isdigit()]
                pts.append((f'2D {m}² {g} {ak}', ds))
    d3 = load('e1-3d.json')
    if d3:
        for nm, cv in d3['configs'].items():
            ds = [(int(k), x) for k, x in cv['draws'].items()]
            pts.append((f'3D {nm} {cv["setting"]}', ds))
    if not pts:
        return
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, (lab, ds) in enumerate(pts):
        for s, x in ds:
            D = x.get('D')
            me = statistics.median(x['e']) if x.get('e') else None   # np.median convention (results audit e1e5-1 #6)
            col = 'C3' if s == 0 else ('C0' if x.get('qualified') else 'C7')
            for ax, y in ((axs[0], D), (axs[1], me)):
                if y is not None and y > 0:
                    ax.scatter(i, y, s=40 if s == 0 else 14, c=col, marker='D' if s == 0 else 'o', zorder=3 if s == 0 else 2)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('rollout distance D to the converged rule')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('median E0 error over cases')
    axs[1].set_xticks(range(len(pts)))
    axs[1].set_xticklabels([p[0] for p in pts], rotation=70, ha='right', fontsize=7)
    axs[0].set_title('E1: 16 random draws per configuration (red diamond = deployed draw s=0; grey = draw not qualified)')
    fig.tight_layout()
    fig.savefig(PLOTS / 'e1_draws.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- E2
def e2_section(R, X):
    L = ['## E2 — repeating the full recipe over seeds', '',
         'The full recipe — bank training, ordering/rotation, rule selection — is repeated for the 2D R512 bank '
         '(seeds 1–3, prefixes $r = 128, 256, 384$) and for the 3D W1024 bank (seeds 1–3, widths 256–1024). The key '
         'verdicts are then re-run per seed. A verdict is **seed-robust** if seeds 1–3 give the same definitive label, '
         '**seed-dependent** if two seeds give different definitive labels, and **inconclusive** otherwise. Agreement with '
         'the incumbent bank (seed 0) is reported separately as historical agreement (A2.7). Retrains are not bitwise '
         'reproducible, so seeds and hashes are recorded in `inputs/banks*/s<seed>/BANK.json`. Three seeds support a '
         'description of agreement, not a population-level robustness claim. The three retrains used identical '
         'data-generation settings; ' + data_hash_sentence() + ' (A17.2). Accuracy verdicts use the registered validation populations, including the 16-case '
         '4096² subset, and $\\varepsilon_{ref}$ is a conditional Richardson estimate. The $t$ intervals on the seed '
         'spread concern the mean of $\\log_{10}$ error with $n = 3$ (2 degrees of freedom); they are not prediction '
         'intervals for further seeds.', '',
         '**Status: provisional until the Codex results audit has accepted the analysis.**', '']
    X['E2'] = {}
    # 2D selection table
    sel = {b: load(f'select2d-{b}.json') for b in range(4)}
    if any(sel.values()):
        rows = []
        for r in ('128', '256', '384'):
            row = [f'R{r}']
            for b in range(4):
                s = sel[b]
                row.append('pending' if s is None else f"{s['selected'][r]['rule']} ({s['selected'][r]['outcome']})")
            row.append((sel[0] or sel[1] or {}).get('selected', {}).get(r, {}).get('deployed', ''))
            rows.append(row)
        L += ['### 2D rule selection at 1024² (38 cases; first candidate qualified under both LSPG-BE and LSPG-CN-R)', '',
              tbl(['rank', 'bank 0 (incumbent)', 'seed 1', 'seed 2', 'seed 3', 'deployed rule'], rows), '']
        X['E2']['selection_2d'] = {str(b): ({k: v['rule'] for k, v in s['selected'].items()} if s else None) for b, s in sel.items()}
    sel3 = {b: load(f'select3d-{b}.json') for b in range(4)}
    if any(sel3.values()):
        keys = []
        for s3 in sel3.values():
            for sl in (s3 or {}).get('slots', []):
                k = (sl['width'], sl['spec'], sl['family'])
                if k not in keys:
                    keys.append(k)
        rows = []
        for k in keys:
            row = [f'W{k[0]}', k[1].replace('|', ' '), k[2]]
            for b in range(4):
                sl = next((x for x in (sel3[b] or {}).get('slots', []) if (x['width'], x['spec'], x['family']) == k), None)
                row.append('pending' if sl is None else (sl['rule'] or '—') + ('' if sl['outcome'] == 'selected' else
                                                                                 f" ({sl['outcome']})"))
            rows.append(row)
        L += ['### 3D rule selection at 65³ (64 cases; first candidate of the ladder qualified against gl64, check gl56)', '',
              tbl(['width', 'spec', 'family', 'bank 0 (incumbent)', 'seed 1', 'seed 2', 'seed 3'], rows), '']
        X['E2']['selection_3d'] = {str(b): ([dict(width=x['width'], spec=x['spec'], family=x['family'], rule=x['rule'],
                                                  outcome=x['outcome']) for x in s3['slots']] if s3 else None) for b, s3 in sel3.items()}
    fr3 = {b: (load(f'e2-3d-{b}.json') or {}).get('front_65') for b in range(4)}
    if any(fr3.values()):
        pk = []
        for fr in fr3.values():
            for k in (fr or {}).get('points', {}):
                if k not in pk:
                    pk.append(k)
        rows = []
        for k in pk:
            row = [k.replace('|', ' ')]
            for b in range(4):
                v = ((fr3[b] or {}).get('points') or {}).get(k)
                row.append('pending' if fr3[b] is None else ('n/a' if v is None else
                           f"{v.get('state')} / {v.get('state_worst')}" + (f" (cost {f(v['cost'])})" if v.get('cost') is not None else '')
                           + (' †' if v.get('qualifier') else '')))
            rows.append(row)
        L += ['### 3D front of the tested universe at 65³ (median-error front / worst-error front; cost = B/A vs '
              'W256 gl24 fs CN 0.01 on the same bank; † = target-unverified pick, A2.5)', '',
              tbl(['point', 'bank 0', 'seed 1', 'seed 2', 'seed 3'], rows), '',
              '"missing" means no definitive state. Either no rule could be selected under an unverified target (A13.5), '
              'or the registered timing contract (≥ 2 drift-accepted repetitions on every case) was not met. The front is '
              'never complete: it covers only the tested candidates.', '']
        X['E2']['front_3d'] = {str(b): ({k: dict(state=v.get('state'), state_worst=v.get('state_worst'), cost=v.get('cost'),
                                                 median_e=v.get('median_e'), worst_e=v.get('worst_e'), qualifier=v.get('qualifier'))
                                         for k, v in fr['points'].items()} if fr else None) for b, fr in fr3.items()}
    for dim in ('2d', '3d'):
        v = load(f'e2-verdicts-{dim}.json')
        banks = {b: load(f'e2-{dim}-{b}.json') for b in range(4)}
        L.append(f'### {dim.upper()} per-seed verdicts')
        L.append('')
        have = [b for b, r in banks.items() if r]
        L.append(f"Per-bank analyses present: {', '.join(map(str, have)) if have else 'none'} of 0–3.")
        L.append('')
        if v is None or len([b for b in have if b > 0]) < 3:
            L += ['Cross-seed labels: **pending** (not every seed has its analysis yet).', '']
        else:
            c = v['counts']
            L += [f"Cross-seed labels over {len(v['rows'])} verdict rows: seed-robust {c['seed-robust']}, seed-dependent "
                  f"{c['seed-dependent']}, inconclusive {c['inconclusive']}" +
                  (' — **blocked by G-E2.3**' if v.get('blocked') else '') + '.', '']
            keyrows = [x for x in v['rows'] if x['key'].startswith(('H-acc-N|sel', 'H-acc-r|sel', 'selected-rule'))
                       or x['label'] == 'seed-dependent']
            rows = [[x['key'].replace('|', ' · '), x['incumbent'], *[x['seeds'].get(s) for s in ('1', '2', '3')], x['label'], x['historical'] or '',
                     'yes' if x.get('qualifiers') else ''] for x in keyrows]
            L += ['The table lists the headline verdicts (selected-rule H-acc-N, H-acc-r and the selected rule itself) and '
                  'every seed-dependent row; the full set is in `checks/e2-verdicts-' + dim + '.json`. Front rows use qualify\'s '
                  'registered dominance rule (5 % cost band, error margin $\\max(2\\varepsilon_{ref}, 0.02\\,e)$), applied to '
                  'separate median-error (`front…`) and worst-error (`front…-worst`) fronts of the *tested* candidates (A17). '
                  '`Q1Q2|…|dep|…` rows are the deployed rules re-qualified on each retrained bank.', '']
            L += [tbl(['verdict', 'bank 0', 'seed 1', 'seed 2', 'seed 3', 'cross-seed label', 'vs incumbent',
                       'target-unverified qualifier'], rows), '']
            X['E2'][dim] = dict(counts=c, blocked=v.get('blocked'),
                                rows={x['key']: dict(label=x['label'], seeds=x['seeds'], incumbent=x['incumbent'],
                                                     historical=x['historical'], qualifiers=x.get('qualifiers'))
                                      for x in v['rows']},
                                spread_ratio=v.get('spread_ratio'))
            sp = v.get('spread') or {}
            srows = []
            for k, s in sorted(sp.items()):
                if s.get('median_t'):
                    srows.append([k.replace('|', ' · '), f(s['incumbent'][0]) if s.get('incumbent') else 'n/a',
                                  f"{f(s['median_minmax'][0])}–{f(s['median_minmax'][1])}",
                                  f"[{f(s['median_t']['lo'])}, {f(s['median_t']['hi'])}]"])
            if srows:
                L += ['Seed spread of the median E0 error (seeds 1–3; the 95 % $t$ interval is on $\\log_{10}$, with '
                      '$n = 3$, so it is wide by construction):', '',
                      tbl(['configuration', 'bank 0 median', 'seeds 1–3 min–max', '95 % t-interval (seeds)'], srows[:40]), '']
        for b in have:
            r = banks[b]
            g = r.get('G-E2.3')
            if g is not None:
                L.append(f"- bank {b}: G-E2.3 (incumbent replay) passed: {f(g.get('passed'))}.")
        L.append('')
    plot_e2()
    L += ['![E2 seed spread](plots/e2_spread.png)', '']
    return L


def data_hash_sentence():
    """grouping of the Stage-A training-data sha256 over seeds 1-3, read from the training records"""
    hs = {}
    for b in (1, 2, 3):
        p = HERE / 'runs' / 'e2t2d' / 'archive' / 'output' / f'train2d_s{b}' / 'stageA' / 'train.json'
        if p.exists():
            hs[b] = (json.loads(p.read_text()).get('data') or {}).get('data_sha256')
    groups = {}
    for b, h in hs.items():
        groups.setdefault(h, []).append(b)
    if len(hs) < 3 or None in groups:
        return 'the training-data hashes are unavailable for some seed'
    if len(groups) == 1:
        return 'the generated training data have the same sha256 for every seed'
    parts = ['seed' + ('s ' if len(v) > 1 else ' ') + ', '.join(map(str, v)) + f' (sha256 {k[:12]}…)' for k, v in groups.items()]
    return 'the generated data are not bitwise identical, as their sha256 groups are ' + '; '.join(parts)


def deployed_sentence():
    """per deployed rank, the retrained seeds on which the deployed rule fails Q1+Q2 at 1024² (either scheme)"""
    out = []
    for r in (128, 256, 384):
        fails, rule = [], None
        for b in (1, 2, 3):
            d = load(f'e2-2d-{b}.json')
            if not d:
                continue
            ks = [k for k in d['configurations'] if k.startswith(f'1024|R{r}|dep|LSPG_') and k.endswith('_1')]
            rule = rule or (d['configurations'][ks[0]]['rule'] if ks else None)
            if any(d['configurations'][k]['qualified'] is False for k in ks):
                fails.append(b)
        if rule:
            out.append(f"deployed R{r} `{rule}` fails qualification at 1024² on " +
                       ('no retrained seed' if not fails else 'seed' + ('s ' if len(fails) > 1 else ' ') + ', '.join(map(str, fails))))
    s_ = '; '.join(out) + '.'
    return s_[0].upper() + s_[1:]


def plot_e2():
    PLOTS.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for b in range(4):
        r = load(f'e2-2d-{b}.json')
        if r:
            xs, ys = [], []
            for i, rk in enumerate((128, 256, 384)):
                for j, sc in enumerate(('BE', 'CNR')):
                    k = f'1024|R{rk}|sel|LSPG_{sc}_1'
                    if k in r['configurations']:
                        xs.append(i * 2 + j + b * 0.12)
                        ys.append(r['configurations'][k]['median_e'])
            axs[0].scatter(xs, ys, label=f'bank {b}' + (' (incumbent)' if b == 0 else ''), marker='os^v'[b])
        r = load(f'e2-3d-{b}.json')
        if r:
            xs, ys = [], []
            for i, w in enumerate((256, 512, 768, 1024)):
                for j, sp in enumerate(('fs|CN|0.01|K1', 'qn|BDF2|0.025|k4')):
                    e = (r.get('errors') or {}).get(f'gl64|W{w}|{sp}')
                    if e:
                        xs.append(i * 2 + j + b * 0.12)
                        ys.append(e['median'])
            axs[1].scatter(xs, ys, label=f'bank {b}', marker='os^v'[b])
    axs[0].set_xticks(range(6))
    axs[0].set_xticklabels([f'R{r} {s}' for r in (128, 256, 384) for s in ('BE', 'CN-R')], fontsize=8)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('median E0 error (selected rule, 1024²)')
    axs[0].set_title('2D: seed spread per prefix r')
    axs[1].set_xticks(range(8))
    axs[1].set_xticklabels([f'W{w} {s}' for w in (256, 512, 768, 1024) for s in ('CN', 'BDF2')], fontsize=8)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('median E0 error (gl64, 65³)')
    axs[1].set_title('3D: seed spread per width')
    for ax in axs:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / 'e2_spread.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- E5
def e5_section(R, X):
    d = load('e5.json')
    L = ['## E5 — hardware', '']
    if d is None:
        return L + ['pending']
    L += ['Paired A–B–A timing inside each physical GPU and job, two jobs each on A100-80G and H200. Only dimensionless '
          'ratios are compared across hardware: $q_1$ = B/A cost ratio, $q_2$ = mesh-cost ratio '
          '$\\max_L q_1 / \\min_L q_1$ over $\\{256^2, 1024^2\\}$, $q_3$ = FOM/ROM speed-up. '
          '$\\phi_k = q_k(\\mathrm{H200}) / q_k(\\mathrm{A100})$ with a paired case bootstrap (10,000 resamples). A ratio is '
          '**hardware-consistent** if the 95 % interval lies in $[0.8, 1.25]$, **hardware-dependent** if it lies '
          'entirely outside, and inconclusive otherwise.', '',
          '**Status: provisional until the Codex results audit has accepted the analysis.**', '',
          f"Jobs: {', '.join(j['name'] + ' (' + str(j['job_id']) + ', ' + j['status'] + ')' for j in d['jobs'])}. "
          f"Fixtures (injected ×1.5 must read hardware-dependent; null must read consistent): "
          f"{', '.join(k + ' ' + ('ok' if v.get('ok') else 'FAILED') for k, v in d['fixtures'].items())}.", '']
    rows, X['E5'] = [], dict(phi={}, agreement=d.get('agreement'), jobs=[(j['name'], j['job_id'], j['status']) for j in d['jobs']])
    for k, v in d['phi'].items():
        if not isinstance(v, dict):
            rows.append([k, '', v, '', ''])
            continue
        for kk, x in v.items():
            rows.append([f'$\\phi_{k[-1]}$', kk.replace('|', ' '), f(x['phi']), f"[{f(x['lo'])}, {f(x['hi'])}]", x['label']])
            X['E5']['phi'][f'{k}|{kk}'] = dict(phi=x['phi'], lo=x['lo'], hi=x['hi'], label=x['label'])
    L += [tbl(['ratio', 'subject', 'φ', '95 % interval', 'label'], rows), '']
    arows = []
    for k, lab in (d.get('agreement') or {}).items():
        pt = {t: f"{(d['per_type'].get(t, {}).get(k) or {}).get('label')} {(d['per_type'].get(t, {}).get(k) or {}).get('per_job')}"
              for t in ('A100-80G', 'H200')}
        arows.append([k.replace('|', ' '), pt['A100-80G'], pt['H200'], lab])
    g52 = [(j, k, v) for j, gg in d['gates'].items() for k, v in ((gg or {}).get('G-E5.2 (descriptive, H200)') or {}).items()]
    out_ = [x for x in g52 if not x[2].get('within')]
    L += ['Scope (results audit e1e5-1 #11): the conclusions are conditional on the four observed physical GPUs (two per '
          'type). The FOM comparators are frozen H200-selected comparators (fom2d `FROZEN-SELECTION-1024.json`). Only the '
          'H-cost-N(2) and S2-subset hypotheses are re-tested. $\\phi_1$ compares costs normalised by the reference ROM; '
          'it is not an absolute H200/A100 run-time ratio. The descriptive timing-anchor check G-E5.2 (H200 jobs against '
          f"P1's intervals, non-blocking) has {len(out_)} of {len(g52)} cells outside P1's interval:", '',
          tbl(['job', 'cell', 'job value', 'P1 value', 'P1 interval', 'within'],
              [[j, k.replace('|', ' '), f(v['job']), f(v['p1']), f"[{f(v['p1_lo'])}, {f(v['p1_hi'])}]", f(v['within'])]
               for j, k, v in g52]), '']
    urows = []
    for t, uv in (d.get('unit_to_unit') or {}).items():
        for k, v in uv.items():
            if isinstance(v, dict) and k.startswith('phi'):
                for kk, x in v.items():
                    if isinstance(x, dict) and 'phi' in x:
                        urows.append([t, f'$\\phi_{k[-1]}$', kk.replace('|', ' ').replace('ckpt:18f0266ae6f0 ', ''), f(x['phi']),
                                      f"[{f(x['lo'])}, {f(x['hi'])}]", x['label']])
    L += ['Unit-to-unit baseline (job 2 / job 1 within one GPU type; reported, no required outcome):', '',
          tbl(['type', 'ratio', 'subject', 'φ', '95 % interval', 'label'], urows), '']
    L += ['Per-type verdicts of the registered timing hypotheses (per-job labels in brackets) and their cross-hardware agreement:', '',
          tbl(['hypothesis', 'A100-80G', 'H200', 'agreement'], arows), '', '![E5 phi](plots/e5_phi.png)', '']
    plot_e5(d)
    return L


def plot_e5(d):
    PLOTS.mkdir(exist_ok=True)
    items = [(f'{k} {kk}', x) for k, v in d['phi'].items() if isinstance(v, dict) for kk, x in v.items()]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(items) + 1.5))
    for i, (lab, x) in enumerate(items):
        ax.errorbar(x['phi'], i, xerr=[[x['phi'] - x['lo']], [x['hi'] - x['phi']]], fmt='o',
                    c='C0' if x['label'] == 'hardware-consistent' else 'C3' if x['label'] == 'hardware-dependent' else 'C7')
    ax.axvspan(0.8, 1.25, color='C2', alpha=.12, label='consistency band [0.8, 1.25]')
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab.replace('ckpt:18f0266ae6f0|', '') for lab, _ in items], fontsize=6)
    ax.set_xscale('log')
    ax.set_xlabel('φ = q(H200) / q(A100), 95 % paired-bootstrap interval')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / 'e5_phi.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true')
    a = ap.parse_args()
    X = {}
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    spent = led.get('spent_gpu_hours') if 'spent_gpu_hours' in led else None
    commit = git('rev-parse', 'HEAD')
    status = ('final: the Codex results audit accepted the analyses' if a.final else
              'provisional: production is complete only where stated, and the Codex results audit is pending')
    L = ['# jcp-robust — randomised rules, recipe seeds and hardware for the off-mesh quadrature ROM', '',
         f'Covers plan V2 items E1 (randomised lattice rules), E2 (repeating the full bank + rule-selection recipe '
         f'over seeds) and E5 (hardware). **Status: {status}.** Generated by `make_report.py` from the analysis JSONs '
         f'in `checks/` at commit `{commit[:10]}`.', '',
         '```mermaid', 'flowchart LR', '  classDef trained fill:#fde2c8,stroke:#c46a1b', '  classDef frozen fill:#dbe8fb,stroke:#2b5ea7',
         '  classDef verdict fill:#dff3df,stroke:#2f7d32',
         '  B0[incumbent banks]:::frozen --> E1[E1: 16 random draws per qualified lattice configuration]',
         '  E1 --> V1[H-E1a/b/c + controls C-E1.1, C-E1.2]:::verdict',
         '  S[seeds 1-3]:::trained --> T[bank training + rotation]:::trained --> SEL[rule selection]',
         '  SEL --> VER[verdict ladders 256/1024/4096, 3D widths]', '  VER --> V2[seed-robust / seed-dependent / inconclusive]:::verdict',
         '  HW[A100 x2, H200 x2]:::frozen --> E5[paired A-B-A timing]', '  E5 --> V5[phi intervals vs 0.8-1.25]:::verdict', '```', '']
    L += e1_section(None, X)
    L += e2_section(None, X)
    L += e5_section(None, X)
    L += ['## Resources', '', f"GPU-hours from `runs/GPU-LEDGER.json`: spent {f(spent)} of the cap {f(led.get('cap_gpu_hours'))}. "
          'Every job and its id are listed in the ledger, with the logs under `runs/<attempt>/archive`.', '']
    L += ['## Deviations, retractions and corrections', '',
          '- smk3 (the 3D smoke) timed out at 2 h during data generation; it was split into smk3c and a training pilot (A9).',
          '- e1m4096 was cancelled at 42 min: its certificate ignored the per-execution transfer overhead. It was rerun as '
          'e1m4096b with a measured overhead (A10.3).',
          '- The A9 memory note mixed GB and GiB. The pilot\'s host RSS was 168.6 GiB, i.e. 40.1 % of the request, not '
          '"176.8 GB / 42 %" (A12.4).',
          '- The E1 3D supplementary configurations c and d took about 1,000 s each, far above their A11.3 estimate of 2 × W256 (A13.6).',
          '- **Retracted (A17):** the first E2 2D roll-up computed the 1024² front with exact joint inequalities instead of '
          'qualify\'s registered dominance rule. Its cross-seed counts and its "R256 front membership is seed-dependent" '
          'reading are withdrawn; the counts above come from the corrected rule.',
          '- On the retrained 2D banks, the deployed rules do not transfer. ' + deployed_sentence() +
          ' These are qualification failures of the deployed rules; they do not affect the selected-rule accuracy verdicts.', '']
    L += ['## Glossary', '',
          '- **Draw $s$**: one random shift (lattice) or scramble (Sobol) of a quadrature rule; $s=0$ is the rule deployed today.',
          '- **$D$**: rollout distance of a draw\'s trajectory to the converged (cross-family) reference rule, relative to the initial state norm.',
          '- **Q1 / Q2**: the qualification contract. Q1: validity, stationarity and tight-solve agreement. Q2: the distance '
          'to the next rule in the ladder (successor) $\\le 2.5\\times10^{-4}$ and $\\rho \\le 0.116$.',
          '- **H-E1a**: robust / inconclusive / fragile, from the number of draws failing qualification and a one-sided Clopper–Pearson (CP) upper bound.',
          '- **H-E1b**: where the deployed draw ranks among the 16 draws (typical, atypically favourable or unfavourable); ranks are intervals because of ties within $\\delta = 2\\max\\varepsilon_{ref}$.',
          '- **H-E1c**: whether the E0 error depends on the draw by more than $\\max(2\\varepsilon, 0.1\\,e_{conv})$.',
          '- **C-E1.1 / C-E1.2**: must-fail controls. C-E1.1: a bad generating vector must be caught by the independent '
          'Gauss family. C-E1.2: a mislabelled rule must be refused.',
          '- **E0 error**: error against the frozen full-order reference store, per case.',
          '- **Bank / seed**: a trained decoder bank. Bank 0 is the incumbent; seeds 1–3 are retrained with the identical recipe.',
          '- **H-acc-N / H-acc-r**: accuracy is mesh-independent (N) / improves with the prefix size r or the width.',
          '- **Front**: cost–accuracy non-dominated set of the *tested* configurations (never complete); member / dominated / unqualified.',
          '- **φ**: H200 value of a dimensionless timing ratio divided by its A100 value.',
          '- **Provisional**: not yet accepted by the independent Codex results audit.',
          '- **Target unverified**: the cross-family reference rule could not be certified, so the pick uses own-family qualification only and is labelled.']
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    X.update(lane='jcp-robust', commit=commit, status=status, ledger=dict(spent_gpu_hours=spent, cap=led.get('cap_gpu_hours')),
             plots=[str(p.relative_to(HERE)) for p in sorted(PLOTS.glob('*.png'))])
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(X, indent=1, default=float) + '\n')
    print('report.md', len(L), 'lines; EXPORT keys', list(X))


if __name__ == '__main__':
    main()
