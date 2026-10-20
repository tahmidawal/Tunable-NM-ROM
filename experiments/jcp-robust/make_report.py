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
    esc = lambda c: str(c).replace('|', '\\|')                  # report audit 1 #6: no raw pipes inside cells
    out += ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


FINAL = [False]
EXTRA = {}          # report audit 1 #11: material qualifications and key numbers carried into EXPORT.json


def STATUS_LINE():
    return ('**Status: final — accepted by the Codex results audits (see the audits list).**' if FINAL[0] else
            '**Status: provisional until the Codex results audit has accepted the analysis.**')


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
          STATUS_LINE(), '']
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
    L += e1_reading()
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
    EXTRA['E1_draw_distributions'] = [dict(zip(['dim', 'configuration', 'D0', 'D_median_iqr', 'D_minmax', 'E0_deployed',
                                                'E0_median_iqr', 'E0_minmax'], r)) for r in out]
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


def e1_reading():
    """the E1 reading paragraph, every claim derived from the analysis files (report audit 1 #10)"""
    BAR = 2.5e-4
    fail, succ_only, tgt_max = 0, 0, 0.
    for m in (256, 1024, 4096):
        d = load(f'e1-2d-{m}.json') or {}
        for g, gv in d.get('groups', {}).items():
            for ak, av in gv['arms'].items():
                if av.get('diagnostic') or not ak.startswith('LSPG'):
                    continue
                for k, x in av['rules'].items():
                    if not (k.startswith('d') and k[1:].isdigit()):
                        continue
                    tgt_max = max(tgt_max, x.get('D') or 0.)
                    if x.get('qualified') is False:
                        fail += 1
                        if (x.get('succ_max') or 0) > BAR and (x.get('D') or 0) <= BAR and not x.get('prod_failed'):
                            succ_only += 1
    v = load('e1-verdicts.json') or {}
    lab = lambda r: (r.get('H_E1c') or {}).get('label') if isinstance(r.get('H_E1c'), dict) else r.get('H_E1c')
    insens = all(lab(r) == 'insensitive' for r in v.get('rows', []) if r['label'] == 'verdict')
    L = [f"Reading: {succ_only} of the {fail} failing 2D draws fail only on the draw-matched successor distance "
         f"(above {f(BAR)} while the distance to the independent cross-family target stays within it). The largest "
         f"target distance over all primary 2D draws is {f(tgt_max)}. The successor test runs on every case, which is "
         "stricter than qualify's historical subset test. The fragile labels therefore concern successor-rule convergence, "
         "not accuracy against the independent family; H-E1c is " + ('insensitive in every verdict cell.' if insens else
                                                                       'not insensitive in every cell (see the table).')]
    d3 = load('e1-3d.json') or {}
    for nm, cv in d3.get('configs', {}).items():
        b_ = (cv.get('verdicts') or {}).get('H_E1b') or {}
        if isinstance(b_, dict) and str(b_.get('label', '')).startswith('atypically'):
            st = b_.get('stats', {})
            q0 = (cv['draws'].get('0') or {}).get('qualified')
            dr = st.get('D') or ['?', '?']
            L.append(f"In 3D, {nm} is {b_['label']} through the rank of $D_0$ ({dr[0]}–{dr[1]} of {len(cv['draws'])}); its "
                     f"E0-error ranks are {(st.get('median_e') or [None, None, '?'])[2]} (median) and "
                     f"{(st.get('worst_e') or [None, None, '?'])[2]} (worst), and the deployed draw "
                     f"{'still qualifies' if q0 else 'does not qualify'}.")
    sup = [c['name'] for c in d3.get('config_supplementary', [])]
    g12 = {nm: ((cv.get('controls') or {}).get('G-E1.2') or {}).get('passed') for nm, cv in d3.get('configs', {}).items()}
    L.append('The CP column gives the bound over all 16 draws and over the 15 fresh draws: draw 0 was the selected rule, so it '
             'is selection-conditioned (A1.5). $\\varepsilon_{ref}$ is a conditional Richardson estimate (DESIGN §1). '
             f"Supplementary 3D configurations ({', '.join(sup) or 'none'}) have no archived historical run, so their "
             f"replay gate G-E1.2 is unavailable ({', '.join(f'{k}: {g12.get(k)}' for k in sup)}); they are reported apart "
             'from the registered population.')
    return [' '.join(L), '']


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
            sup_ = nm in {c['name'] for c in d3.get('config_supplementary', [])}
            pts.append((f'3D {nm} {cv["setting"]}' + (' (supplementary)' if sup_ else ''), ds))
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
    axs[0].set_title('E1: random draws per configuration (red diamond = deployed draw s=0; grey = draw not qualified)', fontsize=9)
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
         'description of agreement, not a population-level robustness claim. The three 2D retrains used identical '
         'data-generation settings; for the 2D Stage-A training data, ' + data_hash_sentence() + ' (A17.2). Accuracy verdicts use the registered validation populations, including the 16-case '
         '4096² subset, and $\\varepsilon_{ref}$ is a conditional Richardson estimate. The $t$ intervals on the seed '
         'spread concern the mean of $\\log_{10}$ error with $n = 3$ (2 degrees of freedom); they are not prediction '
         'intervals for further seeds.', '',
         STATUS_LINE(), '']
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
        if sel[0]:
            g21 = {r: v for r, v in sel[0]['selected'].items()}
            mis = [f"R{r} (selects {v['rule']}, deployed {v['deployed']})" for r, v in g21.items() if not v.get('equals_deployed')]
            L += ['G-E2.1 (non-blocking anchor, incumbent rule recovery): the procedure recovers the deployed rule at ' +
                  ', '.join(f'R{r}' for r, v in g21.items() if v.get('equals_deployed')) + '; it does not at ' +
                  (', '.join(mis) if mis else 'no rank') + '. This anchor never blocks (DESIGN A2.4).', '']
            X['E2']['G_E2_1'] = {r: dict(selected=v['rule'], deployed=v['deployed'], recovered=bool(v.get('equals_deployed')))
                                 for r, v in g21.items()}
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
                lab_ = sl and (sl['outcome'] if not (sl['outcome'] == 'none within ladder' and sl.get('target_certified') is not True)
                               else 'none within ladder, target unverified: not a definitive absence')
                row.append('pending' if sl is None else (sl['rule'] or '—') + ('' if sl['outcome'] == 'selected' else f' ({lab_})'))
            rows.append(row)
        cert = {w: [b for b in range(4) for x in (sel3[b] or {}).get('slots', []) if x['width'] == w and x['outcome'] == 'selected']
                for w in (768, 1024)}
        L += ['### 3D rule selection at 65³ (64 cases; first candidate of the ladder qualified against gl64, check gl56)', '',
              tbl(['width', 'spec', 'family', 'bank 0 (incumbent)', 'seed 1', 'seed 2', 'seed 3'], rows), '',
              f"Wide widths: certified W768 selections on banks {sorted(set(cert[768])) or 'none'}, certified W1024 "
              f"selections on banks {sorted(set(cert[1024])) or 'none'}. Every other wide-width pick is a diagnostic pick "
              '(target unverified) or no qualified rung. The wide-width selected-rule *selection* verdicts are therefore '
              'inconclusive rather than seed-dependent. Wide-width front rows that do carry a seed-dependent label are built on '
              'target-unverified picks and keep the † qualifier: they compare diagnostic points, not certified selections.', '']
        L += g_e22(sel3[0])
        X['E2']['selection_3d'] = {str(b): ([dict(width=x['width'], spec=x['spec'], family=x['family'], rule=x['rule'],
                                                  outcome=x['outcome'], target_certified=x.get('target_certified'))
                                             for x in s3['slots']] if s3 else None) for b, s3 in sel3.items()}
    R3 = {b: load(f'e2-3d-{b}.json') for b in range(4)}
    if all(R3.values()):
        rows = []
        for sp in ('fs|CN|0.01|K1', 'qn|BDF2|0.025|k4'):
            for tr in ('256->512', '512->768', '768->1024', 'range_256_1024'):
                cells = []
                for b in range(4):
                    x = R3[b]['h_acc_r'].get(f'gl64|{sp}', {}).get(tr)
                    cells.append(x.get('outcome') if isinstance(x, dict) else x)
                rows.append([sp.replace('|', ' '), tr, *cells])
        v3 = load('e2-verdicts-3d.json') or {}
        sr = v3.get('spread_ratio') or {}
        L += ['### 3D accuracy vs width on the fixed rule gl64 (H-acc-r, isolates the bank; Q2 against gl80 on cases 0–7)', '',
              tbl(['spec', 'transition', 'bank 0', 'seed 1', 'seed 2', 'seed 3'], rows), '',
              'Spread ratio (cross-seed spread of the lower width\'s median E0 error / median |paired change| of the transition; '
              '< 1 means the seed spread is small against the width step):', '',
              tbl(['spec', 'transition', 'spread ratio'], [[k.split('|', 1)[1].rsplit('|', 1)[0].replace('|', ' '), k.rsplit('|', 1)[1],
                                                           f(v) if not isinstance(v, str) else v] for k, v in sorted(sr.items())]), '']
        from e2verdicts import tint
        rows = []
        for sp in ('fs|CN|0.01|K1', 'qn|BDF2|0.025|k4'):
            for w in (256, 512, 768, 1024):
                es = [R3[b]['errors'].get(f'gl64|W{w}|{sp}') for b in range(4)]
                if not all(es):
                    continue
                for stat in ('median', 'worst'):
                    vals = [e[stat] for e in es[1:]]
                    t = tint(vals)
                    rows.append([sp.replace('|', ' '), f'W{w}', stat, f(es[0][stat]), f'{f(min(vals))}–{f(max(vals))}',
                                 f"[{f(t['lo'])}, {f(t['hi'])}]" if t else 'n/a'])
        span = {}
        for sp in ('fs|CN|0.01|K1', 'qn|BDF2|0.025|k4'):
            for w in (256, 512, 768, 1024):
                es = [R3[b]['errors'].get(f'gl64|W{w}|{sp}') for b in (1, 2, 3)]
                if all(es):
                    for stat in ('median', 'worst'):
                        vs = [e[stat] for e in es]
                        span.setdefault(stat, []).append((max(vs) / min(vs), f"{sp.split('|')[1]} W{w}"))
        mx = {k: max(v) for k, v in span.items()}
        L += ['Seed spread of the E0 error on the fixed rule gl64 (median and worst case over the 64 cases; the $t$ interval is on '
              'the mean of $\\log_{10}$ error over seeds 1–3, $n = 3$, wide by construction). Largest max/min ratio over seeds 1–3: '
              f"median error {f(mx['median'][0], 4)} ({mx['median'][1]}), worst-case error {f(mx['worst'][0], 4)} ({mx['worst'][1]}).", '',
              tbl(['spec', 'width', 'statistic', 'bank 0', 'seeds 1–3 min–max', '95 % t-interval (seeds)'], rows), '']
        X['E2']['gl64_3d'] = dict(h_acc_r={str(b): R3[b]['h_acc_r'] for b in range(4) if R3[b]},
                                  errors={str(b): R3[b]['errors'] for b in range(4)}, spread_ratio=sr)
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
                                spread_ratio=v.get('spread_ratio'), spread=v.get('spread'))
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
                      tbl(['configuration', 'bank 0 median', 'seeds 1–3 min–max', '95 % t-interval (seeds)'], srows), '']
            if dim == '2d' and v.get('spread_ratio'):
                L += ['2D spread ratio (cross-seed spread of the lower rank\'s median E0 error / median |paired change| of the '
                      'transition; > 1 means the seed spread exceeds the rank step even where every seed improves):', '',
                      tbl(['kind · mesh · scheme · transition', 'spread ratio'],
                          [[k.replace('|', ' · '), f(x) if not isinstance(x, str) else x] for k, x in sorted(v['spread_ratio'].items())]), '']
        for b in have:
            r = banks[b]
            g = r.get('G-E2.3')
            if g is not None:
                L.append(f"- bank {b}: G-E2.3 (incumbent replay) passed: {f(g.get('passed'))}.")
        L.append('')
    plot_e2()
    L += ['![E2 seed spread](plots/e2_spread.png)', '']
    return L


def g_e22(s0):
    """G-E2.2 (non-blocking, DESIGN 3.3): the incumbent's 3D picks vs jcp-dims' m* table for the same cells
    (m = quadrature points: gl n -> n^3, lat N -> N)"""
    p = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/runs/EXPORT.json')
    if not (s0 and p.exists()):
        return ['G-E2.2 (incumbent vs jcp-dims m*): unavailable.', '']
    W = json.loads(p.read_text()).get('width_mstar_3d') or {}
    pts = lambda r: (int(r[2:]) ** 3 if r.startswith('gl') else int(r[3:])) if r else None
    rows, agree, absent = [], [], []
    for sl in s0['slots']:
        d = ((W.get(f"W{sl['width']}|{sl['family']}") or {}).get('per_spec') or {}).get(sl['spec'])
        if d is None:
            absent.append(f"W{sl['width']} {sl['spec'].replace('|', ' ')} {sl['family']}")
            continue
        mine, theirs = pts(sl['rule']), d.get('m')
        ok = mine == theirs
        agree.append(ok)
        unv = sl.get('target_certified') is not True
        if not ok:
            lab = 'differs'
        elif mine is None:
            lab = 'both none (unverified absence: target not certified)' if unv else 'both none (certified absence)'
        else:
            lab = 'same m (pick target-unverified: not a certified agreement)' if unv else 'same m (certified)'
        rows.append([f"W{sl['width']}", sl['spec'].replace('|', ' '), sl['family'], f"{sl['rule'] or 'none'} ({sl['outcome']})",
                     d.get('outcome'), lab])
    import collections
    cnt = collections.Counter(r[-1] for r in rows)
    EXTRA['G_E2_2'] = [dict(zip(['width', 'spec', 'family', 'this_lane', 'dims_mstar', 'label'], r)) for r in rows]
    return [f'G-E2.2 (non-blocking anchor): the incumbent\'s picks against jcp-dims\' $m^*$ table for the same cells. Of '
            f"{len(rows)} compared cells: " + ', '.join(f'{k} {v}' for k, v in cnt.items()) + '. Cells without a dims entry: '
            f"{', '.join(absent) if absent else 'none'}.", '',
            tbl(['width', 'spec', 'family', 'this lane (incumbent)', 'jcp-dims m*', 'G-E2.2'], rows), '']


def summary():
    """headline counts, generated from the verdict files"""
    import collections
    D = lambda c: ', '.join(f'{k} {v}' for k, v in c.items())
    L = ['## Summary', '']
    v = load('e1-verdicts.json')
    if v:
        for d_ in ('2D', '3D'):
            rs = [r for r in v['rows'] if r['dim'] == d_ and r['label'] == 'verdict' and not r.get('supplementary')]
            ca = collections.Counter((r['H_E1a']['label'] if isinstance(r['H_E1a'], dict) else r['H_E1a']) for r in rs)
            cb = collections.Counter((r['H_E1b']['label'] if isinstance(r['H_E1b'], dict) else r['H_E1b']) for r in rs)
            cc = collections.Counter((r['H_E1c']['label'] if isinstance(r['H_E1c'], dict) else r['H_E1c']) for r in rs)
            L.append(f"- **E1 {d_}** ({len(rs)} registered verdict cells): H-E1a {D(ca)}; H-E1b {D(cb)}; H-E1c {D(cc)}.")
    for d_ in ('2d', '3d'):
        w = load(f'e2-verdicts-{d_}.json')
        if w:
            hl = [x for x in w['rows'] if x['key'].startswith(('H-acc-N|sel', 'H-acc-r|sel', 'H-acc-r|gl64'))]
            ch = collections.Counter(x['label'] for x in hl)
            sr = [x for x in w['rows'] if x['key'].startswith('selected-rule')]
            cs = collections.Counter(x['label'] for x in sr)
            L.append(f"- **E2 {d_.upper()}** ({len(w['rows'])} verdict rows, seeds 1–3): {D(w['counts'])}. Headline accuracy rows "
                     f"(H-acc-N / H-acc-r): {D(ch)}. Selected-rule rows: {D(cs)}.")
    e5 = load('e5.json')
    if e5:
        cp = collections.Counter(x['label'] for v_ in e5['phi'].values() if isinstance(v_, dict) for x in v_.values())
        ca = collections.Counter((e5.get('agreement') or {}).values())
        L.append(f"- **E5** ({len(e5['jobs'])} jobs, A100-80G and H200): φ intervals {D(cp)}; cross-hardware agreement of "
                 f"the registered verdicts {D(ca)}.")
    L += ['', 'How to read these counts:',
          '- E1 "fragile" concerns convergence to the draw-matched successor rule, not accuracy: the accuracy-sensitivity '
          'verdict H-E1c is listed beside it. The reading paragraph in the E1 section quantifies this from the data.',
          '- E2 2D: the selected-rule accuracy verdicts (H-acc-N, H-acc-r) are the seed-robust headline. The selected rule itself, '
          'and whether the *deployed* rules still qualify on a retrained bank, vary with the seed; see the E2 sections for '
          'the per-rank lists.',
          '- E2 3D: the fixed-rule gl64 rows isolate the bank; the selected-rule rows are sparse at W768 and W1024, where most '
          'slots have no certified rule, so most wide-width rows are inconclusive rather than evidence either way.',
          '- E5: the ratios are dimensionless and normalised; the conclusions are conditional on the four observed GPUs, '
          'the frozen H200-selected comparators and the two re-tested hypotheses.',
          '', 'Per-item tables, qualifications and every number follow; the plain-language glossary is at the end.', '']
    return L


def led_state(att):
    for j in json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text()).get('jobs', []):
        if j['attempt'] == att:
            return f"{j.get('state')}, {f(j.get('elapsed_hours'), 3)} h of a {f(j.get('time_limit_hours'), 3)} h limit"
    return 'not in the ledger'


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
    axs[1].set_xticklabels([f'W{w} {s}' for w in (256, 512, 768, 1024) for s in ('CN', 'BDF2')], fontsize=7, rotation=45, ha='right')
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
          STATUS_LINE(), '',
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
    EXTRA['E5_unit_to_unit'] = [dict(zip(['type', 'ratio', 'subject', 'phi', 'interval', 'label'], r)) for r in urows]
    EXTRA['E5_G_E5_2_anchors'] = [dict(v, job_label=j, cell=k) for j, k, v in g52]
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
    FINAL[0] = bool(a.final)
    X = {}
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    spent = led.get('spent_gpu_hours') if 'spent_gpu_hours' in led else None
    commit = git('rev-parse', 'HEAD')
    status = ('final: the Codex results audit accepted the analyses' if a.final else
              'provisional: production is complete and the results audits accepted with fixes; the final report audit is pending')
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
    L += summary()
    L += e1_section(None, X)
    L += e2_section(None, X)
    L += e5_section(None, X)
    jrows = []
    for j in led.get('jobs', []):
        js = HERE / 'runs' / j['attempt'] / 'JOBSPEC.json'
        gt = json.loads(js.read_text()).get('gpu') if js.exists() else 'n/a'
        gh = j.get('elapsed_hours') or 0.        # the ledger's elapsed_hours are GPU-hours (GPUs x wall, all requeues)
        jrows.append([j['attempt'], j['job_id'], gt, j.get('gpus', 1), j.get('state'), f(gh / (j.get('gpus') or 1), 3), f(gh, 3)])
    L += ['## Resources', '', f"GPU-hours from `runs/GPU-LEDGER.json`: spent {f(spent)} of the cap {f(led.get('cap_gpu_hours'))} "
          '(cap raised from 75 by the coordinator, A18). Namespace `/cluster/tufts/paralab/tawal01/jcp-robust/`. Every job '
          'ran in its own directory; the logs are under `runs/<attempt>/archive`, and raw arrays over 100 MB stay in '
          '`burgers-archive/jcp-robust/<attempt>/` with sha256 listings.', '',
          tbl(['attempt', 'job id', 'GPU', 'GPUs', 'state', 'wall h', 'GPU-h'], jrows),
          '', f"Column sum of GPU-h: {f(sum((j.get('elapsed_hours') or 0.) for j in led.get('jobs', [])), 6)}; ledger spent total: "
          f"{f(spent, 6)}.", '']
    X['jobs'] = [dict(attempt=r[0], job_id=r[1], gpu=r[2], state=r[4]) for r in jrows]
    L += ['## Deviations, retractions and corrections', '',
          f"- smk3 (the 3D smoke) hit its time limit ({led_state('smk3')}) during data generation; it was split into smk3c "
          'and a training pilot (A9).',
          f"- e1m4096 was cancelled ({led_state('e1m4096')}): its certificate ignored the per-execution transfer overhead. "
          'It was rerun as e1m4096b with a measured overhead (A10.3).',
          '- The A9 memory note mixed GB and GiB when quoting the smoke\'s host RSS; A12.4 restates it in GiB against the '
          '336 GiB acceptance ceiling (420G / 1.25).',
          f"- The certificate for e13d underestimated the E1 3D supplementary configurations c and d, whose cost was "
          f"estimated as 2 × W256 (A11.3). The job still finished inside its limit ({led_state('e13d')}) (A13.6).",
          '- Transient login-node authentication failures interrupted one submission (e2t3d3) and one staging (e2f3). '
          'Both failed before sbatch with the reservation released, and were resubmitted cleanly (A15).',
          '- **Retracted (A17):** the first E2 2D roll-up computed the 1024² front with exact joint inequalities instead of '
          'qualify\'s registered dominance rule. Its cross-seed counts and its "R256 front membership is seed-dependent" '
          'reading are withdrawn; the counts above come from the corrected rule.',
          '- On the retrained 2D banks, the deployed rules do not transfer. ' + deployed_sentence() +
          ' These are qualification failures of the deployed rules; they do not affect the selected-rule accuracy verdicts.', '']
    L += ['## Glossary', '',
          '**E1**',
          '- **Draw $s$**: one random Cranley–Patterson shift (lattice) or Owen scramble (Sobol) of a quadrature rule; $s=0$ is '
          'the rule deployed today. Rule names: `fib<n>` / `lat<n>` are rank-1 lattice rules with $n$ points, `gauss<n>` / `gl<n>` '
          'tensor Gauss rules ($n$ per dimension in 3D), `sob<n>` Sobol points; the suffix `s<k>` is draw $k$.',
          '- **$D$**: maximum over cases of the rollout distance between a draw\'s trajectory and the converged cross-family '
          'reference rule (gauss384 in 2D, gl64 in 3D), relative to the initial-state norm.',
          '- **Successor distance**: the same distance to the next, larger rule of the same family with the same draw.',
          '- **$\\rho$**: the tested-advection indicator of a rule against the converged (or successor) rule at the draw\'s own '
          'reached states (Q2 bar 0.116).',
          '- **Q1 / Q2**: the qualification contract. Q1: valid, stationary and agreeing with tight solves. Q2: successor '
          'distance $\\le 2.5\\times10^{-4}$ and $\\rho \\le 0.116$.',
          '- **H-E1a**: robust / inconclusive / fragile, from the number of draws failing Q1+Q2 and one-sided Clopper–Pearson (CP) '
          'upper bounds on the failure rate (over all 16 draws, and over the 15 fresh draws excluding the selected draw 0).',
          '- **H-E1b**: where the deployed draw ranks among the 16. Ranks are intervals because of ties: distances $D$ are ranked '
          'with tie margin $\\delta = 0$, E0 errors with $\\delta = 2\\max\\varepsilon_{ref}$. Typical, atypically favourable '
          'or atypically unfavourable.',
          '- **H-E1c**: per eligible case, $\\Delta_i = \\max_s |e_{s,i} - e_{0,i}|$ (largest departure of any draw\'s E0 error '
          'from the deployed draw\'s) against the margin $\\max(2\\varepsilon_{ref}, 0.1\\,e_{conv})$, where $e_{conv}$ is '
          'the converged rule\'s error. Insensitive if at least 90 % of eligible cases are within the margin.',
          '- **C-E1.1 / C-E1.2**: must-fail controls. C-E1.1: a bad generating vector must be caught by the independent Gauss '
          'family. C-E1.2: a mislabelled rule must be refused. **G-E1.x**: integrity gates (identity, reproduction of the '
          'archived deployed run, permuted replay).',
          '- **Eligible cases**: cases whose case-kind is rankable in the E0 reference store (non-rankable 2D case-kinds are excluded).',
          '- **IQR**: inter-quartile range (75th minus 25th percentile) over the draws.',
          '**Models, schemes and settings**',
          '- **$r$ / R128…R384**: number of leading bank modes used by the 2D ROM (prefix of the ordered R512 bank). **W256…W1024**: '
          'width of the 3D W1024 bank used. **M**: number of quadrature points. **M2_512**: a different 3D bank of width 512.',
          '- **LSPG / GAL**: least-squares Petrov–Galerkin / Galerkin projection. **BE / CN / CN-R / BDF2**: backward Euler, '
          'Crank–Nicolson, Crank–Nicolson started with two backward-Euler (Rannacher) steps, BDF2. Factors 1, 2, 2.5 multiply '
          'the base time step.',
          '- **3D solver specs** (`<solver>|<scheme>|<dt>|<K>`): `fs` = fixed-sweep LSPG solver with K cached-predictor '
          'Levenberg–Marquardt sweeps per step (K1 = one sweep); `qn` = QN-k solver, quasi-Newton iterations with a frozen '
          'Gauss–Newton matrix (k4 = at most four); `ad` = the adaptive reference solves used for the tight/tighter checks.',
          '**E2**',
          '- **Bank / seed**: a trained decoder bank. Bank 0 is the incumbent; seeds 1–3 are retrained with the identical recipe '
          'and only the model and subset seeds changed.',
          '- **H-acc-N**: the accuracy at fixed $r$ does not degrade from 256² to 1024² and 4096² (equivalence test). **H-acc-r**: '
          'accuracy improves with $r$ (2D) or with the width (3D), per transition and over the range.',
          '- **Selected rule / deployed rule**: the rule the registered selection procedure picks on a bank / the rule deployed '
          'today (from the incumbent).',
          '- **Front**: cost–accuracy non-dominated set of the *tested* qualified configurations (never complete), computed '
          'separately for median and worst-case E0 error with qualify\'s dominance rule (5 % cost band, error margin '
          '$\\max(2\\varepsilon_{ref}, 0.02\\,e)$). States: member / dominated / unqualified / absent / missing.',
          '- **Seed-robust / seed-dependent / inconclusive**: seeds 1–3 give the same definitive label / different definitive '
          'labels / anything else. "vs incumbent" is the separate historical comparison with bank 0.',
          '- **Spread ratio**: cross-seed spread of the lower step\'s median error divided by the median paired change of the step.',
          '- **$t$ interval**: 95 % Student-$t$ interval for the mean of $\\log_{10}$ error over the three seeds (2 degrees of freedom).',
          '- **Target unverified (†)**: the cross-family reference could not be certified (gl56 vs gl64 check failed), so the pick '
          'uses own-family qualification only and is labelled; a "none" under an unverified target is not a definitive absence.',
          '- **G-E2.1 / G-E2.2 / G-E2.3**: incumbent anchors: deployed-rule recovery in 2D / agreement with jcp-dims\' $m^*$ in 3D '
          '(both non-blocking) / replay of the archived incumbent runs (blocking).',
          '**E5**',
          '- **FOM**: full-order model. **CC**: the frozen same-mesh central-difference comparator (`fd|CC|L1024|…`); '
          '**PS**: the frozen sine pseudo-spectral comparator (`ps|N96|…`), both selected on H200 by jcp-fom2d. '
          '**S**: median FOM time / median ROM time. **S2-subset**: pass iff the lower 95 % bound of $S \\ge 2$, fail iff the '
          'upper bound is below 2. **H-cost-N(2)**: P1\'s mesh-cost statistic on {256², 1024²} only: pass iff the upper 95 % '
          'bound of $R = \\max/\\min$ cost is $\\le 1.25$, fail iff the lower bound exceeds 1.25.',
          '- **$q_1, q_2, q_3$**: B/A cost ratio, mesh-cost ratio, FOM/ROM speed-up. **φ**: H200 value divided by A100 value. '
          'Bootstrap interval: 95 % paired case bootstrap, 10,000 resamples.',
          '**General**',
          '- **E0 error**: per case, the maximum over output times of $\\lVert u - R\\rVert / \\lVert R(t_0)\\rVert$ against the frozen '
          'full-order reference store $R$. **$\\varepsilon_{ref}$**: the reference\'s '
          'own error, a conditional Richardson estimate.',
          '- **Resource columns**: wall h = elapsed wall-clock hours; GPU-h = GPUs × wall hours over all attempts.',
          '- **Provisional / final**: before / after acceptance by the independent Codex results audits.']
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    X.update(EXTRA)
    X.update(lane='jcp-robust', commit=commit, status=status, ledger=dict(spent_gpu_hours=spent, cap=led.get('cap_gpu_hours')),
             plots=[str(p.relative_to(HERE)) for p in sorted(PLOTS.glob('*.png'))])
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(X, indent=1, default=float) + '\n')
    print('report.md', len(L), 'lines; EXPORT keys', list(X))


if __name__ == '__main__':
    main()
