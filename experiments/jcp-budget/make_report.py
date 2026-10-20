"""Generate report.md, plots/*.png and runs/EXPORT.json from the lane's analysis JSONs (never a hand-typed number).

Inputs are named in report_inputs.json (committed):
  refs          anaref output over every reference job (2D and 3D FD + PS, T = 0.5 and T = 1.0)
  budget2d      {mesh: [ana2d outputs]}          (1024^2: both partitions, A8)
  meshcmp       ana2d --compare output
  b7b8          {mesh: ana7 output}              (T = 0.5; 1024^2 merged)
  b7b8_t1       ana7 output at T = 1.0 (optional)
  b3            ana3d output
  smokes        [smoke acceptance JSONs]
  ledger        runs/GPU-LEDGER.json
  deferred      [{item, reason}]
  audits        [{file, verdict}]

    python make_report.py [--inputs report_inputs.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ING = ('Q', 'T', 'F', 'S', 'I', 'M')
ING_NAME = dict(Q='quadrature', T='time package', F='form (+ stopping rule)', S='solver tolerance', I='IC projection',
                M='test-space size', e_P='bank floor', e_dyn_star='ideal dynamics')


def last_amendment():
    import re
    return max(int(x) for x in re.findall(r'^## Amendment A(\d+)', (HERE / 'DESIGN.md').read_text(), re.M))


WIN_IDX = dict(W1=(1, 6), W2=(6, 11), W3=(11, 21))


def win_meds(v, w_):
    """median over cases eligible in window w_ of the window error, fixed-t0 and current-time normalisation"""
    a, b = WIN_IDX[w_]
    el = [x for x in v['rows'] if x.get(f'elig_{w_}') and 'e_rel_t' in x and len(x['e_rel_t']) >= b]
    if not el:
        return '—', '—'
    return g(np.median([x[f'e_{w_}'] for x in el])), g(np.median([max(x['e_rel_t'][a:b]) for x in el]))


def mono(v):
    st = v['stability']
    return f"{st.get('monotone_violations_rom', '—')} / {st.get('monotone_violations_ref', '—')}"


def J(p):
    return json.loads((HERE / p).read_text()) if p else None


def g(x, d=3):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if x == 0:
        return '0'
    return f'{x:.{d}g}' if abs(x) < 1e4 else f'{x:.3e}'


def pct(x):
    return '—' if x is None else f'{100 * x:.3g} %'


def table(head, rows):
    esc = lambda c: str(c).replace('|', ' ')        # member keys contain '|': never break the markdown table
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def plots(inp, B2, B7, B3, B7T1=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'plots').mkdir(exist_ok=True)
    made = []
    # 1. B2 main effects at 1024^2 (median over eligible cases), log scale, never stacked
    b = B2.get('1024')
    if b:
        mks = sorted(b['members'])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        w = .8 / (len(ING) + 2)
        keys = list(ING) + ['e_P', 'e_dyn_star']
        for k_, key in enumerate(keys):
            vals = [(b['members'][m]['summary']['median_main'].get(key) if key in ING
                     else b['members'][m]['summary'].get(f'median_{key}')) or np.nan for m in mks]
            ax.bar(np.arange(len(mks)) + (k_ - len(keys) / 2) * w, vals, w, label=ING_NAME[key])
        ax.set_yscale('log')
        ax.set_xticks(np.arange(len(mks)))
        ax.set_xticklabels(mks, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('median over cases of max_k ||effect|| / n0')
        ax.set_title('B2 at 1024^2: one-ingredient effects, floor and ideal dynamics (E0 R_ST, conditional)')
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout()
        fig.savefig(HERE / 'plots' / 'b2_main_1024.png', dpi=130)
        plt.close(fig)
        made.append('plots/b2_main_1024.png')
    # 2. B7 error against time at 1024^2 (median over cases with the new reference)
    c = B7.get('1024')
    if c:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for mk, v in sorted(c['b7'].items()):
            el = [r for r in v['rows'] if 'e_t_new' in r and r.get('elig_W1') and r.get('elig_W2')]   # eligible population
            if not el:
                continue
            lab = mk + ('' if v['qualified_at_T'] else ' (not qualified at T)') + f' n={len(el)}'
            for ax, key in zip(axs, ('e_t_new', 'e_rel_t')):
                E = np.array([r[key] for r in el])
                t = np.arange(E.shape[1]) * .05
                ax.plot(t[1:], np.median(E[:, 1:], axis=0), marker='o', ms=3, label=lab, ls='--' if 'GAL' in mk else '-')
        for ax, yl in zip(axs, ('median e(t) = ||u(t) - R(t)|| / ||R(t0)||', 'median ||u(t) - R(t)|| / ||R(t)||')):
            ax.axvline(.25, color='grey', lw=.8)
            ax.set_yscale('log')
            ax.set_xlabel('t (training horizon 0.25 marked)')
            ax.set_ylabel(yl)
        axs[0].set_title('B7 at 1024^2, eligible cases: fixed-t0 normalisation')
        axs[1].set_title('same runs: current-time normalisation')
        axs[0].legend(fontsize=5.5, ncol=2, loc='lower left')
        fig.tight_layout()
        fig.savefig(HERE / 'plots' / 'b7_error_time_1024.png', dpi=130)
        plt.close(fig)
        made.append('plots/b7_error_time_1024.png')
        # 3. floor over time per bank width: median and worst over eligible cases, every B7 panel (DESIGN 5.3)
        panels = [('2D $1024^2$ references, T = 0.5', c, ('W1', 'W2'))]
        if B7T1:
            panels.append(('2D $1024^2$ references, T = 1.0 (W3 provisional)', B7T1, ('W1', 'W2', 'W3')))
        if B3 and B3.get('b7'):
            panels.append(('3D $65^3$ references, T = 0.5', B3, ('W1', 'W2')))
        fig, axs = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2))
        axs = np.atleast_1d(axs)
        for ax, (title, src, wins) in zip(axs, panels):
            seen = set()
            for mk, v in sorted(src['b7'].items()):
                r = mk.split('|')[0]
                if r in seen:
                    continue
                P = [x['e_P_t'] for x in v['rows'] if 'e_P_t' in x and all(x.get(f'elig_{w_}') for w_ in wins)]
                if P:
                    seen.add(r)
                    P = np.array(P)
                    t = np.arange(P.shape[1]) * .05
                    ln, = ax.plot(t, np.median(P, axis=0), marker='o', ms=3, label=f'{r} median, n={len(P)}')
                    ax.plot(t, np.max(P, axis=0), ls='--', color=ln.get_color(), label=f'{r} worst')
            ax.axvline(.25, color='grey', lw=.8)
            if 'W3' in wins:
                ax.axvspan(.5, 1., color='grey', alpha=.12)
            ax.set_yscale('log')
            ax.set_xlabel('t (training horizon 0.25 marked)')
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=6)
        axs[0].set_ylabel('bank floor e_P(t) = ||(I - Pi) R(t)|| / ||R(t0)||')
        fig.tight_layout()
        fig.savefig(HERE / 'plots' / 'b7_floor_time.png', dpi=130)
        plt.close(fig)
        made.append('plots/b7_floor_time.png')
        # 4. sampling ratio s
        if c.get('b8'):
            fig, ax = plt.subplots(figsize=(7, 4))
            s_ = [r_['W1']['s'] for v in c['b8'].values() for r_ in v['rows'] if 'W1' in r_ and r_['W1'].get('s')]
            if s_:
                ax.hist(s_, bins=30)
            ax.axvline(1.1, color='k', ls=':')
            ax.axvline(.9, color='k', ls=':')
            ax.set_xlabel('s = e_fine / e_257 (W1, all members and cases)')
            ax.set_ylabel('count')
            ax.set_title('B8 output-sampling ratio at 1024^2 (1025^2 route vs 257^2 lattice)')
            fig.tight_layout()
            fig.savefig(HERE / 'plots' / 'b8_sampling_1024.png', dpi=130)
            plt.close(fig)
            made.append('plots/b8_sampling_1024.png')
    if B3:
        mks = sorted(B3['members'])
        fig, ax = plt.subplots(figsize=(8, 4))
        keys = ['Q', 'T', 'S', 'F', 'e_P', 'e_dyn_star']
        w = .8 / len(keys)
        for k_, key in enumerate(keys):
            vals = [(B3['members'][m]['summary']['median_main'].get(key) if key in ('Q', 'T', 'S', 'F')
                     else B3['members'][m]['summary'].get(f'median_{key}')) or np.nan for m in mks]
            ax.bar(np.arange(len(mks)) + (k_ - len(keys) / 2) * w, vals, w,
                   label='form (at the tight-solver end)' if key == 'F' else ING_NAME.get(key, key))
        ax.set_yscale('log')
        ax.set_ylabel('median over cases of max_k ||effect|| / n0')
        ax.set_xticks(np.arange(len(mks)))
        ax.set_xticklabels(mks, rotation=25, ha='right', fontsize=7)
        ax.set_title('B2 in 3D (65^3, v16): effects, floor, ideal dynamics')
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(HERE / 'plots' / 'b2_main_3d.png', dpi=130)
        plt.close(fig)
        made.append('plots/b2_main_3d.png')
    return made


def merge_b2(files):
    out = dict(members={}, gates=[], controls={})
    for f in files:
        j = J(f)
        out['members'].update(j['members'])
        out['gates'].append(j['gates'])
        out['controls'].update({k: v for k, v in j['controls'].items() if v})
        out['mesh'] = j['mesh']
        out['coverage'] = out.get('coverage', []) + [j.get('coverage')]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='report_inputs.json')
    a = ap.parse_args()
    I = J(a.inputs)
    R = J(I['refs'])
    B2 = {m: merge_b2(fs) for m, fs in I['budget2d'].items()}
    MC = J(I.get('meshcmp'))
    B7 = {m: J(f) for m, f in I['b7b8'].items()}
    B7T1 = J(I.get('b7b8_t1'))
    B3 = J(I.get('b3'))
    L = J(I['ledger'])
    made = plots(I, B2, B7, B3, B7T1)
    L_ = []
    w = L_.append
    w('# jcp-budget: error budget, longer horizons and physical diagnostics for the off-mesh Burgers ROM (B2, B7, B8)')
    w('')
    w('**Status:** generated by `make_report.py` from the lane\'s analysis JSONs; every table and label below is generated from them and from the ledger; the correction list is prose that cites the DESIGN amendments holding its measurements. '
      'Accuracy numbers are scored against order-verified second-order references (E0 for $t\\le0.25$, this lane\'s own '
      'references beyond); every $\\varepsilon_{\\rm ref}$ is a **conditional Richardson estimate**, not a bound. Post-hoc '
      'labels carried unchanged: E0 A10; combined A10, A11, retracted A15. GAL-CN rows are a registered diagnostic scheme. '
      f'Pre-registration `DESIGN.md` with amendments A1–A{last_amendment()}.')
    w('')
    # ------------------------------------------------------------------ what ran
    w('## What ran')
    w('')
    rows = [(j['attempt'], j.get('job_id'), j.get('state'), g(j.get('elapsed_hours'))) for j in L['jobs']]
    w(table(['attempt', 'job id', 'state', 'elapsed GPU-h'], rows))
    w('')
    w(f"Spent {g(L.get('spent_gpu_hours'))} of {g(L.get('cap_gpu_hours'))} GPU-h (Slurm accounting, `runs/GPU-LEDGER.json`). "
      'Every result-producing job ran on H200 under `/cluster/tufts/paralab/tawal01/jcp-budget/` with `jax_backend=gpu` in its '
      'log (smoke and collection checks); `ref3` was cancelled 30 s after it started, before any case, and has no collected '
      'log (A9). `ref2b`, `ref2c` and `ref2t1` were cancelled and partially collected (A9, A14).')
    w('')
    rows = []
    for j in L['jobs']:
        f = HERE / 'checks' / f"resources-{j['attempt']}.json"
        if f.exists() and j.get('elapsed_hours') is not None:
            rc = json.loads(f.read_text())
            rows.append((j['attempt'], g(rc['projection_hours']), g(rc['time_limit_hours']), g(j['elapsed_hours']), j.get('state'),
                         g(j['elapsed_hours'] / rc['projection_hours'])))
    if rows:
        w('**Certificates against elapsed time** (projection and time limit from `checks/resources-<job>.json`, elapsed from '
          'Slurm; cancelled jobs ran only part of their configuration):')
        w('')
        w(table(['job', 'projected h', 'time limit h', 'elapsed h', 'state', 'elapsed / projected'], rows))
        w('')
    # ------------------------------------------------------------------ references
    w('## New references (B7, B8)')
    w('')
    cs = R['cases']
    for (d, T) in sorted({(c['dim'], c['horizon']) for c in cs}):
        sel = [c for c in cs if c['dim'] == d and c['horizon'] == T]
        acc = sum(c['outcome'] == 'accepted' for c in sel)
        w(f"- **{d}D, $T={g(T)}$:** {len(sel)} cases, {acc} accepted; refined {sum(bool(c['refined']) for c in sel)}, "
          f"tight retries {sum(bool(c['tight']) for c in sel)}; G-R2 (agreement with E0 on $t\\le0.25$) passed on "
          f"{sum(c['G_R2']['passed'] for c in sel)}; G-R3 (fine route = lattice at shared nodes) "
          f"{sum(bool(c['G_R3_flag']) for c in sel if c['fine_requested'])} of {sum(c['fine_requested'] for c in sel)}.")
        bad = {}
        for c in sel:
            if c['outcome'] != 'accepted':
                bad.setdefault(c['outcome'], []).append(f"{c['cohort']} {c['case']}")
        if bad:
            w('  - not accepted (non-rankable): ' + '; '.join(f"{k}: {', '.join(v)}" for k, v in sorted(bad.items())))
        for wdw in ('W1', 'W2', 'W3'):
            e = [c['judge']['windows'][wdw]['eps'] for c in sel if wdw in c['judge']['windows'] and c['judge']['windows'][wdw]['eps']]
            if e:
                w(f"  - window {wdw}: median $\\varepsilon_{{\\rm ref}}$ {pct(np.median(e))}, worst {pct(max(e))} "
                  f"(conditional Richardson estimate)")
    w('')
    if R.get('tiers'):
        w('Pseudo-spectral cross-check (E0 procedure, per window) and the resulting tier (A2.3):')
        w('')
        rows = []
        for d, v in R['tiers'].items():
            for T, ws in v.items():
                for wdw, x in ws.items():
                    rows.append((d, T, wdw, x['tier'], ', '.join(f'{k}: {vv}' for k, vv in x['verdicts'].items())))
        w(table(['dim', 'horizon', 'window', 'tier', 'per-case verdicts'], rows))
        w('')
    s = R['summary']
    w(f"Reference gates: G-R1 {g(s.get('G_R1'))}; G-R2 evaluable failures: {', '.join(s.get('G_R2_evaluable_failures') or []) or 'none'}"
      f" (not evaluable: {', '.join(s.get('G_R2_not_evaluable') or []) or 'none'}; A13); G-R3 {g(s.get('G_R3'))}; offline judge agreement "
      f"{g(s.get('judge_agree'))}; C-R1 (first-order ladder rejected) {g(s.get('C_R1'))}; $T=1$ prefix {g(s.get('prefix_T1'))}.")
    w('')
    # ------------------------------------------------------------------ B2
    w('## B2: error budget')
    w('')
    w('Per case and output time, relative to $n_0=\\lVert R(t_0)\\rVert$ on the scoring lattice: one-ingredient effects '
      '$I_x=u_x-u_\\varnothing$ (switching quadrature to Gauss $320^2$, the time package to BDF2 at $\\Delta t_0/8$, the '
      'form, the solver tolerance to tight, the initial condition to the full-grid least-squares projection, the test space to '
      '$M=2048$), pairwise interactions $I_{xy}=u_{xy}-u_x-u_y+u_\\varnothing$, the floor $e_P=\\lVert(I-\\Pi)u_{\\rm ref}\\rVert$ '
      'and the ideal dynamics $e_{\\star,\\rm dyn}=\\lVert u_\\star-\\Pi u_{\\rm ref}\\rVert$. Each is the max over the five '
      'output times; tables give medians over eligible cases. Norms are **not additive** (the per-time identity '
      '$e_v^2=e_P^2+\\lVert v-\\Pi u_{\\rm ref}\\rVert^2$ is gate G-B0).')
    w('')
    for mesh in sorted(B2, key=int):
        b = B2[mesh]
        w(f'### ${mesh}^2$')
        w('')
        rows = []
        for mk in sorted(b['members']):
            sm = b['members'][mk]['summary']
            mm = sm['median_main']
            rows.append((mk + (' (diag.)' if b['members'][mk]['diagnostic'] else ''), sm['dominant'],
                         *[g(mm.get(x)) for x in ING], g(sm.get('median_e_P')), g(sm.get('median_e_dyn_star')),
                         g(sm.get('median_U_star')), g(sm.get('median_e_tot')), f"{sm['n_complete']}/{sm['n_eligible']}"))
        w(table(['member', 'dominant', *ING, 'e_P', 'e_dyn*', 'U*', 'e_tot', 'complete/eligible'], rows))
        w('')
        rows = []
        for mk in sorted(b['members']):
            pl = b['members'][mk]['summary']['pairs']
            two = b['members'][mk]['summary']['two_largest']
            key = ''.join(sorted(two, key=ING.index)) if len(two) == 2 else None
            rows.append((mk, ', '.join(two), pl.get(key, '—') if key else '—',
                         ', '.join(f'{k}' for k, v in pl.items() if v == 'interacting') or 'none'))
        w(table(['member', 'two largest main effects', 'their pair', 'interacting pairs'], rows))
        w('')
        gts = b['gates']
        w('Gates: ' + '; '.join(
            f"G-B0 {g(x['G_B0']['passed'])} (worst {g(x['G_B0']['worst'])}), G-B2 {g(x['G_B2']['passed'] if isinstance(x['G_B2'], dict) else x['G_B2'])}, "
            f"G-B3 {g(x['G_B3']['passed'])} ({x['G_B3']['n']} reproductions, worst {g(x['G_B3']['worst'])}), "
            f"G-B6 {g(x['G_B6']['passed'])}, B0 mismatched-normaliser {g(x['B0_real_mismatch']['passed'])}" for x in gts) + '.')
        w('')
        ctl = b['controls']
        if ctl:
            w('Controls: ' + '; '.join(f"{k.replace('_', '-')} {g(v.get('fired'))}" for k, v in ctl.items() if isinstance(v, dict)
                                       and 'fired' in v) + '.')
            w('')
    if MC:
        w('### Mesh invariance of the budget (sub16)')
        w('')
        rows = [(mk, v['label'], ', '.join(f'{m}: {d}' for m, d in v['dominant'].items())) for mk, v in sorted(MC['members'].items())]
        w(table(['member', 'same dominant at every mesh?', 'dominant per mesh'], rows))
        w('')
    if B3:
        w('### 3D ($65^3$ lattice, cohort v16: val64 cases 0–15)')
        w('')
        rows = []
        for mk in sorted(B3['members']):
            sm = B3['members'][mk]['summary']
            rows.append((mk, sm['dominant'], *[g(sm['median_main'].get(x)) for x in ('Q', 'T', 'S', 'F')], g(sm.get('median_e_P')),
                         g(sm.get('median_e_dyn_star')), g(sm.get('median_U_star')),
                         ', '.join(f'{k}: {v}' for k, v in sm['pairs'].items())))
        w(table(['member', 'dominant', 'Q', 'T', 'S', 'F (tight end)', 'e_P', 'e_dyn*', 'U*', 'pairs'], rows))
        w('')
        w('Gates: ' + ', '.join(f"{k.replace('_', '-')} {g(v.get('passed'))}" for k, v in B3['gates'].items()) +
          '. Controls: ' + ', '.join(f"{k.replace('_', '-')} {g((v or {}).get('fired', (v or {}).get('fired_any')))}" for k, v in B3['controls'].items()) + '.')
        w('')
        w('Not reported by `ana3d`: G-B5 does not apply in 3D (no IC ingredient); G-B1 and G-B6 are not computed in 3D — the '
          'bitwise reproduction of every deployed corner is G-B3, and the logical corner settings were reconstructed '
          'independently in the results audit (`audits/codex-results-1.md`), not by a gate of this lane.')
        w('')
    # ------------------------------------------------------------------ B7
    w('## B7: longer horizons')
    w('')
    w('The bank was trained on $t\\le0.25$; beyond it the model extrapolates in time. Q1/Q2 are certified over every step to '
      '$T$ (horizon-specific labels); growth $g=e_{W_2}/e_{W_1}$ uses this lane\'s references for both windows; labels carry '
      'the cross-check tier (A2.3).')
    w('')
    for mesh in sorted(B7, key=int):
        c = B7[mesh]
        w(f"### ${mesh}^2$, $T={g(c['horizon'])}$")
        w('')
        rows = []
        for mk, v in sorted(c['b7'].items()):
            gr = v['growth'].get('W2/W1', {})
            e2, e2r = win_meds(v, 'W2')
            rows.append((mk, g(v['Q1']['pass']), v['Q2']['label'], g(v['qualified_at_T']), g(v['stability']['pass']), mono(v),
                         gr.get('label', '—'), g(gr.get('median')), f"{gr.get('n', 0)}", e2, e2r))
        w(table(['member', 'Q1 at T', 'Q2 at T', 'qualified at T', 'stable (outputs)', 'energy rises ROM / ref', 'growth W2/W1',
                 'median g', 'n', 'median e_W2 (fixed t0)', 'median e_W2 (current t)'], rows))
        w('')
        fo = c['front_order']
        w(f"Front order W1 → W2: all members {fo['all']['label']} (Kendall $\\tau_b$ {g(fo['all'].get('kendall_tau_b'))}); "
          f"LSPG members {fo['lspg']['label']} ($\\tau_b$ {g(fo['lspg'].get('kendall_tau_b'))}).")
        w('')
    if B3 and B3.get('b7'):
        w('### 3D, $T=0.5$')
        w('')
        rows = [(mk, g(v['Q1']['pass']), v['Q2']['label'], g(v['qualified_at_T']), g(v['stability']['pass']), mono(v),
                 v['growth'].get('W2/W1', {}).get('label', '—'), g(v['growth'].get('W2/W1', {}).get('median')),
                 v['growth'].get('W2/W1', {}).get('n', 0), *win_meds(v, 'W2')) for mk, v in sorted(B3['b7'].items())]
        w(table(['member', 'Q1 at T', 'Q2 at T', 'qualified at T', 'stable', 'energy rises ROM / ref', 'growth W2/W1', 'median g',
                 'n', 'median e_W2 (fixed t0)', 'median e_W2 (current t)'], rows))
        w('')
        fo = B3.get('front_order') or {}
        if fo:
            w(f"Front order W1 → W2 (all 3D members): {fo.get('label', '—')} (Kendall $\\tau_b$ {g(fo.get('kendall_tau_b'))}, "
              f"n = {fo.get('n', '—')}).")
            w('')
    if B7T1:
        w('### $1024^2$, $T=1.0$ (sub16)')
        w('')
        rows = [(mk, g(v['Q1']['pass']), v['Q2']['label'], g(v['stability']['pass']), mono(v),
                 v['growth'].get('W2/W1', {}).get('label', '—'), v['growth'].get('W3/W2', {}).get('label', '—'),
                 g(v['growth'].get('W3/W2', {}).get('median')), v['growth'].get('W3/W2', {}).get('n', 0), *win_meds(v, 'W3'))
                for mk, v in sorted(B7T1['b7'].items())]
        w(table(['member', 'Q1 at T', 'Q2 at T', 'stable', 'energy rises ROM / ref', 'growth W2/W1', 'growth W3/W2 (provisional)',
                 'median g W3/W2', 'n (W3/W2)', 'median e_W3 (fixed t0, provisional)', 'median e_W3 (current t, provisional)'], rows))
        fo, f3 = B7T1.get('front_order', {}), B7T1.get('front_order_W2W3', {})
        w('')
        w(f"Front order W1 → W2 (all / LSPG): {fo.get('all', {}).get('label', '—')} / {fo.get('lspg', {}).get('label', '—')} "
          f"(Kendall $\\tau_b$ {g(fo.get('all', {}).get('kendall_tau_b'))} / {g(fo.get('lspg', {}).get('kendall_tau_b'))}). "
          f"Front order W2 → W3 (all / LSPG; W3 tier {f3.get('all', {}).get('tier', '—')}, so provisional, no verdict): "
          f"{f3.get('all', {}).get('label', '—')} / {f3.get('lspg', {}).get('label', '—')}.")
        w('')
    # ------------------------------------------------------------------ B8
    w('## B8: physical diagnostics and output-sampling adequacy')
    w('')
    for mesh in sorted(B7, key=int):
        c = B7[mesh]
        if not c.get('b8'):
            continue
        w(f'### ${mesh}^2$')
        w('')
        rows = []
        for mk, v in sorted(c['b8'].items()):
            rr = v['rows']
            n_all = len(rr)
            s_ = [r_['W1']['s'] for r_ in rr if 'W1' in r_ and r_['W1'].get('s')]
            us = [r_['undershoot'] for r_ in rr]
            os_ = [r_['overshoot'] for r_ in rr]
            dens = [x for r_ in rr for x in r_['band_density'][1:] if not isinstance(x, str)]
            Er = [abs(x) for r_ in rr for x in r_['E_ratio_minus1'][1:]]
            ic = [r_['ic_over_floor'] for r_ in rr if isinstance(r_.get('ic_over_floor'), (int, float))]
            rows.append((mk, v['adequacy_W1']['label'], v['adequacy_W2']['label'], g(np.median(s_)) if s_ else '—',
                         g(max(Er)) if Er else '—', f"{sum(x > 1e-3 for x in us)}/{len(us)}", f"{sum(x > 1e-3 for x in os_)}/{len(os_)}",
                         g(np.median(dens)) if dens else '—', g(np.median(ic)) if ic else '—', n_all))
        w(table(['member', 'sampling W1', 'sampling W2', 'median s (W1)', 'max abs(E_rom/E_ref − 1)', 'undershoot > 1e-3',
                 'overshoot > 1e-3', 'median near-wall density (cases × times)', 'IC error / floor at t0', 'cases (descriptive)'],
                rows))
        w('')
        w('The sampling labels use eligible case-windows only. The descriptive columns (median $s$, energy, extrema, near-wall '
          'density, IC error) pool every stored case, including the non-rankable references listed under "New references"; '
          'so does the $s$ histogram.')
        w('')
        if c.get('lane_statement_W1'):
            ls = c['lane_statement_W1']
            st = (f"in W1 the $257^2$ scoring lattice is adequate for the {ls['lspg_members']} LSPG members of the qualified "
                  "2D front, relative to the $1025^2$ route" if ls['all_adequate'] and ls['ranking'] == 'adequate'
                  else 'not established for the LSPG members')
            w(f"Front-ranking adequacy (W1, LSPG members): {ls['ranking']}. Lane statement: **{st}** (GAL-CN diagnostic "
              "members are not covered by it; their labels are in the table).")
            w('')
    # ------------------------------------------------------------------ deferred, retractions
    w('## Deferred items')
    w('')
    for d in I.get('deferred', []):
        w(f"- {d['item']}: {d['reason']}")
    w('')
    w('## What was wrong or corrected')
    w('')
    for c in I.get('corrections', []):
        w(f'- {c}')
    w('')
    w('## Audits')
    w('')
    w(table(['file', 'verdict'], [(a_['file'], a_['verdict']) for a_ in I.get('audits', [])]))
    w('')
    for m_ in made:
        w(f'![{m_}]({m_})')
        w('')
    w('## Glossary')
    w('')
    for t_, d_ in GLOSSARY:
        w(f'- **{t_}**: {d_}')
    (HERE / 'report.md').write_text('\n'.join(L_) + '\n')
    export = dict(lane='jcp-budget', label='E0 R_ST and this lane\'s R_ST: conditional Richardson estimates',
                  ledger=dict(spent=L.get('spent_gpu_hours'), cap=L.get('cap_gpu_hours')),
                  b2={m: {mk: dict(dominant=v['summary']['dominant'], median_main=v['summary']['median_main'],
                                   e_P=v['summary'].get('median_e_P'), e_dyn_star=v['summary'].get('median_e_dyn_star'))
                          for mk, v in b['members'].items()} for m, b in B2.items()},
                  b2_mesh=MC and {mk: v['label'] for mk, v in MC['members'].items()},
                  b7={m: {mk: dict(qualified_at_T=v['qualified_at_T'], stable=v['stability']['pass'],
                                   growth={k: x['label'] for k, x in v['growth'].items()}) for mk, v in c['b7'].items()}
                      for m, c in B7.items()},
                  b8={m: {mk: dict(W1=v['adequacy_W1']['label'], W2=v['adequacy_W2']['label']) for mk, v in c.get('b8', {}).items()}
                      for m, c in B7.items()},
                  lane_statement={m: dict(scope='W1, LSPG members only, relative to the 1025^2 route',
                                          lspg_all_adequate=c.get('lane_statement_W1', {}).get('all_adequate'),
                                          ranking=c.get('lane_statement_W1', {}).get('ranking')) for m, c in B7.items()},
                  b3=B3 and {mk: v['summary']['dominant'] for mk, v in B3['members'].items()},
                  b7_T1=B7T1 and dict(members={mk: dict(Q1=v['Q1']['pass'], Q2=v['Q2']['label'], stable=v['stability']['pass'],
                                                         growth={k: x['label'] for k, x in v['growth'].items()})
                                               for mk, v in B7T1['b7'].items()},
                                      front_order_W1W2={k: v['label'] for k, v in B7T1.get('front_order', {}).items()},
                                      front_order_W2W3={k: v['label'] for k, v in B7T1.get('front_order_W2W3', {}).items()}),
                  front_order_T05={m: {k: v['label'] for k, v in c.get('front_order', {}).items()} for m, c in B7.items()},
                  plots=made, deferred=I.get('deferred', []))
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(export, indent=1, default=float) + '\n')
    print('report.md', len(L_), 'lines;', made)


GLOSSARY = [
    (r'B2 / B7 / B8', r'the plan items: error budget; longer horizons; physical diagnostics and output-sampling adequacy.'),
    (r'member', r'one qualified configuration of the 2D front at one mesh, written `R<r> <form> <scheme> <factor>`: bank width $r$, reduced form, time scheme and time-step factor (step = factor × 0.005).'),
    (r'LSPG / GAL', r'least-squares and Galerkin reduced equations; GAL-CN rows ("diag.") are a registered diagnostic scheme, not part of the LSPG front the lane statement covers.'),
    (r'BE / CN / CNR / BDF2', r'backward Euler; Crank–Nicolson; Crank–Nicolson with Rannacher start-up; second-order backward differentiation.'),
    (r'3D member tokens', r'`W256` / `W512`: 3D bank width; `gl24`: 24-point Gauss–Legendre rule per axis; `lat8192` / `lat16384`: lattice rule with that many nodes; `fs` / `qn`: fixed-sweep and quasi-Newton solvers; then the time scheme and the time step; `K1` / `k4`: solver iteration setting.'),
    (r'ingredient letters', r'2D: Q quadrature (Gauss $320^2$; check rule Gauss $640^2$); T time package (BDF2 at $\Delta t_0/8$); F form with its stopping rule; S solver tolerance (tight); I initial condition (full-grid least-squares projection); M test space ($M=2048$ sine functions). 3D: Q quadrature (Gauss–Legendre `gl56`; check rule `gl64`), T, S, and F measured only at the tight-solver end; there is no I or M ingredient in 3D.'),
    (r'main effect $I_x$, pair $I_{xy}$', r'the change caused by switching one ingredient; the part of a two-switch change not explained by the two single switches. Table values are medians over eligible cases of the maximum over output times of the norm divided by $n_0$.'),
    (r'$n_0$', r'norm of the reference at the initial time on the scoring lattice; errors are divided by it unless "current t" is stated.'),
    (r'$e_P$ (floor)', r'distance from the reference to the bank span on the scoring lattice; no reduced solution can beat it.'),
    (r'$e_{\star,\rm dyn}$ (e_dyn*)', r'distance of the ideal-end trajectory (all numerics converged, Galerkin) from the projected reference: the error of the reduced dynamics itself.'),
    (r'$U_\star$ (U*)', r"the ideal end's own numerical uncertainty (half-step and tighter-tolerance replays). A trajectory effect is resolved only if it is at least $3U_\star$; $e_P$ must exceed $2\varepsilon_{\rm ref}$; $e_{\star,\rm dyn}$ must satisfy both."),
    (r'e_tot', r"the deployed configuration's error against the E0 reference (maximum over the five output times, divided by $n_0$)."),
    (r'complete/eligible', r'cases with every candidate available / cases whose reference is eligible; the dominance rule counts complete cases only.'),
    (r'dominant', r'per case, the largest resolved candidate, or "tied" if it exceeds another resolved candidate by no more than the pair threshold ($3U_\star$ between effects, $2\varepsilon_{\rm ref}$ between reference quantities, their sum when mixed). A component is dominant if it is the per-case winner on more than half of the complete cases, otherwise "no single dominant". "inconclusive (incomplete)" if fewer than 75 % of eligible cases are complete; "inconclusive (population)" below 8 cases (6 at $4096^2$).'),
    (r'pair labels', r'per case: interacting if $|I_{xy}|>0.25\max(|I_x|,|I_y|)$ and $\ge3U_\star$; additive if $\le0.25\max$ with the larger main effect resolved. Over cases: interacting if more than 25 % of eligible cases interact, additive if at least 75 % are additive, otherwise mixed; "inconclusive (population)" if fewer than the minimum population have the pair available; undefined if the corner does not exist (3D F×S).'),
    (r'mesh label', r'same / differs: the dominant component agrees / disagrees at $256^2$, $1024^2$, $4096^2$ on the shared sub16 cases; inconclusive if any mesh has no single dominant.'),
    (r'$\varepsilon_{\rm ref}$ (eps_ref)', r"Richardson estimate of the reference's own error; conditional, not a bound."),
    (r'W1, W2, W3', r'output-time windows $(0,0.25]$, $(0.25,0.5]$, $(0.5,1]$.'),
    (r'Q1 / Q2 at T', r'solver qualification (stationarity, distance to a tight solve) and quadrature qualification (agreement with a finer rule), certified over every step to the horizon $T$; "qualified at T" needs both.'),
    (r'growth g, n', r'per case $g=e_{W_2}/e_{W_1}$ (or $e_{W_3}/e_{W_2}$), using cases whose errors exceed twice their reference uncertainty in both windows. Label from the bootstrap 95 % interval of the median: no growth (upper ≤ 1), bounded growth (lower > 1, upper ≤ 2), accumulating (lower > 2), otherwise between labels; "inconclusive (population)" below the minimum population. n: cases used.'),
    (r'fixed t0 / current t', r'the window error divided by $n_0$, or by the reference norm at the same output time; the reference decays, so the second is larger.'),
    (r'front order', r'whether the ranking of members by median window error is the same in two windows over their common eligible cases: preserved (no resolved pair reversed), not preserved, inconclusive (no resolved pair).'),
    (r'tier', r'validated (pseudo-spectral cross-check passed) → verdicts; unvalidated → provisional values, no verdict; suspended → descriptive only.'),
    (r'non-rankable', r'a reference case that its judge did not accept, whose G-R2 is not evaluable, or that is A9 / A14 terminal; it enters no verdict or eligible population (B8 descriptive columns, labelled as such, still include it).'),
    (r'stable (outputs)', r'every run valid and, at every recorded output time, the reduced kinetic energy no more than $10^{-3}$ (relative) above its initial value; between output times nothing is checked.'),
    (r'energy rises ROM / ref', r'number of consecutive output-time pairs, summed over cases, at which the energy of the reduced solution / of the reference rises by more than $10^{-3}$ relative.'),
    (r'Kendall $\tau_b$', r'rank correlation of the members\' median errors in the two windows (1 = same order).'),
    (r'$s$', r'ratio of the error measured on the $1025^2$ lattice to the error on the $257^2$ scoring lattice.'),
    (r'sampling label', r'per case-window the two routes are consistent if they agree within 10 % plus twice their reference uncertainties (reference-limited windows are dropped); adequate if at least 90 % are consistent, inadequate below 50 %, otherwise mixed; "inconclusive (population)" below the minimum population of case-windows.'),
    (r'E_rom/E_ref', r'kinetic energy $\tfrac12\lVert u\rVert^2$ of the reduced solution over that of the reference at the same output time; the table gives the largest deviation from 1 over cases and times.'),
    (r'undershoot / overshoot > 1e-3', r'cases whose reduced solution goes below zero (undershoot) or above the initial maximum (overshoot) by more than $10^{-3}$ of the initial maximum.'),
    (r'near-wall density', r'error energy per node in the band within 1/32 of the walls, relative to the whole-interior error energy per node; the median pools cases and output times.'),
    (r'IC error / floor at t0', r'initial-condition error of the deployed run over the bank floor at $t_0$.'),
    (r'G-… / C-…', r'gates (must pass) and must-fail controls of DESIGN section 7 and amendments.'),
]


if __name__ == '__main__':
    main()
