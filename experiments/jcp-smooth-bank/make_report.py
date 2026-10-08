"""Generate report.md and plots/*.png for jcp-smooth-bank from the pulled run JSONs (never hand-typed numbers).

Inputs: runs/<train job>/archive/output/<arm>/train.json and runs/<eval job>/archive/output/{eval,timing}/.
Decision rules: DESIGN.md A1.5 as amended by A2.4 and A4 (implemented in `verdicts`).

    python make_report.py --train tr1 --eval ev1 [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS = ('acc', 'fast')
BARS = (0.116, 0.06, 0.01)
GAUSS = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256)
FIB = (987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368)
MMAX = dict(G=256 ** 2, F=46368)
LABEL = dict(frozen_deployed='frozen bank, deployed rotation', frozen_lane='frozen bank, lane rotation',
             base='base (retrain, σ=4)', sob01='sob01 (σ=4, λ=0.1)', sig2='sig2 (σ=2)', sig1='sig1 (σ=1)',
             sob001='sob001 (σ=4, λ=0.01)', sob1='sob1 (σ=4, λ=1)', coarse='coarse (129-node data)',
             base_s1='base_s1 (seed 1)', comb='comb')
ORDER = ['frozen_deployed', 'frozen_lane', 'base', 'sob01', 'sig2', 'sig1', 'sob001', 'sob1', 'base_s1', 'coarse', 'comb']


def fmt(x, p=2):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    if x == 0:
        return '0'
    return f'{x:.{p}e}' if (abs(x) < 1e-2 or abs(x) >= 1e4) else f'{x:.{p + 1}g}'


def pct(x):
    return '—' if x is None else f'{100 * x:.3f}'


def mfmt(m, fam):
    return f'>{MMAX[fam]}' if m is None else str(int(m))


def load(train, ev):
    T = {}
    for p in sorted((HERE / 'runs' / train / 'archive/output').glob('*/train.json')):
        T[p.parent.name] = json.loads(p.read_text())
    E = {}
    for p in sorted((HERE / 'runs' / ev / 'archive/output/eval').glob('eval_*.json')):
        d = json.loads(p.read_text())
        E[d['label']] = d
    tp = HERE / 'runs' / ev / 'archive/output/timing/timing.json'
    TM = json.loads(tp.read_text()) if tp.exists() else None
    meta = [json.loads(p.read_text()) for p in sorted((HERE / 'runs' / ev / 'archive/output/eval').glob('meta_task*.json'))]
    return T, E, TM, meta


# --------------------------------------------------------------------- metrics ----

def rows(E, b, s, arm):
    return [r for r in E[b]['settings'][s]['rows'] if r['arm'] == arm]


def worst(E, b, s, arm, key):
    rs = rows(E, b, s, arm)
    v = [r[key] for r in rs if r.get(key) is not None]
    return max(v) if v and len(v) == len(rs) else None


def metrics(E, b, s):
    S = E[b]['settings'][s]
    P = S['projection']['S']
    g = rows(E, b, s, 'gref')
    out = dict(
        E_S=worst(E, b, s, 'gref', 'ref_S_evolved'), E_ST=worst(E, b, s, 'gref', 'ref_ST_evolved'),
        e_val_med=P['value']['median'], e_val_max=P['value']['max'],
        e_grad_med=P['grad_D2']['median'], e_grad4_med=P['grad_D4']['median'], e_grad_max=P['grad_D2']['max'],
        fd_unc_med=P['fd_target_uncertainty']['median'],
        mstar={k: v for k, v in S['mstar'].items()}, tail=S.get('tail', {}), C4=S.get('C4', {}),
        spectra=S.get('spectra', {}).get('aggregate', {}), trust=S['trust'], A_sv=S.get('A_singular_values'),
        cases=len(g))
    # C6 (A4): every required rollout finite with <= 2 non-accepted exits; tight-solver sensitivity on dev6
    bad = [f"{r['arm']}:{r['cohort']}{r['case']}" for r in S['rows']
           if r['arm'] not in ('gauss8',) and (not r['finite'] or r.get('nonaccepted_exits', 0) > 2)]
    tight = [r.get('vs_gref_evolved') for r in S['rows'] if r['arm'] == 'gref_tight']
    out['C6_failures'] = bad
    out['tight_max'] = max(tight) if tight else None
    c4 = S.get('C4', {})
    out['C4_pass'] = bool(c4) and all(v['check_rho_max'] <= 1e-5 and v['flux_rho_max'] <= 1e-5 for v in c4.values())
    g768 = [r.get('vs_gref_evolved') for r in S['rows'] if r['arm'] == 'gauss768']
    out['g768_max'] = max(g768) if g768 else None
    return out


def eligible(E, T, b, s, M):
    m = M[b][s]
    why = []
    if not m['C4_pass']:
        why.append('C4')
    if m['g768_max'] is None or m['g768_max'] > 1e-6:
        why.append('gref-vs-768 rollout')
    if m['C6_failures']:
        why.append(f"C6 ({len(m['C6_failures'])} rollouts)")
    if m['tight_max'] is None or m['tight_max'] > 1e-3:
        why.append('solver-sensitive')
    for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad_med', 'e_grad4_med'):
        if m[k] is None or not math.isfinite(m[k]) or m[k] <= 0:
            why.append(f'{k} missing')
    tr = T.get(b, {}).get('rotation') if b in T else None
    if tr and (tr['R_G_condition_number'] > 1e12 or tr['rotated_bank_orthonormality_deviation_at_train_mesh'] > 1e-6):
        why.append('rotation conditioning')
    if not E[b]['C5a']['passed']:
        why.append('C5a')
    return why


def red(m_base, m_t, fam):
    """A2.4 reduction factor of m*."""
    if m_t is None:
        return 1.0 if m_base is None else 0.0
    return (MMAX[fam] if m_base is None else m_base) / m_t


def verdicts(E, T, M):
    out = {}
    base = 'base'
    ref = 'frozen_lane'
    if base not in M:
        return out
    noise = {}
    for s in SETTINGS:
        for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad_med', 'e_grad4_med'):
            if ref in M and M[ref][s][k] and M[base][s][k]:
                noise[(s, k)] = abs(M[base][s][k] / M[ref][s][k] - 1)
    out['noise'] = {f'{s}|{k}': v for (s, k), v in noise.items()}
    resolved = lambda s, k, d: d is not None and abs(d) > 2 * noise.get((s, k), 0.)
    dlt = lambda t, s, k: (M[t][s][k] / M[base][s][k] - 1) if (M[t][s][k] and M[base][s][k]) else None
    for t in M:
        if t in (base, ref, 'frozen_deployed'):
            continue
        v = dict(eligible={s: eligible(E, T, t, s, M) for s in SETTINGS})
        base_ok = {s: not eligible(E, T, base, s, M) for s in SETTINGS}
        ok = {s: base_ok[s] and not v['eligible'][s] for s in SETTINGS}
        d = {s: {k: dlt(t, s, k) for k in ('E_S', 'E_ST', 'e_val_med', 'e_grad_med', 'e_grad4_med')} for s in SETTINGS}
        v['delta'] = d
        # H1 (A1.5 + A4)
        h1g = all(ok[s] and d[s]['e_grad_med'] is not None and d[s]['e_grad_med'] <= -.2 and d[s]['e_grad4_med'] <= -.2
                  and resolved(s, 'e_grad_med', d[s]['e_grad_med']) and resolved(s, 'e_grad4_med', d[s]['e_grad4_med'])
                  for s in SETTINGS)
        h1v = all(ok[s] and d[s]['e_val_med'] <= .10 for s in SETTINGS)
        h1e = any(ok[s] and d[s]['E_S'] <= -.05 and resolved(s, 'E_S', d[s]['E_S']) and
                  all(d[o]['E_S'] <= .02 for o in SETTINGS if o != s) for s in SETTINGS)
        v['H1'] = dict(grad=h1g, value=h1v, rollout=h1e, passed=bool(h1g and h1v and h1e))
        # H2 descriptive (A1.5 + A2.4)
        h2 = {}
        for bb in (0.06, 0.01):
            for s in SETTINGS:
                mb, mt = M[base][s]['mstar'][f'gref|G|{bb}'], M[t][s]['mstar'][f'gref|G|{bb}']
                cb, ct = M[base][s]['mstar'][f'common|G|{bb}'], M[t][s]['mstar'][f'common|G|{bb}']
                mref = M[ref][s]['mstar'][f'gref|G|{bb}'] if ref in M else 'na'
                rf = red(mb, mt, 'G')
                other = [o for o in SETTINGS if o != s][0]
                rf_o = red(M[base][other]['mstar'][f'gref|G|{bb}'], M[t][other]['mstar'][f'gref|G|{bb}'], 'G')
                same_dir = (ct is not None and (cb is None or ct <= cb)) or (ct is None and cb is None)
                h2[f'{s}|{bb}'] = dict(reduction=rf, other_reduction=rf_o, common_same_direction=same_dir,
                                       resolved=(mref == mb), passed=bool(ok[s] and ok[other] and rf >= 1.5 and rf_o >= 1.0
                                                                          and same_dir and mref == mb))
        v['H2'] = h2
        v['H2_passed'] = any(x['passed'] for x in h2.values())
        v['useful_winner'] = bool((v['H1']['passed'] or v['H2_passed']) and all(
            ok[s] and M[t][s]['E_S'] <= 1.1 * M[base][s]['E_S'] and M[t][s]['E_ST'] <= 1.1 * M[base][s]['E_ST']
            for s in SETTINGS))
        score = 0.
        for s in SETTINGS:
            mb, mt = M[base][s]['mstar']['gref|G|0.06'], M[t][s]['mstar']['gref|G|0.06']
            score += math.log2(((MMAX['G'] if mb is None else mb)) / (2 * MMAX['G'] if mt is None else mt))
        v['rank_score'] = score
        out[t] = v
    return out


# ----------------------------------------------------------------------- plots ----

def plots(E, M, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    banks = [b for b in ORDER if b in M]
    cols = {b: c for b, c in zip(banks, plt.rcParams['axes.prop_cycle'].by_key()['color'] * 3)}
    # frontier
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for j, s in enumerate(SETTINGS):
        for b in banks:
            for i, (key, ylab) in enumerate((('e_val_max', 'projection floor (worst, vs S)'),
                                            ('E_S', 'converged-quadrature rollout error (worst, vs S)'))):
                for bb, mk in ((0.06, 'o'), (0.116, 's')):
                    m = M[b][s]['mstar'][f'gref|G|{bb}']
                    x = m if m is not None else 2 * MMAX['G']
                    ax[i, j].scatter(x, M[b][s][key], color=cols[b], marker=mk, s=50,
                                     facecolors='none' if m is None else cols[b],
                                     label=f'{b} (ρ≤{bb})' if (i == 0 and j == 0) else None)
                ax[i, j].set_xscale('log'); ax[i, j].set_yscale('log')
                ax[i, j].set_xlabel('Gauss points needed m* (own reached states)'); ax[i, j].set_ylabel(ylab)
                ax[i, j].set_title(f'{s}')
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Error floor vs quadrature points needed (one marker per bank and bar; hollow = not reached)')
    fig.tight_layout(); fig.savefig(out / 'frontier.png', dpi=140); plt.close(fig)
    # ladders
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for j, s in enumerate(SETTINGS):
        for b in banks:
            R = E[b]['settings'][s]['rho']
            ax[j].plot([p * p for p in GAUSS], [R[f'gauss{p}']['gref']['max'] for p in GAUSS], '-o', ms=3, color=cols[b], label=f'{b} Gauss')
            ax[j].plot(FIB, [R[f'fib{n}']['gref']['max'] for n in FIB], '--', color=cols[b], alpha=.6)
        for bb in BARS:
            ax[j].axhline(bb, color='k', lw=.6, ls=':')
        ax[j].set_xscale('log'); ax[j].set_yscale('log'); ax[j].set_title(f'{s}: worst ρ on own reached states (solid Gauss, dashed Fibonacci)')
        ax[j].set_xlabel('points m'); ax[j].set_ylabel('worst ρ')
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'ladders.png', dpi=140); plt.close(fig)
    # spectra
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, s in enumerate(SETTINGS):
        for b in banks:
            env = M[b][s]['spectra'].get('f|envelope_median')
            if env:
                ax[j].semilogy(env, color=cols[b], label=b)
        ax[j].set_title(f'{s}: median Chebyshev envelope of f = u(u_x+u_y)'); ax[j].set_xlabel('degree j'); ax[j].set_ylim(1e-16, 2)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'spectra.png', dpi=140); plt.close(fig)


def e2e_plot(E, M, TM, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    banks = [b for b in ORDER if b in M]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    w = .38
    for j, s in enumerate(SETTINGS):
        x = np.arange(len(banks))
        ax[j].bar(x - w / 2, [100 * M[b][s]['E_ST'] for b in banks], w, label='vs ST (space+time ref.)')
        ax[j].bar(x + w / 2, [100 * M[b][s]['E_S'] for b in banks], w, label='vs S (space-only ref.)')
        ax[j].set_xticks(x); ax[j].set_xticklabels(banks, rotation=30, fontsize=8)
        ax[j].set_ylabel('worst error % (PROVISIONAL)'); ax[j].set_title(f'{s}: converged-quadrature rollout error')
    ax[0].legend(fontsize=8)
    if TM:
        lab, val = [], []
        for k, v in TM['median_ms'].items():
            lab.append(k.replace('|', ' ')); val.append(v)
        ax[2].barh(np.arange(len(val)), val); ax[2].set_yticks(np.arange(len(val))); ax[2].set_yticklabels(lab, fontsize=6)
        ax[2].set_xlabel('median query ms (one GPU, A-B-A)'); ax[2].set_title(f"cost at m* rules ({TM['gpu']})")
    fig.tight_layout(); fig.savefig(out / 'e2e.png', dpi=140); plt.close(fig)


# ---------------------------------------------------------------------- report ----

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='tr1')
    ap.add_argument('--eval', default='ev1')
    ap.add_argument('--out', default='report.md')
    a = ap.parse_args()
    T, E, TM, meta = load(a.train, a.eval)
    M = {b: {s: metrics(E, b, s) for s in SETTINGS} for b in E}
    V = verdicts(E, T, M)
    plots(E, M, HERE / 'plots')
    e2e_plot(E, M, TM, HERE / 'plots')
    banks = [b for b in ORDER if b in M]
    L = []
    w = L.append
    w('# Smooth and derivative-trained banks for the off-mesh Burgers 2D ROM (lane C3, round 1)\n')
    w('Retrains the Burgers 2D coordinate-network bank with a gradient-matching (Sobolev) loss and with smaller '
      'initial random-Fourier-feature scales, and measures what that does to the quadrature points the off-mesh ROM '
      'needs and to its accuracy. Status: **round 1, development/validation cohorts (dev6 ∪ val32) only; every '
      'accuracy number is PROVISIONAL** because the references are first-order (ST = space+time refined, S = '
      'space-only refined; both reported). Generated by `make_report.py` from the run JSONs; pre-registration '
      '`DESIGN.md` (amendments A1–A4).\n')
    w('## What ran\n')
    w('| job | role | GPU | job id | content |')
    w('|---|---|---|---|---|')
    tj = next(iter(T.values()), {}) if T else {}
    w(f"| `{a.train}` | training | {tj.get('gpu', '—')} | {tj.get('job_id', '—')} | {', '.join(sorted(T))} |")
    if meta:
        w(f"| `{a.eval}` | evaluation at ${E[banks[0]]['mesh']}^2$ | {meta[0]['gpu']} | {meta[0]['job_id']} | {', '.join(banks)} |")
    if TM:
        w(f"| `{a.eval}` (timing) | paired timing | {TM['gpu']} | {TM['job_id']} | m* rules of every bank |")
    w('')
    w('## Training\n')
    w('| bank | σ (initial) | λ | steps | train h | recon mean (train) | train LS floor median / max | ‖B_j‖ median init→final | ‖B_j‖ max init→final | R2a | FD stencil sensitivity (train, median) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for b in [x for x in ORDER if x in T]:
        t = T[b]; tr = t['train']
        r2a = ', '.join(f"{x['step']}: {x['got']} vs {x['want']}" for x in tr.get('R2a', [])) or 'n/a'
        w(f"| {b} | {t['sigma']} | {t['lam_sob']} | {tr['steps_done']} | {tr['seconds'] / 3600:.2f} | {fmt(tr['recon_rel_l2_mean'])} | "
          f"{fmt(t['rotation']['train_ls_floor']['median'])} / {fmt(t['rotation']['train_ls_floor']['max'])} | "
          f"{tr['B_init']['0.5']:.2f} → {tr['B_final']['0.5']:.2f} | {tr['B_init']['1']:.2f} → {tr['B_final']['1']:.2f} | {r2a} | "
          f"{fmt(t['data']['fd_target_uncertainty']['median'])} |")
    w('')
    for s in SETTINGS:
        Rp = E[banks[0]]['settings'][s]['R_prime']; Mm = E[banks[0]]['settings'][s]['M']
        w(f'## Setting `{s}` ($R\'={Rp}$, $M={Mm}$)\n')
        w('### Representation and derivatives (discrete, 257² nodes, vs the S reference; PROVISIONAL)\n')
        w('| bank | projection floor median | projection floor worst | gradient error median (D2) | gradient error median (D4) | gradient error worst (D2) | FD stencil sensitivity (ref., median) |')
        w('|---|---|---|---|---|---|---|')
        for b in banks:
            m = M[b][s]
            w(f"| {b} | {fmt(m['e_val_med'])} | {fmt(m['e_val_max'])} | {fmt(m['e_grad_med'])} | {fmt(m['e_grad4_med'])} | {fmt(m['e_grad_max'])} | {fmt(m['fd_unc_med'])} |")
        w('')
        w('### Quadrature points needed (worst ρ over the population ≤ bar, confirmed at every larger rung)\n')
        w('| bank | Gauss m*(0.116) own / common | Gauss m*(0.06) own / common | Gauss m*(0.01) own / common | Fibonacci m*(0.06) own | tail ϱ̂ (R²) | target checks C4 |')
        w('|---|---|---|---|---|---|---|')
        for b in banks:
            m = M[b][s]; ms = m['mstar']; tl = m['tail'].get('gref', {})
            tail = f"{tl['varrho']:.4f} ({tl['r2']:.2f})" if tl.get('varrho') else 'no resolved tail'
            w(f"| {b} | {mfmt(ms['gref|G|0.116'], 'G')} / {mfmt(ms['common|G|0.116'], 'G')} | {mfmt(ms['gref|G|0.06'], 'G')} / {mfmt(ms['common|G|0.06'], 'G')} | "
              f"{mfmt(ms['gref|G|0.01'], 'G')} / {mfmt(ms['common|G|0.01'], 'G')} | {mfmt(ms['gref|F|0.06'], 'F')} | {tail} | {'pass' if m['C4_pass'] else 'FAIL'} |")
        w('')
        w('Worst ρ at selected Gauss rungs (own reached states; in brackets: restricted to the first 320 tests, Hari\'s $M$):\n')
        sel = (32, 48, 64, 96, 128)
        w('| bank | ' + ' | '.join(f'Gauss {p}²' for p in sel) + ' |')
        w('|---|' + '---|' * len(sel))
        for b in banks:
            R = E[b]['settings'][s]['rho']
            w(f'| {b} | ' + ' | '.join(f"{fmt(R[f'gauss{p}']['gref']['max'])} ({fmt(R[f'gauss{p}']['gref']['max_320'])})" for p in sel) + ' |')
        w('')
        w('### Smoothness (Chebyshev, 64 own reached states)\n')
        w('| bank | u: n(1e-8) median / max | f: n(1e-8) median / max | f: n(1e-4) median | f classification (geometric/algebraic/inconclusive/unresolved) |')
        w('|---|---|---|---|---|')
        for b in banks:
            sp = M[b][s]['spectra']
            if not sp:
                continue
            cl = sp.get('f|class', {})
            w(f"| {b} | {sp['u|n_1e-08']['median']} / {sp['u|n_1e-08']['max']} | {sp['f|n_1e-08']['median']} / {sp['f|n_1e-08']['max']} | {sp['f|n_1e-04']['median']} | "
              f"{cl.get('geometric', 0)}/{cl.get('algebraic', 0)}/{cl.get('inconclusive', 0)}/{cl.get('unresolved', 0)} |")
        w('')
        w('### End-to-end (worst over the 38 cases, evolved error %, PROVISIONAL; ST beside S)\n')
        arms = ['gref', 'gauss32', 'gauss48', 'gauss64', 'gauss96', 'fib1597', 'fib4181', 'fib6765', 'lat64']
        w('| bank | ' + ' | '.join(arms) + ' |')
        w('|---|' + '---|' * len(arms))
        for b in banks:
            cells = []
            for arm in arms:
                st_, s_ = worst(E, b, s, arm, 'ref_ST_evolved'), worst(E, b, s, arm, 'ref_S_evolved')
                cells.append(f'{pct(st_)} / {pct(s_)}')
            w(f'| {b} | ' + ' | '.join(cells) + ' |')
        w('\nEach cell: ST / S.\n')
    if TM:
        w('## Cost at each bank\'s own m* rules (median ms per query, one GPU, paired A–B–A, dev6 × 3 repetitions)\n')
        w('| setting | bank | rule | m | median ms | outputs equal to the evaluation rollout |')
        w('|---|---|---|---|---|---|')
        for k, v in TM['median_ms'].items():
            s, b, rule = k.split('|')
            inv = [x for x in TM['invocations'] if x['setting'] == s and x['label'] == b and x['rule'] == rule]
            w(f"| {s} | {b} | {rule} | {inv[0]['m']} | {v:.1f} | {sum(x['output_matches_phase1'] for x in inv)}/{len(inv)} |")
        w('')
    w('## Gates and controls\n')
    if meta:
        c = meta[0]['controls']
        w(f"- C2 (bandwidth ordering, manufactured bumps): {'pass' if c['C2']['passed'] else 'FAIL'} ({c['C2']['n8_w02']} < {c['C2']['n8_w005']}); "
          f"C3 (kink classified algebraic): {'pass' if c['C3']['passed'] else 'FAIL'}; C5b (difference stencil response): {'pass' if c['C5b']['passed'] else 'FAIL'}.")
    for b in banks:
        el = {s: eligible(E, T, b, s, M) for s in SETTINGS}
        c5 = E[b]['C5a']['rel_error_by_step']
        w(f"- `{b}`: C5a min rel. error {fmt(min(c5.values()))}; " + '; '.join(
            f"{s}: C4 {'pass' if M[b][s]['C4_pass'] else 'FAIL'}, gref-vs-768 rollout {fmt(M[b][s]['g768_max'])}, "
            f"tight-solver distance {fmt(M[b][s]['tight_max'])}, eligibility {'ok' if not el[s] else 'FAILED: ' + ', '.join(el[s])}"
            for s in SETTINGS) + '.')
    w('')
    w('## Verdicts (DESIGN A1.5, A2.4, A4; PROVISIONAL, development/validation; one seed per arm)\n')
    if 'noise' in V:
        w('Noise yardstick (|base / frozen-lane − 1|): ' + ', '.join(f'{k} {v:.3f}' for k, v in V['noise'].items()) + '.\n')
    for t, v in V.items():
        if t == 'noise':
            continue
        h2 = '; '.join(f"{k}: ×{x['reduction']:.2f} ({'pass' if x['passed'] else 'no'}{'' if x['resolved'] else ', unresolved'})" for k, x in v['H2'].items())
        w(f"- **{t}**: H1 {'PASS' if v['H1']['passed'] else 'fail'} (gradient {v['H1']['grad']}, value {v['H1']['value']}, rollout {v['H1']['rollout']}); "
          f"H2 {'PASS' if v['H2_passed'] else 'fail'} [{h2}]; useful winner: {'YES' if v['useful_winner'] else 'no'}; rank score {v['rank_score']:.2f}. "
          + ' '.join(f"Δ{s}: E_S {pct(v['delta'][s]['E_S'])}%, e∇ {pct(v['delta'][s]['e_grad_med'])}%, e_val {pct(v['delta'][s]['e_val_med'])}%." for s in SETTINGS))
    w('')
    w('## Plots\n')
    for p in ('frontier', 'ladders', 'e2e', 'spectra'):
        w(f'![{p}](plots/{p}.png)\n')
    w('## Glossary\n')
    for k, v in GLOSSARY:
        w(f'- **{k}** — {v}')
    (HERE / a.out).write_text('\n'.join(L) + '\n')
    (HERE / 'report_numbers.json').write_text(json.dumps(dict(metrics=M, verdicts=V), indent=1, default=str))
    print('wrote', a.out)


GLOSSARY = [
    ('bank', 'the frozen coordinate network G(x) whose columns span the reduced trial space; u(x) = G(x)c.'),
    ('σ (initial)', 'the standard deviation of the random Fourier frequencies B at initialisation; B is trained afterwards, so the final ‖B_j‖ are also reported.'),
    ('λ', 'weight of the gradient-matching (Sobolev) term in the training loss.'),
    ('base', 'the original recipe retrained in this lane with the same data and seed: the comparator for every treatment.'),
    ('frozen bank, deployed / lane rotation', 'the bank used by the 2D off-mesh lane, evaluated with that lane\'s rotation (replication) and with this lane\'s rotation procedure (comparator for base).'),
    ('acc / fast', 'linear settings: span of the first R\'=384 (acc) or 128 (fast) importance-ordered bank columns, tested against M=1536 or 512 sine functions.'),
    ('ρ', 'relative error of a quadrature rule\'s tested advection vector against the converged target (Gauss 640²), per state.'),
    ('own / common states', 'own = the states the bank\'s own converged rollout reaches (k=1..50, 38 cases); common = the S reference fields projected onto each bank (same physical states for every bank).'),
    ('m*(b)', 'smallest rule size whose worst ρ is ≤ b and stays ≤ b at every larger rung; ">N" means no rung reached it.'),
    ('tail ϱ̂', 'effective geometric convergence parameter fitted to the median ρ over the resolved Gauss rungs (descriptive, not a certified analyticity width).'),
    ('projection floor', 'relative error of the least-squares projection of the reference field onto the bank span, on the 257² nodes.'),
    ('gradient error (D2 / D4)', 'relative error of the projection\'s exact gradient against second- / fourth-order central differences of the reference field.'),
    ('FD stencil sensitivity', '‖D2 − D4‖/‖D4‖: how much the derivative target itself depends on the stencil (an indicator, not a bound).'),
    ('converged-quadrature rollout error (E_S, E_ST)', 'worst error over cases and output times of the ROM with the converged rule (Gauss 640²) against the S or ST reference.'),
    ('ST / S', 'refined reference solutions at 8192²: ST also refines time (Δt/16), S keeps the ROM\'s Δt (space-only).'),
    ('n(ε)', 'Chebyshev degree beyond which the coefficient envelope of a field stays below ε; a bandwidth.'),
    ('C2–C6, R0–R2', 'pre-registered gates and controls (DESIGN A1.6, A2.5, A4).'),
    ('PROVISIONAL', 'the references are first-order; accuracy numbers can move when second-order references exist.'),
    ('H1 / H2 / useful winner', 'pre-registered hypotheses (Sobolev helps; smoothness reduces points) and the gate for promoting a bank (DESIGN A1.5).'),
]


if __name__ == '__main__':
    main()
