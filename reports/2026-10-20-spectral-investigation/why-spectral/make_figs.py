"""Figures for report.md, drawn only from this folder's analysis JSONs (CPU, matplotlib). Writes figs/*.png."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C
from a6_cost import w_ps

# reference categorical palette (dataviz skill, light mode), fixed order
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'lines.linewidth': 2, 'savefig.dpi': 160, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})


def J(n):
    return json.loads((C.OUT / n).read_text())


def first_N(es, t):
    for N in sorted(es, key=int):
        if es[N] is not None and es[N] <= t:
            return int(N)
    return None


def fig_spectra():
    a1 = J('a1_spectra.json')['cohorts']
    e2 = sorted(a1['E0_2d'], key=lambda r: r['s'])
    pick = [('E0 2D, smallest s', e2[0]), ('E0 2D, median s', e2[len(e2) // 2]), ('E0 2D, largest s', e2[-1]),
            ('A−', sorted(a1['reg_Am'], key=lambda r: r['s'])[8]), ('C1', sorted(a1['reg_C1'], key=lambda r: r['s'])[8]),
            ('C2', sorted(a1['reg_C2'], key=lambda r: r['s'])[8])]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for j, (ti, title) in enumerate(((0, 't = 0 (initial field)'), (1, 't = 0.05'), (5, 't = 0.25'))):
        for i, (lab, r) in enumerate(pick):
            tl = np.array(r['tail_t'][ti])
            N = np.arange(1, len(tl) + 1)
            ax[j].semilogy(N[1:], np.maximum(tl[1:], 1e-12), color=COL[i], label=f"{lab} (s = {r['s']:.0f})")
        ax[j].set_title(title, color=INK, fontsize=9)
        ax[j].set_xlabel('N (modes 1..N−1 kept per axis)')
        ax[j].axhline(1e-2, color=INK2, lw=0.8, ls=':')
        ax[j].set_ylim(1e-9, 2)
    ax[0].set_ylabel('best N-mode sine error, relative to ‖u(t₀)‖')
    ax[2].legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(C.FIGS / 'fig1_sine_tails.png')


def fig_N_vs_s():
    a1 = J('a1_spectra.json')['cohorts']
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    groups = [('E0_2d', 'E0 2D'), ('reg_N0', 'N0'), ('reg_Am', 'A−'), ('reg_C1', 'C1'), ('reg_C2', 'C2'), ('E0_3d', 'E0 3D')]
    for i, (k, lab) in enumerate(groups):
        rows = a1[k]
        s = [r['s'] for r in rows]; n = [r['N_proj']['0.01'] for r in rows]
        ax[0].scatter(s, n, s=14, color=COL[i], label=lab, edgecolor='#fcfcfb', linewidth=0.5)
    allr = [r for k, _ in groups[:5] for r in a1[k]]
    s = np.array([r['s'] for r in allr]); n = np.array([r['N_proj']['0.01'] for r in allr], float)
    p = np.polyfit(np.log(s), np.log(n), 1)
    xs = np.geomspace(5, 1100, 50)
    ax[0].plot(xs, np.exp(p[1]) * xs ** p[0], color=INK2, lw=1.2, ls='--', label=f'2D fit N ∝ s^{p[0]:.2f}')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('steepness s = a/ν'); ax[0].set_ylabel('N for a 1 % sine floor')
    ax[0].set_title('projection floor of the references', color=INK, fontsize=9)
    ax[0].legend(fontsize=7)
    # solver error
    ps2 = J('a2_e02d.json')['rows']
    S = [r['s'] for r in ps2.values()]; N1 = [first_N(r['e'], 1e-2) for r in ps2.values()]
    N3 = [first_N(r['e'], 1e-3) for r in ps2.values()]
    ax[1].scatter(S, N1, s=14, color=COL[0], label='E0 2D, 1 %')
    ax[1].scatter(S, N3, s=14, color=COL[1], label='E0 2D, 0.1 %', marker='s')
    try:
        reg = J('a2_reg.json')['rows']
        S = [r['s'] for r in reg.values()]
        ax[1].scatter(S, [first_N(r['e'], 1e-2) or np.nan for r in reg.values()], s=14, color=COL[0], marker='^', label='regimes, 1 %')
        ax[1].scatter(S, [first_N(r['e'], 1e-3) or np.nan for r in reg.values()], s=14, color=COL[1], marker='v', label='regimes, 0.1 %')
    except FileNotFoundError:
        pass
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('steepness s = a/ν'); ax[1].set_ylabel('smallest N reaching the error')
    ax[1].set_title('spectral solver error against the references', color=INK, fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGS / 'fig2_N_vs_steepness.png')


def fig_step_cost():
    c6 = J('a6_cost.json')
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    pts = [('spectral 2D (N = 8–256)', c6['fit_ps2'], COL[0], 'o'), ('spectral 3D (N = 16–64)', c6['fit_ps3'], COL[2], 's'),
           ('ROM 2D (r = 128–384)', c6['fit_rom2'], COL[1], 'D')]
    for lab, f, col, mk in pts:
        W = np.array([p[0] for p in f['points']]); t = np.array([p[1] for p in f['points']])
        ax.scatter(W, 1e3 * t, color=col, marker=mk, s=22, label=lab, zorder=3, edgecolor='#fcfcfb', linewidth=0.5)
        xs = np.geomspace(W.min() / 3, W.max() * 30, 60)
        ax.plot(xs, 1e3 * (f['tau0'] + xs / f['P']), color=col, lw=1.2, alpha=0.8)
    r3 = c6['rom3']['W_C_W768']
    ax.scatter([r3['W_contraction']], [1e3 * r3['tau']], color=COL[3], marker='^', s=30, label='ROM 3D (W_C_W768, per step)', zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('modelled floating-point work per time step'); ax.set_ylabel('measured time per step (ms, H200)')
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(C.FIGS / 'fig3_step_cost.png')


def fig_bank():
    b3 = J('a3_bank.json')['rows']
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, src in enumerate(('E0', 'N0', 'Am', 'C1', 'C2')):
        rr = [r for r in b3 if r['src'] == src]
        ax.scatter([r['s'] for r in rr], [r['K_match_over_r']['384'] for r in rr], s=16, color=COL[i],
                   label=src.replace('Am', 'A−'), edgecolor='#fcfcfb', linewidth=0.5)
    ax.axhline(1, color=INK2, lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('steepness s = a/ν')
    ax.set_ylabel('sines needed / bank functions\n(to match the r = 384 bank floor)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGS / 'fig4_bank_vs_sines.png')


def fig_syn():
    s5 = J('a5_synthetic.json')
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labs = dict(F1='F1 front, 1 parameter', F2='F2 front, 2 parameters', F3='F3 bump, 2 parameters', F4='F4 bump width, 1 parameter')
    for i, fam in enumerate(('F1', 'F2', 'F3', 'F4')):
        rr = [r for r in s5['rows'] if r['family'] == fam]
        d = [r['delta'] for r in rr if r['rho']['0.01']]
        rho = [r['rho']['0.01'] for r in rr if r['rho']['0.01']]
        ax.plot(d, rho, marker='o', color=COL[i], label=labs[fam], ms=5)
        ra = [r['rho_aniso']['0.01'] for r in rr if r['rho']['0.01']]
        ax.plot(d, ra, marker='s', color=COL[i], ls='--', lw=1.4, ms=4)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
    ax.set_xlabel('feature width δ (sharper →)')
    ax.set_ylabel('ρ = sine unknowns / POD rank (1 %)\nsolid: square blocks, dashed: anisotropic')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGS / 'fig5_synthetic_compression.png')


def fig_cross():
    m7 = J('a7_model.json')
    rn = m7['rho_needed']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for j, (d, cfgs) in enumerate(((2, (('nps20|nrom50', 'ROM 50 steps'), ('nps20|nrom20', 'ROM 20 steps'))),
                                   (3, (('nps10|nrom15', 'spectral 10 steps, ROM 15'), ('nps25|nrom15', 'spectral 25 steps, ROM 15'))))):
        for i, (cf, lab) in enumerate(cfgs):
            for k_s, ls in (('rho_S1', '-'), ('rho_S2', '--')):
                pts = sorted((int(k.split('|')[3][1:]), v[k_s]) for k, v in rn.items()
                             if k.startswith(f'd{d}|{cf}|') and k.endswith('|fit') and v[k_s])
                if pts:
                    ax[j].plot(*zip(*pts), ls=ls, color=COL[i], marker='o', ms=3,
                               label=f"{lab}, {'S = 1' if k_s == 'rho_S1' else 'S = 2'}")
        for k, v in m7['ours'].items():
            if (d == 3) == k.startswith('3D'):
                ax[j].scatter([v['N_joint']], [v['rho']], color=COL[4], marker='*', s=60, zorder=4)
        ax[j].annotate('our ROMs', xy=(list(v['N_joint'] for k, v in m7['ours'].items() if (d == 3) == k.startswith('3D'))[0],
                                      max(v['rho'] for k, v in m7['ours'].items() if (d == 3) == k.startswith('3D'))),
                       xytext=(4, 6), textcoords='offset points', fontsize=7, color=INK2)
        ax[j].set_xscale('log'); ax[j].set_yscale('log')
        ax[j].set_xlabel('spectral resolution N needed for the accuracy target')
        ax[j].set_title(f'{d}D', color=INK, fontsize=9)
        ax[j].legend(fontsize=7, loc='lower right')
    ax[0].set_ylabel('compression ρ = (N−1)^d / r needed')
    fig.tight_layout()
    fig.savefig(C.FIGS / 'fig6_required_compression.png')


if __name__ == '__main__':
    C.FIGS.mkdir(exist_ok=True)
    fig_spectra(); fig_N_vs_s(); fig_step_cost(); fig_bank(); fig_syn(); fig_cross()
    print('figures written')
