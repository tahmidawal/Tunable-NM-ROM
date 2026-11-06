"""R2/R4 development figure (Re 95) generated only from saved ROM-lane arrays.

(a) phase-sensitive velocity error e_u(t) of every arm over the 1.7 s horizon;
(b) lift and (c) drag over the last 0.7 s: reference, POD r16, retrained bank r16;
(d) R2 projection floors vs POD rank with the frozen and retrained banks and the R4 rollout errors.
Usage: /home/tahmid/Dev/.venv/bin/python fig_rom_dev.py <out.png>
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROM = HERE.parents[3] / '2026-11-03-cylinder-unsteady-rom/experiments/cylinder-unsteady-rom/artifacts'
SER = ROM / 'r4-bank-qcheck01/results/r4'
BLUE, ORANGE, AQUA, YELLOW, MAG = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
ARMS = [('pod_r8', 'POD r=8', AQUA), ('pod_r16', 'POD r=16', BLUE), ('pod_r24', 'POD r=24', YELLOW),
        ('bank_r16', 'retrained bank r=16', ORANGE), ('frozen_bank_r4', 'frozen steady bank r=4', MAG)]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def main(out):
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURF)
    ax = axs[0, 0]; style(ax)
    for name, lab, col in ARMS:
        s = np.load(SER / f'series_re95_{name}.npz')
        ax.semilogy(s['t_save'] - s['t_save'][0], s['e_u'], color=col, lw=1.6, label=lab)
    ax.axhline(1e-2, color=INK2, lw=1.0, ls='--'); ax.text(1.69, 1.15e-2, 'target (RMS)', ha='right', color=INK2, fontsize=8)
    ax.set_xlabel('time since initial projection [s]'); ax.set_ylabel('$e_u(t)$ (phase-sensitive)')
    ax.set_title('(a) development Re 95: velocity error over 1.7 s (~5 periods)', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower right')
    ref = np.load(SER / 'series_re95_pod_r16.npz')
    t = ref['t']; keep = t >= t[-1] - 0.7
    for ax, key, rk, name in ((axs[0, 1], 'CL', 'CL_ref', '$C_L$'), (axs[1, 0], 'CD', 'CD_ref', '$C_D$')):
        style(ax)
        ax.plot(t[keep], ref[rk][keep], color=INK, lw=2.4, label='reference (P4/P3)')
        for nm, lab, col in (('pod_r16', 'POD r=16', BLUE), ('bank_r16', 'retrained bank r=16', ORANGE)):
            s = np.load(SER / f'series_re95_{nm}.npz')
            ax.plot(s['t'][keep], s[key][keep], color=col, lw=1.2, ls=(0, (4, 2)) if nm == 'bank_r16' else '-', label=lab)
        ax.set_xlabel('t [s]'); ax.set_ylabel(name)
        ax.set_title(f"({'b' if key == 'CL' else 'c'}) {name} over the last 0.7 s of the horizon", loc='left', fontsize=9.5)
    axs[0, 1].legend(frameon=False, fontsize=7.5, loc='lower left')
    ax = axs[1, 1]; style(ax)
    fl = json.loads((ROM / 'r2-export01/results/R2-FLOORS.json').read_text())['arms']
    r4 = json.loads((SER / 'R4-re95.json').read_text())['arms']
    ranks = [2, 4, 8, 12, 16, 20, 24]
    ax.semilogy(ranks, [fl[f'pod_r{r}']['velocity_basis_only']['median'] for r in ranks], color=BLUE, lw=2, marker='o', ms=6,
                mec=SURF, mew=1.5, label='POD projection floor (median)')
    ax.semilogy([8, 16, 24], [r4[f'pod_r{r}']['e_u']['rms'] for r in (8, 16, 24)], color=BLUE, lw=1.2, ls='--', marker='s', ms=6,
                mec=SURF, mew=1.5, label='POD rollout $e_u$ (RMS)')
    ax.semilogy([4], [fl['frozen_bank_r4']['velocity_basis_only']['median']], marker='D', ms=8, color=MAG, ls='none',
                label='frozen steady bank floor')
    ax.semilogy([4], [r4['frozen_bank_r4']['e_u']['rms']], marker='x', ms=9, color=MAG, ls='none', mew=2, label='frozen bank rollout')
    ax.semilogy([16], [r4['bank_r16']['e_u']['rms']], marker='^', ms=9, color=ORANGE, ls='none', label='retrained bank rollout (r=16)')
    ax.axhline(5e-3, color=INK2, lw=1.0, ls=':'); ax.text(2, 4.2e-3, 'floor rule (median)', ha='left', color=INK2, fontsize=8)
    ax.set_xlabel('velocity basis rank r'); ax.set_ylabel('relative $L^2$ velocity error')
    ax.set_title('(d) R2 floors and R4 rollout errors vs rank', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.suptitle('G5 ROM development results (Re 95, held-out Re 105 still sealed)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200, facecolor=SURF)


if __name__ == '__main__':
    main(sys.argv[1])
