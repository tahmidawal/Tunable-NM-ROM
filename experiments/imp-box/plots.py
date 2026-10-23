"""imp-box plots from the retained production cells (the make_report.py cell mapping: <W>_<lev>t, the A16 time-stage
restart, preferred over <W>_<lev>): S vs family width per level and target (median and worst), one fixed colour and
marker per reduced arm; hollow markers = the S carries a search qualification (incomplete FOM search or an A13
ladder-minimum label); panels without any retained cell are marked deferred. S = 1 and S = 0.5 reference lines."""
from __future__ import annotations

import json
from pathlib import Path

from make_report import cell_dirs, ladder_min

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WIDTHS = {'W08': 0.08, 'W16': 0.16, 'W28': 0.28, 'W40': 0.40}
LEVELS = ('A1', 'A2', 'A3', 'A4')
TAUS = (0.01, 0.03, 0.1)
ARMS = {'podg': ('R-PODG', 'o', '#1f77b4'), 'ecsw': ('R-ECSW', 's', '#ff7f0e'), 'trans': ('R-TRANS', '^', '#2ca02c'),
        'bank_lspg': ('B-LSPG', 'v', '#d62728'), 'bank_e4': ('B-E4', 'D', '#9467bd')}


def main():
    cells = HERE / 'runs' / 'cells'
    F = {}
    for key, (name, cdir) in cell_dirs(cells).items():
        W, lev = key.split('_')
        F[(W, lev)] = json.loads((cdir / 'final.json').read_text())
    out = HERE / 'plots'
    out.mkdir(exist_ok=True)
    for stat in ('median', 'worst'):
        fig, axes = plt.subplots(len(TAUS), len(LEVELS), figsize=(13, 8), sharex=True, sharey=True, squeeze=False)
        for i, tau in enumerate(TAUS):
            for j, lev in enumerate(LEVELS):
                ax = axes[i][j]
                for arm, (lab, mk, col) in ARMS.items():
                    xs, ys, qual = [], [], []
                    for W, wv in WIDTHS.items():
                        fz = F.get((W, lev))
                        if not fz or not fz.get('scorable', False):
                            continue
                        k = f'{arm}|{tau}|{stat}'
                        se = fz['S'].get(k) or {}
                        s = se.get('S')
                        if s is not None:
                            fsx = [fz['selection'].get(f'{f}|{tau}|{stat}', {}).get('status') for f in ('spec', 'specad')]
                            xs.append(wv)
                            ys.append(s)
                            qual.append(any(x not in ('finalist', None) for x in fsx) or bool(ladder_min(fz, se)))
                    if xs:
                        ax.plot(xs, ys, color=col, lw=1)
                        for x_, y_, q_ in zip(xs, ys, qual):
                            ax.plot([x_], [y_], marker=mk, color=col, mfc='none' if q_ else col, ls='none')
                if not any((W, lev) in F for W in WIDTHS):
                    ax.text(0.5, 0.5, 'deferred (A14/A15)', transform=ax.transAxes, ha='center', va='center',
                            fontsize=9, color='0.4')
                ax.axhline(1.0, color='k', lw=0.8)
                ax.axhline(0.5, color='k', lw=0.8, ls=':')
                ax.set_yscale('log')
                if i == 0:
                    ax.set_title(lev)
                if j == 0:
                    ax.set_ylabel(f'S at τ={tau:g} ({stat})')
                if i == len(TAUS) - 1:
                    ax.set_xlabel('family width W')
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=col, marker=mk, label=lab) for lab, mk, col in ARMS.values()]
        hs.append(Line2D([], [], color='0.3', marker='o', mfc='none', ls='none',
                         label='hollow: incomplete FOM search or ladder-minimum label'))
        fig.legend(handles=hs, loc='lower center', ncol=6, fontsize=8)
        fig.suptitle(f'Speed-up S (upper bound vs the tested FOM ladder; H200; val32; conditional spectral references), {stat} error')
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out / f'S_vs_W_{stat}.png', dpi=120)
        plt.close(fig)
    print('plots written', sorted(p.name for p in out.glob('*.png')))


if __name__ == '__main__':
    main()
