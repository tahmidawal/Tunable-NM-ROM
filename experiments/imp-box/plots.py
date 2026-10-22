"""imp-box plots from the collected production cells (runs/cells/<W>_<lev>/out/cell/final.json): S vs family width per
level and target (median and worst), one marker per reduced arm; S = 1 and S = 0.5 reference lines. PNG in plots/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WIDTHS = {'W08': 0.08, 'W16': 0.16, 'W28': 0.28, 'W40': 0.40}
LEVELS = ('A1', 'A2', 'A3', 'A4')
TAUS = (0.01, 0.03, 0.1)
ARMS = {'podg': ('R-PODG', 'o'), 'ecsw': ('R-ECSW', 's'), 'trans': ('R-TRANS', '^'), 'bank_lspg': ('B-LSPG', 'v'),
        'bank_e4': ('B-E4', 'D')}


def main():
    cells = HERE / 'runs' / 'cells'
    F = {}
    for W in WIDTHS:
        for lev in LEVELS:
            f = cells / f'{W}_{lev}' / 'out' / 'cell' / 'final.json'
            if f.exists():
                F[(W, lev)] = json.loads(f.read_text())
    out = HERE / 'plots'
    out.mkdir(exist_ok=True)
    for stat in ('median', 'worst'):
        fig, axes = plt.subplots(len(TAUS), len(LEVELS), figsize=(13, 8), sharex=True, sharey=True, squeeze=False)
        for i, tau in enumerate(TAUS):
            for j, lev in enumerate(LEVELS):
                ax = axes[i][j]
                for arm, (lab, mk) in ARMS.items():
                    xs, ys = [], []
                    for W, wv in WIDTHS.items():
                        fz = F.get((W, lev))
                        if not fz or not fz.get('scorable', False):
                            continue
                        s = (fz['S'].get(f'{arm}|{tau}|{stat}') or {}).get('S')
                        if s is not None:
                            xs.append(wv)
                            ys.append(s)
                    if xs:
                        ax.plot(xs, ys, marker=mk, label=lab)
                ax.axhline(1.0, color='k', lw=0.8)
                ax.axhline(0.5, color='k', lw=0.8, ls=':')
                ax.set_yscale('log')
                if i == 0:
                    ax.set_title(lev)
                if j == 0:
                    ax.set_ylabel(f'S at τ={tau:g} ({stat})')
                if i == len(TAUS) - 1:
                    ax.set_xlabel('family width W')
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f'Speed-up S (upper bound vs the tested FOM ladder; H200; val32; PROVISIONAL), {stat} error')
        fig.tight_layout()
        fig.savefig(out / f'S_vs_W_{stat}.png', dpi=120)
        plt.close(fig)
    print('plots written', sorted(p.name for p in out.glob('*.png')))


if __name__ == '__main__':
    main()
