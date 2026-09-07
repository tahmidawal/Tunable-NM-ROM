"""Plot fresh verified reference fields, never model predictions or old waves."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--verification', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.verification/'result.json').read_text())
    assert result['passed']
    # This is the predeclared off-center crossed-width control, selected without
    # consulting any learned model. Saved observations follow the source manifest.
    case, n = 2, 256
    times = np.asarray([0, 7, 24, 48])*.05
    fig, axes = plt.subplots(2, len(times), figsize=(12, 6.7), constrained_layout=True)
    descriptions = [('dirichlet', 'Reflective walls'), ('absorbing', 'Absorbing boundary')]
    for row, (bc, description) in enumerate(descriptions):
        prefix = f'gaussian_compact_coverage_{bc}_{case}'
        with np.load(args.verification/(prefix+'_arrays.npz')) as saved:
            fields = saved[f'{prefix}_{n}_u']
        if bc == 'dirichlet':
            fields = np.pad(fields, ((0, 0), (1, 1), (1, 1)))
        for column, (t, u) in enumerate(zip(times, fields)):
            ax = axes[row, column]
            limit = max(float(np.max(abs(u))), 1e-12)
            im = ax.imshow(u.T, origin='lower', extent=(0, 1, 0, 1),
                           cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
            ax.set_title(f't = {t:g}')
            ax.set_xlabel('x')
            ax.set_ylabel(description+'\ny' if column == 0 else 'y')
            ax.set_xticks([0, .5, 1])
            ax.set_yticks([0, .5, 1])
            colorbar = fig.colorbar(im, ax=ax, shrink=.8, pad=.03)
            colorbar.set_label('u')
            colorbar.ax.tick_params(labelsize=8)
    fig.suptitle('Fresh reference solutions for an off-center localized pulse\nEach panel has its own amplitude scale; these are reference fields, not decoder predictions.', fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=180)
    fig.savefig(args.out.with_suffix('.pdf'))
    provenance = dict(source_result=str((args.verification/'result.json').resolve()),
                      source_job=result['provenance']['job_id'],
                      control_index=case, intervals=n, saved_times=times.tolist(),
                      interpretation='Fresh FOM reference only. No learned result is displayed.')
    args.out.with_suffix('.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(str(args.out.with_suffix('.png')))


if __name__ == '__main__':
    main()
