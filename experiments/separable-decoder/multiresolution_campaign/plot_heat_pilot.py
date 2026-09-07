"""Plot the independently audited development heat pilot; no hard-coded results."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.input.read_text())
    rows = review['rows']
    meshes = sorted({row['intervals'] for row in rows})
    methods = sorted({row['method'] for row in rows})
    palette = ['#1b6b9c', '#cf7330', '#773f8e']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
    for method, color in zip(methods, palette):
        groups = [[row for row in rows if row['intervals'] == mesh and row['method'] == method]
                  for mesh in meshes]
        error_medians = [100*statistics.median(row['common_grid_time_max_errors']['relative_current']
                                             for row in group) for group in groups]
        error_maxima = [100*max(row['common_grid_time_max_errors']['relative_current']
                               for row in group) for group in groups]
        costs = [1000*statistics.median(row['query_median_seconds'] for row in group)
                 for group in groups]
        label = 'FOM: direct sine propagation' if method.startswith('fom') else 'NM-ROM: '+method.split('rom_cn_')[1]
        axes[0].plot(meshes, error_medians, 'o-', color=color, label=label)
        axes[0].plot(meshes, error_maxima, 'x--', color=color, alpha=.75)
        axes[1].plot(meshes, costs, 'o-', color=color, label=label)
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(meshes, [str(n) for n in meshes])
        ax.set_xlabel('Intervals per axis')
        ax.grid(alpha=.2, which='both')
    axes[0].set_ylabel('Time-maximum current-field relative L2 error (%)')
    axes[0].set_title('Common observation grid; solid median, dashed worst', fontsize=10)
    axes[1].set_ylabel('Complete host-field query (ms)')
    axes[1].set_title('Median of case repetition medians; same GPU', fontsize=10)
    axes[1].legend(loc='best', fontsize=8)
    cases = len({row['case'] for row in rows})
    fig.suptitle(f'Heat frozen-weight pilot — {cases} development cases, not final confirmation', fontsize=12)
    for extension in ('png', 'pdf'):
        fig.savefig(args.out_prefix.with_suffix('.'+extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
