"""Plot hash-verified scalar projections; this is not a PDE validation figure."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.metadata/'RECEIPT.json').read_text())
    contract = json.loads(args.contract.read_text())
    ranks = contract['rom']['velocity_prefixes']
    expected_re = contract['case2d']['development_Re']
    rows = {}
    for item in receipt['files']:
        path = args.metadata/item['name']
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('Progress record differs from collected receipt')
        if path.name.endswith('_model.json'):
            projections = json.loads(path.read_text())['record']['development_projection']
            if [row['Re'] for row in projections] != expected_re:
                raise ValueError('Exact development population required')
            rows[path.stem.removesuffix('_model')] = projections
    data = {'status': 'provisional_representation_only',
            'job_id': int(receipt['queue'].split('|')[0]),
            'receipt_sha256': sha(args.metadata/'RECEIPT.json'),
            'contract_sha256': sha(args.contract), 'rows': rows,
            'generator_sha256': sha(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'plot-data.json').write_text(json.dumps(data, indent=2)+'\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    labels = [('velocity_relative', 'Velocity projection error',
               r'$\|u-u_{ref}\|_{L^2}/\|u_{ref}\|_{L^2}$',
               'representation_development_velocity_gate'),
              ('pressure_scaled', 'Pressure projection error',
               r'$\|p-p_{ref}\|_{L^2}/(U^2\sqrt{V})$',
               'representation_development_pressure_scaled_gate')]
    for axis, (metric, title, ylabel, gate_key) in zip(axes, labels):
        for kind, label, color, marker in [('bank', 'Neural bank', '#D55E00', 'o'),
                                           ('POD', 'POD', '#0072B2', 's')]:
            values = [max(row[metric] for row in rows[f'{kind}_r{rank}']) for rank in ranks]
            if not all(value > 0 for value in values):
                raise ValueError('Positive errors required for this log plot')
            axis.plot(ranks, values, marker=marker, color=color, label=label,
                      linestyle='-' if kind == 'bank' else '--', linewidth=2)
        gate = contract['rom'][gate_key]
        axis.axhline(gate, color='#555555', linestyle=':', label=f'Gate ({gate:g})')
        axis.set(xscale='log', yscale='log', title=title,
                 xlabel='Velocity basis prefix r', ylabel=ylabel)
        axis.set_xticks(ranks, [str(rank) for rank in ranks])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(axis='y', alpha=.18)
        axis.legend(frameon=False, fontsize=9)
    axes[1].text(.03, .03, 'Shared pressure space: curves coincide',
                 transform=axes[1].transAxes, fontsize=9)
    fig.suptitle('Fixed-channel development: representation checks', fontsize=16)
    fig.text(.5, .015, f"Worst of the {len(expected_re)} development cases. Projection only; larger POD solves/validation pending. Job {data['job_id']}.",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(args.output/'projection-checks.png', dpi=170)
    fig.savefig(args.output/'projection-checks.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
