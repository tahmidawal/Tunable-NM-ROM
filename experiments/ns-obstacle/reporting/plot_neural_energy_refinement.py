"""Plot completed, independently audited energy-weighted trace refinement records."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.metadata / name).read_text())
    review = json.loads(args.review.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    names = ['state-00000-initial.json', 'ACCEPTED.jsonl', 'CONFIG.json',
             'FINAL.json', 'MODEL-RECEIPT.json', 'SELECTED-FULL-REPLAY.json']
    assert review['schema'] == 'ns-obstacle-energy-trace-refinement-saved-audit-v1'
    selected_paths = [Path(path) for path, value in review['provenance'].items()
                      if value == review['selected_model_sha256']
                      and Path(path).name == 'spatial-bank-raw.npz']
    assert len(selected_paths) == 1
    audited_directory = selected_paths[0].parent
    for name in names:
        assert review['provenance'][str(audited_directory / name)] == sha(args.metadata / name), name
    receipt = read('MODEL-RECEIPT.json')
    initial = read('state-00000-initial.json')
    accepted = [json.loads(line) for line in
                (args.metadata / 'ACCEPTED.jsonl').read_text().splitlines()]
    assert review['passed'] and review['objective_variant'] == 'energy_supervised_trace_error_lambda_1'
    assert review['selected_model_sha256'] == receipt['bank_sha256']
    assert len(accepted) == review['accepted_steps']
    steps = np.array([0] + [row['iteration'] for row in accepted])
    assert np.array_equal(steps, np.arange(len(steps)))
    metrics = [initial['metrics']] + [row['metrics'] for row in accepted]
    combined = np.array([initial['objective']] +
                        [row['full_post_update_objective'] for row in accepted])
    base = np.array([row['base_objective'] for row in metrics])
    trace = np.array([row['supervised_divergence_objective_weighted'] for row in metrics])
    assert np.allclose(combined, base + trace, rtol=1e-12, atol=1e-15)
    assert combined[0] == review['initial_objective']
    assert combined[-1] == review['selected_objective'] == combined.min()
    assert combined[-1] == receipt['selected_checkpoint']['objective']
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), layout='constrained')
    for values, label in ((combined, 'Combined'), (base, 'Energy base'), (trace, 'Added trace error')):
        assert np.isfinite(values).all() and np.all(values >= 0)
        axes[0].semilogy(steps, values, label=label)
    legacy = np.array([row['legacy_base_objective'] for row in metrics])
    assert np.isfinite(legacy).all() and np.all(legacy >= 0)
    axes[0].semilogy(steps, legacy, '--', label='Legacy base (different weights)')
    axes[0].set(title='Energy objective and legacy monitor', ylabel='Weighted squared errors')
    axes[0].legend()
    panels = [('value_relative_squared_by_mode', 'Velocity-mode error', 'Relative mode error [%]'),
              ('gradient_relative_squared_by_mode', 'Gradient-mode error', 'Relative mode error [%]'),
              ('divergence_error_relative_squared_by_mode', 'Trace-gradient error', 'Error / teacher gradient norm [%]')]
    for ax, (key, title, ylabel) in zip(axes[1:], panels):
        squared = np.asarray([row[key] for row in metrics])
        assert np.isfinite(squared).all() and np.all(squared >= 0)
        for mode in range(squared.shape[1]):
            ax.semilogy(steps, 100 * np.sqrt(squared[:, mode]), label=f'Mode {mode + 1}')
        ax.set(title=title, ylabel=ylabel)
    axes[-1].legend(ncol=2)
    for ax in axes:
        ax.set_xlabel('Additional accepted L-BFGS steps')
        ax.grid(alpha=.2)
    fig.suptitle('Energy-weighted refinement: recorded training diagnostics')
    fig.supxlabel(f"Termination: {review['status']}. Trace error is relative to the FE teacher, not zero.\n"
                  'Legacy base uses different weights and is not optimized. Training-mode errors are not actual ROM flow errors.', fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output / ('training-progress.' + extension), dpi=150)
    plt.close(fig)
    (args.output / 'PLOT-MANIFEST.json').write_text(json.dumps({
        'generator_sha256': sha(Path(__file__)), 'review_sha256': sha(args.review),
        'inputs': {name: sha(args.metadata / name) for name in names},
        'model_sha256': receipt['bank_sha256'],
        'scope': 'Accepted-state training diagnostics; no actual flow errors or timing comparison.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
