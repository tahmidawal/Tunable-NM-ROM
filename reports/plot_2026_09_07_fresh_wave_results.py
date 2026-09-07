"""Plot stored primary-run accuracy; no simulation, fitting or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--result', action='append', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    loaded = [(p, json.loads(p.read_text())) for p in args.result]
    fig, axes = plt.subplots(2, len(loaded), figsize=(13, 8.4), constrained_layout=True, squeeze=False, sharex='row')
    colors = ['#245c9b', '#d66b24']
    rows = ['pod_k', 'pod_r', 'learned_bank_linear_r', 'mlp', 'mlp_velocity', 'quadratic', 'quadratic_velocity']
    provenance = []
    seed_labels = '/'.join(str(s) for s in loaded[0][1]['config']['optimizer_seeds'])
    for col, (path, run) in enumerate(loaded):
        assert run['completed'] and not run['final_test_opened']
        bc, results = next(iter(run['boundary_results'].items()))
        labels = [f"POD {run['config']['latent']}", f"POD {run['config']['rank']}", f"Learned bank {run['config']['rank']}", 'MLP', 'MLP + velocity', 'Quadratic', 'Quadratic + velocity']
        # Baseline labels contain their role, rather than assuming file order.
        baseline = {b['label']: b for b in results['linear_baselines']}
        keys = list(baseline)
        baseline_rows = [next(b for b in baseline.values() if 'pod' in b['label'].lower() and b['rank']==run['config']['latent']), next(b for b in baseline.values() if 'pod' in b['label'].lower() and b['rank']==run['config']['rank']), baseline['learned_bank_linear_r']]
        for row, metric in enumerate(('displacement', 'energy_state')):
            ax = axes[row, col]
            for i, b in enumerate(baseline_rows):
                summary = b[metric]
                ax.plot([100*summary['median'], 100*summary['worst']], [i, i], color='#555555', lw=1.5)
                ax.plot(100*summary['median'], i, 'o', color='#555555', markersize=5)
                ax.plot(100*summary['worst'], i, '|', color='#555555')
            for arm in results['arms']:
                i = rows.index(arm['name'])
                repeat = run['config']['optimizer_seeds'].index(arm['optimizer_seed'])
                y = i+(repeat-.5)*.21
                summary = next(s for s in arm['rollout']['summaries'] if s['dt']==arm['rollout']['primary_dt'])[metric]
                unresolved = not arm['rollout']['refinement_passed']
                ax.plot([100*summary['median'], 100*summary['worst']], [y, y], color=colors[repeat], ls='--' if unresolved else '-', lw=1.4)
                ax.plot(100*summary['median'], y, 'o', color=colors[repeat], markerfacecolor='white' if unresolved else colors[repeat], markersize=5)
                ax.plot(100*summary['worst'], y, '|', color=colors[repeat])
            ax.axvline(100*run['config']['accuracy_target'], color='#b73138', lw=1, ls=':')
            ax.set_xscale('log')
            ax.set_yticks(range(len(labels)), labels)
            ax.set_ylim(len(labels)-.5, -.5)
            ax.grid(axis='x', alpha=.15)
            ax.set_xlabel(('Displacement' if metric=='displacement' else 'Energy norm of state error')+' (%)')
            ax.set_title(('Reflective walls' if bc=='dirichlet' else 'Absorbing boundary')+' — primary step')
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), job=run['provenance']['job_id']))
    fig.suptitle(f'Fresh wave accuracy on unseen validation trajectories\nDot = median trajectory maximum; line ends at worst case. Blue/orange = seeds {seed_labels}.\nOpen dots / dashed lines fail the original time-step check; the dotted line is the accuracy ceiling.', fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=170)
    fig.savefig(args.out.with_suffix('.pdf'))
    args.out.with_suffix('.json').write_text(json.dumps(dict(inputs=provenance, caveats=['Intervals are median-to-worst ranges, not confidence intervals.', 'Linear bank models have more phase-space coordinates than the nonlinear heads.', 'Separate finer-step diagnostics do not replace these original primary results.']), indent=2)+'\n')
    print(str(args.out.with_suffix('.png')))


if __name__=='__main__':
    main()
