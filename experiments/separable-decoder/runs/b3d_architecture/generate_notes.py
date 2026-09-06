"""Generate architecture-screen notes from complete and failed job JSONs.

Run with the absolute project Python. Optional positional paths include results
read from other approved worktrees; output always stays in this worktree.
"""
import argparse
import json
from pathlib import Path


def percentage(value):
    return f'{100*value:.4f}%'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('results', type=Path, nargs='*')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    inputs = args.results or sorted(here.glob('*/out/result.json'))
    assert inputs, 'no results available'
    lines = [
        '# Burgers 3D architecture screen', '',
        'Generated from the linked raw run JSONs. These are provisional validation '
        'results; a representation screen does not establish rollout accuracy or wave transfer.', '',
        'The four new architecture worktrees await base/name confirmation. This initial '
        'table records the common MLP controls. Update the campaign-status prose when those arms run.', '',
        '| Model | Width | Optimizer seed | Mean | Median | Worst | Above 15% | Unconverged | Tangent mean | Trainable parameters |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    details, failures = [], []
    for path in inputs:
        report = json.loads(path.read_text())
        if report.get('kind') != 'b3d_architecture':
            continue
        config = report['config']
        label = config['name']
        width = config['model_config'].get('width', '—')
        if not report.get('complete'):
            failures.append(f'- [{path.parent.parent.name}]({path.resolve()}): incomplete; '
                            f'{report.get("failure_type", "no recorded exception")}: '
                            f'{report.get("failure_message", "inspect job logs")}.')
        for run in report.get('runs', []):
            if not run.get('fits'):
                failures.append(f'- {label}, optimizer seed {run["seed"]}: no fully checked validation fit.')
                continue
            fit = run['fits'][-1]
            summary = fit['summary']
            lines.append(f'| {label} | {width} | {run["seed"]} | '
                f'{percentage(summary["mean"])} | {percentage(summary["median"])} | '
                f'{percentage(summary["worst"])} | {fit["outliers_above_15pct"]} | '
                f'{fit["nonstationary"]} | {percentage(fit["tangent_summary"]["mean"])} | '
                f'{run["training"]["trainable_parameter_count"]} |')
            invariants = [x['invariant_stationarity'] for x in fit['geometry']]
            details.append(f'- {label}, width {width}, seed {run["seed"]}: '
                f'initial-state mean {percentage(fit["groups"]["initial"]["mean"])}, '
                f'later-state mean {percentage(fit["groups"]["later"]["mean"])}, '
                f'maximum invariant stationarity {max(invariants):.6e}, '
                f'maximum budget change {run["budget_change_max"]:.6e}. '
                f'Job {config["slurm_job"]}, {config["gpu"]}, source commit '
                f'`{config["commit"]}`. [Raw result]({path.resolve()}).')
    lines += ['', 'All rows use the fixed learned bank and common linear initialization. '
        'Equal updates are not equal compute; the wider control has more parameters. '
        'Optimizer repeats share the same data. No cross-job timing comparison is made.', '']
    lines += details
    lines += ['', 'Incomplete attempts:'] + ([''] + failures if failures else ['', 'None among the input files.'])
    lines += ['', '## Glossary', '',
        '- **Model / width:** coefficient-generating head and units in each MLP hidden layer.',
        '- **Optimizer seed:** fitting randomness; it does not regenerate an independent data cohort.',
        '- **Mean / median / worst:** full-field relative reconstruction error over validation states.',
        '- **Above 15%:** count of validation states exceeding the unchanged worst-error gate.',
        '- **Unconverged:** selected fits whose inherited normalized-gradient measure exceeds 1e-6.',
        '- **Tangent mean:** mean fraction of the actual PDE velocity outside the decoder tangent space.',
        '- **Trainable parameters:** shared model weights; per-snapshot free latent codes are counted separately in JSON.',
        '- **Initial / later states:** validation at the initial time, or after time stepping.',
        '- **Invariant stationarity:** relative residual projected into available decoder directions.',
        '- **Budget change:** largest relative error change between the two latent-solver attempt budgets.',
        '- **Bank / latent code:** fixed learned spatial features, and reduced coordinates controlling their coefficients.',
        '- **Validation / rollout:** unseen states for architecture selection, and online prediction through time.', '']
    output = here.parents[1] / 'B3D-ARCH-NOTES.md'
    output.write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
