"""Generate this arm's notes from raw JSONs; no manually transcribed results."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / 'runs/b3d_architecture/screen40/out/result.json'
CONTROL = ROOT / 'runs/b3d_architecture/mlp128/out/result.json'


def summary(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), worst=float(a.max()))


def pct(value):
    return f'{100*value:.4f}%'


def main():
    candidate = json.loads(SCREEN.read_text())
    control = json.loads(CONTROL.read_text())
    assert candidate['complete'] and control['complete']
    assert candidate['config']['model'] == 'b3d_arch_encoder'
    assert candidate['config']['optimizer_seeds'] == [200, 201]
    lines = ['# Burgers shared-encoder architecture screen', '',
             'Generated from the raw run JSONs. Results are provisional pending the '
             'coordinator\'s independent result review; this validation screen establishes '
             'neither rollout accuracy nor reflective-wave transfer.', '',
             'The candidate uses exactly the control decoder architecture. It replaces '
             'independent training codes with a shared offline encoder of solution '
             'coefficients. All rows use the same frozen bank, cohort, common affine '
             'initialization, optimizer schedule and multistart fitting protocol.', '',
             '| Training scheme | Seed | Validation mean | Median | Worst | Above 15% | Unconverged | Tangent mean | Training mean |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, report in [('Free codes, matched MLP', control), ('Shared encoder', candidate)]:
        for run in report['runs']:
            fit = run['fits'][-1]
            s = fit['summary']
            lines.append(f'| {name} | {run["seed"]} | {pct(s["mean"])} | '
                         f'{pct(s["median"])} | {pct(s["worst"])} | '
                         f'{fit["outliers_above_15pct"]} | {fit["nonstationary"]} | '
                         f'{pct(fit["tangent_summary"]["mean"])} | '
                         f'{pct(run["training"]["final"]["mean"])} |')
    lines += ['', 'The primary table uses latent fits from zero and the codes of the '
              'same predeclared training states. It does not use the validation '
              'snapshot encoder as an initial guess. Local fits are not established '
              'global minima.', '',
              '| Seed | Direct encoder mean | Median | Worst | Above 15% | Initial-state mean | Later-state mean |',
              '|---|---:|---:|---:|---:|---:|---:|']
    initial = np.asarray(candidate['validation_states']['sid']) % 51 == 0
    for run in candidate['runs']:
        direct = np.asarray(run['direct_encoder_error'])
        s = summary(direct)
        lines.append(f'| {run["seed"]} | {pct(s["mean"])} | {pct(s["median"])} | '
                     f'{pct(s["worst"])} | {int(np.sum(direct>.15))} | {pct(direct[initial].mean())} | '
                     f'{pct(direct[~initial].mean())} |')
    lines += ['', 'Direct encoder reconstruction is a separate snapshot diagnostic '
              'that consumes the solution coefficients. It is not an online PDE '
              'prediction or a cost result.', '',
              '| Seed | Decoder parameters | Encoder parameters | Free training coordinates | Initial-state fitted mean | Later-state fitted mean | Minimum Jacobian rank | Maximum invariant stationarity | Maximum budget change |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in candidate['runs']:
        d, fit = run['architecture_diagnostics'], run['fits'][-1]
        lines.append(f'| {run["seed"]} | {d["decoder_parameter_count"]} | '
                     f'{d["encoder_parameter_count"]} | {run["training"]["optimized_code_count"]} | '
                     f'{pct(fit["groups"]["initial"]["mean"])} | '
                     f'{pct(fit["groups"]["later"]["mean"])} | '
                     f'{min(g["rank"] for g in fit["geometry"])} | '
                     f'{max(g["invariant_stationarity"] for g in fit["geometry"]):.6e} | '
                     f'{run["budget_change_max"]:.6e} |')
    dec = candidate['runs'][0]['architecture_diagnostics']['decoder_parameter_count']
    ctl = control['runs'][0]['training']
    assert dec == ctl['trainable_parameter_count']
    controls_by_seed = {r['seed']:r for r in control['runs']}
    mean_wins = sum(r['fits'][-1]['summary']['mean'] <
                    controls_by_seed[r['seed']]['fits'][-1]['summary']['mean']
                    for r in candidate['runs'])
    tangent_wins = sum(r['fits'][-1]['tangent_summary']['mean'] <
                       controls_by_seed[r['seed']]['fits'][-1]['tangent_summary']['mean']
                       for r in candidate['runs'])
    repeats = len(candidate['runs'])
    lines += ['', f'The matched control has {ctl["trainable_parameter_count"]} decoder '
              f'parameters and {ctl["optimized_code_count"]} independently optimized '
              'training coordinates. The encoder adds offline shared parameters '
              'without increasing decoder capacity. Equal updates are not equal '
              'compute; no cross-job timing comparison is made.', '',
              f'The shared encoder improves fitted validation mean in {mean_wins} '
              f'of {repeats} repeats and tangent mean in {tangent_wins} of '
              f'{repeats} repeats. It does not consistently improve reconstruction '
              'or establish a passing representation. The direct encoder has an '
              'additional approximation gap relative to latent fitting.', '',
              'Acceptance status:', '']
    for run in candidate['runs']:
        fit = run['fits'][-1]
        failed = []
        if fit['summary']['mean'] > .05:
            failed.append('mean error')
        if fit['summary']['worst'] > .15:
            failed.append('worst error')
        if fit['nonstationary']:
            failed.append('normalized-gradient convergence')
        if run['budget_change_max'] >= .01:
            failed.append('budget stability')
        label = ', '.join(failed) if failed else 'none of these preliminary checks'
        lines.append(f'- Seed {run["seed"]}: fails {label}. The inherited full pilot '
                     'has not run; POD comparison and all operator/negative controls '
                     'are still required for promotion.')
    config = candidate['config']
    lines += ['', f'Source commit `{config["commit"]}`, job {config["slurm_job"]}, '
              f'GPU {config["gpu"]}, backend `{config["backend"]}`, '
              f'f64 `{config["x64"]}`, matrix precision `{config["matmul_precision"]}`. '
              'The two optimizer repeats share one data seed. No final-test data '
              'were opened.', '',
              '[Encoder raw result](runs/b3d_architecture/screen40/out/result.json). '
              '[Matched-control raw result](runs/b3d_architecture/mlp128/out/result.json). '
              '[Source and artifact verification](runs/b3d_architecture/screen40/verification.json).', '',
              'The original A100-80GB request was canceled while pending and '
              'replaced with an available A100-40GB resource request. The scientific '
              'configuration was unchanged. [Cancellation record]'
              '(runs/b3d_architecture/screen/completion.log).', '',
              '## Glossary', '',
              '- **Training scheme / shared encoder / free codes:** how training '
              'snapshots obtain their reduced coordinates: a shared solution-to-code '
              'network or a separate optimized vector for every snapshot.',
              '- **Seed:** optimizer randomness; these repeats use the same data.',
              '- **Mean / median / worst:** relative full-field reconstruction '
              'errors over the validation cohort.',
              '- **Above 15% / unconverged:** counts exceeding the inherited '
              'worst-error or normalized-gradient requirements.',
              '- **Tangent mean:** mean relative physical velocity missing from '
              'the decoder derivative range.',
              '- **Training mean:** reconstruction error using assigned training '
              'codes, rather than latent-fitted validation codes.',
              '- **Direct encoder:** solution-coefficient encoding without a '
              'subsequent latent fit; not an online PDE solve.',
              '- **Initial / later states:** validation at the initial time or '
              'after evolution.',
              '- **Decoder / encoder parameters:** weights in the coefficient '
              'generator or offline snapshot-to-coordinate network.',
              '- **Free training coordinates:** per-snapshot optimized unknowns '
              'in addition to shared model weights.',
              '- **Jacobian rank / invariant stationarity:** number of independent '
              'decoder derivative directions and the residual projected along '
              'those directions.',
              '- **Budget change:** maximum relative reconstruction change when '
              'the latent-solver attempt budget is doubled.',
              '- **Bank / POD comparison / pilot:** frozen spatial patterns, '
              'comparison with the inherited linear projection baseline, and '
              'the complete numerical acceptance checks.',
              '- **f64:** double-precision arithmetic.', '']
    (ROOT/'B3D-ENCODER-NOTES.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
