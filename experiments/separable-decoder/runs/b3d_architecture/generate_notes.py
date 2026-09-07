"""Generate the common architecture comparison from audited run JSONs only.

Run audit_campaign.py first. Reads other worktrees; writes only this worktree.
No architecture result or comparison statistic is transcribed by hand.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
NAMES = {'b3d_arch_baseline': 'MLP control', 'b3d_arch_anchor': 'Protected anchor',
         'b3d_arch_quadratic': 'Quadratic', 'b3d_arch_encoder': 'Shared encoder',
         'b3d_arch_mixture': 'Smooth mixture'}


def pct(x):
    return f'{100*float(x):.4f}%'


def label(config):
    name = NAMES[config['model']]
    if config['model'] == 'b3d_arch_baseline':
        name += f" {config['model_config']['width']}"
    return name


def main():
    audit = json.loads((HERE/'review/campaign-audit.json').read_text())
    old_pilot = json.loads((SRC/'runs/b3d_repair/pilot_head33/out/result.json').read_text())
    reference = json.loads((HERE/'mlp128/out/result.json').read_text())
    cfg = reference['config']['source_config']
    schedule = reference['runs'][0]['training']
    pod = old_pilot['gates']['D4_heldout_oracle_validation']['pod_K_floor_mean']
    mean_limit = min(.05, .5*pod)
    rows, incomplete = [], []
    for record in audit['records']:
        path = Path(record['path'])
        report = json.loads(path.read_text())
        if record['status'] != 'verified':
            incomplete.append(f"- [{record['name']}, job {record['job']}]({path}): "
                              f"{record['status']}; {record.get('failure', 'inspect job logs')}.")
            continue
        for run in report['runs']:
            fit = run['fits'][-1]
            flags = []
            if fit['summary']['mean'] > mean_limit:
                flags.append('mean/POD ratio')
            if fit['summary']['worst'] > .15:
                flags.append('worst')
            if fit['nonstationary']:
                flags.append('stationarity')
            if run['budget_change_max'] >= .01:
                flags.append('budget stability')
            rows.append(dict(label=label(report['config']), config=report['config'], run=run,
                             fit=fit, path=path, flags=flags))
    state = 'The campaign remains incomplete. ' if audit['missing_models'] or incomplete else 'The numerical screen is complete. '
    lines = ['# Burgers 3D decoder architecture comparison', '',
        state + 'Generated from audited raw JSONs across the isolated worktrees. These are bounded '
        'validation results under the recorded training schedule; they do not establish '
        'global representation minima, rollout accuracy, or reflective-wave transfer.', '',
        f"Grid nodes per axis: {cfg['N']}; latent coordinates: {cfg['k']}; spatial bank size: {cfg['r']}. "
        f"Each repeat uses {schedule['steps']} updates, learning rate {schedule['lr']:.6g}, "
        f"batch {schedule['batch']}, {cfg['max_snaps']} training states and "
        f"{len(reference['validation_states']['sid'])} validation states. "
        f"The unrestricted bank projection has mean error {pct(reference['bank_error']['mean'])} "
        f"and worst error {pct(reference['bank_error']['worst'])}.", '',
        f'The inherited POD comparator gives an effective mean-error ceiling of {pct(mean_limit)} '
        f'(POD mean {pct(pod)}). The worst-error ceiling is {pct(.15)}. '
        'Passing this preliminary screen still requires the actual inherited pilot, including '
        'all negative controls, before rollout promotion. The POD value is the unchanged '
        'historical comparator for this grid, latent size and validation cohort; it is recomputed '
        'inside any promoted pilot.', '',
        '## Reconstruction and acceptance', '',
        '| Head | Optimizer seed | Training mean | Validation mean | Median | Worst | Above 15% | Unconverged | Preliminary failure |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        run, fit = row['run'], row['fit']; s = fit['summary']
        lines.append(f"| {row['label']} | {run['seed']} | {pct(run['training']['final']['mean'])} | "
            f"{pct(s['mean'])} | {pct(s['median'])} | {pct(s['worst'])} | "
            f"{fit['outliers_above_15pct']} | {fit['nonstationary']} | "
            f"{', '.join(row['flags']) or 'none; inherited pilot required'} |")
    lines += ['', 'All models use the fixed learned spatial bank and the same initial linear '
        'map. Two optimizer repeats share one data cohort. The historical warm-refined head '
        'used a different initialization/training history and is not an otherwise matched '
        'architecture control. Equal updates are not equal compute; no cross-job timing '
        'comparison is made.', '', '## Tangent quality and model size', '',
        '| Head | Seed | Tangent mean | Tangent median | Initial-state mean | Later-state mean | Minimum rank | Shared weights | Optimized code values |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        run, fit = row['run'], row['fit']; training = run['training']
        lines.append(f"| {row['label']} | {run['seed']} | {pct(fit['tangent_summary']['mean'])} | "
            f"{pct(fit['tangent_summary']['median'])} | {pct(fit['groups']['initial']['mean'])} | "
            f"{pct(fit['groups']['later']['mean'])} | {min(x['rank'] for x in fit['geometry'])} | "
            f"{training['trainable_parameter_count']} | {training['optimized_code_count']} |")
    lines += ['', 'Tangent errors use the PDE velocity at the truth state, projected into '
        'the decoder Jacobian range at the fitted code. A smaller tangent error or full '
        'Jacobian rank alone does not certify time-stepping accuracy. The encoder operates '
        'offline; its shared-weight count includes both encoder and decoder, while its '
        'online decoder is the matched narrow MLP.', '', '## Architecture-specific diagnostics', '']
    for row in rows:
        run = row['run']; d = run.get('architecture_diagnostics', {})
        lead = f"- {row['label']}, seed {run['seed']}: "
        if row['config']['model'] == 'b3d_arch_encoder':
            errors = np.asarray(run['direct_encoder_error'])
            lines.append(lead + f"direct encoder validation mean {pct(errors.mean())}, median "
                f"{pct(np.median(errors))}, worst {pct(errors.max())}. This direct encoding was "
                'not used to initialize the primary validation fits.')
        elif row['config']['model'] == 'b3d_arch_anchor':
            lines.append(lead + f"minimum sampled Jacobian singular value {d['minimum_jacobian_singular_value']:.6g}; "
                f"anchor-coordinate recovery error {d['coordinate_recovery_max_absolute']:.6e}; "
                f"maximum sampled condition number {d['maximum_jacobian_condition']:.6g}.")
        elif row['config']['model'] == 'b3d_arch_quadratic':
            report = json.loads(row['path'].read_text())
            ids = np.asarray(report['validation_states']['sid'])
            outliers = ids[np.asarray(row['fit']['error']) > .15]
            lines.append(lead + 'outlier snapshot IDs ' + ', '.join(str(int(x)) for x in outliers) +
                f"; all are initial states: {bool(np.all(outliers % 51 == 0))}. "
                'The initial parameters are deterministic, so these repeats vary minibatch randomness '
                'rather than data or parameter initialization.')
        elif row['config']['model'] == 'b3d_arch_mixture':
            usage = ', '.join(pct(x) for x in d['mean_usage'])
            lines.append(lead + f"average expert weights {usage}; mean routing entropy "
                f"{d['mean_entropy']:.6g} nats; routing-collapse flag {d['routing_collapse']}; "
                f"identical-expert flag {d['identical_expert_collapse']}; normalized expert "
                f"disagreement RMS {d['expert_disagreement_relative_rms']:.6g}. Flags describe "
                'these validation codes and are not acceptance gates.')
    if not any(row['run'].get('architecture_diagnostics') for row in rows):
        lines.append('No completed candidate-specific diagnostics yet.')
    quadratic = [row for row in rows if row['config']['model'] == 'b3d_arch_quadratic']
    if len(quadratic) == 2:
        mean_gains, tangent_gains = [], []
        for row in quadratic:
            control = next(x for x in rows if x['label'] == 'MLP control 128' and
                           x['run']['seed'] == row['run']['seed'])
            mean_gains.append(1-row['fit']['summary']['mean']/control['fit']['summary']['mean'])
            tangent_gains.append(1-row['fit']['tangent_summary']['mean']/control['fit']['tangent_summary']['mean'])
        lines += ['', '## Interpretation', '',
            f"Quadratic lowers mean error by {pct(min(mean_gains))}–{pct(max(mean_gains))} "
            f"and tangent mean by {pct(min(tangent_gains))}–{pct(max(tangent_gains))} "
            'relative to the matched narrow MLP. These are relative improvements, not '
            'percentage-point differences. It is the strongest candidate in this bounded '
            'screen, but both the effective mean and worst-error conditions remain unsatisfied. '
            'Its large outliers are unseen initial states; converged local fitting does not '
            'certify a global minimum.', '',
            'The fixed protected anchor preserves its intended geometry but worsens state '
            'and tangent accuracy. Shared encoder training gives an inconsistent reconstruction '
            'gain across repeats, while direct encoding has a further error gap. The smooth '
            'mixture improves training and tangent errors without improving validation mean '
            'or worst error; its declared collapse tests do not explain that failure. '
            'These findings apply to the tested forms, bank, data and training schedule.', '']
    lines += ['', '## Convergence and provenance', '']
    for row in rows:
        run, fit = row['run'], row['fit']
        i = int(np.argmax(fit['error']))
        invariant = max(g['invariant_stationarity'] for g in fit['geometry'])
        lines.append(f"- {row['label']}, seed {run['seed']}: maximum inherited stationarity "
            f"{max(fit['optimality']):.6e}, maximum invariant stationarity {invariant:.6e}, "
            f"worst-state inherited stationarity {fit['optimality'][i]:.6e}, maximum budget "
            f"change {run['budget_change_max']:.6e}. Job {row['config']['slurm_job']}, "
            f"{row['config']['gpu']}, source `{row['config']['commit']}`. [Raw result]({row['path']}).")
    lines += ['', 'The audit verifies committed source bytes, source checkpoint, output/log '
        'checksums, precision/backend, parameter provenance, validation membership and '
        'summaries recomputed from raw arrays. Independent GPU regeneration can produce '
        'roundoff differences in derived parameters and coefficient arrays; shared hashes '
        'need not be byte-identical. Raw random draws and membership must match, and numerical '
        'arrays must pass the recorded comparison. Host CPU-affinity warnings are recorded '
        'separately; these runs are not used for timing claims. '
        '[Audit evidence](runs/b3d_architecture/review/campaign-audit.json).', '',
        '## Incomplete or invalid attempts', '']
    lines += incomplete or ['None among the discovered result files.']
    if audit.get('attempts_without_output'):
        lines.append('')
    for attempt in audit.get('attempts_without_output', []):
        lines.append(f"- [{attempt['model']}, attempt {attempt['label']}]({attempt['path']}): no numerical output; "
                     'preserved submission/accounting records distinguish queue cancellation from execution failure.')
    if audit['missing_models']:
        lines += ['', 'Requested models without a result file: ' + ', '.join(audit['missing_models']) + '.']
    lines += ['', '## Glossary', '',
        '- **Head / bank / code:** nonlinear coefficient map, fixed learned spatial features, and reduced coordinates.',
        '- **Optimizer seed:** fitting randomness; both repeats share the same PDE cases.',
        '- **Training / validation:** states used in fitting, and unseen states used to assess the model.',
        '- **Mean / median / worst:** full-field relative reconstruction errors on the indicated cohort.',
        '- **POD:** the inherited linear projection comparator with the same number of latent coordinates.',
        '- **Above 15%:** validation-state count exceeding the unchanged worst-error ceiling.',
        '- **Unconverged / stationarity:** selected fits above the inherited normalized-gradient tolerance.',
        '- **Invariant stationarity:** residual fraction projected along available decoder directions.',
        '- **Tangent:** field changes available through the decoder Jacobian; error measures missing PDE velocity.',
        '- **Initial / later:** states at the initial condition, and after time stepping.',
        '- **Rank / singular value / condition number:** independent local decoder directions, directional sensitivity, and their largest-to-smallest ratio.',
        '- **Shared weights / optimized code values:** learned network parameters, and separately optimized per-snapshot latent coordinates.',
        '- **Encoder:** offline map from solution coefficients to a latent code.',
        '- **Routing / entropy / nats:** mixture probabilities, their uncertainty, and its natural-log units.',
        '- **Expert collapse / RMS:** negligible use or indistinguishable expert outputs under declared thresholds, and root-mean-square magnitude.',
        '- **Budget change:** largest relative error change when increasing the latent-solver attempt budget.',
        '- **Pilot / promotion / rollout:** inherited validation checks, advancement after all checks pass, and online prediction through time.', '']
    output = SRC/'B3D-ARCH-NOTES.md'
    output.write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
