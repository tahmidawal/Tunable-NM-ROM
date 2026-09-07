"""Generate the mixture-arm notes exclusively from committed raw result fields."""
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]


def pct(value):
    return f'{100*value:.6f}%'


def main():
    own_path = HERE/'screen40/out/result.json'
    own = json.loads(own_path.read_text())
    audit = json.loads((HERE/'screen40/audit.json').read_text())
    pilot_path = SRC/'runs/b3d_repair/pilot_head33/out/result.json'
    pod = json.loads(pilot_path.read_text())['gates']['D4_heldout_oracle_validation']['pod_K_floor_mean']
    mean_limit = min(.05,.5*pod)
    lines = ['# Smooth-mixture Burgers 3D architecture screen', '',
        'Generated from the raw validation artifacts. Scientific conclusions remain '
        'provisional pending the coordinator\'s independent campaign audit; this screen '
        'does not establish rollout accuracy or reflective-wave transfer.', '',
        'The two-expert smooth mixture retains the fixed spatial bank, latent dimension, '
        'training objective, cohort, and common affine initialization. The control rows '
        'below use the matching optimizer seeds and update schedule. Their runs use '
        'different GPU instances; no elapsed-time comparison is made.', '',
        '| Model | Seed | Training mean | Validation mean | Median | Worst | Above worst gate | Unconverged | Tangent mean | Parameters |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    sources = []
    for label in ['mlp128','mlp192','screen40']:
        path = HERE/label/'out/result.json'
        result = json.loads(path.read_text())
        if not result['complete']:
            lines.append(f'| {label} | Incomplete | | | | | | | | |')
            continue
        display_label = 'mixture' if label == 'screen40' else label
        for run in result['runs']:
            fit = run['fits'][-1]
            s = fit['summary']
            lines.append(f'| {display_label} | {run["seed"]} | {pct(run["training"]["final"]["mean"])} | '
                f'{pct(s["mean"])} | {pct(s["median"])} | {pct(s["worst"])} | '
                f'{fit["outliers_above_15pct"]} | {fit["nonstationary"]} | '
                f'{pct(fit["tangent_summary"]["mean"])} | {run["training"]["trainable_parameter_count"]} |')
        c = result['config']
        sources.append(f'- [{label} raw result]({path.relative_to(SRC)}): '
                       f'job {c["slurm_job"]}, {c["gpu"]}, commit `{c["commit"]}`.')
    lines += ['', f'The unrestricted bank mean error is {pct(own["bank_error"]["mean"])}. '
        f'The retained POD comparison gives an effective mean limit of {pct(mean_limit)}, '
        f'using POD mean {pct(pod)} from the '
        f'[inherited reference]({pilot_path.relative_to(SRC)}). The worst-error limit '
        f'remains {pct(.15)}. These are representation-screen requirements; a successful '
        'screen still requires the inherited pilot and all negative controls.', '',
        '| Mixture seed | Initial mean | Later mean | p95 | Mean passes | Worst passes | Normalized-gradient passes | Budget relative change | Invariant stationarity max | Minimum rank |',
        '|---|---:|---:|---:|---|---|---|---:|---:|---:|']
    for run in own.get('runs',[]):
        if not run.get('fits'):
            continue
        fit = run['fits'][-1]
        s = fit['summary']
        lines.append(f'| {run["seed"]} | {pct(fit["groups"]["initial"]["mean"])} | '
            f'{pct(fit["groups"]["later"]["mean"])} | {pct(s["p95"])} | '
            f'{s["mean"]<=mean_limit} | {s["worst"]<=.15} | {fit["nonstationary"]==0} | '
            f'{run["budget_change_max"]:.9e} | '
            f'{max(g["invariant_stationarity"] for g in fit["geometry"]):.9e} | '
            f'{min(g["rank"] for g in fit["geometry"])} |')
    completed_fits = [run['fits'][-1] for run in own.get('runs',[]) if run.get('fits')]
    failed = [fit['summary']['mean'] > mean_limit or fit['summary']['worst'] > .15
              or fit['nonstationary'] > 0 for fit in completed_fits]
    verdict = ('Every completed mixture repeat fails the representation screen.'
               if failed and all(failed) else
               'The representation-screen outcome must be judged per repeat in the table above.')
    lines += ['', verdict + ' This bounded outcome concerns the declared architecture '
        'and training protocol; it does not establish that all mixtures fail.', '',
        'The selected fits are local multistart solutions, not certified global '
        'minima. The same errors, all starts and both solver budgets are retained in '
        'the JSON. Tangent errors measure the actual truth-state Burgers velocity '
        'outside the available decoder directions; they do not certify curvature or rollout.', '',
        '| Mixture seed | Mean expert usage | Mean entropy | Saturated >0.95 | Saturated >0.99 | Relative expert disagreement RMS | Routing collapse | Identical experts |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for run in own.get('runs',[]):
        if 'architecture_diagnostics' not in run:
            continue
        d = run['architecture_diagnostics']
        usage = ', '.join(f'{x:.9f}' for x in d['mean_usage'])
        lines.append(f'| {run["seed"]} | {usage} | {d["mean_entropy"]:.9f} | '
            f'{d["saturation_counts"]["0.95"]} | {d["saturation_counts"]["0.99"]} | '
            f'{d["expert_disagreement_relative_rms"]:.9f} | {d["routing_collapse"]} | '
            f'{d["identical_expert_collapse"]} |')
    lines += ['', 'Every validation routing weight, entropy and expert disagreement '
        'is saved. Routing-collapse and identical-expert flags use the declared '
        'descriptive thresholds in the [arm design](B3D-MIXTURE-DESIGN.md). Neither '
        'is a scientific acceptance gate, and healthy routing does not imply useful '
        'specialization. Equal updates and approximate parameter-count controls do '
        'not establish equal compute.', '', 'Source artifacts:', ''] + sources
    lines += ['', '[Implementation-agent provenance and independent NumPy checks]'
        '(runs/b3d_architecture/screen40/audit.json) verify pulled checksums, source '
        'bytes, GPU/f64/highest configuration, seed/state membership, unchanged bank '
        'and QR, checkpoint hashes, independent coefficient reconstruction, and '
        'routing diagnostics. The coordinator\'s separate scientific audit remains '
        'pending. ' + ('Host CPU-affinity warnings are retained in the logs. '
        if audit['host_affinity_warning'] else 'No host CPU-affinity warning occurred in this job. ') +
        'Backend preflight confirms GPU execution. No pilot, test cohort, larger grid, '
        'wave run, or online timing was opened for this arm.', '',
        '## Glossary', '',
        '- **Model / seed:** head architecture and optimizer randomness; optimizer '
        'repeats reuse the same generated data cohort.',
        '- **Training / validation:** snapshots used for fitting, and unseen states '
        'reserved for selecting the architecture.',
        '- **Mean / median / p95 / worst:** average, middle, 95th percentile, and '
        'largest relative full-field reconstruction errors.',
        '- **Above worst gate:** number of validation states exceeding the retained '
        'worst-error limit.',
        '- **Unconverged / normalized gradient:** selected latent fits failing the '
        'inherited first-order stopping requirement.',
        '- **Tangent mean:** average fraction of the true PDE velocity unavailable '
        'through an infinitesimal latent change.',
        '- **Parameters:** trained shared head weights, excluding separately '
        'optimized per-snapshot latent codes.',
        '- **Initial / later:** validation at initial time, or after time stepping.',
        '- **POD / bank / QR:** the inherited linear projection comparator, the '
        'fixed learned spatial features, and orthonormal coordinates for that bank.',
        '- **Budget / multistart:** latent-solver attempt limit and several initial '
        'guesses; the minimum-error fit is kept even if unconverged.',
        '- **Invariant stationarity:** normalized reconstruction residual projected '
        'onto the decoder\'s available local directions.',
        '- **Rank:** numerically independent Jacobian directions under the declared '
        'SVD threshold; full rank does not guarantee small error.',
        '- **Expert / usage:** one residual head, and its average smooth routing weight.',
        '- **Entropy / saturation:** routing uncertainty in natural-log units, and '
        'a near-exclusive routing weight.',
        '- **Disagreement RMS:** root mean square of expert-output differences '
        'normalized by the fixed training output scale.',
        '- **Collapse / identical experts:** the separate, cohort-specific usage '
        'and output-similarity flags defined in the design.',
        '- **Pilot / rollout:** inherited validation acceptance checks, and online '
        'time evolution from an initial condition.',
        '- **f64 / highest:** double-precision arithmetic and the repository\'s '
        'required highest matrix-multiplication precision.', '']
    output = SRC/'B3D-MIXTURE-NOTES.md'
    output.write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
