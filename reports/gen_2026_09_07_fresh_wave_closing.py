"""Generate exact closing evidence from immutable run JSONs, for the canonical log."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads((args.runs/'verify03/cluster/out/verification/result.json').read_text())
    assert reference['passed']
    lines = ['## 2026-09-07', '', '### Fresh absorbing and reflective wave head comparison — completed bounded campaign', '',
             'User-approved wave branch/worktree was created from `3f8ccc4`, using its own cluster namespace. Implementation owner wrote only the wave tree; coordinator wrote only the repair tree plus this canonical log and canonical reports. All old wave code, banks, checkpoints, data and conclusions remain untrusted historical material. Fresh FOM, learned coordinate network, heads and actual reduced evolution were implemented independently. No merge was performed.', '',
             'Reference attempts `verify01` and `verify02` failed their declared resolution gates and are retained. Before neural training, the localized family was explicitly revised to a Gaussian core with a smooth compact taper and narrower center range; no Gaussian descriptor enters the head. Expanded fresh verification and independent spectral/field audits accepted the bounded pilot. This does not prove uniform accuracy outside its tested family.', '',
             f"Accepted reference job `{reference['provenance']['job_id']}`, source `{reference['provenance']['source_commit']}`, passed {len(reference['gates'])} declared gates. Two cluster full-pipeline preflights and the local GPU component checks also passed before scientific training.", '',
             'The table below is generated from original primary-run JSONs. Errors are mean / median / worst of each trajectory\'s maximum energy norm of state error, normalized by the initial energy norm. Outliers include every trajectory above the declared accuracy ceiling; the full acceptance rule also requires displacement, velocity, fit and temporal checks.', '',
             '| Boundary | Head | Optimizer seed | Energy error mean | Median | Worst | Outliers | Original time check | Full target |',
             '|---|---|---|---|---|---|---|---|---|']
    campaigns = []
    for label, bc in [('reflective01', 'dirichlet'), ('absorbing02', 'absorbing')]:
        r = json.loads((args.runs/label/'cluster/out/campaign/result.json').read_text())
        cleanup = json.loads((args.runs/label/'cleanup.json').read_text())
        assert cleanup['remote_deleted_and_absence_checked'] and r['completed'] and not r['final_test_opened']
        entry = r['boundary_results'][bc]
        campaigns.append((label, r, entry))
        for arm in entry['arms']:
            s = next(s for s in arm['rollout']['summaries'] if s['dt']==arm['rollout']['primary_dt'])['energy_state']
            lines.append(f"| {bc} | {arm['name']} | {arm['optimizer_seed']} | {s['mean']:.17g} | {s['median']:.17g} | {s['worst']:.17g} | {s['outliers']} | {arm['rollout']['refinement_passed']} | {arm['accuracy_passed']} |")
    lines += ['', 'All original heads fail the full provisional engineering target. All original trajectories complete; completion and a small energy-balance defect do not establish accurate evolution. MLP results pass the original time check. Reflective quadratic cases require the separately reported continuation below.', '']
    for label, r, entry in campaigns:
        b = next(b for b in entry['linear_baselines'] if b['label']=='learned_bank_linear_r')
        c = r['config']
        lines.append(f"`{label}` job `{r['provenance']['job_id']}`, device `{r['provenance']['device_kind']}`, source `{r['provenance']['source_commit']}`: {c['n']} intervals, learned rank {c['rank']}, latent {c['latent']}, training seed {c['train_seed']} / count {c['train_count']}, validation seed {c['validation_seed']} / count {c['validation_count']}. Unrestricted learned-bank linear energy-error median {b['energy_state']['median']:.17g}, worst {b['energy_state']['worst']:.17g}; all its physical-error outlier counts are zero. This higher-dimensional comparator does not isolate head structure from dimension.")
        nonstationary = [(a['name'], a['optimizer_seed'], a['latent_fit']['nonstationary']) for a in entry['arms'] if a['latent_fit']['nonstationary']]
        lines.append('Selected nonstationary snapshot fits: '+repr(nonstationary)+'. These are explicitly retained failures, not discarded observations.')
    lines += ['', '#### Frozen-checkpoint numerical continuation', '',
              'Exact original latent position and velocity, trained heads, learned bank and operators were retained. The old finest step was repeated for cross-GPU parity before the extra steps. No retraining, refitting or final-cohort opening occurred. The original primary-step verdict was not replaced. All input checkpoint hashes were reconciled with the completed parent archive.', '',
              '| Attempt / job | Finest pair passing cases | Unresolved cases | Worst finest-pair state difference | Finest-step state error median | Worst | Resolved-subset state error median | Resolved-subset outliers |',
              '|---|---|---|---|---|---|---|---|']
    for label in ['refquad20001', 'refquadvel20002', 'refquad20101', 'refquadvel20101']:
        r = json.loads((args.runs/label/'cluster/out/refinement/result.json').read_text())
        assert r['old_fine_parity_passed']
        assert json.loads((args.runs/label/'cleanup.json').read_text())['remote_deleted_and_absence_checked']
        pairs = [p for p in r['adjacent_refinement'] if p['fine_dt']==r['diagnostic_config']['rom_dts'][-1]]
        good = {p['case'] for p in pairs if p['passed']}
        bad = [p['case'] for p in pairs if not p['passed']]
        summary = r['fine_diagnostic']['summaries'][-1]['energy_state']
        cases = [x for x in r['fine_diagnostic']['cases'] if x['dt']==r['diagnostic_config']['rom_dts'][-1] and x['case'] in good]
        # Keep exact saved case-key schema visible if it ever changes.
        errors = sorted(x['max_energy_state_error'] for x in cases)
        midpoint = len(errors)//2
        median = (errors[midpoint-1]+errors[midpoint])/2 if len(errors)%2==0 else errors[midpoint]
        outliers = sum(x>r['original_config']['accuracy_target'] for x in errors)
        lines.append(f"| {label} / {r['provenance']['job_id']} | {len(good)} / {len(pairs)} | {bad} | {max(p['max_energy_state_difference'] for p in pairs):.17g} | {summary['median']:.17g} | {summary['worst']:.17g} | {median:.17g} | {outliers} / {len(errors)} |")
    lines += ['', 'The continuation remains a bounded numerical diagnostic; unresolved cases are not claimed globally converged. The first continuation used regenerated scales differing only at roundoff; the independent audit recomputed full-field metrics using original stored normalizations. Later attempts used the stored scales exactly.', '',
              'Independent reviewers checked primary summaries, fit selection and stationarity, phase masks, completion, energy and refinement aggregation. Coordinator NumPy/SciPy audits independently reconstructed the neural bank, mass/edge-stiffness/face-damping operators, head values/Jacobians, reconstruction and tangent field errors, rollout energies, and continuation field metrics. Audit scripts and JSONs live in the repair campaign `review/` directory. Trained checkpoints and native outputs are tracked on the wave branch; full checked archives, logs, submissions, cancellation records and cleanup receipts are tracked on the repair branch.', '',
              'Every accepted run logged GPU backend, f64 and highest precision. No cross-job wall-clock comparison is made. Original absorbing job `3338489` was canceled before starting through the numeric-ID helper; its staged source was archived and cleaned. Local stage `refquadvel20001` was never submitted and was superseded by its hardened successor. Every executed cluster directory was checksum-pulled and deleted after completion.', '',
              'Retractions/limits: no earlier wave finding is reinstated; no new head succeeds at the full target; original reflective quadratic primary trajectories are not globally time converged. Optimizer repeats share one data split, and MLP/quadratic parameter counts differ. The final cohort remains unopened. These are scalar two-dimensional waves with separate learned weights per boundary; no three-dimensional wave, Navier–Stokes or grid-independent online-cost result follows.', '',
              'Next proposed controlled experiment: retain the verified FOM and learned bank, compare a latent-dimension ladder with matching linear dimensions, then assess wave-state/dynamics-aware training. Velocity-tangent fitting alone is insufficient here. Further architecture experiments and merging the wave branch into the repair branch remain open pending the user\'s choice.', '',
              'Canonical generated report: `reports/2026-09-07-fresh-wave-head-transfer.md`; source generator, figures and reproducibility manifest are beside it. The report and this closing table derive numerical values directly from immutable JSONs.', '']
    args.out.write_text('\n'.join(lines))


if __name__=='__main__':
    main()
