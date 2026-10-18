"""Generate the dated NM-ROM handoff from audited result JSONs; no GPU work.

The manifest freezes branch metadata on the first generation. Reproduction checks
every scientific input hash before regenerating the same document.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(value):
    return f'{100 * value:.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    repair = repo / 'worktrees/2026-09-06-burgers3d-repair'
    wave = repo / 'worktrees/2026-09-06-wave-head-transfer'
    runs = repair / 'experiments/separable-decoder/runs'
    campaign = runs / 'fresh_wave_campaign'
    report = repo / 'reports/2026-09-07-nmrom-handoff.md'
    manifest_path = report.with_suffix('.manifest.json')
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    inputs = {}

    def read(path):
        path = Path(path)
        key = str(path.relative_to(repo))
        digest = sha(path)
        if old is not None and old['inputs_sha256'].get(key) != digest:
            raise RuntimeError(f'Handoff input changed or was added: {key}')
        inputs[key] = digest
        return json.loads(path.read_text())

    def link(path, label=None):
        path = Path(path)
        target = '../' + str(path.relative_to(repo))
        return f'[{label or path.name}]({target})'

    def primary(arm):
        return next(s for s in arm['rollout']['summaries']
                    if s['dt'] == arm['rollout']['primary_dt'])

    audit = read(runs / 'b3d_architecture/review/campaign-audit.json')
    assert not audit['missing_models']
    b3d = [(r, read(repo / Path(r['path']).relative_to(repo))) for r in audit['records']]
    assert all(record['status'] == 'verified' for record, _ in b3d)
    pilot = read(runs / 'b3d_repair/pilot_head33/out/result.json')
    waves = [(label, read(campaign / label / 'cluster/out/campaign/result.json'))
             for label in ['reflective01', 'absorbing02']]
    refinements = [(label, read(campaign / label / 'cluster/out/refinement/result.json'))
                   for label in ['refquad20001', 'refquadvel20002',
                                 'refquad20101', 'refquadvel20101']]
    fields = read(repo / 'reports/figures/2026-09-07-fresh-wave-fields.json')
    read(repo / 'reports/animations/2026-09-07-wave-evolution.json')
    read(campaign / 'review/final-review-disposition.json')
    for label, _ in refinements:
        assert read(campaign / f'review/{label}-audit.json')['passed']
    for _, run in waves:
        p = run['provenance']
        assert run['completed'] and not run['final_test_opened']
        assert p['jax_backend'] == 'gpu' and p['x64'] and p['matmul_precision'] == 'highest'
    cfg = waves[0][1]['config']
    bc_names = {'dirichlet': 'Reflective', 'absorbing': 'Absorbing'}
    head_names = {'mlp': 'MLP', 'quadratic': 'Quadratic',
                  'mlp_velocity': 'MLP + velocity', 'quadratic_velocity': 'Quadratic + velocity'}
    b3d_names = {'b3d_arch_baseline': 'MLP', 'b3d_arch_anchor': 'Protected anchor',
                'b3d_arch_quadratic': 'Quadratic', 'b3d_arch_encoder': 'Shared encoder',
                'b3d_arch_mixture': 'Smooth mixture'}
    slugs = ['2026-09-04-separable-tensor-consolidated', '2026-09-06-burgers3d-repair',
             '2026-09-06-wave-head-transfer', '2026-09-06-b3d-anchor',
             '2026-09-06-b3d-quadratic', '2026-09-06-b3d-encoder', '2026-09-06-b3d-mixture']
    if old:
        snapshot = old['repository_snapshot']
    else:
        snapshot = {'main_before_handoff': subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), 'worktrees': {}}
        for slug in slugs:
            tree = repo / 'worktrees' / slug
            snapshot['worktrees'][slug] = {
                'branch': subprocess.check_output(['git', '-C', str(tree), 'branch', '--show-current'], text=True).strip(),
                'commit': subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()}

    lines = [
        '# NM-ROM handoff: separable decoders, Burgers 3D and fresh waves', '',
        'Dated handoff snapshot for the results finalized on 2026-09-07. The completed '
        'architecture screens and fresh-wave results are independently reviewed, bounded '
        'validation evidence; the engineering acceptance targets remain provisional and '
        'no tested compressed head passes its full target. This handoff adds no numerical experiment.', '',
        '**Read [LAB-LOG.md](../LAB-LOG.md) first in the next session.** It remains the '
        'canonical record, including retractions and later changes. This document is a '
        'fixed transfer note, not a second mutable project-status file.', '',
        '## What the next person needs to know', '',
        '- The separable architecture is implemented: a learned coordinate network supplies '
        'spatial functions, and a small latent head supplies their coefficients. The '
        'spatial bank supports substantially better representation than the tested compressed heads.',
        '- Burgers 3D: the quadratic head is the strongest mean-reconstruction improvement '
        'in the matched screen, but unseen initial states still produce large errors. '
        'Representation gates block promotion to rollout and cost claims.',
        '- Fresh reflective and absorbing waves: the full-bank linear model meets the '
        'declared initial-normalized physical-error ceilings. Every tested compressed head '
        'misses the full target. Good snapshot fitting and energy balance do not establish accurate evolution.',
        '- All earlier wave experiments are excluded from trusted evidence by the user. '
        'The fresh benchmark, its verification and its checkpoints are the only wave lineage to continue.',
        '- Wave dynamics are currently scalar and two-dimensional. Three-dimensional '
        'waves and Navier–Stokes remain future extensions; surface plots do not change the PDE dimension.', '',
        '## User intent and decisions to preserve', '',
        'The user wants the NM-ROM framework to transfer across PDEs and dimensions, '
        'starting with the current smooth/localized data style. The immediate sequence '
        'includes Burgers 3D, ordinary and reflective waves, and later incompressible '
        'Navier–Stokes. Broader initial-condition families are a later question. '
        'Do not condition the neural head on Gaussian centers, widths or other family '
        'descriptors. A Gaussian-like data generator is allowed; a family-specific '
        'online parameter fit is not the selected direction. Separate trained weights '
        'per PDE/boundary are the tested scope; unchanged-weight transfer is unproven.', '',
        'The most recent visualization wording was “Can I not get it in images please.” '
        'This was interpreted as requesting still images showing successive evolution '
        'times. That follow-up was interrupted by this handoff request, so a newly '
        'arranged still-image sequence remains open. Existing static field grids, GIFs '
        'and an interactive time viewer are already saved below.', '',
        '## Architecture and mathematical contract', '',
        r'The general decoder is $u(x;z)=b(x)\sum_{j=1}^{r}g_j(x)h_j(z)$. '
        r'The factor $b$ enforces the appropriate essential boundary constraint; '
        r'$g$ is a trained coordinate network, $h$ is the coefficient head, and '
        r'$z\in\mathbb{R}^{k}$ is the state solved for online. '
        'The coordinate network is independent of the latent code, so spatial '
        'operators can be assembled or projected offline. The bank is learned; '
        'POD appears only as an explicit baseline. Mass-weighted QR changes bank '
        'coordinates without changing its span. Coefficient PCA initializes the '
        'heads and codes; it does not replace the coordinate network with POD.', '',
        'Earlier non-wave implementation includes weak residuals, empirical quadrature '
        'for nonlinear terms, exact reduced linear operators, and tensor contractions '
        'on applicable PDEs. That lineage is in the consolidated branch and canonical '
        'log. Its older cost numbers are not evidence of speed for this Burgers 3D '
        'screen or the fresh wave campaign.', '',
        'For waves, displacement and velocity share a consistent latent state:', '',
        r'$$u=Gh(z),\qquad v=GJ_h(z)w,\qquad \dot z=w,$$', '',
        r'$$\dot w=\underset{a}{\operatorname{argmin}}\,'
        r'\left\|J_h(z)a+H_h(z)[w,w]+D_rJ_h(z)w+K_rh(z)\right\|_2^2.$$', '',
        'Here the mass-orthonormal bank is fixed, the reduced stiffness and boundary '
        'damping operators are fixed for each physical case, and the Hessian term '
        'accounts for curvature of the decoder. The least-squares solve uses QR '
        'without ridge regularization; each Runge–Kutta stage has rank and finite-value '
        'guards. Velocity is the Jacobian lift of latent velocity, not an independently '
        'decoded field. The full-bank linear comparator is propagated independently '
        'by a matrix exponential.', '',
        r'The reference satisfies $M\dot v+Cv+Ku=0$, $\dot u=v$, with '
        r'$E(u,v)=\tfrac12(v^TMv+u^TKu)$ and $\dot E=-v^TCv$. '
        'Mass uses tensor-product trapezoid weights; stiffness uses edge differences; '
        'absorber damping includes every boundary face and corner contribution. '
        'Fixed Dirichlet walls cause sign-inverting reflection. The first-order '
        'Sommerfeld absorber is approximate, especially for oblique incidence. '
        'Its boundary-model reflection is separate from discretization error. '
        'Its constant-displacement nullspace also requires an independent mean-field check.', '',
        '## Burgers 3D: completed representation screen', '']

    reference = b3d[0][1]
    source_cfg = reference['config']['source_config']
    schedule = reference['runs'][0]['training']
    pod = pilot['gates']['D4_heldout_oracle_validation']['pod_K_floor_mean']
    # Same declared ceiling as the audited architecture-note generator.
    mean_limit = min(.05, .5 * pod)
    worst_limit = .15
    lines += [
        f"The fixed bank uses {source_cfg['N']} grid nodes per axis, {source_cfg['r']} "
        f"spatial functions and {source_cfg['k']} latent coordinates. Each matched repeat "
        f"uses {schedule['steps']} updates, learning rate {schedule['lr']}, batch "
        f"{schedule['batch']}, {source_cfg['max_snaps']} training states and "
        f"{len(reference['validation_states']['sid'])} validation states. The unrestricted "
        f"bank projection has mean {pct(reference['bank_error']['mean'])}% and worst "
        f"{pct(reference['bank_error']['worst'])}% relative field error.", '',
        f"The preliminary mean ceiling is {pct(mean_limit)}% from the inherited POD "
        f"comparison; the worst-state ceiling is {pct(worst_limit)}%. These remain "
        'provisional promotion criteria: even a screen pass requires the full inherited '
        'pilot, including negative controls and a recomputed POD comparator.', '',
        'Errors below are per-state relative field L2 reconstruction errors after '
        'multistart latent fitting, summarized across validation states. They are not '
        'rollout errors. Percent displays are rounded; linked JSONs retain full precision.', '']
    rows = []
    quadratic_rows = []
    for record, data in b3d:
        name = b3d_names[data['config']['model']]
        if data['config']['model'] == 'b3d_arch_baseline':
            name += f" {data['config']['model_config']['width']}"
        for run in data['runs']:
            fit = run['fits'][-1]
            s = fit['summary']
            rows.append([name, run['seed'], pct(s['mean']), pct(s['median']), pct(s['worst']),
                         fit['outliers_above_15pct'], fit['nonstationary']])
            if name == 'Quadratic':
                control = next(r for r in reference['runs'] if r['seed'] == run['seed'])
                quadratic_rows.append((1-s['mean']/control['fits'][-1]['summary']['mean'],
                                       1-fit['tangent_summary']['mean']/control['fits'][-1]['tangent_summary']['mean']))
    lines += [table(['Head', 'Seed', 'Mean %', 'Median %', 'Worst %',
                     'Outliers', 'Nonstationary fits'], rows), '',
        f"Quadratic reduces mean reconstruction error by {pct(min(v[0] for v in quadratic_rows))}%–"
        f"{pct(max(v[0] for v in quadratic_rows))}% and mean tangent error by "
        f"{pct(min(v[1] for v in quadratic_rows))}%–{pct(max(v[1] for v in quadratic_rows))}% "
        'relative to the matched narrow MLP. These are relative reductions, not '
        'percentage-point changes. It still fails both mean and worst-state criteria. '
        'Its large outliers are unseen initial states, so a rollout cannot be rescued '
        'merely by giving it an inaccurate starting representation.', '',
        'The protected anchor achieves its intended geometry but worsens accuracy. '
        'Shared encoder training gives inconsistent gains, and direct encoder output '
        'has an additional reconstruction gap. The smooth mixture improves training '
        'and tangent fitting without improving validation mean or worst error; the '
        'recorded collapse tests do not explain its failure. The wider MLP is also '
        'not an improvement. These findings are limited to the tested forms and budget.', '',
        'Optimizer repeats share a data cohort. Quadratic initialization is deterministic; '
        'its repeats vary minibatch randomness. Local stationarity is not proof of '
        'globally optimal fitting. The earlier warm-refined incumbent had a different '
        'training history and must not be presented as a matched architecture control. '
        'No candidate is promoted to a Burgers 3D rollout, final test or cost claim.', '',
        'Detailed state/tangent/geometry diagnostics and source-job provenance: ' +
        link(repair / 'experiments/separable-decoder/B3D-ARCH-NOTES.md') + '. '
        'Independent campaign audit: ' + link(runs / 'b3d_architecture/review/campaign-audit.json') + '.', '',
        '## Fresh waves: verified reference, failed compressed-head target', '',
        'The first fresh verification attempts, `verify01` and `verify02`, failed '
        'their resolution gates and remain historical failures. Before any scientific '
        'neural comparison, the family was changed to a Gaussian core with a smooth '
        'compact cutoff and narrower center coverage. No old wave code, bank, '
        'checkpoint, data or gate was inherited as trusted infrastructure.', '',
        f"Reference `verify03`, job `{cfg['reference_evidence']['job_id']}`, source "
        f"`{cfg['reference_evidence']['source_commit']}`, passed "
        f"{cfg['reference_evidence']['gate_count']} declared gates and independent review. "
        'Its acceptance is bounded to the recorded family and mesh. Absorber '
        'reference uncertainty is empirical/conditional, not a uniform theorem.', '',
        table(['Frozen setting', 'Value'], [
            ['Mesh intervals per axis', cfg['n']], ['Learned bank / latent / weak test dimensions',
             f"{cfg['rank']} / {cfg['latent']} / {cfg['weak_tests']}"],
            ['Training / validation / sealed final trajectories',
             f"{cfg['train_count']} / {cfg['validation_count']} / {cfg['final_count']}"],
            ['Training / validation / sealed final seeds',
             f"{cfg['train_seed']} / {cfg['validation_seed']} / {cfg['final_seed']}"],
            ['Optimizer seeds', ', '.join(map(str, cfg['optimizer_seeds']))],
            ['Final time / saved-time interval', f"{cfg['end_time']} / {cfg['observation_dt']}"],
            ['Primary time step', cfg['rom_primary_dt']],
            ['Original time-step ladder', ', '.join(map(str, cfg['rom_dts']))],
            ['Bank / head training updates', f"{cfg['bank_steps']} / {cfg['head_steps']}"],
            ['Gaussian width range', str(cfg['parameter_ranges']['gaussian_standard_deviation'])],
            ['Compact support half-width range', str(cfg['parameter_ranges']['support_half_width'])],
        ]), '',
        f"The provisional engineering target requires every validation trajectory to "
        f"complete with displacement, velocity and energy-state errors at most "
        f"{pct(cfg['accuracy_target'])}%, with finest-step-pair differences at most "
        f"{pct(cfg['rom_refinement_target'])}% on each physical scale. Selected latent "
        'fits must also satisfy stationarity, rank and doubled-budget stability gates. '
        'These are validation results; the final cohort is still sealed.', '',
        '### How to read the wave errors', '',
        r'Let $\delta u=u_{\rm ROM}-u_{\rm ref}$, $\delta v=v_{\rm ROM}-v_{\rm ref}$, '
        r'$U_0=\|u_{\rm ref}(0)\|_M$, $E_0=E(u_{\rm ref}(0),v_{\rm ref}(0))$, '
        r'and $V_0=\sqrt{2E_0}$. The stored errors are', '',
        r'$$e_u(t)=\frac{\|\delta u(t)\|_M}{U_0},\qquad '
        r'e_v(t)=\frac{\|\delta v(t)\|_M}{V_0},\qquad '
        r'e_E(t)=\sqrt{\frac{E(\delta u(t),\delta v(t))}{E_0}}.$$', '',
        '**Every rollout table below summarizes each trajectory’s maximum over saved '
        'times, then takes the mean, median and worst across trajectories.** '
        'It is not a time average or a continuous-time supremum. The energy-state '
        'metric is the energy norm of the state difference, not the difference '
        'between two solution energies. Normalization is fixed by the initial state.', '',
        r'Instantaneous relative displacement error instead divides by '
        r'$\|u_{\rm ref}(t)\|_M$. An absorbing field can become very small, making '
        'that ratio large while the initial-normalized error is small. The illustrated '
        'case below demonstrates the distinction; it is not a cohort summary.', '']
    rows = []
    for bc, field in fields['boundaries'].items():
        m = field['metrics']['mlp']
        rows.append([bc.title(), field['case'], field['optimizer_seed'], f"{field['times'][-1]:.8g}",
                     pct(m['displacement'][-1]), pct(m['current_field_displacement'][-1])])
    lines += [table(['Boundary', 'Case', 'Seed', 'Time', 'Error / initial norm %',
                     'Error / current norm %'], rows), '',
        '### Fresh linear baselines', '',
        r'This table and the original nonlinear rollout table report the energy-state '
        r'error $e_E$. The complete wave report also gives displacement and velocity '
        'errors separately.', '']
    rows = []
    for _, run in waves:
        for bc, entry in run['boundary_results'].items():
            for base in entry['linear_baselines']:
                s = base['energy_state']
                rows.append([bc_names[bc], 'Learned bank' if base['label']=='learned_bank_linear_r' else 'Fresh POD',
                             base['rank'], pct(s['mean']), pct(s['median']), pct(s['worst']), s['outliers']])
    lines += [table(['Boundary', 'Linear model', 'Displacement dimension', 'Mean %', 'Median %',
                     'Worst %', 'Energy outliers'], rows), '',
        f"The full learned-bank model has {2*cfg['rank']} displacement-plus-velocity "
        f"state coordinates versus {2*cfg['latent']} for the compressed wave heads. "
        'It meets all declared initial-normalized physical-error ceilings, but this '
        'dimension mismatch leaves compression and nonlinear dynamics coupled. '
        'The fresh POD baseline at the small dimension also struggles. Neither '
        'comparison proves that a nonlinear head cannot work.', '',
        '### All original nonlinear rollout results', '']
    rows = []
    passing = []
    for _, run in waves:
        for bc, entry in run['boundary_results'].items():
            passing.append(f"{bc_names[bc]}: {sum(a['accuracy_passed'] for a in entry['arms'])}/{len(entry['arms'])}")
            for arm in entry['arms']:
                s = primary(arm)['energy_state']
                rows.append([bc_names[bc], head_names[arm['name']], arm['optimizer_seed'],
                             pct(s['mean']), pct(s['median']), pct(s['worst']), s['outliers'],
                             'pass' if arm['rollout']['refinement_passed'] else 'unresolved'])
    lines += [table(['Boundary', 'Head / objective', 'Seed', 'Mean %', 'Median %', 'Worst %',
                     'Energy outliers', 'Original time check'], rows), '',
        '**Full target passes — ' + '; '.join(passing) + '.** '
        'All original trajectories completed. All MLP runs and absorbing quadratic '
        'runs pass their original time-step checks, so their large errors cannot '
        'be dismissed as the observed time-step failure affecting reflective quadratic runs. '
        'Velocity-tangent training improves fitting diagnostics without consistently '
        'improving actual evolution.', '']
    capacities = {arm['kind']: arm['parameter_count'] for arm in waves[0][1]['boundary_results']['dirichlet']['arms']}
    lines += [f"MLP has {capacities['mlp']} head parameters; quadratic has "
              f"{capacities['quadratic']}. Equal update counts do not make this a "
              'parameter- or compute-matched comparison. Optimizer repeats share '
              'the same data split. Nonstationary snapshot fits are retained as '
              'gate failures in the full report; later snapshot fits are not '
              'the initial fits used to start the rollout.', '',
        '### Frozen-checkpoint continuation of reflective quadratic runs', '',
        'Follow-ups retained each original bank, head, physical operators and '
        'stored initial latent position and velocity. They repeated the original '
        'finest step for hardware parity and then reduced the step further. '
        'No retraining, refitting or final-test opening occurred; the original '
        'primary-step verdict remains unchanged.', '']
    rows = []
    for label, data in refinements:
        assert not any(data[k] for k in ['retraining_performed', 'initial_fitting_performed', 'final_test_opened'])
        fine_dt = data['diagnostic_config']['rom_dts'][-1]
        s = next(s for s in data['fine_diagnostic']['summaries'] if s['dt'] == fine_dt)['energy_state']
        pairs = data['fine_diagnostic']['finest_two_refinement']
        rows.append([head_names[data['name']], data['optimizer_seed'], data['provenance']['job_id'],
                     f"{sum(p['passed'] for p in pairs)}/{len(pairs)}",
                     ', '.join(str(p['case']) for p in pairs if not p['passed']) or 'none',
                     pct(s['median']), pct(s['worst']), s['outliers']])
    lines += [table(['Head / objective', 'Seed', 'Job', 'Finest-pair passes',
                     'Unresolved case', 'Finest-step energy median %', 'Worst %', 'Energy outliers'], rows), '',
        f"The added steps are {', '.join(str(v) for v in refinements[0][1]['diagnostic_config']['rom_dts'][1:])}. "
        'Every temporally resolved case still misses accuracy. The remaining case '
        'must stay labeled unresolved; its error cannot be attributed entirely '
        'to architecture. The first continuation regenerated normalization scales '
        'with roundoff differences; the independent audit recomputed metrics with '
        'the original scales. Later continuations used the stored scales exactly.', '',
        '## Saved code, evidence and visualizations', '',
        'The complete wave write-up is ' + link(repo / 'reports/2026-09-07-fresh-wave-head-transfer.md') +
        '. It contains all displacement, velocity, representation, phase, fitting, '
        'reference-uncertainty and refinement tables omitted from this compact handoff.', '',
        table(['Location', 'Contents'], [
            [link(wave / 'experiments/fresh-wave-head', 'Fresh wave implementation'),
             '`fresh_fom.py`, `fresh_models.py`, `fresh_learning.py`, `fresh_rom.py`, '
             '`fresh_evaluate.py`, `fresh_campaign.py` and `fresh_checkpoint_refine.py`'],
            [link(wave / 'experiments/fresh-wave-head/campaign-config.json', 'Frozen campaign config'),
             'Family, seeds, schedules, gate definitions; use with `FROZEN-MATH.json`'],
            [link(repair / 'experiments/separable-decoder', 'Burgers architecture implementation'),
             '`b3d_arch_bench.py`, `b3d_arch_baseline.py`, `b3d_arch_pilot.py`; '
             'candidate implementations/checkpoints live in their owning worktrees'],
            [link(campaign, 'Complete fresh-wave campaign archive'),
             'Submissions, outputs, source/checksum manifests, cleanup receipts and movie export'],
            [link(campaign / 'review', 'Independent review evidence'),
             'Model/operator/full-field/refinement audit scripts and JSONs; final review disposition'],
            [link(repo / 'reports/figures/2026-09-07-fresh-wave-reflective-displacement-fields.png', 'Reflective still-image grid'),
             'Reference, full-bank linear, MLP and absolute-error fields at saved times'],
            [link(repo / 'reports/figures/2026-09-07-fresh-wave-absorbing-displacement-fields.png', 'Absorbing still-image grid'),
             'Same layout; velocity grids and PDF exports are beside these files'],
            [link(repo / 'reports/animations/2026-09-07-wave-evolution.html', 'Interactive wave evolution'),
             'Standalone play/pause, time slider, boundary/view and playback-speed controls'],
            [link(repo / 'reports/animations/2026-09-07-reflective-wave-surface.gif', 'Reflective animation') + ' / ' +
             link(repo / 'reports/animations/2026-09-07-absorbing-wave-surface.gif', 'Absorbing animation'),
             'Reference and saved MLP prediction, fixed amplitude scales; top views also available'],
        ]), '',
        'Movies contain actual saved-time observations, without temporal interpolation. '
        'The same selected validation case is used for both boundaries. Movie export '
        'regenerated only the reference fields and checked parity against the original '
        'snapshots and rollout metrics. It did not train a model or recompute a ROM.', '',
        'Historical validation completed before this handoff: Burgers benchmark, '
        'architecture-component and actual-head integration checks; wave FOM, model, '
        'learning and evaluation component checks; cluster preflights; independent '
        'raw-array and provenance audits; GIF frame decoding and browser-control checks. '
        'This documentation session only regenerates tables and checks document inputs/links. '
        'It does not rerun numerical tests.', '',
        '### Exact primary result files', '']
    for record, _ in b3d:
        lines.append(f"- Burgers `{record['name']}`, job `{record['job']}`: " + link(Path(record['path'])) + '.')
    for label, run in waves:
        p = run['provenance']
        lines.append(f"- Waves `{label}`, job `{p['job_id']}`, {', '.join(p['device_kind'])}, "
                     f"source `{p['source_commit']}`: " + link(campaign / label / 'cluster/out/campaign/result.json') + '.')
    for label, data in refinements:
        lines.append(f"- Continuation `{label}`, source `{data['provenance']['source_commit']}`: " +
                     link(campaign / label / 'cluster/out/refinement/result.json') + '.')
    lines += ['', 'Native wave checkpoints are tracked on the wave branch. Complete '
              'checked cluster archives and root audits are tracked on the repair branch. '
              'Architecture checkpoints remain with their owning candidate branches. '
              'All completed campaign directories were checksum-pulled and removed '
              'from the cluster; local archives are the recovery source.', '',
        '## Branch map and ownership at handoff', '',
        f"Canonical documentation is on `main`, which was at "
        f"`{snapshot['main_before_handoff']}` immediately before this handoff. "
        'Main is the frozen public scientific baseline with the known broken heat '
        'rollout; do not use it as an experiment base. Branch hashes below identify '
        'the saved state before this documentation addition, not floating future tips.', '']
    roles = ['Consolidated earlier non-wave lineage; contains excluded historical wave material',
             'Root coordinator, controls, complete archives, independent audits and plotting',
             'Fresh wave science implementation, native outputs and trained checkpoints',
             'Protected anchor candidate', 'Quadratic candidate', 'Shared encoder candidate', 'Smooth mixture candidate']
    rows = []
    for slug, role in zip(slugs, roles):
        state = snapshot['worktrees'][slug]
        rows.append([link(repo / 'worktrees' / slug, slug), f"`{state['commit'][:12]}`", role])
    lines += [table(['Worktree under repository root', 'Saved commit', 'Role'], rows), '',
        'Each listed experiment branch is `exp/<worktree-directory-name>`. '
        'Read across trees as needed, but write scientific work only in the owning '
        'tree. The canonical root lab log and reports are the required shared-document '
        'destinations. No experiment branches were merged. The question of merging '
        'the wave branch into the repair branch was previously raised and remains '
        'unanswered; do not merge implicitly.', '',
        '## Recommended next experiment — proposed, not run', '',
        '**First test the effect of latent dimension with the current verified wave '
        'reference and frozen learned bank.** The full-bank comparison changes '
        'dimension and evolution structure together. A controlled dimension study '
        'is the clearest way to reduce that ambiguity before adding more head complexity.', '',
        '1. Freeze the current reference implementation, data split, learned bank, '
        'physical metrics and original results. Choose a small increasing set of '
        'latent dimensions up to the bank dimension before training. Keep the final '
        'cohort sealed during model selection.',
        '2. At each dimension, fit an affine head and the current MLP in the same '
        'learned bank. Give them the same training membership and initialization '
        'subspace, and record parameter counts and budgets. Retain fresh POD as a '
        'separate comparator. Matching displacement dimension also matches the '
        'displacement-plus-velocity state dimension.',
        '3. Measure unrestricted bank projection, best recorded multistart head fit, '
        'tangent error, initial-condition fit, and autonomous rollout separately. '
        'Keep rank, nonstationary fits, failed cases, all timestep repetitions and '
        'outliers. This distinguishes lack of representation from poor evolution '
        'without treating a local fit as a proven global optimum.',
        '4. Preserve initial-normalized metrics for comparison and also show absolute '
        'field error and instantaneous relative error. Declare how vanishing '
        'reference amplitudes are flagged before examining new results. Monitor '
        'phase and the absorbing mean as well as energy; do not relabel the old '
        'acceptance metric after seeing results.',
        '5. If accurate reconstruction is available at a useful dimension but '
        'autonomous evolution still fails, test a separately controlled '
        'trajectory-aware training objective or state-manifold design. The existing '
        'velocity-tangent penalty is insufficient evidence of successful dynamics. '
        'If affine and nonlinear models both need nearly the full bank, reconsider '
        'the desired compression before claiming an architecture fix.', '',
        'For Burgers 3D, prioritize representation of the unseen initial-state '
        'outliers under the fixed-bank protocol, preserving separate initial/later '
        'state summaries. Re-run the full pilot before any rollout promotion. '
        'Keep larger-PDE expansion and speed benchmarking behind demonstrated '
        'accuracy. These are recommendations, not newly authorized cluster jobs.', '',
        '## Operational instructions for resuming', '',
        '- Read the canonical log and repository [AGENTS.md](../AGENTS.md). '
        'For a new wave experiment, propose the current fresh-wave branch as the '
        'base because it owns the verified implementation and checkpoints. Obtain '
        'the required base/worktree/namespace agreement before creating a new tree. '
        'For simultaneous experiments, propose separate owners and namespaces first.',
        '- Use `/home/tahmid/Dev/.venv/bin/python` locally and '
        '`/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python` on the cluster. '
        'Run real numerical work on the cluster GPU partition with f64 and '
        '`JAX_DEFAULT_MATMUL_PRECISION=highest`; require `jax_backend=gpu` in every log. '
        'Local GPU work is only short, bounded smoke testing through `jaxrun`.',
        '- Use a unique paralab job directory, regenerate data from its seed, stage '
        'committed code directly with checksums, and check `squeue` before and after '
        'submission. The existing approved wave namespace is '
        '`/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/`. Do not infer '
        'permission to reuse it for a new concurrent campaign.',
        '- Existing repair `cluster/stage_fresh_wave.py`, `submit_fresh_wave.py` and '
        '`collect_fresh_wave.py` preserve source/job/archive provenance. Pull and '
        'verify results before removing the exact remote directory. Never use '
        'bare `scancel`; use the applicable guarded helper with exact numeric IDs.',
        '- Preserve weak-form minimization, sufficiently more test modes than '
        'latent coordinates, decoder-output quadrature fitting when used, and '
        'the FOM-exact upwind operator in Burgers weak advection. Do not substitute '
        'random strong-form collocation. Hyper-reduce initialization too before '
        'claiming a grid-independent online path.',
        '- Compare cost and accuracy from the same solver invocation and GPU. '
        'Warm the GPU before timing and persist repetition arrays. No cross-job '
        'timing inference is supported here; the fresh wave cold start remains '
        'full-field and there is no grid-independent speed claim.',
        '- Preserve archives, the unopened final cohort and unrelated user edits. '
        'Append results and retractions to the canonical lab log before closing '
        'every session. Ask about merges when branches finish; do not merge unprompted.', '',
        'The handoff session starts no simulations and leaves no numerical job to '
        'monitor. New experiments, branch merging and the still-image evolution '
        'follow-up are the remaining decisions/tasks; this document does not '
        'mark them completed.', '',
        '## Reproducing this document', '',
        'Tables and numerical prose are generated from the linked run JSONs. '
        'The adjacent manifest records their SHA-256 hashes, frozen branch metadata '
        'and the output hash. Regeneration stops if an input has changed. Existing '
        'worktrees or restored equivalent archived paths are required; this command '
        'does no training, fitting or simulation.', '',
        '```bash',
        '/home/tahmid/Dev/.venv/bin/python reports/gen_2026_09_07_nmrom_handoff.py \\',
        f'  --repo {repo}',
        '```', '',
        'Run from the repository root. Source: '
        '[gen_2026_09_07_nmrom_handoff.py](gen_2026_09_07_nmrom_handoff.py); manifest: '
        '[2026-09-07-nmrom-handoff.manifest.json](2026-09-07-nmrom-handoff.manifest.json).', '',
        '## Plain-language glossary', '',
        '- **NM-ROM / ROM / FOM:** nonlinear-manifold reduced-order model / reduced '
        'model / full-order reference model. A PDE is a partial differential equation.',
        '- **Spatial bank, rank, r:** stored learned spatial functions; rank describes '
        'how many are linearly independent. **Latent dimension, k:** number of '
        'coordinates in the compressed displacement state. Waves also evolve latent velocity.',
        '- **Head, MLP, quadratic, affine:** the map from latent coordinates to bank '
        'coefficients; respectively a feed-forward neural network, a polynomial '
        'including quadratic products, or a linear map plus a constant offset.',
        '- **Protected anchor / shared encoder / smooth mixture:** tested heads that '
        'preserve a fixed coordinate component, share a learned field-to-code map, '
        'or blend coefficient predictions from multiple experts.',
        '- **POD / PCA / QR:** data-derived linear basis / principal-component '
        'coordinates / orthogonal matrix factorization. Their role must be stated; '
        'none makes the learned spatial network a POD network.',
        '- **Reconstruction / projection / tangent error:** error in fitting a '
        'snapshot / error using arbitrary coefficients in a specified linear span / '
        'error representing physical velocity using the decoder Jacobian.',
        '- **Jacobian / Hessian / manifold:** first derivative / second derivative '
        'of the decoder / the set of fields produced by varying its latent code.',
        '- **Rollout / autonomous evolution:** advancing the reduced state through '
        'time from its initial fit, without refitting to the reference at later times.',
        '- **M, K, C, reduced operators:** mass weights, spatial stiffness and '
        'boundary damping in the reference equations, and their projections into '
        'the learned bank. **Weak test modes:** functions used to average/project '
        'the PDE residual. This use of mass M differs from test-count notation '
        'used in some earlier project documents.',
        '- **Empirical quadrature / NNLS / hyper-reduction:** weighted spatial '
        'sampling fitted to decoder outputs / fitting nonnegative weights / '
        'evaluating the online residual without traversing the full grid. '
        '**Cold start:** computing the initial reduced state.',
        '- **Mean / median / worst:** arithmetic average / middle value / largest '
        'error. Burgers rows pool validation states; wave rollout rows first take '
        'each trajectory’s maximum over saved times. Percent columns multiply '
        'dimensionless errors by one hundred.',
        '- **Outlier / nonstationary fit:** a case above the stated error ceiling '
        '(nonfinite cases also fail) / a selected optimization result failing the '
        'recorded local convergence condition. Neither is removed from acceptance.',
        '- **Seed / repeat / validation / sealed final cohort:** recorded random '
        'number initialization / optimization run / data used to assess and select '
        'models / reserved data still unopened for final evaluation.',
        '- **Time check / finest-pair passes / unresolved case:** comparison of '
        'trajectories at the smallest declared timesteps / count meeting that '
        'tolerance / case still failing it. Passing a time check does not mean '
        'the physical solution is accurate. Case indices start at zero.',
        '- **Initial-normalized / instantaneous relative / energy-state error:** '
        'division by fixed initial scales / division by the current reference '
        'field norm / energy norm of the difference in displacement and velocity. '
        '**Phase:** position within an oscillation; undefined when its amplitude vanishes.',
        '- **Gate / provisional target / negative control:** required check / '
        'declared engineering threshold without universal scientific status / '
        'deliberately invalid input that should trigger a failure.',
        '- **Dirichlet / Sommerfeld / nullspace:** fixed displacement wall / '
        'approximate outgoing-wave boundary rule / states an operator does not '
        'detect. **Navier–Stokes:** equations for fluid velocity and pressure; '
        'incompressibility imposes a divergence constraint.',
        '- **Worktree / commit / job / checkpoint / SHA-256:** separate working '
        'directory for a branch / saved code revision / scheduler run identifier / '
        'stored trained model state / content fingerprint used to verify files.',
        '- **f64 / highest precision / GPU preflight:** double precision / required '
        'matrix-multiplication setting / check that a cluster calculation is '
        'actually using the accelerator before accepting its results.', '']

    if old is not None and set(inputs) != set(old['inputs_sha256']):
        raise RuntimeError('Handoff input set changed')
    report.write_text('\n'.join(lines))
    manifest = {
        'purpose': 'Immutable dated handoff; LAB-LOG.md remains canonical',
        'numerical_work_performed': False,
        'repository_snapshot': snapshot,
        'inputs_sha256': dict(sorted(inputs.items())),
        'generator_sha256': sha(Path(__file__)),
        'report_sha256': sha(report),
        'command': ['/home/tahmid/Dev/.venv/bin/python', str(repo / 'reports' / Path(__file__).name),
                    '--repo', str(repo)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'report': str(report), 'input_files': len(inputs),
                      'report_sha256': sha(report), 'bytes': report.stat().st_size}))


if __name__ == '__main__':
    main()
