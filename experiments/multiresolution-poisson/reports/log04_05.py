"""Generate the closing canonical-log entry from both completed audited pilots."""
import argparse,json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('cell',type=Path);ap.add_argument('--artifact-commit',required=True);a=ap.parse_args()
c=a.cell;root=c/'runs';d4=json.loads((root/'pilot04/result.json').read_text());s4=json.loads((root/'pilot04/summary.json').read_text())
d5=json.loads((root/'pilot05/result.json').read_text());s5=json.loads((root/'pilot05/summary.json').read_text())
selection=json.loads((root/'pilot04/SPEED-SELECTION.json').read_text())['selected_continuation']
assert s4['audit']['passed'] and s5['audit']['passed']
lines=['## 2026-09-07','',
    '### Poisson continuation factorial and projection/initialization speed study closed','',
    'The user retained separate worktrees and requested continued development. This session stayed in the approved Poisson tree, branch and remote namespace; no branch or merge was created. '
    f"The reviewed result archive and generated findings are committed through `{a.artifact_commit}`. No further Poisson GPU study is launched this round.",'',
    f"Training factorial job `{d4['provenance']['job_id']}` used immutable source `{d4['provenance']['commit']}`; speed factorial job `{d5['provenance']['job_id']}` used `{d5['provenance']['commit']}`. "
    'Each used its own attempt directory, GPU preflight, float64/highest precision, paired complete-query repetitions and burn-in. Source and checkpoint bytes were staged from committed history with direct checksums; data were regenerated from seeds. Sealed final sources stayed closed.','',
    f"The training factorial crossed original coverage with expanded coverage and shared-global with full-field per-snapshot relative normalization. All {len(d4['training']['arms'])} fixed endpoints reached {d4['config']['training_steps_each']} updates. "
    f"Original reproduction used the first {d4['config']['original_training_prefix_count']} sources of the original {d4['config']['original_draw_count']} draw; expanded coverage added {d4['config']['additional_training_count']} independently seeded sources. "
    f"Training uses {d4['config']['training_nodes_per_axis']} nodes, while queries use {d4['config']['intervals']} intervals. Both bank and head were continued, along with training codes, under matched updates and batches rather than equal visits per source. "
    f"Actual offline source/code-fit/compile/train/diagnostic elapsed was {d4['training']['offline_seconds_including_data_codefit_compilation_and_diagnostics']:.9g} seconds. Offline durations are not a paired warm training-speed comparison.",'',
    f"The owner audit checked {s4['audit']['row_count']} training-study query invocations, {s4['audit']['distinct_timed_field_hashes']} unique timed fields and {s4['audit']['diagnostic_panels']} reference-only bank/head diagnostic panels. "
    f"Maximum independent CPU query-metric difference was {s4['audit']['maximum_independent_cpu_metric_difference']:.9g}. The root coordinator separately audited every timed field and independently reproduced the endpoint selection.",'',
    'Training stationary-control physical errors, generated from native data:','',
    '| Intervals | Development cohort | Model | Median / worst physical error | Invalid / nonstationary |',
    '|---:|---|---|---:|---:|']
for s in s4['summaries']:
    if s['cohort']=='all_development' or s['arm']!='rom_modular' or s['tau']!=0:continue
    lines.append(f"| {s['intervals']} | {s['cohort']} | {s['model']} | {s['physical_median']:.9g} / {s['physical_max']:.9g} | {s['invalid_count']} / {s['nonstationary_count']} |")
lines+=['',f"The predeclared minimax rule across all sources and both meshes selects `{selection['model']}`, SHA256 `{selection['checkpoint_sha256']}`, with worst empirically adjusted error {selection['worst_adjusted_error']:.9g}. "
    'This choice uses every development case, not the previously inspected narrow-source example. Relative normalization changes the tradeoff between worst-case and median errors; expanded coverage does not uniformly improve the fixed-compute endpoints. Bank/head fits remain local reference-only diagnostics and are never online initializations.','',
    f"The speed factorial retains the original and selected checkpoint, crossing sine-product versus forward-DST source projection with mean-code versus nearest training-prediction initialization. It records {s5['audit']['primary_invocations']} repeated primary calls and {s5['audit']['stationary_invocations']} separate single-call stationary controls. "
    'The cache uses only training codes and decoder-predicted weak coefficients. Lookup, projection, guarded linear solve/fallback, nonlinear evolution, decoding and full host-field transfer are charged. Cache/operator assembly and compilation are offline.','',
    s5['audit']['timing_boundary'],'',
    'The residual threshold remains relative to each chosen start. A nearer start can demand a tighter absolute threshold and force a stationary exit. Initial/final residuals, absolute thresholds, stop reasons and nearest-index gaps are retained. Projection parity compares the same initialization choice; different mean/nearest local minima are evaluated directly rather than rejected by equality.','',
    '| Intervals | Checkpoint | Projection | Initialization | Primary query ms | Worst physical error | Invalid / nonstationary | Paired same-grid DST/ROM |',
    '|---:|---|---|---|---:|---:|---:|---:|']
for s in s5['summaries']:
    if s['cohort']!='all_development' or s['panel']!='primary' or not s['model']:continue
    lines.append(f"| {s['intervals']} | {s['model']} | {s['projection']} | {s['initialization']} | {1000*s['latency_seconds']:.9g} | {s['physical_max']:.9g} | {s['invalid_count']} / {s['nonstationary_count']} | {s['same_grid_dst_median_case_cost_ratio']:.9g} |")
lines+=['','Both complete-query development envelopes use median case-median latencies and medians of per-case cost ratios. The efficient FOM panel includes same-grid DST and charged coarse-grid interpolation. Every source must meet the solver, parity and empirical reference-adjusted accuracy gates; reference differences are not rigorous continuum bounds. Separate stationary controls do not enter speed selection.','',
    '| Intervals | Target | Selected speed-study ROM | ROM / selected FOM ms | Paired FOM/ROM |',
    '|---:|---:|---|---:|---:|']
for e in s5['envelope']:
    if e['cohort']!='all_development':continue
    r,f=e['rom'],e['fom'];label='unattained' if not r else f"{r['model']} / {r['projection']} / {r['initialization']}"
    rt='—' if not r else f"{1000*r['latency_seconds']:.9g}";ft='—' if not f else f"{1000*f['latency_seconds']:.9g}"
    ratio='—' if e['median_case_cost_ratio'] is None else f"{e['median_case_cost_ratio']:.9g}"
    lines.append(f"| {e['intervals']} | {e['target']} | {label} | {rt} / {ft} | {ratio} |")
lines+=['',f"The speed-study CPU audit checks {s5['audit']['distinct_preserved_field_hashes']} preserved full fields, reconstructs neural outputs from saved coordinates and weights, verifies the training-only cache and selected codes, and recomputes physical errors, residuals and stationarity. "
    f"Maximum query-metric discrepancy is {s5['audit']['maximum_independent_cpu_metric_difference']:.9g}, decoded-field relative discrepancy {s5['audit']['maximum_decoder_field_relative_difference']:.9g}, and stationarity discrepancy {s5['audit']['maximum_stationarity_absolute_difference']:.9g}. "
    f"Projection-gate failures: {s5['audit']['projection_gate_failures']}; primary fallbacks: {s5['audit']['timed_fallbacks']}; stationary fallbacks: {s5['audit']['stationary_fallbacks']}. "
    'All failures remain in native records and errors. The audit distinguishes exact saved source-parameter hashes from tiny cross-architecture exponential roundoff in independent seed regeneration.','']
for label in ('pilot04','pilot05'):
    clean=json.loads((root/label/'cleanup.json').read_text());archive=json.loads((root/label/'ARCHIVE.json').read_text())
    assert clean['remote_deleted_and_absence_checked']
    lines.append(f"- `{label}`: archive {archive['bytes']} bytes in {len(archive['ordered_parts'])} checked parts, SHA256 `{clean['archive_sha256']}`. Exact remote `{clean['remote']}` deleted and absence checked.")
lines+=['','All raw JSON, source manifests, logs, native audits, PNG/PDF figures and generated findings are preserved under `worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/`. The four trained checkpoints are also directly tracked under `runs/pilot04/checkpoints/`, not only inside the checked archive.','',
    'No prior numerical result is retracted. Cross-architecture exact input-hash regeneration is not asserted; recorded within-job source hashes, exact persisted draw prefixes and numerical CPU reconstruction are the supported checks. These bounded development studies do not establish an optimal training recipe, a globally optimal head fit, rigorous physical certification or a paper-complete speed claim. Further training, solver changes and final confirmation remain open; stop for coordinator review with separate worktrees.','']
(root/'pilot05/LAB-LOG-ENTRY.md').write_text('\n'.join(lines));print(root/'pilot05/LAB-LOG-ENTRY.md')
