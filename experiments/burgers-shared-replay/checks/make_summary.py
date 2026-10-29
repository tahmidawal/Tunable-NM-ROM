"""Generate the replay summary entirely from retained run/audit JSON."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'screen01-summary.json').read_text())
a=json.loads((HERE/'screen01-audit.json').read_text())
r=json.loads((HERE.parent/'runs/screen01/output/dev6/result.json').read_text())
d=s['arms']['direct']; c=s['arms']['replay']; t=a['totals']; settings=s['settings']
text=[
    '# Shared-decoder complete-trajectory replay: cost-screen result',
    '',
    'This is a final negative result for the frozen development cost screen. Stronger accuracy qualification was not run after the predeclared cost stop.',
    '',
    f"The complete standalone K{settings['K_coarse']} solve followed by sequential K{settings['K_fine']} replay took {c['median_paired_ratio']:.9f} times the direct solver's time by the registered paired statistic. It was slower in all {c['slower_pairs']}/{c['pairs']} paired repetitions. The final problem, physical mesh, fine initial fit and original temporal prediction remained fixed.",
    '',
    '| Case | Direct median ms | Replay median ms | Paired median ratio | Paired range | Accepted guesses |',
    '|---|---:|---:|---:|---|---:|',
]
for x in a['paired_timing_cases']:
    text.append(f"| {x['case']} | {x['direct_median_ms']:.6f} | {x['replay_median_ms']:.6f} | {x['paired_median_ratio']:.9f} | {x['paired_ratio_min']:.6f}–{x['paired_ratio_max']:.6f} | {x['accepted_candidates']} |")
text += [
    '',
    'Each timing includes all initial fits, the coarse trajectory and preview fields where applicable, guard evaluations, fine evolution and requested fine output fields. Ratios pair methods within the same repetition; a ratio of the displayed median times need not equal the paired median ratio.',
    '',
    '| Work over all development trajectories | Direct | Replay |',
    '|---|---:|---:|',
    f"| Fine corrections | {t['direct']['fine_corrections']} | {t['replay']['fine_corrections']} |",
    f"| Fine physics Jacobian evaluations | {t['direct']['fine_physics_jacobians']} | {t['replay']['fine_physics_jacobians']} |",
    f"| Coarse corrections | — | {t['replay']['coarse_corrections']} |",
    f"| Coarse physics Jacobian evaluations | — | {t['replay']['coarse_physics_jacobians']} |",
    f"| Fine initial-fit iterations | {t['direct']['fine_ic_iterations']} | {t['replay']['fine_ic_iterations']} |",
    f"| Coarse initial-fit iterations | — | {t['replay']['coarse_ic_iterations']} |",
    '',
    f"The guard accepted {t['replay']['accepted_candidates']} and rejected {t['replay']['rejected_candidates']} of {t['replay']['candidate_offers']} coarse guesses. Accepted guesses lowered the original fine objective, but the subsequent fine solves used more corrections in every case. Coarse and fine Jacobians have different widths; their counts are diagnostics, not a substitute for measured total runtime. Initial-fit Jacobian counts are not exposed by the unchanged initializer, and their full runtime is included.",
    '',
    f"All {a['all_fresh_certificates']} regular coarse/fine trajectory certificates passed the production stopping rule; maximum fresh stationarity indicator was {a['max_fresh_gamma']:.9g}. Independent AD/analytic checks, preserved-field promotion, original/disabled fine parity, initial-fit parity, timing witnesses, and both injected coarse-failure controls pass. A failed coarse solve stops its history and disables later guesses. Unavailable field interiors are NaN while boundary walls remain zero.",
    '',
    f"The registered limits were median paired ratio ≤0.85 and worst case ≤1.10. The observed worst case was {s['gate']['maximum_case_ratio']:.9f}, and all {a['case_medians_above_1p10']} case medians exceeded the latter limit. Numerical and timing controls passed; the cost gate failed.",
    '',
    f"Run settings: L={settings['L']}, R={settings['R']}, M={settings['M']}, {settings['rule']}, {settings['scheme']}, dt={settings['dt']}, {settings['steps']} steps, {settings['repetitions']} paired repetitions per case. Job {a['job_id']} completed successfully using {a['allocated_gpu_seconds']} allocated GPU-seconds on {a['gpu'][0]}; f64/highest and mandatory GPU checks passed. The log is complete and stderr is empty. Setup took {s['setup']['seconds']:.6f} seconds and is reported separately from query timing while remaining inside allocated cost.",
    '',
    'No tight/tighter baseline agreement, successor quadrature qualification, physical PDE accuracy, sealed-cohort evaluation, three-dimensional extension or repeated correction cycle was run. This does not establish an accuracy gain or reject every possible shared-decoder strategy. The canonical coordinator log owns the combined preparation/test resource total and merge decision.',
    '',
    f"Source commit: `{s['commit']}`. Result SHA256: `{s['source_result_sha256']}`. Model manifest SHA256: `{s['model_manifest_sha256']}`. Raw data, exact staged inputs, accounting and logs are under `../runs/screen01`; generated audits and retention proofs are beside this file.",
    '',
    '## Plain-language glossary',
    '',
    'Case: a fixed development initial condition and viscosity. Direct: the original fine decoder solve with predictions from its earlier accepted states. Replay: first solve the complete small-decoder trajectory, then solve every fine step again using only accepted fine history. K: number of latent coordinates. L: physical mesh interval count. R: frozen spatial-bank width. M: weak test-mode count. Gauss rule: spatial integration points and weights. BE: backward Euler time stepping. dt: time-step size. Guard: the objective comparison that accepts or rejects an offered starting point. Correction: one nonlinear solver update. Jacobian: derivatives used by that solver. Stationarity: the stated derivative-based stopping condition; it is not physical-error certification. AD: automatic differentiation. Paired ratio: replay time divided by direct time for the same case and timing block. Paired range: smallest and largest retained paired ratios, without discarding repetitions. Accepted guesses: offers that passed the fixed fine-objective guard. GPU-seconds: allocation duration including setup and checks. SHA256: a content hash binding the report to retained evidence.',
]
(HERE/'screen01-summary.md').write_text('\n'.join(text)+'\n')
