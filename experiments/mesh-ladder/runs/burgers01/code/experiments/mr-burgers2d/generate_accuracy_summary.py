"""Render audited paired Burgers tables to stdout; numbers come only from JSON."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('record',type=Path);a=p.parse_args();d=json.loads((a.record/'PANEL.json').read_text());assert d['audit']['passed']
print('# Fixed-bank Burgers accuracy and solver comparison\n')
print('These are audited development measurements from one GPU allocation. The spatial bank and network capacity are fixed; the refined head adds initial-field training with decoded-state replay. Final paper cases remain unopened.\n')
print(f"Scientific source `{d['source_commit']}`, job `{d['job_id']}`, GPU `{d['gpu']}`. Reported timings are pooled medians of all retained repetitions.\n")
for cohort in ['opened development','fresh development','all']:
 print(f'## {cohort.capitalize()}\n')
 print('| Intervals | Method | GPU ms | Worst fixed-initial error % | Initial error % | Stationary queries | Physical target |')
 print('|---:|---|---:|---:|---:|---:|---|')
 for row in d['rows']:
  if row['cohort']!=cohort:continue
  stationary='—' if row['stationary_invocations'] is None else f"{row['stationary_invocations']}/{row['invocations']}"
  physical='pass' if row['physical_pass'] else ('provisional: reference check fails' if not row['reference_pass'] else 'fail')
  print(f"| {row['intervals']} | {row['name']} | {row['gpu_ms']:.6f} | {100*row['worst_fixed_initial']:.6f} | {100*row['worst_initial_error']:.6f} | {stationary} | {physical} |")
 print()
print('## Fine-grid training reconstruction\n')
print('| Quantity | Median error % | Worst error % |\n|---|---:|---:|')
for label,key in [('Frozen bank projection floor','floor'),('Original head fitted training codes','before'),('Refined head fitted training codes','after'),('Drift from original decoded replay targets','replay_after')]:
 r=d['training'][key];print(f"| {label} | {100*r['median']:.6f} | {100*r['maximum']:.6f} |")
print('\nReplay measures preservation of original decoder outputs at recorded training codes. It is not error against newly regenerated PDE trajectories.\n')
print('## Same-job speed ratios on all development cases\n')
print('| Intervals | ROM | FOM | FOM / ROM GPU time | Both meet physical and numerical criteria |\n|---:|---|---|---:|---|')
for r in d['comparisons']:
 if r['cohort']=='all':print(f"| {r['intervals']} | {r['rom']} | {r['fom']} | {r['gpu_ratio']:.6f} | {'yes' if r['both_physical_and_numerical_pass'] else 'no'} |")
print('\n## Glossary\n')
print('- **Intervals:** grid spaces per axis; nodes per axis are one larger.\n- **Fixed-initial error:** field-error norm divided by the reference initial-field norm, maximized over saved times and cases.\n- **Initial error:** reconstruction error at the supplied starting time.\n- **Stationary queries:** queries whose initial fit and every reduced time step meet the normalized gradient criterion.\n- **Physical target:** the predeclared error target including the empirical spatial/time reference allowance.\n- **Reference check:** observed reference differences must shrink and their allowance must fit the declared budget; this is empirical, not a rigorous continuum bound.\n- **Frozen / trained:** unchanged original head / head refined using training initial fields and decoded-state replay.\n- **Accepted / stationary:** original stall-based optimizer / larger-budget optimizer with explicit stationarity stopping.\n- **FFT loose / tight:** iterative full-grid Newton–BiCGStab controls with the configured loose/tight nonlinear and linear tolerances.\n- **FOM / ROM:** full-grid PDE solver / nonlinear reduced model.\n- **Replay / projection floor:** original decoded training targets / error outside the fixed learned spatial span.\n- **GPU time:** complete supplied-field-to-full-output query on the GPU.\n- **Development:** cases used for method development; the independent final cohort remains sealed.')
