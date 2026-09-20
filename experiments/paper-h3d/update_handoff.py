"""Generate one audited comparison excerpt from retained raw result files."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert a.attempt.isalnum()
root=Path(__file__).resolve().parent;run=root/'runs'/a.attempt
record=json.loads((run/'archive/out/result.json').read_text());summary=json.loads((run/'archive/out/summary.json').read_text())
field=json.loads((run/'audit-local.json').read_text());panel=json.loads((run/'audit-panel.json').read_text());collected=json.loads((run/'COLLECTED.json').read_text())
assert record['complete'] and field['passed'] and panel['passed'] and collected['checksums_verified'] and collected['removed']
keep={'dst_exact','linear_bank_galerkin_exact','pod128_exact'}
keep|={f'nmrom_K{k}_q{q}_dense' for k in (8,16,32) for q in (0,96)}
keep|={x['name'] for x in record['operators']}
lines=[f'## Audited {a.attempt} comparison', '',f'Generated from `runs/{a.attempt}/archive/out/result.json` and its summary; job `{record["job_id"]}`, source `{record["source_commit"]}`. '
       f'Independent checks passed for {field["checked_fields"]} saved fields, {panel["paired_invocations"]} paired invocations and {panel["summary_rows"]} summary rows. The exact remote attempt is removed. Final data remain unopened.', '',
       '| Intervals | Method | Median device ms | Worst evolved current-relative error (%) | Nonstationary cases |','|---:|---|---:|---:|---:|']
for row in summary['rows']:
 if row['method'] in keep or row['method'].endswith('_native_grid_interpolated'):
  lines.append(f'| {row["intervals"]} | {row["method"]} | {row["device_ms_median"]:.6f} | {100*row["same_grid_current_evolved_worst"]:.6f} | {row["cases_with_nonstationary_solves"]} |')
lines+=['','These are development comparisons on one frozen bank, with separately trained heads. The larger head reaches the linear-bank error floor at substantial query cost; direct DST remains more accurate and the linear bank/POD controls remain much faster. The original weak DeepONet and failed direct-transfer variants are retained. Further head initialization and operator training are development work, not final confirmation.']
text='\n'.join(lines)+'\n';path=root/'HANDOFF.md';source=path.read_text();start='<!-- GENERATED_LATEST_COMPARISON -->';end='<!-- END_GENERATED_LATEST_COMPARISON -->'
if start in source:
 left,right=source.split(start,1);_,right=right.split(end,1);source=left+start+'\n'+text+end+right
else:source=source.replace('## Glossary\n',start+'\n'+text+end+'\n\n## Glossary\n')
path.write_text(source)
