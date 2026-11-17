"""Generate pilot report directly from committed JSON evidence; no hand-entered results."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text())
ref=read('reference-assessment.json');rp=read('evidence/repair01-result.json');dyn=read('evidence/dynamics01-result.json');ae=read('evidence/ae01-result.json');out=read('evidence/output01-result.json');ledger=read('cluster/ledger.json')
best=min([x for x in rp['rows'] if x['tag']=='mixed_training_repair' and x['kind']=='neural'],key=lambda x:max(x['evolved']))
lines=['# Burgers quadtree pilot: scoped negative result','',
'This completed development pilot does not establish an accurate, faster quadtree NM-ROM. It retains a verified reference pipeline, reproduced incumbents, trained classical/neural local representations, matched-data controls and actual dense-output PDE measurements. These are development results, not untouched-test confirmation.','',
'## Decision','',
f"Stop adaptive-selector development for these checkpoints: after both registered repairs, the best worst evolved projection error is {100*max(best['evolved']):.5f}%, and {sum(e<=.005 for e in best['evolved'])}/{len(best['evolved'])} cases meet the 0.5% screen. This configuration uses coarse rank {best['r']}, depth {best['depth']}, local prefix {best['k']} and {best['columns']} active coefficients. Projection uses the pilot grid and known snapshots; it is not a PDE result.",
'', 'The fixed-space rollout also fails the physical target and is slower than the same-allocation classical control. This does not rule out larger training sets, different neural representations, specialized local networks, nonlinear local coordinates or the separate localized family. None of those unrun extensions is claimed complete.','',
'## Reference and output qualification','',
f"All {len(ref['rows'])} registered development cases have regular second-order finite-difference space/time ladders. The largest conditional reference uncertainty is {100*max(x['conditional_uncertainty'] for x in ref['rows']):.6f}%; the largest independent spectral disagreement is {100*max(x['cross_difference'] for x in ref['rows']):.6f}%. Manufactured fixtures show the expected order and every numerical residual passed. These are numerical convergence assessments, not rigorous error bounds.",
'', 'The reference extension was checked on successively denser grids and every final reference field was saved on exactly 4096×4096 boundary-inclusive nodes. Independent NumPy audits reproduced the stored projection errors and the saved dense rollout errors, including deliberate wrong-normalization detection.','',
'## Actual PDE diagnostic, common input/output','',
'All methods start from the same resident 4096×4096 input array and return a resident final field of that shape. Restriction, initialization, stepping and decoding are charged. Fixed ROMs use backward Euler and exact sign-upwind on 128 intervals, with four orthonormal sine tests per coefficient. Every saved step met the stationarity criterion. The following warm ROM measurements are one invocation per case; spectral rows use medians of five repetitions. They are exploratory costs, not the registered final confidence-interval speed test.','',
'| Method | Worst evolved error, internal scoring (%) | Worst final error, actual4096 (%) | Median warm query (ms) |','|---|---:|---:|---:|']
for method in ['pod128','pod192','localpoly','localneural']:
 z=[x for x in dyn['rows'] if x['method']==method]
 lines.append(f"| {method} | {100*max(max(x['error'][1:]) for x in z):.5f} | {100*max(x['final4096_error'] for x in z):.5f} | {1000*np.median([x['warm_seconds'] for x in z]):.3f} |")
for N in [64,128]:
 z=[x for x in dyn['spectral'] if x['N']==N];er=max(max(max(y) for y in x['dense4096_errors']) for x in z)
 lines.append(f"| Spectral N{N} (all evolved errors on actual4096) | {100*er:.5f} | {100*max(max(y[-1] for y in x['dense4096_errors']) for x in z):.5f} | {1000*np.median([np.median(x['seconds']) for x in z]):.3f} |")
lines+=['','The ROMs are inaccurate at these settings; the table is not a matched-accuracy speedup comparison. Global reusable basis/test tables are prepared outside the query. Their preparation time was not measured, so recorded cold-query times are compilation-inclusive subtotals, not full cold setup costs. Tree selection, coarsening, hyper-reduced local equations and final-panel timing remain unqualified.','',
'## Tuned classical output policy','',
'This separate paired allocation compares exact sine synthesis with boundary-aligned bilinear extension of a smaller spectral field. Both return actual4096 output. It cannot supply cross-allocation ROM timing ratios.','',
'| Output policy | Median query (ms) | Worst final physical error (%) |','|---|---:|---:|']
for m in ['sine','bilinear256']:
 lines.append(f"| {m} | {1000*np.median([np.median(x['times'][m]) for x in out['rows']]):.3f} | {100*max(max(x['final_errors'][m]) for x in out['rows']):.5f} |")
lines+=['',f"Largest interpolation discrepancy over all required evolved times: {100*max(max(x['interpolation_differences']) for x in out['rows']):.5f}%. Output-policy tuning does not change the reference or accuracy requirement.",'',
'## Nonlinear comparator and training limits','',
'The historical frozen nonlinear head and direct bank were reproduced on exposed development cases. New global POD and local models used the same train32/dev8 snapshots. Local POD uses an approximate interpolated residual basis, not an asserted optimal smooth basis. The local neural bank completed the initial throughput pilot and a full mixed-configuration training repair; all checkpoints and failures are retained.','',
'POD-autoencoder encoder reconstruction is not best manifold projection. Its linear POD truncation was reported separately; the following floors are measured on the pilot grid. No autoencoder rollout superiority or failure beyond these tested representations is inferred.','',
'| POD lift | Worst evolved POD truncation (%) | Best tested encoder reconstruction worst error (%) |','|---|---:|---:|']
for lift in [128,256]:
 z=[x for x in ae['rows'] if x['lift']==lift]
 lines.append(f"| {lift} | {100*np.asarray(z[0]['truncation_error'])[:,[2,4,6,8,10]].max():.5f} | {100*min(max(x['evolved_max']) for x in z):.5f} |")
diag=H/'evidence/diagnosis01-result.json'
if diag.exists():
 d=json.loads(diag.read_text());lines+=['','## Discretization diagnosis','','| Full-order upwind mesh | Step | Worst evolved error (%) |','|---|---:|---:|']
 for L,dt in sorted(set((x['L'],x['dt']) for x in d['rows'])):
  z=[x for x in d['rows'] if x['L']==L and x['dt']==dt];lines.append(f"| {L} | {dt:g} | {100*max(max(x['error'][1:]) for x in z):.5f} |")
 lines+=['','This isolates the bias of the fixed upwind/backward-Euler diagnostic. Shared rollout error must not be attributed entirely to the learned features. These full-order times are not used in cross-job speed ratios.']
sec=sum(x.get('elapsed_seconds',0) for x in ledger);cpusec=sum(x.get('elapsed_seconds',0)*x.get('allocated_cpus',0) for x in ledger)
lines+=['','## Resources and reproducibility','',f"{len(ledger)} owned allocations, {sec} allocated A100 GPU-seconds ({sec/3600:.6f} GPU-hours), {cpusec/3600:.6f} allocated CPU-hours. Compilation and allocation overhead are included. Local GB10 tests were limited to three sub-minute smoke calls; their combined configured wall-time ceiling was 165 seconds, but total local GPU time was not instrumented. No local production training or experiment ran.",'',
'Branch: `exp/2026-11-17-burgers-quadtree`, forked from the prescribed consolidated baseline. Code-only GitHub mirrors preserve every implementation commit. `vendor/PROVENANCE.json` ties every inherited source/checkpoint to Git and SHA256. `cluster/ledger.json` records job IDs, limits and source commits. `evidence/*-retention.json` binds the local and owned remote raw arrays, checkpoints, logs and scripts. `AMENDMENTS.md` records prospective changes; `DECISION-A0.md` scopes the stop. Untouched final-test solutions were never generated, no unrelated jobs or archived results were changed, and no branches were merged.','',
'## Glossary','',
'- **Projection:** fit a known field; this is a representation diagnostic, not a forecast or physics solve.','- **POD / POD-autoencoder:** a linear snapshot basis / a nonlinear latent map inside that basis.','- **Rank or prefix:** number of active basis coefficients; distinct from network width and latent dimension.','- **Stationarity:** the weak-residual optimization reached its declared derivative criterion; this does not imply physical accuracy.','- **Warm query:** solve and requested output using already prepared reusable tables.','- **Reference uncertainty:** a conditional numerical convergence estimate for the comparison solution.','- **Actual4096:** exactly 4096 nodes per axis, including boundaries, rather than 4096 intervals.','']
(H/'RESULTS.md').write_text('\n'.join(lines))
print(H/'RESULTS.md')
