**RESULTS-CONFIRMED for round 3.** All four round-2 required fixes are satisfied. No new blocking results defect was found.

| Round-2 required fix | Verdict | Evidence |
|---|---|---|
| Correct pooled plot statistic | **CORRECT** | Now pools qualified seeds within each case before taking the case median. K=16 BE gives **1.440246%**, matching EXPORT’s pooled comparison. |
| Complete initial-fit diagnostics and 4096² fit | **CORRECT** | All **35** initial-fit summaries and their per-case records match the checksum-verified accuracy-job JSON. The 4096² fit explicitly uses seed 0. |
| Limit fit interpretation; mark unusable timings | **CORRECT** | Unsupported causal/power-law statements were removed. Cost-plot open markers identify unusable timings; its title explicitly says descriptive. |
| Preserve verified conclusions | **CORRECT** | F5 remains **inconclusive**; K=16 BE remains **matched-dimension trade-off**; r=128 remains **at no more cost, at least as accurate**. |

The affected results checks are:

1. **E0 errors and Q1/Q2 — CORRECT, carried forward.** Configuration and per-case exports are unchanged. Round 2’s raw-array verification of H16s0 BE, H32s1 BE, L128 BE, L384 CNR and failing H4s0 BE therefore remains applicable. I did **not** download or rescore these trajectories again.

2. **Qualification pools — CORRECT, unchanged.** Both schemes retain K=4: none; K=8: none; K=16: seeds 0,1; K=32: seed 1. Only K=16 meets the two-qualified-seed requirement.

3. **Q-LH-dim / Q-LH-cost — CORRECT.** Paired tests, cost bands, Holm decisions and rung results are unchanged from the verified round-2 export. The changed plot now agrees with their registered pooled error statistic.

4. **F5 — CORRECT, unchanged.** The frozen family and its inputs remain identical; “inconclusive” and the exploratory qualification remain appropriate.

5. **Timing and cost — CORRECT.** All three meshes’ cost-fit outputs reproduce exactly, including the bootstrap results. The newly supplied 4096² descriptive fit gives:
   - \(p=0.119403\), interval **[0.079782, 0.165211]**;
   - relative RMS **0.131935**;
   - **9,910/10,000** accepted bootstrap fits.

   Its per-case fused costs match the checksum-verified timing-job JSON. This descriptive fit includes unusable configurations, which the table and plot identify; it does not establish qualified deployment scaling. H-cost-N gates and trajectory-consistency records are unchanged.

6. **Training, Q-repro and initial fit — CORRECT.** Training, reproduction and A1.7 results are unchanged. The new initial-fit reporting correctly separates initialisation from rollout: H16s0 BE has median initial-fit iterations **17**, median recorded stationarity **6.177091e-7**, maximum **9.814974e-7**, and median rollout iterations **82**.

7. **256² and cross-mesh panels — CORRECT.** Their underlying results are unchanged. The 256² descriptive cost fit also reproduces exactly.

8. **Report/export consistency and deliverables — CORRECT.** With filesystem writes intercepted, regeneration reproduced `report.md` and **all five PNGs byte-for-byte**. Only `initial_fit` and `cost_vs_K` changed among EXPORT’s top-level values. The previously missing diagnostics are now supplied.

**Ranked required fixes: none.**

No repository or cluster files were changed, and no jobs were submitted. Downloaded JSONs passed archive checksums; scratch peaked below 2 MB and was deleted. The pre-existing `HEARTBEAT.json` modification remains untouched.