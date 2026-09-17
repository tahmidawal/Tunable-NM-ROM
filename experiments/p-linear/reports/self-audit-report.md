# Self-audit of the p-linear report against the raw JSONs (substitute for the Codex report audit)

Codex (gpt-6-astra) is unavailable until 2026-09-19 11:33 (usage limit; LANE-PROTOCOL.md notice). Per that notice and DESIGN §A9 this is a written self-audit: one row per claim the report and the lab-log entry make, the JSON field it rests on, the check run, and the value read. Every value below is read from `summary.json`, `verdicts.json`, the three `audit.json` files, the run `result.json` files, `FAILURE.json`, and the two check JSONs by `generate_entries.py`; none is typed. The independence guarantee is weaker than a second model family.

| # | claim | rests on | check | value read |
|---|---|---|---|---|
| 1 | 1024 verdict DEGENERATE (D1 ∧ D2 ∧ D3) | verdicts.json["1024"]; audit.json criterion.degenerate | generator and NumPy audit derive it independently from invocation rows | generator D1=True, D2_strict=True, D3=True; audit degenerate=True |
| 2 | 1024 D1 cost span | verdicts.json["1024"].D1_span vs audit criterion.D1_cost_span | two implementations agree | 1.553958 vs 1.553958 |
| 3 | 256 D1 fails literally, intent not met | verdicts.json["256"].{D1_span,falsified_literal,falsified_intent}; audit criterion | two implementations agree | span 3.173x; literal True/True, intent False/False |
| 4 | top rung = cheapest and most accurate at both meshes | verdicts.json[*].{cheapest,top,D2_lowest} | string equality | 1024: cheapest `d_linear_qr_m4@new_K32` top `d_linear_qr_m4@new_K32`; 256: cheapest `d_linear_qr_m4@new_K32` top `d_linear_qr_m4@new_K32` |
| 5 | top rung reaches the bank floor | summary rows worst_same_grid of `d_linear_qr_m4@new_K32` and `bank_floor@new_K32` | relative difference | 1024: 7.421302e-03 vs floor 7.421273e-03; 256: 7.458632e-03 vs 7.458604e-03 |
| 6 | bank floor independently rebuilt | audit.json numpy_bank_floor (plin1024b) | NumPy QR of the 1046529x512 bank, no JAX | max abs difference 6.75e-11, worst 7.421273e-03 |
| 7 | every reported error recomputed from saved fields | audit.json recomputed_errors (all three jobs) | sha256 of every field checked, error recomputed | 1024: 1440 errors/480 fields, worst 1.53e-16; 256: 1368/456, 3.36e-16; head: 324/108, 1.06e-16 |
| 8 | GPU backend and precision | result.json {backend,x64,matmul_precision,gpu} | audit backend check | 1024: gpu/NVIDIA H200; 256: gpu/NVIDIA A100-PCIE-40GB; head: gpu/NVIDIA A100-PCIE-40GB; x64 True, precision highest |
| 9 | no ratio across GPUs | result.json gpu per job | the report prints a GPU-per-job table; every span/non-dominated set is per job | H200 (1024) vs A100-PCIE-40GB (256, head): no cross-mesh cost number exists in summary.json (all rows carry one job_id) |
| 10 | fidelity gates | result.json gates; audit fidelity_gates_recomputed | all pass at 1e-9, recomputed from fields | 1024: 23/23, worst 5.52e-12; 256: 18/18, worst 1.70e-13; head: 2/2 |
| 11 | consistency pairs | result.json consistency | all pass at the stated tolerances | 1024: 6/6; 256: 6/6; q512 eliminated vs direct 2.48e-07 (1024) |
| 12 | POD-512 and DST dominate the neural points | summary rows worst_same_grid/median_total_ms of e_pod512_m4@trainset, dst_direct | read from rows; non-dominated flags | 1024: POD-512 0.1838 % at 6.972 ms, DST 3.248 ms; top rung 0.7421 % at 4.424 ms |
| 13 | CG is the slowest full-order route | summary rows median_total_ms of cg_* | read from rows | 1024: 62.5 ms (1e-2) to 121.4 ms (1e-8) vs DST 3.248 ms |
| 14 | q = 0 three layers agree | summary augmented_best_found_q0@new_K32, q0_m256@new_K32, bank_floor; result reconstruction.dense_vs_projected_oracle (256) | oracle q=0 has no V; dense cross-check at 256 | 1024: oracle 3.1139 % vs solved 3.1146 %; 256 dense-vs-projected 2.21e-11 |
| 15 | augmented oracle q > 0 RETRACTED (A10) | summary oracle rows retracted flag; checks/2026-09-17-oracle-metric-scale.json; checks/2026-09-17-oracle-fix-check-64.json | V^T V measured per mesh; fix verified locally | retracted rows: 1024 5/6, 256 0/6; V^T V diag 64/256/1024: 0.0630/1.0079/16.126; fixed q=R vs floor rel 0.0e+00, q0 identical True, all True |
| 16 | plin1024 retraction is a GPU OOM, not disk-full | runs/plin1024/FAILURE.json | log complete, ends in RESOURCE_EXHAUSTED; disk recorded | FAILED 1:0 at 00:14:27 on NVIDIA A100-PCIE-40GB (40960 MiB); disk 91% (not disk-full); timed numbers False |
| 17 | H1 reproducibility | verdicts.json head.H1_relative; result arms | 5 % bar | 0.00 % relative (control 3.1212 % vs primary 3.1212 %) -> pass |
| 18 | H2 verdict | verdicts.json head.{H2_max_drop,H2_min_ratio} | 20 % drop bar; 2x ratio bar | max drop 34.2 % (`K64_w256_L3`), min ratio 2.755x -> partial movement; validation picks `K32_w256_L2` |
| 19 | head arms solved at 1024 do not enter the linear-case table | DESIGN section 5 (enters only if ratio < 2x) | H2_min_ratio | 2.755x >= 2 -> none enters; the head rows are on the A100 job and are never compared to the H200 job |
| 20 | one job per directory, squeue before/after | runs/*/SUBMISSION.json | one_job_per_directory flags | plin256: True, plin1024: True, plin1024b: True, plhead1: True |
| 21 | remote directories deleted | ssh listing after collection | namespace listing empty | p_linear_20260917/ contains no attempt directories (verified 2026-09-17 10:55 EDT) |
| 22 | job count | runs/*/SUBMISSION.json | count | 4 of 8 (one retracted) |

## Findings of this self-audit

1. **One retraction found by this audit and not by any gate**: the augmented best-found oracle for q > 0 (row 15). No gate covered it because it is untimed and not in D1–D3; the bracket floor ≤ best-found ≤ solved was stated in DESIGN §3 but never asserted in code. It is now checked per rung by the generator and the offending values are struck through and flagged in `summary.json`.
2. The 1024 and 256 verdicts differ on D1 only through the top rung's cost advantage (row 2 vs 3), and agree on D2, D3, monotonicity and on `falsified_intent = False`. The paper's claim is decided by `falsified_intent` per §A8, declared before the 1024 job ran.
3. Nothing in the timed rows, gates, POD/DST/CG rows, or the head job is affected by the oracle defect (rows 5–13, 17–19 rest on other fields).
4. Not verified here: the eliminated `q512_m4` arm's cost (an inert-iteration artefact, DESIGN §A4) is reported but excluded from the ladder line; a reader who disagrees with §A4 can recompute D1 from the `q512_m4@new_K32` row in `summary.json`.
