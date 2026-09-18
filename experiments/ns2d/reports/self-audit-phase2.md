# Self-audit — Phase 2 attempt `ns201` (job 3783796) and the §A4 amendment

**This is a written self-audit by the lane agent, not an independent-model audit.** Codex is
quota-blocked until 2026-09-19 11:33 (DESIGN §A5). Each claim below names the JSON field it
rests on and the check that was run. The mechanical check is `audit_phase2.py`
(`artifacts/ns201/audit.json`, 67 checks); its six MISMATCH rows are a finding, not a bug in
the audit, and are explained under claim 4.

| # | claim | rests on | check run | status |
|---|---|---|---|---|
| 1 | The `ns201` bank has numerical rank 128 of R=256 at every evaluation mesh | `gates.B-ORTH_N{64,128,256}.rank` | `audit_phase2.py` recomputes the bank in NumPy from the pickled weights (integer Fourier features → SiLU MLP → mean-zero columns) and its SVD rank; equals 128 at all three meshes | verified |
| 2 | The rank cap is structural: rank(G) ≤ g_hidden because the g-MLP's last layer is linear | `config.ARCH.g_hidden` = 128; `sep_common.apply_mlp` (last layer `x @ w + b`) | read the code; the parent lane's `sep_burgers_r3.py` header records the same landmine and the `G_HIDDEN >= 2R` fix; smoke on the patched driver gives rank 16/16 for R=16, g_hidden=32 | verified by reading + smoke; the cluster rerun is the confirmation at scale |
| 3 | H-ORACLE fails: oracle median 0.203 vs POD-16 median 0.240 (ratio 1.19, bar 2.0) at every mesh | `oracle.{N}.oracle_median`, `.podK_median`; `gates.H-ORACLE_N{N}` | audit recomputes both medians from the saved per-state arrays (`oracle_N{N}.npz`) and the pass rule; matches to 1e-9; the JSON per-state lists equal the npz arrays exactly | verified |
| 4 | The reported oracle numbers are contaminated by ≤ 1.1e-2 relative (64²), 7.6e-3 (128²), 3.0e-3 (256²) because they were computed through R_b⁻¹ of a singular R_b | `oracle_N{N}.npz: Z, e_oracle`; ns101 `dev8_N{N}.npz` | audit recomputes ‖G h(z) − u‖/n₀ exactly at the saved codes on the 48 archived states; a debug run reproduced the driver's formula in NumPy (|c|max ≈ 1e12, cancellation) and showed the exact norm equals the rank-128 projection formula to all digits; the patched driver reports `formula_vs_field_worst_rel` = 1.3e-16 on a full-rank smoke | verified; contamination does not change any verdict |
| 5 | The bank floor gate B-FLOOR passes (ratio 1.82–1.85 ≤ 2) but is itself computed with a 256-column QR whose 128 extra columns are roundoff directions | `floors.{N}.bank.256.per_case_worst_fixed`, `gates.B-FLOOR_N{N}` | audit recomputes the per-case floor for the 8 archived cases with a rank-128 projection: differs by 2.0e-2 (64²), 7.4e-3 (128²), 2.6e-3 (256²), recorded as value-level checks; the gate arithmetic from the stored numbers matches exactly | verified with the caveat recorded |
| 6 | The 256² cohort hash mismatch on `pax049` is node-dependent roundoff, not a code change | `gates.B-DATA_{train,dev}_N256` (ns201, `pax049`) vs the same gates in the `ns202` log (`pax106`, PASS, same commit `dddb41fb`) | compared the two job logs; 64² and 128² dev hashes match on both nodes | plausible, not proven by value: `ns201` saved no fields. The value gate (§A4, ≤1e-8 on the archived 8 trajectories) is asserted in every later job; the audit's claim-4 agreement on 256² states (3e-3, dominated by the cancellation effect) does not bound it tightly |
| 7 | `dev8_eval_ref.npz` is Phase 1's data | cut from `runs/ns101/archive/.../dev8_N{64,128,256}.npz` (job 3780151, archived and checksum-verified) at output indices [0,5,10,15,20,25] | audit checks the archived `physical` rows equal a column-by-column NumPy redraw of seed 20260918 (max abs diff 0) | verified |
| 8 | H-SOLVED passes: single-start median within 1.5× of the oracle (0.213 vs 0.203) | `oracle.{N}.single_start_median` | audit recomputes from the npz arrays and the pass rule | verified (but the number carries the claim-4 contamination) |
| 9 | Training reconstruction: rel-L2 mean 0.209, median 0.206, max 0.621, 30 000 steps, 3202 s | `training.*` | reported by the driver from the fitted codes; not independently recomputable (training fields not saved). Consistent with the oracle level (0.20) | not independently verified |
| 10 | The reruns `ns203`/`ns204` change only `G_HIDDEN` (2R), the head width/depth (512×3) and the step count (100 000) | `configs/phase2-k{16,32}r{256,512}-v2.env` vs `phase2-k{16,32}r{256,512}.env`; `runs/ns20{3,4}/PROVENANCE.json` | `diff` of the env files; staged bytes equal the committed blobs at `beffbb1b` | verified |

## What this self-audit cannot establish

- Whether the H-ORACLE bar (oracle ≤ ½ POD-K) is reachable on this family by this decoder:
  the parent lane's evidence on Burgers (longer training ≈ 2×) is the only calibration.
- Whether the rank-capped bank materially biased the *head's* training (the head was trained
  against a 128-dimensional effective span while nominally 256): a full-rank rerun answers it.
- Claim 6 by value, until a job on a mismatching node reports `value_worst_rel`.

## Addendum — `ns203` (job 3787319, the §A4 rerun) and `ns202` (job 3783797)

Still a written self-audit (Codex quota-blocked until 2026-09-19 11:33). Mechanical check:
`audit_phase2.py` → `artifacts/ns203/audit.json` (67 checks, **all match**) and
`artifacts/ns202/audit.json` (67 checks, the same six oracle-formula MISMATCH rows as `ns201`,
≤ 1.04e-2 relative, explained by claim 4 above — `ns202` ran before §A4).

| # | claim | rests on | check run | status |
|---|---|---|---|---|
| 11 | `ns203`'s bank has numerical rank 256 = R at every mesh; B-RANKCAP and B-ORTH pass | `gates.B-RANKCAP`, `gates.B-ORTH_N{64,128,256}.rank`, `.cond_Rb` | audit recomputes the NumPy bank from the pickled weights and its SVD rank: 256 at all three meshes; κ(R_b)=44.4 | verified |
| 12 | H-ORACLE FAILS at every mesh: oracle median 0.2034/0.2026/0.2024 vs POD-16 0.2417/0.2404/0.2402, ratio 1.19 (bar 2.0) | `oracle.{N}.oracle_median`, `.podK_median`, `gates.H-ORACLE_N{N}` | audit recomputes ‖G h(z) − u‖/n₀ exactly at the saved codes on the 48 archived states (worst rel. diff 2.3e-14 at 256²), the medians, and the pass rule | verified — this is the §A4 pre-registered negative |
| 13 | The whitened-formula contamination is gone | `oracle.{N}.formula_vs_field_worst_rel` = 2.5e-14 / 3.2e-14 / 4.7e-14 | audit's exact recomputation agrees with the reported field-space values to 2.3e-14 | verified |
| 14 | B-DATA passes by hash at 64², 128², 256² (train and dev) on `pax050` | `gates.B-DATA_*` (`mode=hash`) | read from the JSON; hashes equal Phase 1's (`pax105`) | verified (hash); the §A4 value path was not exercised on this node |
| 15 | The failure is generalisation, not capacity: training-recon median 0.050 vs held-out oracle median 0.202 (ratio 4.0); at t=0 the oracle beats POD-16 by 1.63×, on evolved times by 1.19× | `training.recon_rel_l2_median`; `oracle.256.per_state_{oracle,podK}` reshaped (64 cases × 6 times) | `generate_ns2d.py` computes the per-time medians and the ratio from the per-state lists; the audit checks those lists equal the npz arrays element-wise; the training median is the driver's own number (claim 9 caveat) | verified for the held-out side; the training side is not independently recomputable |
| 16 | The oracle fit is converged: LM median 39 iterations, no budget exits | `oracle.{N}.oracle_iters_median`, `.oracle_reasons` | **CORRECTED (addendum 3):** `make_lm_fit` maps reason 0 = budget; `oracle_reasons` = {0: 12, 4: 372} at 256², i.e. 12 of 384 states (3.1 %) stopped at the 300-iteration cap | claim as first written was WRONG; see addendum 3 |
| 17 | `ns202` (K=32, R=512, g_hidden=128): rank 128 at every mesh, B-FLOOR fails (ratio 3.7–3.9), H-ORACLE fails (0.118 vs POD-32 0.139, ratio 1.18) | `artifacts/ns202/result.json` gates | audit recomputes rank 128, the floors, the medians and every pass rule; the six oracle-formula rows mismatch by ≤1.04e-2 as for `ns201` | verified; retracted attempt, shown for the record |

**Open after this addendum.** Whether K=32 (`ns204`, job 3787320) also fails H-ORACLE with a
full-rank bank; whether a lower-dimensional family or more trajectories would close the 4.0×
held-out/training gap (a new pre-registration, not done).

## Addendum 2 — `ns204` (job 3787320, K=32, R=512, 2R bank)

Written self-audit; mechanical check `artifacts/ns204/audit.json` (67 checks, **all match**).

| # | claim | rests on | check run | status |
|---|---|---|---|---|
| 18 | rank 512 = R at every mesh, κ(R_b)=141; B-RANKCAP/B-ORTH pass | `gates.B-ORTH_N{64,128,256}.rank`, `.cond_Rb` | audit recomputes the NumPy bank and its SVD rank: 512 at all three meshes | verified |
| 19 | H-ORACLE FAILS: oracle median 0.1218/0.1211/0.1209 vs POD-32 0.1407/0.1396/0.1395, ratio 1.15 (bar 2.0) | `oracle.{N}.oracle_median`, `.podK_median`, `gates.H-ORACLE_N{N}` | audit recomputes ‖G h(z) − u‖/n₀ at the saved codes on the 48 archived states and the pass rule; matches to 1e-9 | verified — pre-registered negative (§A7) |
| 20 | B-DATA passes by hash at all meshes on `pax049` — the node whose 256² hash mismatched in `ns201` | `gates.B-DATA_*` | read from the JSON; §A4's value path was not exercised; the earlier `ns201` mismatch on this node is therefore not reproduced and remains unexplained (claim 6 stays open) | verified (hash) |
| 21 | Held-out / training = 3.2 (0.1209 vs 0.0376); POD-32 / oracle = 0.82 at t=0 and 1.16 on evolved times | `training.recon_rel_l2_median`; `oracle.256.per_state_{oracle,podK}` (64 cases × 6 times) | `generate_ns2d.py` computes from the per-state lists; audit checks lists equal the npz arrays | verified on the held-out side; the training median is the driver's own number |
| 22 | LM median 65 iterations (budget 300); reason codes {0: 18, 4: 366}: **18 of 384 states (4.7 %) are budget exits** (reason 0), the rest stationary (reason 4) | `oracle.256.oracle_iters_median`, `.oracle_reasons`; `ns2d_decoder.make_lm_fit` docstring | read from the JSON and the code map | reported; not fully converged for 4.7 % of states |

## Addendum 3 — correction: some oracle fits are budget exits

Claim 16 said `ns203`'s oracle had "no budget exits". The code map in `ns2d_decoder.make_lm_fit`
is `0 budget, 1 tolerance, 2 tiny step, 3 rejected, 4 stationary`; the stored `oracle_reasons`
are {0: 12, 4: 372} (ns203, every mesh ±1) and {0: 12–18, 4: 366–372} (ns204). So 3.1 % (K=16)
and 3.1–4.7 % (K=32) of the 384 held-out states stopped at the 300-iteration cap. The oracle
error on those states is an *upper bound* on the best fit; the medians (over 384 states) cannot
move by more than the rank of ~18 states and the verdicts (ratio 1.19 / 1.15 vs bar 2.0) stand.
The report now shows the count per mesh. DESIGN §A8 records the correction.

## Addendum 4 — `ns301` (job 3808493, head-only diagnosis, DESIGN §A9/§A10)

Written self-audit; mechanical check `artifacts/ns301/audit.json` (107 checks, 104 match; 3 value-level mismatches explained in §A10 and quantified in `artifacts/ns301/audit_oracle_beaten.json`).

| # | claim | rests on | check run | status |
|---|---|---|---|---|
| 23 | Frozen bank rank 256; B-DATA by hash on `pax052` | `gates.B-ORTH_N256`, `gates.B-DATA_*` | audit recomputes the NumPy bank's SVD rank (256); hashes read from the JSON | verified |
| 24 | Dev-report oracle medians 0.2749/0.2069/0.1983 (plain), 0.2611/0.2102/0.1939 (reg), 0.2466/0.2134/0.2009 (reg_small); POD-16 of the subset 0.2358/0.2296/0.2277 | `arms.*.eval.dev_report.per_state_*` | audit recomputes every median and ratio from the per-state arrays; bank floor on the 48 archived states recomputed exactly (≤1e-9) | verified |
| 25 | Slopes −0.236/−0.215/−0.148 → "ambiguous" by the §A9 rule; successive slopes −0.41→−0.06, −0.31→−0.12, −0.21→−0.09; reg vs plain at n=512 +2.2 %/−1.3 % | `summary.*`, `arms.*` | audit recomputes the LS slope and the verdict rule; `generate_ns2d.py` computes the successive slopes and the reg fraction from the same medians | verified |
| 26 | The oracle is a best-found upper bound: an independent SciPy LM beats it on 1/48 audited states in 3 of 9 arms; medians unchanged to 1e-10 | `audit_oracle_beaten.json` | direct recomputation | verified; recorded as a finding |
| 27 | Train-oracle 0.017–0.129 and training recon 0.024–0.173 per arm | `arms.*.eval.train`, `arms.*.training` | medians and gaps recomputed from per-state arrays; training recon is the driver's own number (training fields not saved) | verified (gap side), reported (recon) |

## Addendum 5 — `ns304` (job 3808502, exploratory q-ladder on the ns204 manifold, DESIGN §A9/§A11)

Written self-audit; mechanical check `artifacts/ns304/audit.json` (`audit_phase3.py`, NumPy only: 351 checks, **all match** — every per-subject/case error metric recomputed from `rom_fields_N256.npz` against `reference_N256.npz`, every aggregate, the timing medians from the retained reps, and the R-LADDER arithmetic).

| # | claim | rests on | check run | status |
|---|---|---|---|---|
| 28 | Job complete: 13 subjects × 8 cases × 4 reps = 416 invocations, no dropped subject, no early exit | log (`INV` lines, `ALL-DONE`), `invocations` in result.json | counted in the log; audit reads every subject×case from the npz | verified |
| 29 | Ladder worst-evolved 0.688/0.708/0.615/0.566/0.331/0.060; median-evolved monotone; inversion q0→q32; gain 11.4×, cost 3.9× | `aggregates.neural_q*`, `gates.R-LADDER` | audit recomputes from fields; generator re-derives monotonicity from the aggregates | verified |
| 30 | POD-LSPG at matched k' better and cheaper at every rung; no neural rung in the non-dominated set; FOM ntol 1e-3 (421 ms, 4.1e-5) dominates all ROMs | `aggregates.pod_k*`, `aggregates.fom_*` | audit recomputes the errors; the Pareto set is computed by `generate_ns2d.py` from the same medians | verified |
| 31 | Decomposition: bank 0.00225, manifold 0.106→0.043→0.0023 (q=0/256/512) on the 48 reference states; energy captured 26/44/68/92/100 % | `decomposition.q*`, `directions.residual_energy_captured` | not independently recomputable without the head (JAX); read from the JSON; formula-vs-field agreement is stored per rung | reported |
| 32 | Zero LM budget exits in every timed rung (R-CONV) | `aggregates.*.budget_exits` | audit sums the per-invocation reasons | verified |
| 33 | Train-cohort hash mismatched on `pax105` and passed by the §A4 inferred rule (dev reference by value 3.6e-16) | `gates.R-DATA_*` | read from the JSON | verified (mode recorded) |
