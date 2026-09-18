# Written self-audit of the b-lowvisc gate report (Codex substitute)

Codex (`gpt-6-astra`) is unavailable until 2026-09-19 11:33 (coordinator notice, verified). Per
DESIGN §10 A1 this written self-audit is the recorded **substitution**, not an equivalent: it is
the same model family as the author and is therefore strictly weaker than the required
independent audit. If this lane is still open after 2026-09-19 11:33, the Codex audit of the
report is run and appended as a dated addendum, retracting anything it overturns.

Every claim in `2026-09-17-b-lowvisc.md`, the JSON field it rests on, and the check run.

| # | claim | field it rests on | check run | outcome |
|---|---|---|---|---|
| 1 | Job completed on a GPU in f64 at `highest` | `result.json: complete, backend, x64, matmul_precision` | audit checks `job_complete`, `jax_backend_gpu`, `x64_and_highest_precision`; log line `jax_backend=gpu`; `sacct` says COMPLETED 0:0 | pass |
| 2 | No trained model entered the gate | `trained_model_used` | audit `no_trained_model_used` | pass |
| 3 | Sealed cohort untouched | `final_cohort_unopened` | audit `final_cohort_unopened` | pass |
| 4 | The two cohorts differ only in viscosity, ν ratio exactly 10 | `family_relation`, `gates.cohorts_differ_only_in_viscosity` | audit redraws both cohorts from the seeds with NumPy and recompares (`max_relative_deviation_from_ten` 3.55e-16) | pass |
| 5 | The incumbent cohort is b-panel's cohort | `gates.incumbent_cohort_matches_abl01_to_one_ulp` (1.97e-16, bitwise false) | value gate not hash gate, per the 1-ulp `np.exp` landmine | pass, bitwise deliberately not required |
| 6 | **Validity**: incumbent POD floors reproduce job 3780638 | `gates.incumbent_pod_floors_reproduce_bpn301`, max rel dev 1.18e-06 | comparator read from committed `comparators/bpn301-summary.json`, never typed; per-rank deviation ≤2.4e-10 below k=512 | pass |
| 7 | **Validity**: incumbent L=256 discretisation error reproduces 4.0265 % | `subject_summary[incumbent__fft_tight__L256].worst_vs_reference_percent` | equals b-panel's figure to the printed digits | pass |
| 8 | Leg (a) floors and ratios | `leg_a_pod_degradation` | audit recomputes every floor by the **Gram-eigenvector route** (different algebra from the job's mode route): max disagreement 1.11e-09 pp (incumbent), 2.97e-12 pp (lowvisc) | pass |
| 9 | G-a passes at 12.092× | `gates.leg_a_pod_degrades` | bar 2.0 pre-registered in DESIGN §6 before the job; replayed by the audit | pass |
| 10 | Singular-value decay claims | `snapshots[*].pod_singular_values`, `pod_tail_fraction` | eigenvalues checked non-increasing by the audit; ratios computed in the generator, not typed | pass |
| 11 | Leg (b) costs, Newton counts, ratios | `leg_b_fom_cost` | timings are same-job, same-GPU, 3 reps with burn-in and device sync, randomised order, all reps retained in `subject_summary.all_gpu_ms`; **no ratio is formed across jobs** | pass |
| 12 | G-b passes at 3.292× | `gates.leg_b_fom_cost_rises` | bar 1.5 pre-registered; every one of the 8 settings exceeds 1.0× | pass |
| 13 | The timed tight subject is the same object as the untimed truth | `gates.timed_tight_matches_untimed_truth` (12/12 bitwise) | this is what makes the same-grid metric comparable to the panel's | pass |
| 14 | **F4 triggered**, 20.8206 % vs 4.0265 %, ratio 5.171× | `subject_summary` at L=256 both families | computed in the generator from the two subject rows; the 3× bar is DESIGN §6 F4, written before the job | **F4 IS TRIGGERED**; reported in the verdict, not buried |
| 15 | The one failed audit check does not move a reported digit | `audit.json: pod_orthonormal_incumbent` 7.21e-05 | independently recomputed here: deviation 4.06e-09 over the leading 256 modes, 1.28e-05 over the leading 500; the k=512 floor recomputed with the full oblique projector `C^T M^-1 C` moves 7.87e-07 relative, ≤5.20e-11 at every other rank; the low-viscosity basis is 1.05e-09 | bounded, **not retracted**, generated into the report |
| 16 | Every reported number is generated | `reports/generate_lowvisc.py` | the report is written only by that script; the four numbers of row 15 were hand-typed in the first draft and were **replaced by generated ones** before commit | fixed during this session |

## What I could not check, and what a real auditor should attack

* **Row 15 is the weakest point.** I bounded the incumbent k=512 floor's sensitivity with one
  alternative projector. A stronger check would recompute the incumbent basis in f128 or by a
  QR/SVD of the snapshot matrix rather than by the Gram, which no artifact in this job supports.
  The headline ratio would have to be wrong by a factor of 6 to fall below the G-a bar, and the
  bound above is 7.87e-07, so the verdict is not in doubt; the *digit* 12.092 at k=512 carries a
  ~1e-6 relative uncertainty that nothing in the report claims otherwise.
* **F4 is a genuine limitation, not a formality.** The low-viscosity cell is under-resolved at
  every mesh probed (20.82 / 13.95 / 8.79 % at L=256/512/1024). Leg (a) may therefore *understate*
  the true low-viscosity Kolmogorov width (numerical diffusion smooths the fronts the basis
  struggles with), but it could equally be argued that the measured degradation is partly an
  artifact of an under-resolved operator. Nothing in this job separates the two. The honest
  statement, which the report makes, is that the same-grid metric is well defined either way and a
  finer-mesh confirmation is owed before the cell is a headline.
* **Legs (a) and (b) are gates, not the result.** Criterion P is untested until stage 3.

## Addendum, stage 2 (2026-09-17, `lvt01` = 3804337) — same substitution, same caveat

| # | claim | field it rests on | check run | outcome |
|---|---|---|---|---|
| 17 | Recipe unchanged from the incumbent | `sep_burgers_r3*.json/config`, `sep_coeff*.json/config`, `hfit_full.json/config+spec` vs push_r3a / dn256b | `recipe_values_match_incumbent`: every listed key equal | pass |
| 18 | Only the data changed | `data.fingerprint` (sum, sumsq, shape) vs the incumbent's | inverted gate: shapes equal, sums differ by >1e-3 relative; job-side `generator-parity.json` passed with bounds (0.001, 0.01) | pass |
| 19 | Training ran to completion, uncapped, on a GPU in f64 | `train.steps_done/time_capped`, `arms.mid.train.steps_done`, `config.backend/x64/matmul_precision` | 300000 / 200000, `gpu`, `True`, `highest` at all three stages | pass |
| 20 | Checkpoint is the one the job hashed | `TRAIN-SHA256.txt`, `OUTPUTS.sha256`, bytes in `checkpoints/`, `EXCLUDED-SHA256.txt` for the npz | all four agree (`9a481e27…` head, `0856f630…` bank, `9e40fe6a…` npz) | pass |
| 21 | The three-layer table is like for like | identical JSON fields (`test_span_floor`, `oracle_test`) from identical scripts on the identical 408-state held-out draw | fields read by the same function for all five runs; no number typed | pass |
| 22 | F2 does not fire (provisional) | `audit.json: f2` | conjunctive over bank floor 9.286×, best-found 1.871×, solved n/a, bar 12.092× | provisional, not firing |
| 23 | "Manifold below the linear floor" | best-found 5.1726 % (held-out) vs POD-512 7.3643 % (development) | **different cohorts** — flagged inline in the report; the panel is the test | not a paper number yet |

**Weakest points, stated.** Row 23 mixes cohorts and is labelled as such. Row 22 evaluates F2's
analogue, not F2; the report says *provisional* every time. The oracle best-found uses the truth
to fit (a representation bound) — it says what the manifold *can* represent, not what the solver
will find; the incumbent's solved/best-found ratio was 1.007 at $q=0$, but nothing yet says the
low-viscosity solver will be as benign. F4 is unchanged and applies to every stage-2 number.

## Addendum, stage 3 (2026-09-17, `lvp01` = 3817807) — same substitution, same caveat

| # | claim | field it rests on | check run | outcome |
|---|---|---|---|---|
| 24 | Every recorded error is right | `audit-panel.json: checks` (b-panel's verbatim NumPy audit) | recomputed from every saved field against the reference and against `fft_tight`; 0 failed of all checks; repetitions bitwise identical; `dense_tight` reproduces `fft_tight` | pass |
| 25 | All 27 subjects converged, panel admissible | `arms[*].converged/admissible`, `total_budget_exits`, `max_joint_stationarity` | re-derived by the audit from exit reasons and stationarity | pass |
| 26 | P fails | `nondominated.gpu_evolved.admissible` = 4 FOM settings | recomputed independently in `panel_tables.py` (set-equal to the audit's) | FAIL, as pre-registered F3 |
| 27 | F2 does not fire | three_layers rows vs `bpn301-summary.json` (`free512` best-found, `q0_M256` best-found and all-times) | ratios 19.0 / 4.48 / 3.27 vs 12.09; conjunctive | does not fire |
| 28 | Ladder monotone, below knob bar | `fixed_M1088` rows | span 1.357× < 2 | monotone; bar not met |
| 29 | vs-reference worst-evolved column | recomputed from fields in `panel_tables.py`; all-times cross-checked against the audit's `worst_reference_percent` | same fields, same norm | pass |
| 30 | best-found is an upper bound here | `solved_over_best_found` = 0.737 at q=0 | the solve beat the 8-start oracle; stated in the report | caveat carried |

**Weakest points.** Row 27's best-found ratio over-states the degradation (row 30). The knob-bar
comparison (row 28) uses b-qxm's 2× bar, which was pre-registered for the incumbent, not restated
in this lane's §6 — it is reported, not gated. F4 applies to every stage-3 number.
