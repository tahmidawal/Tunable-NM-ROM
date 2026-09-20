**The 15 sampled numbers match the summary and artifact JSONs. The report’s verdicts need correction:** its convergence gate differs from DESIGN, a failed gate was waived after results arrived, and several generated conclusions are false. No files were changed and no GPU work or jobs were run.

**Verified numbers**

Paths below are relative to `experiments/b-panel/`:

- **R:** [report](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-panel/experiments/b-panel/reports/2026-09-17-b-panel.md)
- **S:** [reports/summary.json](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-panel/experiments/b-panel/reports/summary.json)
- **A301/A401/A203:** `artifacts/bpn301/audit.json`, `artifacts/bpn401/audit.json`, `artifacts/bpn203/audit.json`.

For ordinary rows, the lookup is `S.rows[subject, metric, job_id].value` → `A.arms[arm=subject][metric]`. Values below are shortened for readability; equality was checked before rounding.

| # | Report location | Job / artifact | Subject; metric key | Reported | Summary = artifact | Result |
|---:|---|---|---|---:|---:|---|
| 1 | R:65 | 3789570 / A301 | `q0_M256_dense_g1em06`; `worst_all_times_percent` | 2.5629% | 2.562871982659394 | Match |
| 2 | R:95 | 3789570 / A301 | `q256_M1088_eqtop_g1em06`; `worst_evolved_percent` | 0.5129% | 0.512919436859756 | Match |
| 3 | R:93 | 3789570 / A301 | `q256_M1088_eqcert_g1em06`; `worst_evolved_percent` | 1.0361% | 1.036116103868751 | Match |
| 4 | R:7, prose | 3789570 / A301 | `pod512_M2048_dense`; `median_gpu_ms` | 2790.8 ms | 2790.828178985976 | Match |
| 5 | R:344, prose | 3789570 / A301 | `fno-large`; `median_gpu_ms` | 7.183 ms | 7.182571920566261 | Match |
| 6 | R:9, prose | 3805065 / A401 | `pod512_M2048_dense`; `worst_evolved_percent` | 0.3328% | 0.332780608690063 | Match |
| 7 | R:579, prose | 3805065 / A401 | `fft_tight`; `worst_reference_percent` | 2.7025% | 2.702451024754954 | Match |
| 8 | R:453 | 3805065 / A401 | `q256_M1088_eqtopxfer_g0p001`; `median_gpu_ms` | 486.341 ms | 486.341400421225 | Match |
| 9 | R:722 | 3805065 / A401 | `q32_M192_eqtopxfer_g1em06`; `rho_max` | 0.1319 | 0.131946389514103 | Match |
| 10 | R:1083 | 3805065 / A401 | Fast q0 / `nt1e-2_dt01`, using `median_gpu_ms` | 2.923× | 40.491704479791 / 13.851479045115 = 2.923276593634 | Match |
| 11 | R:11, prose | 3789572 / A203 | `free512_M1024_dense`; `worst_evolved_percent` | 0.5108% | 0.510791042224290 | Match |
| 12 | R:1084 | 3789572 / A203 | `q0_M64_eqxfer_g0p001`; `median_gpu_ms` | 32.2 ms | 32.229534466751 | Match |
| 13 | R:1003 | 3789572 / A203 | `q128_M576_eqxfer_g1em06`; `rho_max` | 0.1702 | 0.170173935310172 | Match |
| 14 | R:1004 | 3789572 / A203 | `q256_M1088_eqxfer_g1em06`; `rho_max` | 0.3239 | 0.323875464059935 | Match |
| 15 | R:11, prose | 3789572 / A203 | Reduced members of `nondominated_gpu_evolved_admissible` → `nondominated.gpu_evolved.admissible` | 5 | 5 | Matches stored gate* |

\*The count reproduces the implemented gate, which differs from the registered gate.

I also recomputed all **123 JAX-arm timing medians from 2,214 `result.json` invocations**: no discrepancies. Sampled same-grid errors reproduce from the audit’s per-time arrays; transferred-rule certification values match `result.json.transfer[].certification`. The three summary partitions retain the correct job IDs and recorded result hashes.

**Mismatches and unsupported conclusions**

1. **Convergence is evaluated against a relaxed threshold without an amendment.** [DESIGN §5](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-panel/experiments/b-panel/DESIGN.md:185) requires every step’s gradient to be at most \(10^{-6}\), unless it exits on the residual rule. But [audit_panel.py:180](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-panel/experiments/b-panel/audit_panel.py:180) uses each invocation’s `gtol`, including \(10^{-3}\).

   Concrete counterexample: `artifacts/bpn203/result.json`, invocation `name=q0_M64_eqxfer_g0p001, case=0, rep=0`, has `converged=true`, maximum `step_joint_stationarity=0.0009947115635679646`, and all 50 `stop_reasons=4`—gradient exits, not the residual exception.

   Applying the written threshold changes the admissible reduced counts:

   | Mesh | Reported frontier / admissible reduced | Under DESIGN §5 |
   |---|---:|---:|
   | 256² | 0 / 39 | **0 / 27** |
   | 512² | 0 / 31 | **0 / 23** |
   | 1024² | 5 / 20 | **3 / 16** |

   The surviving 1024² reduced frontier arms are the \(10^{-6}\) `eqxfer` arms at q=0,16,32. Its cheapest-admissible-reduced/FOM ratio becomes **2.272019521276**, replacing **1.818384667773**.

2. **The 512² prediction section duplicates six historical records into twelve comparisons.** R:746 reports “6 of 12” capped certifications. However, `artifacts/bpn202-failed/FAILURE.json.completed_before_crash.rule_transfers` contains **six 1024² transfers**, with **three primary certifications**. R:732–743 repeats these records for both 512² rule sets. The generator indexes historical records only by q (`generate_panel.py:444–448`), losing mesh and rule-set identity.

   Consequently, R:749’s verdict that §A5.2 “did not hold” is invalid: §A5.2 named two qrg304 transfers in `bpn201`, not four transfers across two rule sets at 512².

3. **Two counterfactual frontier statements are false.** R:561 and R:920 say removing `free512_M1024_dense` makes `pod256_M1024_dense` non-dominated. Other arms still dominate it:

   | Mesh | Remaining dominator | GPU ms versus POD-256 | Worst all-times % versus POD-256 |
   |---|---|---:|---:|
   | 512² | `q256_M1088_eqtopxfer_g0p001` | 486.341 versus 3488.159 | 2.019542 versus 4.001048 |
   | 1024² | `q128_M576_eqxfer_g0p001` | 128.250 versus 4897.627 | 3.285840 versus 4.141559 |

   Recomputing both frontiers without the free bank leaves POD-256 dominated. The analogous 256² statement about **POD-512** is correct.

4. **Some interpretations exceed their measurements.** R:579’s “at or below [the discretisation error] is indistinguishable” does not follow from comparing one worst-error scalar. Likewise, R:749/1035 infer lost “weight mass” from fewer positive quadrature weights; a support count does not measure weight mass. The numerical observations remain valid.

**Gate audit**

| Verdict group | Registered rule / amendment | Audit finding |
|---|---|---|
| Frozen model, precision, cohort, references, paired timings, repetitions, subject survival | §§3,6,7 | Recorded checks pass across the three accepted jobs. Reference maximum residual is `9.875956374345395e-12`, below `2e-11`. |
| 256² fidelity | §6 | All ten comparisons pass their specified tiers. No relaxation needed. |
| Fast-kernel admission | §3 | Pass at 256² and 512²: relative differences `1.1664953168571243e-13` and `2.006163103631948e-13`; integer vectors identical. |
| POD/free-bank convergence despite strict-flag failures | §5 initial-fit residual exception | Supported for the tight-tolerance arms. The separate acceptance of \(10^{-3}\) evolution gradients violates §5, as detailed above. |
| Non-dominated sets, all four metric pairs | §7 | Recomputed sets match the artifacts under their stored admissibility flags. The primary headline needs the §5 correction. No reduced arm enters an all-times admissible frontier. |
| Reduced-only comparisons | **A4, explicitly after `bpn101` landed** | Properly labelled post-hoc. Arithmetic sets reproduce; the two removal counterfactuals do not. |
| Rule certification and ladder monotonicity | §3.2; A3,A7,A8,A11 | Primary/secondary/none classifications reproduce from the 0.116 thresholds. Monotonicity flags reproduce. Source single-draw caveats are carried. “All converged” at loose tolerance inherits the §5 defect. |
| Acceptance of the 512² panel | §6 mandatory gates; A8 identical-file gate; **A12 after `bpn401` landed** | **Post-data waiver/reinterpretation.** Both `matched_rule_files_bitwise` and its independently recomputed counterpart are false. A12 retains the results by citing §7’s narrower failure clause. That does not resolve §6’s “all must pass” requirement. The failure is disclosed, but acceptance is no longer an unqualified preregistered pass. |
| §A5.2 prediction | A5.2,A7,A10 | Failed capped attempts support the original two-rung prediction. The uncapped 1024² results show that 64 states did not ensure primary certification; they are a changed experiment. The 512² scoring is outside the prediction’s scope. |
| 512² outcome and mesh crossover | A11,A12 | Zero reduced frontier members at 512² reproduces. A hardware-independent crossover location does not follow from these jobs. |
| Physical-error conclusions | §4; A12 presentation addition | All reduced subjects exceed their mesh’s `fft_tight` worst-reference error: 39/39, 39/39, 24/24. This supports that specific scalar comparison, not error additivity or physical indistinguishability. |

A3 changed the transfer-state collector after smoke evidence; A7 changed the fit-state count after failed attempts and before the accepted rerun. These are disclosed methodological changes. **The clearly post-result acceptance change is A12; A4 is a disclosed post-result analysis addition.** I found no amendment authorizing the convergence-threshold change, so its timing cannot be established from the amendment record.

**Cross-job check**

The direct arm-to-arm ratios use timings from the same job:

| Mesh | Job | GPU | Cheapest reduced / cheapest FOM, as reported |
|---|---|---|---:|
| 256² | 3789570 | A100 80GB PCIe | 4.479691669731 |
| 512² | 3805065 | A100 80GB PCIe | 2.923276593634 |
| 1024² | 3789572 | H200 | 1.818384667773 |

This also holds for headline dominance ratios, rule-set comparisons, dense/EQ ratios, and ladder cost spans. Archived batch scripts place the FNO process in the same single-GPU allocation.

**The strict “no ratio crosses jobs or GPUs” check has two exceptions:** R:1093’s **2.46** and **2.92** factors divide the 256² within-job ratios by the 1024² within-job ratios. They are cross-job, cross-GPU *ratios of ratios*. Their arithmetic is correct; normalization does not eliminate hardware dependence.

Additionally, loose-EQ/dense ratios change stopping tolerance as well as quadrature. At 512² q=256, the reported comparison gives **33.3508×**; matching both arms at \(10^{-6}\) gives **20.7064×**.

**Retraction completeness**

**Incomplete.** `checks/retractions.md` does not exist, and the report has no dedicated retractions section. Against the available amendments and dated check notes:

| Required history | Coverage |
|---|---|
| `bpn201` crash and `bpn202` OOM; neither yielded timed results | Covered at R:5 and in the prediction sections. |
| `bpn101` superseded, not withdrawn | Correctly covered at R:5. |
| A3 withdrawal of the frozen-state transfer collector | **Missing as an explicit retraction**, despite `checks/2026-09-17-lab-log-entry.md:63` identifying it. |
| Correction to “the entire reduced frontier” | Correctly reflected in the 256² family counts and free-bank explanation; incorrectly generalized into the two counterfactuals above. |
| Earlier POD nonconvergence interpretation | Covered through the initial-fit residual explanation and strict flags. |
| Fit-state-starvation explanation | Qualified at 1024², but the erroneous 512² prediction scoring must be removed or rewritten. |
| A12 failed identical-file gate | Disclosed, though the post-data acceptance exception needs explicit treatment. |
| A1 independent pre-job audit unavailable | **Not disclosed in the report itself.** |

**Three weakest claims and what a reviewer would ask**

1. **“Five admissible reduced subjects at 1024².”** The count depends on a convergence rule different from DESIGN §5. A reviewer would request corrected flags, frontiers, and cost ratios—or a dated amendment establishing the alternative rule.

2. **“The crossover lies between 512² and 1024².”** Hardware changes from A100 to H200, and the 1024² FOM panel omits the `nt1e-3` controls present at smaller meshes. A reviewer would request matched hardware and comparator sets, plus timing uncertainty, before attributing the transition to mesh refinement.

3. **“The prediction/mechanism is scored by these transferred-rule results.”** The 512² scoring changes mesh, rule set, and fit-state regime and duplicates historical records. A reviewer would request comparisons keyed by `(mesh, rule_set, q, fit-state regime)` and controlled refits before assigning a causal explanation.