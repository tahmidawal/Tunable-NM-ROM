**No numerical table mismatches found.** I recomputed all 132 raw arms, including eight retimes: worst %, median case-maximum %, and median of all 80 timings. All 124 printed rows agree at displayed precision; per-job summaries, combined summary, and recorded hashes agree. All CG failure counts are zero, and every eligible FOM selection and speedup is correct.

Six rows—`pod128_galerkin_exact` and `pod256_galerkin_exact` at every mesh—have **no accuracy-matched FOM**. Their printed ratios correctly use the disclosed, dagger-marked fallback; they are not matched-accuracy speedups.

The independently recomputed order statistics are:

| Mesh | Post-FOM / other sentinel median | Retimed/original: NM-ROM, POD, U-Net | Original v1 maximum deviation | A2 |
|---|---:|---|---:|---|
| 1024² | 0.903229 | 0.929620, 1.001261, 1.003972 | 34.066% | Pass |
| 2048² | 1.054233 | 0.926928, 0.990441, 1.000614 | 12.588% | Pass |
| 4096² | 1.002350 | 1.002123, 1.000165, unavailable | 0.630% | Pass |

Each mesh has 165 post-FOM sentinel measurements; the other pools contain 525/525/480 measurements. All 11 positive controls per mesh fail as required. The saved audits report zero failures, full 16-case sample coverage, and passing factor/POD checks. I verified their arithmetic and recorded checks, **not a fresh field-level SciPy audit**.

All eight operator-training table rows match `result.json`, including validation metrics independently calculated from its error arrays; checkpoint hashes match the evaluated panels.

**Blocker:** None found in the reported arithmetic or actual A2 coverage.

**Major findings**

- **The report overstates the independent audit.** Its “summary vs independent recompute: 0” compares aggregates calculated from the *same stored full-grid error vectors*. Independent fields only cross-check those errors through restricted samples. A1 explicitly promises that the report will disclose this restriction, but it does not. References: report lines **54, 111, 168**; `DESIGN.md:157`; `audit_panel.py:83`; `summarize.py:59`. The saved sample discrepancies pass their thresholds; this is an assurance/disclosure issue, not evidence of wrong errors.

- **Some gates can pass with incomplete or ineffective evidence.** `audit_panel.py:101–106` requires merely a nonempty retime dictionary, not every prescribed retime. Sentinel coverage checks quantity, not unique block coverage or 15 repetitions (`:87–99`). Random-sample checks do not enforce 50,000 unique indices; when reported errors are ≤1e-9, comparison becomes an automatic zero gap (`:60–72`). Actual retime coverage is complete here, but these checks do not guarantee the coverage claimed in A1.

- **Timed-output finiteness is not securely gated.** `panel.py:208` accumulates parity with `max(previous, value)`; a NaN value can leave the previous finite maximum unchanged. The audit also omits explicit finite/positive timing checks (`audit_panel.py:75`). All inspected timings are finite and positive, and recorded parity is zero; these are latent validation holes.

- **“Every operator … still improving” is unsupported.** Report **line 211** makes that inference from checkpoint timing. The 1024² Transolver’s best epoch is 198, versus final epoch 227; validation mean-case-max worsens from **1.3142% to 1.3619%**. All operators exhausted the budget, but that does not establish continuing improvement or convergence. FNO received only **406/358 updates**, approximately **6.34/5.59 full epochs**. These support a budget-limited comparison, not general operator inferiority. References: `runs/tr{1024a,2048}/pull/out/*/result.json`, corresponding `history.json`, report **182–189**.

- **Large speedups apply specifically to CN–CG.** Exact DST is both more accurate and faster than the highlighted batched-fit NM-ROM at every mesh: **2.132 vs 5.148 ms**, **1.933 vs 5.383 ms**, **7.792 vs 11.412 ms**. Excluding DST from “FOM chosen” is disclosed, but a general full-order-solver speedup claim would be unsupported. References: report **14/40, 71/97, 128/152, 208**.

- **A2 establishes pooled stability, not absence of order effects.** Its 1024² ratio is only **0.00323 above the lower acceptance boundary**; NM-ROM retimes improve by approximately **7%** at both smaller meshes. Pooling can hide block-specific effects, and matmul contamination does not establish sensitivity to every FOM-induced effect. A2’s assertion that v1 measures noise rather than carry-over is stronger than the evidence. References: `DESIGN.md:187–200`; report **54, 111**.

**Minor findings / framing limits**

- **Special structure must remain prominent.** Batched fitting uses exact evolution of sine-test moments; its speed is not evidence for a generic nonlinear rollout. POD’s separable-factor construction is numerically supported—snapshot reconstruction errors are about **1.4e-14**, orthogonality errors ≤**4.9e-15**—but its offline efficiency is specific to this family. References: report **203–205**; `DESIGN.md:69–74`; raw `$.factor_model` and `$.pod.ranks`.

- **Hardware and training budgets limit cross-mesh conclusions.** Panel timing uses A100-80GB at 1024² and H200 above it; 1024² operator training used A100-40GB. Equal seconds therefore do not mean equal compute across meshes, and NM-ROM training was not budget-matched here. Disclosed, but scaling or training-fairness claims would need qualification. References: `DESIGN.md:91–98,181–183,203–205`.

- **“Epochs completed” includes a partial final epoch.** The training table faithfully copies that field, but FNO’s 7/6 entries represent 6.34/5.59 passes through the dataset. References: `ops/train_ops.py:176`; report **183,187**.

- **A6 calibration does not quantify the rank-32 penalty.** Rank-16 autotuning ratios **1.007/0.994** are correct, but cannot establish negligible overhead for rank 32. References: `DESIGN.md:239`; report **172**.

- **The cohort is reused, and the headline NM-ROM worst error occurs at initialization.** Both are legitimate disclosed choices, but neither a fresh sealed-test claim nor a pure evolution-accuracy interpretation follows. References: report **3,199**; all three raw results `$.arms.nmrom_q32_field_direct_tol1e-4_chol.same[12][0]`.

**Verdict:** The tables are numerically consistent and pass the amended A2 protocol. They support a narrowly scoped, same-job comparison against tested CN–CG settings, with strong classical linear baselines. They do not establish general NM-ROM superiority, converged operator comparisons, or a fully independent full-grid audit. Correct the audit and “still improving” wording, retain the structural/hardware caveats, and strengthen gate coverage before treating “every gate passed” as comprehensive validation. No files were changed.