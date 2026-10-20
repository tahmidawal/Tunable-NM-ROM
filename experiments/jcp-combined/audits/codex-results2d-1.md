**Overall: WRONG — budget job HOLD.** The 2D numerical selections and hypothesis verdicts reproduce. The submission certificate omits substantial required work, and the new smoke gate accepts incomplete or failed evidence. No files were modified or created.

All accuracy results below remain **PROVISIONAL**, against first-order references.

| Requested item | Verdict | Finding |
|---|---|---|
| 1. G-score and binding | **CORRECT** | Scores bind to `result.json` SHA256 `7774766d…9f500d`. Independently checked all **171,000** stored score/job comparisons: zero failures, maximum difference **8.33×10⁻¹⁷**. Fresh CPU decoding of 648 sampled values also passed. |
| 2. Scheme and step selection | **CORRECT** | Independently reconstructed all 18 scheme×setting selections from scores and raw timing invocations. No mismatches. A2.6/A3.2, eligibility and panel validity are applied correctly. |
| 3. H1/H2/H3/H5 | **NEEDS-RESTATEMENT** | Numerical verdicts reproduce. H3’s required R128/R256 companion comparison is absent from the hypothesis output; supplied below. H5 needs its precise inventory scope retained. |
| 4. Timing and FOM | **CORRECT** | All seven panels pass A5.3; hashes and paired ratios check. FOM implementation is byte-identical to C2’s, with calibrated tolerances and verified steps. |
| 5. Selection cohorts | **CORRECT** | Configs, rows, archives and loader show only **dev6+val32**. No test64 or sealed-cohort selection found in the audited path. |
| 6. Frozen budget arms | **CORRECT** | All six settings contain exactly A1.5’s baseline, diagnostic and two selected arms, with ordered fallbacks. Config matches the frozen file and its SHA256. |

**Selection recomputation.** Global points are BDF2 **3**, CNR **4**, GAL-CN **4**. CNR wins the prescribed raw-sum tiebreak: **0.0104468631**, versus GAL-CN **0.0107473241**. The final fixed-order tiebreak is not reached.

Every setting selects CNR at **Δt₀** for accurate-equal-cost and **2Δt₀** for fast-equal-accuracy:

| Setting | Accurate cost / own BE | Fast cost / own BE |
|---|---:|---:|
| fast | 0.998504 | 0.611213 |
| R128 | 0.998324 | 0.587215 |
| R256 | 0.999134 | 0.560625 |
| acc | 0.997622 | 0.559600 |
| R384 | 0.995964 | 0.550749 |
| R512 | 0.985601 | 0.535542 |

The heuristic median/worst tie thresholds independently reproduce as **0.00121839875 / 0.00353489083**. Several equal-cost decisions sit close to 1, but satisfy the registered rule.

**Hypothesis findings:**

- **H1 — CORRECT verdicts.** R128→R256 passes: TX median falls **52.80%**, with **27 improved / 0 worsened / 11 tied**. R256→R384 is **reference-limited**: only **6/38** resolved. R384→R512 is also **reference-limited**: TX has **0 improved / 5 worsened / 33 tied**; ST has **0/6/32**. This does **not** establish an extended dial.
- **H2 — CORRECT, passes both parts.** Both worst- and median-TX coverage are **42/42 eligible BE arms**. Independently reconstructed numerical fronts match. Common costs along the first three selected rungs increase **0.239006 → 0.567883 → 1.561259**, while median TX error is non-increasing under the tie policy.
- **H3 — CORRECT primary verdict; reporting NEEDS-RESTATEMENT.** Worst case is **val32/19**, with \(t^\ast=0.25\) for ST and TX. “Cancellation not supported” is correct: the CN-to-anchor closeness ratios are **0.45165 / 0.48781**, exceeding **0.2**, despite passing angle/cancellation-strength tests. The missing companion comparison: using the selected accurate arms, **R128 does not remove** the worse ST worst case (**5.142%**, versus fast-BE **4.637%**); **R256 does** (**2.472%**). Budget attribution remains pending.
- **H5 — CORRECT pass under amended A1.9.** Independently reconstructed matching and invocation-paired speedups give **17.70574×**, versus remeasured C2-inventory **3.79728×**, a **4.66274×** improvement, for both ST and TX. The winner is **fast/LSPG-CN/5Δt₀**, matched to **FOM-CN/2Δt₀**—not the frozen CNR arm. A1.9 explicitly permits every eligible ROM arm. Do not describe this as the selected CNR method’s speedup or proof of second-order LSPG-CN. S errors are retained in scores and all ROM aggregate rows.

**Timing/FOM details.** There are **7,830 A–B–A invocations**, all on the same A100-80G UUID. Maximum subject-median drift is **1.061623**, below 1.1. Individual drift excursions exist but do not invalidate A5.3’s subject-median gate. All **608 FOM rows** pass residual verification; all 16 tolerance choices reproduce the calibration rule. “Second-order FOM” means the C2 time integrator; spatial advection remains sign-upwind. CN’s dev6/0 order check retains its pre-asymptotic qualification.

The independently recomputed G-score controls all discriminate: case swap **1.116836**, time shift **0.175700**, ST/S swap **0.00290271**, coordinate shift **0.0223937**.

**Budget job blockers:**

1. **WRONG — incomplete resource inventory and projection.**  
   [c2bud.py’s dry-run](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2bud.py:79) exposes only `budget_runs()`. [project_bud2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:240) consequently omits, per setting:
   - **Nine additional check-rule trajectories × 16 cases**;
   - **16 control-rule calls** and **32 B1 replay calls**;
   - Conditional successor checks, discarded first-pair work, and restoration work if both pairs fail.

   Even the ordinary successful path has **1,338 saved rows per setting**, versus **1,178 inventoried**. Using the certificate’s own measured rates, adding only the omitted ordinary calls raises the margin-adjusted estimate from **21.894 h to 46.141 h**, before additional compilation/setup and conditional work. Successor rule size affects the memory estimate but never the timing estimate. This violates A2.12; the current **48 h** certificate does not establish fit.

2. **WRONG — smoke acceptance does not fail closed.**  
   [smoke_gates](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:294) checks completeness, existing rows’ verification and existing convergence flags. In-memory mutations remained accepted after **setting B1 false**, **deleting a required row**, or **removing an entire setting’s evidence**. It also does not bind the offline B0/G-score machinery checks.

**What does pass for the budget job:** config/frozen/source/smoke hashes, current staging-certificate verification, and the certificate’s arithmetic (**40.821 GB** projected device memory). The actual smoke is good: CPU reanalysis with file output suppressed reproduced **3,888 values**, zero failures; B0 **7.08×10⁻¹²**; B1, inventory and controls pass. **A12 is CORRECT:** the relative mismatch defect is **1.377219**, comfortably above \(10^{-6}\). Grid reproduction was not run in this smoke, as declared.

**Release condition:** complete the invocation inventory—including conditional paths—repair smoke validation, and regenerate a resource certificate that fits the requested allocation. Until then: **HOLD**.