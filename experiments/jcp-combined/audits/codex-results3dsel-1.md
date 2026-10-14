**HOLD c3grid pending correction and regeneration of the resource certificate.** The selections are supported; the blocker is common-panel pricing, not insufficient allocation.

| Audit item | Verdict | Finding |
|---|---|---|
| Pre-registered selection | **CORRECT** | All five settings are unavailable because fixed-sweep CN K=2 fails solver eligibility, leaving every selection pair unresolved. |
| Post-hoc selection | **CORRECT** | Saved statuses, eligibility, controls and family-panel costs support the selections below. |
| A11 extension / A11.3 | **CORRECT** | gl80–gl96 passes; gl112’s archived 42.88-GiB allocation failure supports stopping at gl96. |
| A11.4 justification | **NEEDS-RESTATEMENT** | Numerical reproduction supports accepting the extension, but “initial coefficients differ by one ULP” is too strong. |
| Freeze propagation | **CORRECT** | Preserves label, `posthoc`, effective convergence and null pre-registered convergence. |
| c3grid configuration | **CORRECT** | Contains all five deployed rules, effective converged rules and the post-hoc label; frozen-file hash matches. |
| Per-setting panel pricing change | **CORRECT** | Prices A at fixed-sweep BE dt0 and B at the most expensive timed subject. |
| Common-panel pricing | **WRONG** | Still uses the mean cost of **all grid runs**, contrary to the stated most-expensive-subject policy. |
| Resource certificate | **NEEDS-RESTATEMENT** | Bindings and smoke checks pass, but its projection inherits that pricing defect. |

The independently checked selections are:

| Setting | Converged rule | Lattice m* | Gauss m* | Family costs: lattice / Gauss, seconds | Deployed |
|---|---|---|---|---|---|
| M2_512 | gl56 | lat32768 | gl32 | 0.112938 / 0.112660 | gl32 |
| W256 | gl56 | lat16384 | gl24 | 0.038072 / 0.034867 | gl24 |
| W512 | gl56 | lat32768 | gl32 | 0.126260 / 0.126453 | lat32768 |
| W768 | gl64 | lat32768 | gl40 | 0.279608 / 0.513338 | lat32768 |
| W1024 | gl80, extension | lat65536 | gl48 | 0.923988 / 1.542129 | lat65536 |

Costs reproduce the sum of the two fixed-sweep subjects’ median `tB` values from complete, valid panels. The narrow M2_512 and W512 cost differences determine the prescribed selection; they do not establish a robust performance advantage. `lat256` qualifies for no setting.

For pre-registration, non-control CN failures affect 15 cases for M2_512, seven for W256/W512/W768, and seven for W1024 except eight at gl24. Small successive-rule distances cannot override this solver ineligibility. The post-hoc result establishes agreement for fixed BE and adaptive CN only; subsequent grid-arm eligibility remains necessary.

I independently decoded saved W1024 coefficients through the bank on the full **63³ lattice, CPU/f64/highest**, across all 64 cases:

- Adaptive-CN distances to gl80: lat32768 **3.54380150e−4** and gl40 **2.96661590e−4** fail τ; lat65536 **1.08322587e−4** and gl48 **1.82729761e−4** pass.
- Extension gl80–gl96 distances: BE **1.16809995e−5**, adaptive CN **2.42626014e−5**; both pass **2.5e−5**, with case 5 worst.
- All four gl80 cross-job reproductions pass, with worst distances **6.19e−16–1.52e−15**.

For [A11.4]( /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md:973), the archives contain two cohort hashes. Only case 1’s initial coefficient vector differs: 977 entries, maximum absolute difference **2.77556e−17**, maximum per-case relative difference **1.17790e−16**. This supports numerical equivalence, but not a literal one-ULP bound on individual coefficients or definitive attribution to the peak normaliser. Also, the selector implements a **global** maximum-relative coefficient check, rather than separately normalising each case; these archives pass either interpretation.

The blocking defect is [inventory.py:150](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:150). Common-panel pricing uses **0.81624 s**, the mean over all grid calls; the most expensive actual common subject is **3.23731 s**, with baseline A **0.29921 s**. Pricing the actual 174 subjects and warm-ups conservatively raises that panel from **4,489.56 to 8,081.30 s**, and the margin-adjusted total from **20.94 to 22.44 h**. It still fits 48 h.

Source/config/smoke hashes, archived output checksums, GPU/f64/highest logs and smoke machinery checks pass. The projected device peak reproduces **15.764 GB**. Correct common-panel pricing and regenerate the H200/48 h/200G certificate before submission.

No files were written and no job was submitted.