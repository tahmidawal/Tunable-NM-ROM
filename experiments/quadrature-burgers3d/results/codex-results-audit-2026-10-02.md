**Verdict: the main numerical results and frozen-selection procedure hold up. The report needs corrections to one statistic and several interpretations before acceptance.**

Read-only audit completed; no files, git state, jobs, or network were changed. I reproduced the report in memory, suppressing filesystem writes, and obtained an exact match.

**1. Numerical traceability — CORRECT for the panel tables; WRONG for the reference runtime median.**

Across validation and held-out data, independent recomputation found no mismatches in:

- 188 refined maxima and 188 medians from `per_case`;
- 180 query timing medians from recorded invocations;
- 144 Jacobian timing medians;
- 288 speedup entries, including independently choosing the qualifying FOM;
- same-grid maxima, iteration medians, exit counts, and validation selections.

These representative checks alone cover 24 reported numbers. Errors are percentages; timings are milliseconds:

| Mesh / rank | Selected refined error | Tensor refined error | Selected time | Tensor time |
|---|---:|---:|---:|---:|
| 64³ / 512 | 2.825446 → 2.83 | 10.456294 → 10.46 | 34.096515 → 34.1 | 47.915521 → 47.9 |
| 64³ / 256 | 4.490888 → 4.49 | 10.569550 → 10.57 | 11.147669 → 11.1 | 12.164127 → 12.2 |
| 128³ / 512 | 2.831795 → 2.83 | 6.094473 → 6.09 | 40.563272 → 40.6 | 52.041566 → 52.0 |
| 128³ / 256 | 4.488144 → 4.49 | 6.564893 → 6.56 | 12.136437 → 12.1 | 14.121033 → 14.1 |
| 256³ / 512 | 2.833755 → 2.83 | 3.872918 → 3.87 | 76.685269 → 76.7 | 90.079777 → 90.1 |
| 256³ / 256 | 4.487525 → 4.49 | 5.068480 → 5.07 | 33.046418 → 33.0 | 35.079900 → 35.1 |

Sources: `results/refined-ho.json`, `runs/ho*/code/output/result.json`.

**Actual numerical error:** the 96 reference runtimes have median **47.141053 s**, which should print **47.1 s**, not **47.7 s**. The [generator](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/make_report.py:151) selects `sorted(times)[48]`, the upper middle observation, instead of averaging observations 47 and 48.

The checks above validate aggregation from saved data; they do not constitute a fresh reconstruction of every field or certification state.

**2. Questions (i)–(iii) — CORRECT under the registered rules, with interpretation requiring restatement.**

- **(i), lattice versus Gauss:** all 12 registered comparisons pass: two point counts × two ranks × three meshes. None is unresolved under the continuum-target resolution rule. The reported smallest lattice sizes also follow the registration. However, “4096 reaches the tensor” is an accuracy-only criterion: it does **not** imply selection eligibility. `lat4096_R512` fails the continuum certificate at 256³.
- **(ii), mesh invariance:** the selected arms satisfy all three R2 conditions, not merely the error ratio:

| Arm | Error max/min | Distance 64³→128³ | Distance 128³→256³ | Verdict |
|---|---:|---:|---:|---|
| `gl24_R512` | 1.002941 | 0.071938% | 0.023197% | Pass |
| `lat4096_R256` | 1.000749 | 0.065714% | 0.016391% | Pass |
| `tensor_R512` | 2.699849 | 4.616648% | 2.792707% | Fail |
| `tensor_R256` | 2.085349 | 4.608072% | 2.782948% | Fail |

- **(iii), cost:** the selected off-mesh query is faster than the tensor in all six comparisons. Tensor/selected ratios are **1.405, 1.283, 1.175** at rank 512 and **1.091, 1.164, 1.062** at rank 256, ordered by mesh. The reported FOM speedups follow the specified comparator rules.

But “the speedup against Newton–BiCGStab changes little” is misleading without naming the rule. At 256³/rank 512, the same-grid-rule speedup changes **2.21×→2.59×**, while the refined-rule speedup changes **2.21×→6.45×**, because the qualifying comparator changes.

**3. Gates, controls, and R4 — CORRECT outcomes; some qualifications missing.**

All six production panels record GPU execution, f64 and `highest` precision. Their hard gates and continuum checks pass; both converged-rollout eligibility checks pass at every mesh/rank.

- Both empirical bad-rule controls fire in validation and held-out at every mesh/rank.
- Tensor mesh-target ρ satisfies its threshold everywhere.
- Dense continuum ρ exceeds 0.116 at 64³ and falls below it at finer meshes, as tabulated.
- All six NumPy audits pass, including swapped-reference and perturbed-record fault injections.
- The ROM neighbour timing check is **untested** at 256³. The table discloses this; “passed, ROM neighbour check untested” would satisfy R2’s wording more directly.
- Smoke timing **failed**: drift **1.185924**, neighbour ratio **1.207934**. Production results supersede this diagnostic, but the report’s history should mention it.

**R4 is correctly classified as tolerance-limited.** The four re-check differences are:

`6.187360e-6`, `6.575559e-6`, **`1.021678e-5`**, `6.319570e-6`.

Thus the prescribed unresolved threshold is **`1.021678e-4`**. Printing the maximum as `1.0e-05` immediately before saying it exceeds `1e-5` obscures the crossing; print more digits.

“Bounds the effect” needs restatement: four validation cases establish the observed discrepancy and trigger R4; they do not establish a bound for all 96 cases.

**4. Physical interpretation — NEEDS-RESTATEMENT; one claim is WRONG.**

The core explanation is supported: continuum-advection quadrature removes the backward/upwind advection approximation, the tensor–continuum discrepancy decreases with refinement, and the selected off-mesh errors remain nearly constant. The report also correctly acknowledges that the best tested FOM is more accurate against the primary reference at 256³.

However, [the interpretation paragraph](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/reports/2026-10-01-burgers3d-offmesh-quadrature.md:299) overstates what was established:

- **“Its error is the bank’s and the time step’s”** excludes remaining discrete diffusion, projection/test-space effects, solver error, quadrature error and reference error. No error decomposition or bank-floor measurement here establishes that attribution.
- **“Every non-control rule that passes the continuum certificate” agrees within a few hundredths of a percent — WRONG.** At rank 256, certified `sob16384` has worst refined error **4.6143%**, versus **4.4926%** for `lat32768` at 64³: **0.12165 percentage points** apart. The discrepancy is about **0.124–0.125 points** at finer meshes. Restrict this statement to rules satisfying the rollout-distance criterion.
- **“Cannot fall below the gap”** is too absolute. Quadrature error can partially cancel the stencil discrepancy; the relative errors also use different target norms. Describe the observed limiting discrepancy instead.
- **“Below 0.116 by construction”** is false wording. First-order consistency supports an asymptotic trend; it does not enforce a particular threshold or exact halving.
- The refined-reference discussion treats “about one percent” as established uncertainty. The probe’s **1.69926%** is a refinement discrepancy, not a reference-error bound. The held-out Richardson displacement reaches **2.28104%**.

**Richardson:** correctly disclosed as post-hoc and excluded from selection. But the report never defines its formula,
\[
u_R=2u_{513}-u_{257},
\]
or explains that both spatial and temporal steps were halved. It should explicitly state that cancellation assumes an asymptotic first-order expansion and is not a verified continuum truth.

The rank-256 tensor advantage reversal at 256³ is only **0.58096 percentage points** against the primary reference. Report that measured difference, but avoid interpreting it as established continuum-accuracy superiority under the report’s own uncertainty caveat.

**5. Selection freeze and held-out re-selection — CORRECT.**

Local git history supports the ordering:

1. **`2805fe949`, 01:38:27 EDT:** adds frozen `selection.json`, validation results and held-out configs.
2. **`a38cf0fce`, 01:39:30 EDT:** records staged held-out jobs.
3. **`1967e9f73`, 02:27:39 EDT:** adds held-out results.

All held-out panels record source commit `2805fe949` and the matching selection hash:

`ca69098a38b4f145178a4612c96bc12d425fe8f53aa47583eef2aaf55e5bf524`.

The current selection is byte-identical to `select-val.json`.

The report correctly retains **`gl24_R512`** and discloses that reapplying selection to held-out would choose **`lat8192_R512`**. Independently reproduced. On validation, `lat8192_R512` misses the distance threshold with **0.131164%, 0.127210%, 0.124340%**, versus the allowed **0.1%**. This explains the difference without suggesting a freeze violation.

**6. Reporting requirements — NEEDS-RESTATEMENT / incomplete.**

Title, status, LaTeX equation, coloured Mermaid diagram and ending glossary are present.

Missing or inaccurate:

- Glossary definitions for **Richardson**, **LM its/query** and its aggregation, **Jacobian**, **GFLOP**, **Newton–BiCGStab**, **nt/lt tolerance labels**, and timing/gate terminology.
- The dense glossary and diagram incorrectly imply only validation cases at 128³; **four held-out cases** also ran.
- Registered **ρ p90 and initial-state statistics**, **maximum LM iterations**, and dense **quick-run timings** are omitted.
- Genuine audit fault-injection outcomes are hidden behind “audit pass.”
- Explicit qualification that the lattice result concerns the tested single shifts/scramble, not their distribution.
- The report lives under the experiment directory, not the specified root `reports/` location. The checked-out `CLAUDE.md` lacks the detailed report section; those requirements are present in the supplied instructions and DESIGN §10.

**BLOCKERS:** None found for the saved production results or selection integrity.

**MAJORS:** Correct the certified-rule equivalence claim; qualify causal error attribution and reference uncertainty; define and caveat Richardson; distinguish the two speedup-rule conclusions.

**MINORS:** Fix the reference median, preserve R4 threshold precision, complete the glossary and registered summaries, disclose smoke timing failure, and correct dense-run coverage and report placement.