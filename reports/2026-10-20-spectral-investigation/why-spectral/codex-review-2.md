## Verdict

**NEEDS-FIX**

The measured in-range spectral advantage survives review. The crossover remains a conditional scenario, with unresolved accuracy/cost pairing and a 3D envelope that does not cover censored cases. The completed regime table also contains a cohort mismatch.

No files were changed and no GPU or cluster jobs were launched.

## Numbers checked

Paths below are relative to `reports/2026-10-20-spectral-investigation/why-spectral/`. A1–A7 denote its analysis JSONs; FOM, P1 and REG denote the corresponding lane files.

All **292 `VALUES.json` entries**, including source strings, reproduce from `values.py`. In-memory regeneration exactly reproduces `report.md`. This establishes build consistency, not correctness of every source label.

| Claim | Report value | Recomputed/source value | OK? |
|---|---:|---|:---:|
| E0 2D cases | 38 | A1: 38 | Yes |
| E0 2D minimum steepness | 10.9 | 10.8549475 | Yes |
| E0 2D maximum steepness | 176 | 175.644225 | Yes |
| E0 2D 1% projection median | 11 | 11 | Yes |
| E0 2D 1% projection maximum | 45 | 45 | Yes |
| E0 2D 0.1% projection median | 16 | 16 | Yes |
| E0 2D 0.1% projection maximum | 76 | 76 | Yes |
| E0 3D cases | 68 | A1: 68 | Yes |
| E0 3D 1% projection median | 9 | 9 | Yes |
| E0 3D 1% projection maximum | 22 | 22 | Yes |
| E0 3D 0.1% projection median | 13.5 | 13.5 | Yes |
| E0 3D 0.1% projection maximum | 39 | 39 | Yes |
| C1 1% projection median | 58.5 | 58.5 | Yes |
| C1 0.1% projection maximum | 149 | 149 | Yes |
| C2 maximum steepness | 953 | 953.414320 | Yes |
| C2 1% projection median | 108 | 108.5, rounded to three significant digits | Yes |
| Qualified 2D configurations | 33 | FOM `FRONT-2D.json`: 33 | Yes |
| R128 BE ROM median | 1.1% | FRONT: 1.0953394% | Yes |
| R128 BE ROM worst | 4.84% | FRONT: 4.8433537% | Yes |
| R128 BE charged-step match | 28 | Recomputed from FRONT and A2: 28 | Yes |
| R384 CNR ROM median | 0.198% | FRONT: 0.1978727% | Yes |
| R384 CNR ROM worst | 1.78% | FRONT: 1.7752691% | Yes |
| R384 CNR charged-step match | 40 | Recomputed: 40 | Yes |
| Four selected 3D charged-step matches | 14 | Recomputed against EXPORT, val64: all 14 | Yes |
| N0 ROM median | 0.192% | Pooled baseline: 0.1917264%; eligible **N0 only: 0.1186901%** | **Wrong cohort** |
| N0 ROM worst | 1.78% | Pooled baseline: 1.7752691%; eligible **N0 only: 0.6172561%** | **Wrong cohort** |
| N0 joint match | 32 | Same-cohort eligible N0 comparison: **48** | **No** |
| C1 ROM median / worst | 5.68% / 10.8% | REG: 5.679664% / 10.846115% | Yes |
| C2 ROM median / worst | 16.6% / 30.4% | REG: 16.601444% / 30.367404% | Yes |
| C2 0.1% unresolved cases | 4 | A2 regime rows: 4 | Yes |
| PS2 N24 step time | 0.161 ms | Raw FOM regression: 0.160980669 ms | Yes |
| PS2 N48 step time | 0.155 ms | Raw regression: 0.155492258 ms | Yes |
| PS2 N48 overhead | 2.09 ms | Raw regression: 2.088751723 ms | Yes |
| PS2 N256 step time | 0.272 ms | Raw regression: 0.271984268 ms | Yes |
| PS2 N48 work | 5.13e6 flop | Recomputed: 5,131,371.925 | Model matches |
| R384 evolution | 88.7 ms | P1: 88.7482511 ms | Yes |
| R384 full-field decode | 14.3 ms | P1: 14.2562841 ms | Yes |
| Borrowed Jacobian count | 2.12 | P1 iterations: \(1+\overline{n_{\rm iter}}=2.12\) | Yes |
| R384 work | 4.10e10 flop | Recomputed: 40,973,988,003.84 | Model matches |
| R384 equivalent throughput | 23.1 TFLOP/s | Recomputed: 23.0843918 | Model matches |
| F1, δ=.005, POD rank | 56 | Independent separable SVD/held-out evaluation: 56 | Yes |
| F1 square sine count | 15625 | Independent DST: \(125^2=15625\) | Yes |
| F1 anisotropic sine count | 125 | Independent DST: 125 | Yes |
| F1 anisotropic compression | 2.2 | \(125/56=2.2321429\) | Yes |
| Stability-product minimum | 1.41 | Recomputed: 1.40597209 | Arithmetic only |
| Stability-product median | 2.07 | Recomputed: 2.06717796 | Arithmetic only |
| ROM intercept | 0.403 ms | A6: 0.402751732 ms | Fit only |
| 2D sensitivity, 50 ROM steps | 828 / 540 / 1144 / 576 | Independently recomputed: identical | Model matches |
| 3D sensitivity, 50 ROM steps | 102 / 72 / 126 / 84 | Independently recomputed: identical | Model matches |
| Equal-step crossover upper endpoint | 2.2e4 | **7365.43** for equal steps; printed endpoint also includes 2.5× ROM steps | **Wrong scenario** |
| Regime-table provenance timestep | `5e-4` in VALUES | Actual: .0025 / .002 / .001 | **No** |

Additional numerical checks:

- Raw-field orthonormal DST tails reproduced six stored 2D/3D values within \(6\times10^{-15}\). The \(N\) convention correctly retains modes \(1,\ldots,N-1\).
- Direct R384 bank projection: `0.000152377092875`, versus stored `0.000152377091218`; QR orthogonality error \(1.52\times10^{-13}\).
- CPU charged-step rerun, `val32|5`, \(N=40,\Delta t=.0125\): `0.016032801763498125`, versus saved `0.01603280176349824`.
- The seven/twelve transforms per ETDRK4 stage and the stated point-Jacobian expression agree with the lane code. No new indexing, DST-normalisation, QR, or seconds/milliseconds/microseconds conversion bug was found.

## Findings

**Round-1 disposition**

| Round-1 finding | Status | Evidence |
|---|---|---|
| 1. Intercept treated as hardware bound | **FIXED** | `report.md:278–281` explicitly limits the fit and 3D anchor; sensitivity table added. |
| 2. Censoring and worst-case fitting | **PARTLY** | Counts now disclosed, but censored cases remain excluded from fitting and envelope construction (`a7_model.py:173–179`). |
| 3. Accuracy/cost timestep mismatch | **PARTLY** | In-range matching fixed and reproduced; regime crossover still pairs different step counts. |
| 4. Stability law overgeneralised | **PARTLY** | Boundedness and 3D assumption disclosed (`report.md:286–291`); exclusion argument remains incorrect. |
| 5. Synthetic necessary conditions | **PARTLY** | Anisotropic comparator works; population claims narrowed. Arbitrary rank cutoff still described as a sample-support limit (`report.md:245,262`). |
| 6. Transport theorem misstated | **PARTLY** | Incorrect exponent and universal Fourier-optimality claim removed; “decay is algebraic” remains too broad (`report.md:491`). |
| 7. Mixed-job query decomposition | **FIXED** | Separate total and component profile explicitly identified (`report.md:178–184`). |
| 8. Fit validation/extrapolation | **PARTLY** | Exact prediction model checked, in-sample label and sensitivities added. Finite-rank exponent/iteration sensitivity remains untested. |
| 9. Mechanistic/work-count wording | **FIXED** | Borrowed iteration count, dominant-term rate and footprint qualifiers added (`report.md:160,167,199,205`); LM loop agrees. |
| 10. Exponential/reference-floor claims | **PARTLY** | Main claims qualified (`report.md:80–81,111`); “spatial error only” remains at lines 24 and 117. |
| 11. Build/render/review consistency | **FIXED** | Exact regeneration matches; table delimiters and configuration separators corrected. |

1. **HIGH — The regime crossover still charges fewer steps than its accuracy measurements use.**  
   **Evidence:** `a2_ps.py:DT_REG`; `a7_model.py:155–189`; [report.md:416](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/why-spectral/report.md:416). Actual versus modelled spectral steps, using \(c_{\min}\):

   | Cohort | Accuracy-run steps | Model steps over that cohort |
   |---|---:|---:|
   | N0 | 100 | 20–21 |
   | A− | 100 | 20–36 |
   | C1 | 125 | 46–89 |
   | C2 | 250 | 96–170 |

   Thus “accuracy at those steps is measured” is false. Additionally, `pscore.py:142–144` requires steps aligned to .05 output intervals; arbitrary totals such as 89 or 197 are not directly executable by this implementation.

   **Fix:** Fit resolution using accuracy measured at the charged, executable timestep, or charge the measured timestep. Otherwise explicitly label Scenario 3’s accuracy as unvalidated, including within the sampled steepness range.

2. **HIGH — The new “worst-case envelope” excludes known unresolved cases and is not a cohort envelope.**  
   **Evidence:** `a7_model.py:173–177`; `report.md:414–415`. At 3D 1%, the envelope predicts \(N=21.10\) for `val64|5` and \(23.47\) for `val64|43`, although neither reaches tolerance at any tested resolution through \(N=32\). At 0.1%, **all 19 censored cases** receive envelope predictions below 32. The resolved 0.1% fit reaches only \(s=85.59\).

   **Fix:** Call it an envelope **of resolved observations only**, remove worst-case interpretation, and incorporate censored constraints before publishing a cohort-wide prediction. Treat resolved first-passing resolutions as bracketed observations rather than exact thresholds.

3. **MEDIUM — NEW: N0 matches different ROM and spectral cohorts.**  
   **Evidence:** `values.py:204–215`; `report.md:148,153`; REG `checks/analysis.json`. The fallback uses a **52-case pooled ROM baseline**, while spectral statistics use all **16 N0 cases**. The footnote discloses pooling but does not make the comparison valid. On the same **14 eligible N0 cases**, ROM median/worst are **0.118690% / 0.617256%**, and \(N_{\rm joint}=48\), not 32.

   **Fix:** Compute both methods on identical case IDs and eligibility masks. Keep the pooled baseline in a separate descriptive row if useful.

4. **MEDIUM — The claimed direction of stability-censoring bias is unsupported.**  
   **Evidence:** `report.md:290`; `a7_model.py:165–169`. The three excluded cases imply only
   \[
   c<2.8824,\quad c<2.9794,\quad c<2.6116,
   \]
   under the threshold interpretation. These upper limits exceed the chosen \(c_{\min}=1.406\); they do **not** establish that excluded cases have smaller \(c\) than the model uses. Largest successful tested steps are themselves lower endpoints of brackets, not measured exact stability limits.

   **Fix:** Report the brackets and say the excluded limits are unresolved. Remove “leaving them out makes the spectral solver look cheaper” unless additional measurements establish it.

5. **MEDIUM — NEW: Summary numbers pool incompatible scenarios.**  
   **Evidence:** `values.py:441–443` pools `equal` and `ratio2.5`, but `report.md:461` describes equal steps alone. Equal-step free-unknown crossings span **815.49–7365.43**, not approximately 820–22000. Separately, `values.py:406–409` obtains the quoted 3D compression **1.5e3** from **N=56, 15 ROM steps**; `report.md:453–456` attaches it to resolutions discussed for **20 or 50 ROM steps**.

   **Fix:** Generate each summary from precisely the scenario named in its sentence. Print \(N,n_{PS},n_{ROM}\) beside compression thresholds.

6. **MEDIUM — The “every scanned scenario” conclusion contradicts the tables.**  
   **Evidence:** `report.md:45–49,467–470`. Equal-step, free-unknown scenarios explicitly cross over (`report.md:424,427,430,433,436,439,442,445`), contradicting the asserted requirement that ROM steps stay fixed while spectral steps grow. Every finite-rank prediction also retains cubic ROM work; the scans do not establish a requirement to change that scaling.

   **Fix:** State that fixed ROM steps substantially improve the predicted crossover, and that **none of the tested finite-compression equal-step scenarios** crosses within the scan. Do not promote selected assumptions to properties of every crossing.

7. **LOW — NEW: Provenance still records an obsolete regime timestep.**  
   **Evidence:** `values.py:215`; `VALUES.json:1152`. `tab_reg` claims `dt 5e-4`, despite the completed per-regime runs using .0025/.002/.001.

   **Fix:** Derive this source description from `a2_reg.json["dt"]`, including the pooled-baseline exception if retained.

8. **MEDIUM — Synthetic censoring is still presented as a mathematical sample-size limit.**  
   **Evidence:** `a5_synthetic.py:68–73`; `report.md:245,262`. The code computes an 800-vector SVD and deliberately suppresses ranks above 400. Eight hundred snapshots do not mathematically prohibit ranks 401–800, nor does this experiment establish that those ranks cannot generalise.

   **Fix:** Say “ranks above 400 were excluded by our chosen reliability cutoff.” Preserve the valid anisotropic result, but require sample-size convergence before interpreting censoring as a limitation of the solution family.

9. **MEDIUM — The revised transport citation still overstates algebraic decay.**  
   **Evidence:** `report.md:491`. Arbes–Greif–Urban also treat superalgebraic decay and prove exponential decay under analytic-data assumptions; their result is not uniformly algebraic. See [§5.3, especially Proposition 5.11](https://arxiv.org/html/2305.00066v3).

   **Fix:** Replace “the decay is algebraic” with “the decay depends on data regularity; specific nonsmooth examples have algebraic rates.” Name the half-wave-symmetry assumptions for the exact construction.

## Overclaims

- **“Spatial error only”** (`report.md:24,117`): replace with “small-step error.” Time refinement at selected resolutions does not remove temporal error identically.
- **“A spectral basis is near-optimal”** (`report.md:44`): supported relative to the tested learned bank and sampled families, not among all approximation spaces for these Burgers manifolds.
- **“\(aT\)…is not varied here”** (`report.md:291,463`): amplitude varies across cases, so \(aT\) varies. Say it was **not independently controlled or isolated**.
- **“None of these reports wall-clock times against an efficient full solver”** (`report.md:505`): an exclusion claim covering several abstract-only checks. Narrow it to what the reviewed text establishes and define “efficient.”
- The McGreivy–Hakim figures concern the selected **fluid-related ML-for-PDE papers claiming superiority**, not projection ROMs generally. The revised heading still needs that population qualifier. [Primary study](https://arxiv.org/html/2407.07218v1).
- Dresdner’s **2–4× accuracy at approximately 2× runtime, at the same resolution**, is correctly qualified and supported. [Primary abstract](https://arxiv.org/abs/2207.00556).