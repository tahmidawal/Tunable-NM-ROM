## Verdict

**NEEDS-FIX**

The measured spectral advantage survives. Most round-2 corrections are sound, including the same-case regime comparison and aligned \(c_{\min}\) step counts. Two problems remain: the \(c_{\mathrm{med}}\) sensitivity reuses incompatible accuracy data—and a CPU check actually diverges—and the opening still overstates the compression required.

No files were changed; no GPU or cluster jobs were launched.

## Numbers checked

All **293 entries** in `VALUES.json`, including source strings, reproduce from `values.py`. In-memory template expansion exactly reproduces `report.md`.

Abbreviations: A1/A6/A7 = this report’s analysis JSONs; REG = regimes lane `checks/analysis.json`; FOM = fom2d `checks/match-final.json`; P1 = scaling lane `checks/analysis.json`. Values below were independently aggregated/recomputed unless marked “model/source.”

| Claim | Report value | Recomputed/source value | OK? |
|---|---:|---:|:---:|
| E0 2D cases | 38 | A1: 38 | Yes |
| E0 2D minimum \(s\) | 10.9 | 10.8549475 | Yes |
| E0 2D maximum \(s\) | 176 | 175.644225 | Yes |
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
| C2 maximum \(s\) | 953 | 953.414320 | Yes |
| C2 1% projection maximum | 175 | 175 | Yes |
| N0 eligible/valid cases | 14 | REG: 14 | Yes |
| N0 ROM median | 0.119% | REG: 0.118690110% | Yes |
| N0 ROM worst | 0.617% | REG: 0.617256052% | Yes |
| N0 joint spectral match | 48 | Same-case recomputation: 48 | Yes |
| A− ROM median | 0.883% | REG: 0.882835978% | Yes |
| A− ROM worst | 3.44% | REG: 3.444960146% | Yes |
| C1 ROM median | 5.68% | REG: 5.679664040% | Yes |
| C1 ROM worst | 10.8% | REG: 10.846115052% | Yes |
| C2 ROM median | 16.6% | REG: 16.601443881% | Yes |
| C2 ROM worst | 30.4% | REG: 30.367404492% | Yes |
| PS2 N24 step time | 0.161 ms | FOM regression: 0.160980669 ms | Yes |
| PS2 N48 step time | 0.155 ms | FOM regression: 0.155492258 ms | Yes |
| PS2 N48 overhead | 2.09 ms | FOM regression: 2.088751723 ms | Yes |
| PS2 N256 step time | 0.272 ms | FOM regression: 0.271984268 ms | Yes |
| R384 evolution | 88.7 ms | P1: 88.748251088 ms | Yes |
| R384 full-field decode | 14.3 ms | P1: 14.256284107 ms | Yes |
| Borrowed Jacobian count | 2.12 | P1: \(1+\operatorname{mean}(\mathrm{iterations})=2.12\) | Yes |
| R384 work/step | \(4.10\times10^{10}\) | \(40,973,988,003.84\) flop | Model matches |
| R384 equivalent throughput | 23.1 TFLOP/s | 23.084391806 | Model matches |
| Stability-product minimum | 1.41 | 1.405972091 | Yes |
| Stability-product median | 2.07 | 2.067177962 | Yes |
| Charged-step 2D 1% fit count | 102 | 102 resolved, zero censored | Yes |
| Charged-step 2D 1% exponent | 0.73 | 0.726765167 | Yes |
| Charged-step 2D 0.1% exponent | 0.75 | 0.747354148 | Yes |
| Charged-step 2D 0.1% censoring | 4 | C2 cases 4, 7, 9, 12 | Yes |
| Equal-step free-unknown crossing range | \(1.0\times10^3–7.4\times10^3\) | A7: 1041.39986–7365.43298 | Model/source |
| \(c_{\mathrm{med}}\) accuracy measured at charged steps | Implied by Scenario 3 | **48/102** 2D cases use different counts | **No** |

Additional direct checks:

- Raw-field DST tails reproduced stored values: 2D `val32|5`, \(N=40\): **0.014301441185143458**; 3D `val64|5`, \(N=16\): **0.018697644056341587**. Both agree within \(2\times10^{-16}\).
- Direct R384 bank projection, `val32|0`: **0.000409512108343334**, versus stored **0.000409512106016416**. QR orthogonality error: **\(1.52\times10^{-13}\)**.
- New charged-step run, `C1|0`, \(N=64\), 60 steps: **0.007151038829073026**, exactly reproducing `a2_stab.json`.
- No new tail-index, DST-normalisation, QR-prefix, held-out projection-index, or unit-conversion bug found. The Jacobian expression and LM assembly/solve description agree with `p1comp.py:133–145` and `p1core.py:73–107`.

## Findings

**Round-2 disposition**

Paths below are relative to `reports/2026-10-20-spectral-investigation/why-spectral/`.

| Round-2 finding | Status | Evidence |
|---|---|---|
| 1. Accuracy/cost step mismatch | **PARTLY** | All 102 `a2_stab` counts match the executable \(c_{\min}\) law. The \(c_{\mathrm{med}}\) branch still reuses those errors: see finding 1 below. |
| 2. Envelope excludes censored cases | **FIXED** | `report.md:412–416` explicitly limits the envelope to resolved cases and rejects cohort-wide worst-case interpretation. Censored cases remain unresolved, now disclosed. |
| 3. N0 cohort mismatch | **FIXED** | `values.py:195–215`; `report.md:145–153`. Both methods use identical eligible/valid IDs; 14 cases, \(N_{\mathrm{joint}}=48\). |
| 4. Unsupported stability-bias direction | **FIXED** | `report.md:290–292` gives upper limits and explicitly says their position relative to 1.41 is unknown. |
| 5. Incompatible summary scenarios | **FIXED** | `values.py:411–450`; `report.md:458,463`. Equal-step aggregation is isolated; compression examples now name \(N\) and both step counts. |
| 6. Universal fixed-step requirement | **FIXED** | `report.md:469–473` acknowledges equal-step free-unknown crossings and retains cubic work. |
| 7. Obsolete timestep provenance | **FIXED** | `VALUES.json`’s `tab_reg` source contains the actual per-regime timestep dictionary and eligibility restriction. |
| 8. Synthetic cutoff presented as mathematical limit | **FIXED** | `report.md:245,262` identifies the chosen rank cutoff and requests sample-size convergence. A stale code docstring remains; see Overclaims. |
| 9. Uniform algebraic transport decay | **FIXED** | `report.md:492–493` states symmetry assumptions and distinguishes nonsmooth algebraic examples from analytic exponential decay, consistent with [Arbes–Greif–Urban](https://arxiv.org/html/2305.00066v3). |

1. **MEDIUM — The median-stability sensitivity retains an accuracy/cost mismatch, with an observed divergent case.**

   **Evidence:** `a2_ps.py:128–147` generates accuracy data exclusively with \(c_{\min}\). `a7_model.py:157–185` fits that data once, then substitutes both \(c_{\min}\) and \(c_{\mathrm{med}}\) into the charged step count. Nevertheless, [report.md:416–418](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/why-spectral/report.md:416) says accuracy at those steps is measured and reports the median-\(c\) crossover range.

   The mismatch affects **48/102 cases**. Concrete CPU check: `C2|9`, \(N=256\):
   - \(c_{\min}\): **170 steps**, saved error **0.800454597%**.
   - \(c_{\mathrm{med}}\): **120 steps**, \(\Delta t=0.0020833333333333333\), **non-finite output**.

   **Fix:** Remove the median-\(c\) accuracy-associated crossover range, or label it explicitly as a cost-only sensitivity with unvalidated—and demonstrably failing for one sampled case—accuracy/stability. A valid alternative requires separate accuracy runs and fits for that step law. The main \(c_{\min}\) table is unaffected.

2. **MEDIUM — The opening still asserts a compression requirement contradicted by its own table.**

   **Evidence:** [report.md:41](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-20-spectral-investigation/why-spectral/report.md:41) says the \(s\approx230–1100\) crossover occurs “only if” \(\rho\gtrsim1000\). But `report.md:425,437,440` contains finite-\(\rho=100\) crossings within that range: approximately **510**, **310**, and **450**, respectively. Section 4 correctly acknowledges these exceptions at lines 462 and 471.

   **Fix:** Replace the opening requirement with: “The free-unknown and \(\rho=1000\) scenarios cross around \(s=230–1100\); some \(\rho=100\) scenarios also cross.” Preserve the fixed-step and extrapolation qualifications.

The remaining cost-model limitations are now substantially disclosed: fitted—not physical—intercepts, fixed ROM steps, frozen overhead beyond calibration, borrowed iteration counts, and the single-anchor 3D cubic model. They support conditional scenarios, not a validated regime boundary.

## Overclaims

- **`report.md:8,41`:** “stability limit” remains too strong in the summary. Use “sampled boundedness brackets” and “empirical step-law scenario,” matching the careful explanation in section 4.
- **`report.md:49`:** “a large \(r\) never helps” conflates runtime and accuracy. Say “at fixed \(N\), step counts, and model constants, increasing \(r\) increases runtime.”
- **`a5_synthetic.py:15–16`:** the docstring still calls censoring “the Kolmogorov barrier showing up as a sample requirement.” Replace it with the report’s corrected, discretionary-cutoff wording.
- **`report.md:491`:** qualify exponential decay as applying to **certain parameterized coercive problems**, rather than coercivity alone. The [Greif–Urban abstract](https://arxiv.org/abs/1903.08488) uses the narrower scope.
- The revised McGreivy–Hakim population qualifier and Dresdner same-resolution comparison are supported by their [primary study](https://arxiv.org/html/2407.07218v1) and [abstract](https://arxiv.org/abs/2207.00556), respectively.