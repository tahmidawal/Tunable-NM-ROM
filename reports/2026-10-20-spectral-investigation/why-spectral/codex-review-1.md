## Verdict

**NEEDS-FIX**

The measured result—these deployed ROMs lose to coarse spectral solvers—is supported. Most numerical transcription is sound. The proposed crossover and universal “cannot win” conclusions are not established: they combine censored accuracy fits, unvalidated timestep choices, and extrapolated runtime assumptions.

I made no file changes or job submissions. The `PENDING` regime table is excluded from this verdict.

## Numbers checked

Paths below are relative to `reports/2026-10-20-spectral-investigation/why-spectral/`. A1–A7 denote the corresponding analysis JSONs. FOM and P1 denote the named lane files.

All **264 used entries** in `VALUES.json` reproduce from `values.py`, including their source strings. I additionally checked underlying lane JSONs, recomputed raw-field DST tails in 2D and 3D, recomputed a bank projection, and reran one spectral solve on CPU.

| Claim | Report value | Recomputed/source value | OK? |
|---|---:|---|:---:|
| E0 2D cases | 38 | A1: 38 | Yes |
| E0 2D minimum \(s\) | 10.9 | A1: 10.85495 | Yes |
| E0 2D maximum \(s\) | 176 | A1: 175.64422 | Yes |
| E0 2D 1% projection median | 11 | A1: 11 | Yes |
| E0 2D 1% projection maximum | 45 | A1: 45 | Yes |
| E0 2D 0.1% projection median | 16 | A1: 16 | Yes |
| E0 2D 0.1% projection maximum | 76 | A1: 76 | Yes |
| E0 3D cases | 68 | A1: 68 | Yes |
| E0 3D 1% projection median | 9 | A1: 9 | Yes |
| E0 3D 1% projection maximum | 22 | A1: 22 | Yes |
| E0 3D 0.1% projection median | 13.5 | A1: 13.5 | Yes |
| E0 3D 0.1% projection maximum | 39 | A1: 39 | Yes |
| C1 1% projection median | 58.5 | A1: 58.5 | Yes |
| C1 0.1% projection maximum | 149 | A1: 149 | Yes |
| C2 maximum \(s\) | 953 | A1: 953.41432 | Yes |
| C2 1% projection maximum | 175 | A1: 175 | Yes |
| PS2 \(N=16\), median | 0.0938% | A2: 0.09377186% | Yes |
| PS2 \(N=24\), worst | 6.4% | A2: 6.400139% | Yes |
| PS2 \(N=48\), median | \(5.45\times10^{-4}\)% | A2: 0.000545163% | Yes |
| PS2 \(N=128\), worst | 0.00251% | A2: 0.002505044% | Yes |
| PS3 \(N=8\), median | 1.65% | A2, **val64**: 1.654447% | Yes |
| PS3 \(N=14\), worst | 3.7% | A2, **val64**: 3.698233% | Yes |
| Qualified 2D configurations | 33 | FOM `inputs/FRONT-2D.json`: 33 | Yes |
| R128 BE median / worst | 1.1% / 4.84% | FRONT: 1.095339% / 4.843354% | Yes |
| R384 CNR median / worst | 0.198% / 1.78% | FRONT: 0.197873% / 1.775269% | Yes |
| E0 R384 bank floor | 0.0282% | A3: 0.02824486% | Yes |
| E0 sine/bank ratio | 0.63 | A3: 0.6328125 | Yes |
| C1 R384 bank floor | 3.47% | A3: 3.470544% | Yes |
| C2 R384 bank floor | 9.88% | A3: 9.881812% | Yes |
| PS2 \(N=24\), step time | 0.161 ms | Raw FOM regression: 0.16098067 ms | Yes |
| PS2 \(N=48\), step time | 0.155 ms | Raw FOM regression: 0.15549226 ms | Yes |
| PS2 \(N=48\), overhead | 2.09 ms | Raw FOM regression: 2.08875172 ms | Yes |
| PS2 \(N=256\), step time | 0.272 ms | Raw FOM regression: 0.27198427 ms | Yes |
| R384 evolution | 88.7 ms | P1, **1024 mesh**: 88.748251 ms | Yes |
| R384 full-field decode | 14.3 ms | P1, **4096 mesh**: 14.256284 ms | Yes |
| R384 full query | 97.1 ms | FOM stage 2: 97.073751 ms | Yes |
| Its matched spectral query | 11.7 ms | Same FOM comparison: 11.723279 ms | Yes |
| Its speed ratio | 0.121 | Same comparison: 0.12069085 | Yes |
| Claimed query decomposition | 97.1 ms | Listed components sum to **103.201837 ms** | **No** |
| Stability-product minimum | 1.41 | A2b selected cases: 1.40597209 | Arithmetic only |
| Stability-product median | 2.1 | A2b selected cases: 2.06717796 | Arithmetic only |
| ROM fitted intercept | 0.403 ms | A6: 0.40275173 ms | Fit only |
| 2D free-unknown crossover | 828–876 | A7: 828 / 876 | Model only |
| 3D free-unknown crossover | 100–102 | A7: 102 / 100 | Model only |

Raw checks found no DST normalization or tail-index error: \(N\) consistently retains modes \(1,\ldots,N-1\). The direct R384 bank residual was \(0.000152377092875\), versus stored \(0.000152377091218\); QR orthogonality error was \(1.52\times10^{-13}\). The CPU spectral rerun reproduced its saved error within \(2\times10^{-16}\).

## Findings

1. **HIGH — A fitted intercept is presented as a hardware lower bound.**  
   **Evidence:** `a6_cost.py:148–167`; `a7_model.py:55–59,99`; `report.md:311–313,415–418,442–447`. The ROM fit uses six observations at only three widths, \(r=128,256,384\). Extrapolating its intercept to zero work does not prove that changing quadrature, bank, kernels, or algorithm cannot lower overhead. The 3D model additionally imports this **2D** intercept and assumes cubic growth from one QN configuration.  
   **Fix:** Call this a conditional runtime scenario for the present implementation. Replace “no ROM of this kind can be faster” and class-wide impossibility claims with explicitly conditional statements. Measure small-width behavior and multiple 3D widths before treating the intercept or exponent as established.

2. **HIGH — The steepness fit silently drops hard cases and does not enforce worst-case accuracy.**  
   **Evidence:** `a7_model.py:146–173`; `report.md:390–410`. At 3D 0.1%, `firstN` fails for `val64|5` and `val64|43`; both disappear from the fit. It uses **66/68 cases**, with maximum included \(s=131.865\), not the full range to 180. Ordinary log regression also underpredicts individual resolution requirements: at 2D `val32|5`, the 1% fit predicts \(N=28.83\), whereas the tested requirement is \(N=48\).  
   **Fix:** Treat unresolved thresholds as right-censored and resolved thresholds as intervals between tested resolutions. Publish exclusions and residuals. Use a conservative prediction envelope for worst-case crossover claims, or label the result a typical-case scenario.

3. **HIGH — Accuracy and cost use different timestep configurations.**  
   **Evidence:** `a4_match.py:8–10`; `a7_model.py:84–92,168–183`; `report.md:99,282,307`. Matching uses spectral errors at \(\Delta t=0.0025\) in 2D and 0.01 in 3D. Costs then assume 20 or 10 steps, respectively. A stability threshold does not guarantee the requested error. In FOM’s raw \(N=128\) rows, worst error changes from **0.002505%** at 0.0025 to **0.156842%** at 0.0125: the latter fails a 0.1% target despite being admissible.  
   **Fix:** Determine \(N,\Delta t\), error, and cost jointly for each tolerance. Until then, call these spatial-resolution matches and hypothetical costs—not matched-accuracy speed predictions. Replace “spatial error only” with a qualified time-refinement statement.

4. **HIGH — The stability law is neither a universal bound nor a validated 3D law.**  
   **Evidence:** `a2b_stab.py:20–42`; `a7_model.py:160–181`; `report.md:283–286,439`. Classification checks finiteness and an amplitude threshold at three \(N\) values and three timesteps. Three cases with no successful timestep are excluded from \(c\); their limits remain unknown. The same \(c\) is then applied to 3D without a 3D stability experiment. Taking the minimum also applies the most restrictive sampled product to every hypothetical case, increasing predicted spectral cost.  
   **Fix:** Report timestep brackets and failures, distinguish boundedness from accuracy, and expose sensitivity to \(c\). Restrict the empirical law to tested 2D cohorts. At fixed physical horizon, \(s=a/\nu\) alone is insufficient: rescaling time by amplitude introduces \(aT\) as another parameter.

5. **HIGH — The claimed synthetic “necessary conditions” do not follow from the experiment.**  
   **Evidence:** `a5_synthetic.py:65–86`; `report.md:240–265,425–426`. Training/test generation is separate and the projection indexing is correct. However, empirical POD minimizes **training absolute mean-square error**, not population worst relative error. Ranks above 400 are deliberately discarded, so censored F2/F3 rows cannot establish \(\rho\approx2\), impossibility of compression, or a required nonlinear decoder. F1 is also compared against isotropic square sine blocks despite its separable direction.  
   **Fix:** Limit conclusions to the sampled families and tested ranks. Report censoring without inferring missing compression ratios; add sample-size convergence and an anisotropic spectral comparator before stating general principles.

6. **HIGH — The transport citation states the wrong theorem.**  
   **Evidence:** `report.md:466–469`, also `261`. The claim \(d_N\sim N^{-(r+1/2)}\) for arbitrary \(H^r\) data is incorrect. Arbes–Greif–Urban distinguish general regularity bounds from exact rates for specific piecewise-polynomial data. Their exact sorted-trigonometric construction has symmetry/shift assumptions; it does not prove the report’s universal statement about transport or Burgers.  
   **Fix:** State the theorem’s assumptions and distinguish upper bounds from asymptotic equivalence. Remove the unrestricted “no linear basis beats Fourier by more than a constant.” [Primary paper, §§4–5](https://arxiv.org/html/2305.00066v3).

7. **MEDIUM — Query decomposition mixes measurements and meshes.**  
   **Evidence:** `report.md:176`; `values.py`, keys `q4_R384_rom`, `rom_init_R384`, `rom_ev_R384`, `rom_dec4k_R384`; `a6_cost.py:97–128`. The 97.073751 ms total comes from FOM stage 2; initialization/evolution come from P1 at 1024, decode from P1 at 4096. Even P1’s consistently 4096 components sum to **102.332128 ms**, not 97.1 ms.  
   **Fix:** Present the independently measured query total and P1 component profile separately. Do not join them with an equality or “takes …: init + evolution + decode.”

8. **MEDIUM — Fit checks understate extrapolation uncertainty and partly check a different model.**  
   **Evidence:** `a6_cost.py:148–167`; `a7_model.py:46–75`; `report.md:274–278,288–307`. Query checks reuse fitting observations. The 3D anchor check is almost guaranteed by construction. The ROM check uses each configuration’s actual \(m/r\), while crossover calculations fix \(m/r=42.67\): R128’s reported model check is 22.453 ms, whereas `t_rom(128,2,50)` gives **23.049 ms**. Two throughput fits do not bound extrapolation uncertainty. Spectral overhead is frozen beyond measured \(N\), despite growing setup and synthesis work.  
   **Fix:** Check the exact model used for prediction; label calibration checks as in-sample. Add sensitivity to overhead growth, ROM exponent/intercept, and iteration counts. Describe the two fits as scenarios, not bounds.

9. **MEDIUM — Work counts are reasonable dominant terms, but mechanistic claims exceed them.**  
   **Evidence:** `a6_cost.py:99–114`; `report.md:156,192,196,202`; P1 `p1core.py:73–107` and `p1comp.py:133–145`. The Jacobian expression and seven/twelve spectral transforms per stage agree with code. However, \(n_J=2.12\) comes from **one R384 CN-R case**, then is assigned to every width/scheme. LM performs one initial Jacobian plus one per iteration, and one Cholesky per iteration—not exactly one solve per step. \(W_R\) omits normal-matrix formation, solves, predictors, and other work. Table bytes are not measured device-memory traffic.  
   **Fix:** Label \(n_J\) as a borrowed representative count; obtain configuration/workload counts where available. Describe throughput as dominant-term-equivalent throughput and table bytes as footprint/read-volume estimates.

10. **MEDIUM — Reference-floor and exponential-convergence claims are too categorical.**  
    **Evidence:** `report.md:80–81,99–111`; `a1_spectra.py:58–65`; E0 `DESIGN.md:845–868`. Fitting a finite tail interval supports observed exponential-like decay, not an asymptotic proof. E0 A10 reports approximately \(10^{-9}\) **self-differences**, not a fitted convergence constant. Its conclusions concern particular cases. The small-error plateau has not been decomposed here into reference, temporal, and spectral contributions.  
    **Fix:** Say “approximately exponential over the fitted range” and “consistent with a reference-limited plateau.” Correct the A10 quantity and scope.

11. **LOW — Generated report and source are out of sync; some tables will render incorrectly.**  
    **Evidence:** `report.md:245,300–302,373–380`; `report_template.py`; `review_record.txt`. In-memory regeneration differs by a stability-censoring paragraph after line 286. The synthetic table has six columns but only two delimiter cells; configuration names contain unescaped `|`. The opening says “reviewed,” while §6 says review pending.  
    **Fix:** Regenerate after substantive corrections, use matching table delimiters and escaped pipes, and make review status consistent. The pending regime rerun is not itself a defect.

## Overclaims

- **“A projection ROM … cannot beat … anywhere … \(s\le953\)”** (`report.md:442`): unsupported class-wide conclusion. Supported replacement: *The tested deployed configurations lose in range; the steep configurations fail qualification.*
- **“A crossover needs all of…”** (`443–447`): these are assumptions of selected scenarios, not necessary conditions.
- **“Its per-unknown cost does not grow like \(r^3\)”** (`47`): \(r^3\) is the model’s **total** Jacobian cost; per-unknown cost scales as \(r^2\) under fixed ratios.
- **“Viscosity enters only through \(s\)”** (`439`): neither regression nor the finite-time PDE establishes this.
- **“ROM speed-ups are usually measured against weak baselines”** (`453`): McGreivy–Hakim studied a selected **ML-for-fluid-PDE** literature, not projection ROMs generally. Their 60/76 and 7/10 figures are supported; the heading broadens the population. [Primary paper](https://arxiv.org/html/2407.07218v1).
- **Dresdner’s 2–4× accuracy and approximately 2× runtime** are supported, but explicitly **at the same resolution**; preserve that qualifier. [Primary abstract](https://arxiv.org/abs/2207.00556).