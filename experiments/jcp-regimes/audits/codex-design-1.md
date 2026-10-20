**Overall verdict: NEEDS-RESTATEMENT.** The distributions are feasible and the frozen 2D configurations match, but the design overclaims factor isolation and understates the changes needed to reuse the reference and qualification code.

Read-only audit completed: no files changed, jobs submitted, cohorts opened, or scratch arrays copied.

Evidence below uses these path abbreviations:

- **D**: [regimes DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/DESIGN.md)
- **E0**: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/`
- **Q**: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/`
- **SD**: current worktree’s `experiments/separable-decoder/`
- **Briefs**: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/`

1. **Regime distributions and one-factor attribution — WRONG overall; amplitude feasibility CORRECT.**

   Evaluating the conditional amplitude intervals at both viscosity endpoints gives:

   | Regime | At lower ν endpoint | At upper ν endpoint |
   |---|---:|---:|
   | A+ | [0.5, 2] | [1.99054, 2] |
   | A− | [0.5, 0.63246] | [0.5, 1.41589] |
   | C1 | [0.79057, 1.58114] | [1.76986, 2] |
   | C2 | [0.79245, 1.58489] | [1.77407, 2] |

   All are nonempty and inside `[0.5,2]`; the functions are monotone between endpoints. The quoted global steepness supports are also feasible. **Evidence:** D:44–55.

   However:

   - **C1 already moves both ν and s outside their training supports**, not just C2. The “declared factor only” assertion cannot pass literally. **D:40–42,49,54–57,87–89.**
   - A± impose ν-dependent amplitude laws. Comparing them with N0 changes the joint amplitude–viscosity distribution, even though amplitude stays inside its marginal support. “Inside the box” is not “held constant.”
   - A− and C1 share the same *distribution* of ν, widths and centers, but use independent seeds. This is a defensible distributional amplitude intervention at low viscosity; it does not isolate a pure steepness mechanism. Amplitude also changes transport speed, front-formation time and boundary interaction.
   - \(s=a/\nu\) is defensible as an inverse viscous-front-width proxy after a front forms. It is not the initial gradient, a guarantee of a front by \(T=0.25\), or a complete numerical stiffness measure. The design acknowledges part of this at D:35, but its attribution language remains stronger.

   Passing the proposed σ checks cannot remove these structural confounders.

2. **Training distribution and seeds — WRONG as a bank-training provenance statement; ranges CORRECT.**

   The stated Gaussian ranges match `engines.params_draw` and `burgers2d_film.sample_params`. **Evidence:** `experiments/mr-burgers2d/engines.py:20–25`; `experiments/wave2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py:160–168`.

   But D:20–21 conflates **bank training** with the later **head refit**:

   - The original R512 bank uses the canonical draw; the training driver calls `sample_params(seed=bc.SEED)` and defaults to the full canonical count. **SD/sep_burgers_r3.py:203,293–304.**
   - dn256b appends 4,032 seed-1000 trajectories during coefficient extraction. **SD/cluster/run_dn256b.sbatch:41–47; SD/sep_coeff_extract.py:198–208.**
   - That extraction uses a **frozen bank**, and the emitted checkpoint replaces head parameters while retaining the other checkpoint parameters. **SD/sep_coeff_extract.py:154–158; SD/sep_hfit_run.py:386–408.**

   Thus “the bank was fitted on seeds 0 and 1000” is inaccurate. Distinguish the canonical seed-0 bank training, dense head-refit exposure, and rotation provenance. Checking both descriptor sets for overlap remains conservative.

3. **Seeds, disjointness and sealed/OOD protection — NEEDS-RESTATEMENT.**

   The proposed public seed integers exceed \(2^{31}\), while E0’s secret seeds are generated with `secrets.randbits(31)`. They also differ from the registered historical/public seeds. This is a sound way to avoid **seed-integer equality without reading custody data**. **Evidence:** D:77–95; E0/cohorts.py:35–44; E0/make_seeds.py:47.

   The **three-word argument is not itself a proof of RNG-stream disjointness**. SeedSequence input length does not establish disjoint namespaces; padding/equivalent entropy representations and finite-state hashing preclude that general claim. Here, the explicit namespace word and high-valued second word give a sensible domain-separated construction relative to the registered inputs, with negligible collision risk—not a mathematical guarantee of distinct sampled cases.

   Additional ambiguities:

   - D:89–90 requires checking N0 against test64 descriptors, while D:95 says historical test64 is not used. Distinguish descriptor-only duplicate checking from reading test outcomes.
   - Duplicate checks should cover all generated regimes where overlap is possible, and within-regime duplicates.
   - The no-sealed/OOD access policy is correct. Keep adapters on an explicit dev/validation/regime allowlist; do not enumerate or instantiate E0’s `NEW` cohorts.

4. **Reference validity, refinement, calibration and σ — WRONG acceptance description; otherwise NEEDS-RESTATEMENT.**

   **Low ν does not automatically violate the 2D Péclet gate.** Using the nominal amplitude bound:

   \[
   Pe_{2048}\lesssim s/2048
   \]

   gives maxima approximately 0.098 for A−, 0.244 for C1 and 0.488 for C2—all below 2. The actual gate uses the **running numerical maximum**, so it must still be evaluated. **Evidence:** E0/verdict.py:106–109. The design should not predict Péclet-driven refinement solely from these ν bands.

   The base/refinement levels and non-dyadic order calculation are supported by E0. **E0/refjob.py:40–49,137–153; E0/verdict.py:42–60.** However, D:116–119 states the wrong floor criterion and omits direction consistency. Actual acceptance requires:

   - both adjacent differences to exceed residual-derived floor margins at every output;
   - direction-consistency residual ≤0.5;
   - shared initial field, solver validity, order and Péclet checks.

   It is **not** simply \(d_{23}/n_0\ge10^{-10}\). **E0/verdict.py:15–20,107–126.**

   Reusing the dev6 floor calibration is a declared extrapolation, not validation of that proxy on signed/steep/new regimes. Add representative production/tight/tighter calibration probes or explicitly limit reference confidence when calibration transfer is unverified.

   There are also concrete adapter omissions:

   - `refcore.ic_lattice` bypasses `coh.ic` and reconstructs the original family from `cohort.D`. **E0/refcore.py:239–245.**
   - The spectral path calls the single-Gaussian coefficient initializer directly. **E0/refjob.py:422–423; E0/pscore.py:193–198.**

   Consequently, the promised “two edits only” cannot deliver correct B1/B2 references and spectral cross-checks as currently specified.

   σ is implementable, but needs precise grid-index, normalization and missing-data rules. “Finest accepted level” is undefined for rejected references; the separation rules nevertheless demand counts out of 16. If N0 has too few accepted cases, its range/90th percentile is also unreliable or undefined. **D:125–149.** A fixed interior box reduces boundary contamination but does not guarantee its absence, and five output times can miss transient fronts. Call σ an **output-time interior normalized-gradient diagnostic**, not a complete stiffness certificate.

5. **Frozen ROM configurations and qacc reuse — CORRECT configurations; WRONG “case-provider-only” reuse.**

   I verified the complete FRONT-2D hash:

   `a51ac1a250f00d4b0006a84f2434ab3c96407b1f575f4ffff7a3eb25b7a8b77d`

   All eight LSPG entries and R384 GAL-CN at factor 1 exist and are qualified at 1024. Main rules match D:158–161. **Evidence:** Q/runs/FRONT-2D.json:23599,27630,28286,28963,33655,34309,36309,37636,39636. Successor/check ladders match **Q/configs/acc1024.json:17–18,75–76,133–134,190–192**.

   But:

   - `p1common.select` builds its lookup only for dev6/val32; new regime keys fail. **Q/p1common.py:91–105.**
   - qacc hard-codes dev6 as the tighter/check subset. Replacing `case_list` does not implement regime cases 0–3. **Q/qacc.py:150–152,204–217.**
   - IC construction assumes a five-component Gaussian descriptor. **Q/qacc.py:114–116.**
   - Cohort hashes and reference routing also need generalization. **Q/p1common.py:87–88; Q/qscore.py:57–75.**

   Reuse is sound after explicit adapters for selection, IC construction, normalization, hashing and reference identity. Preserve the existing nodal-input/cold-quadrature sampling contract for new families.

6. **Q1/Q2 per regime — NEEDS-RESTATEMENT.**

   The LSPG thresholds and pooled-step interpretation match COMMON-BRIEF §3. Q2 on every regime case is stronger than the original subset policy. **D:183–188; Briefs/COMMON-BRIEF.md:49–60.**

   Two corrections are necessary:

   - “Verbatim” is false for GAL: qualification uses \(\eta\le10^{-8}\), not literal \(\gamma\le10^{-3}\). **Q/qscore.py:29,35–38.**
   - Explicitly carry tight/tighter validity, successor/check validity, independent recomputation and disagreement rejection—not merely distance thresholds. **Q/qscore.py:175–195.**

   Qualification must include **all sampled cases**, even when references fail. D:195’s “over eligible cases” must not scope the subsequent validity/qualification rules.

7. **Statistical decisions and post-hoc freedom — NEEDS-RESTATEMENT.**

   Case-level bootstrap, 10,000 replicates, fixed seed and predeclared factor-2 bands are sensible. **D:200–214.** Factor 2 is a chosen practical margin, not an externally justified accuracy standard.

   Remaining issues:

   - The pooled baseline mixes fresh N0 with previously exposed development/validation cases. Its ratio estimates degradation against that **specific mixture**, not a fresh training-distribution population. **D:182.**
   - For N0 itself, numerator and denominator overlap; independent resampling is wrong unless N0-versus-I is explicitly excluded.
   - Maxima from 16 versus up to 54 cases have unequal sample-size effects. Keep the worst ratio descriptive and avoid interpreting it as comparable tail risk.
   - Reference acceptance can preferentially remove difficult cases. “HOLDS” must explicitly mean **holds on reference-eligible cases**, with coverage and failure counts adjacent.
   - Multiple comparisons are substantial. Only the distance-trend paragraph disclaims family-wise claims; the degradation labels need the same exploratory designation or a frozen multiplicity procedure.
   - Define precedence among FAILS, NOT QUALIFIED, “not usable” and HOLDS; handling of zero/undefined bootstrap denominators; and ties/non-finite values in trend calculations.
   - Pooled A−/C1/C2 trends cannot identify a pure steepness effect.

8. **Controls capable of failing on real data — NEEDEDS-RESTATEMENT.**

   **Verdict: NEEDS-RESTATEMENT.** The under-resolved-rule and zero-budget controls are genuine numerical interventions, not forged output flags. The existing qacc implementation recomputes distances and certificates from their trajectories. **Q/qacc.py:293–338.**

   However, D:219’s “each … at real size” conflicts with the synthetic/unit controls at D:224–227,233. Synthetic order/Péclet fixtures test verdict logic; they do not demonstrate that an incorrectly resolved real PDE reference is rejected.

   Freeze the rank, cases and schemes used for each control, the outcome if a negative control does not fire, and a real reference-side degradation control. R-real merely records an outcome; it does not require detecting a defect. Also test that B1/B2 nodal, scoring-lattice and spectral initializations represent the same normalized field.

9. **Budget against E0 timings — NEEDS-RESTATEMENT; ordinary ST estimate is reasonable.**

   E0’s logged ST cases include **448, 392, 341 and 325 seconds**. **E0/runs/fd2ddev6p0/logs/5217591.out:30,38,46,54.** Thus approximately 0.12 GPU-h/case and approximately 2 GPU-h per ordinary 16-case panel are consistent with existing evidence.

   The weak point is the expensive tail. E0 documents a tight 8192-level attempt taking approximately **3.5 hours**, versus approximately 282 seconds for production. **E0/DESIGN.md:896–900,925.** The 0.2–0.45 GPU-h estimates do not reserve for that path.

   Two reference pilots—C2 and B2—cannot certify costs “by regime,” especially high-viscosity floor behavior. Include calibration probes, retry/tight/refinement costs, compilation and σ-memory overhead in the certificate. The ledger/cap gate is correct, but an optimistic certificate does not enforce a conservative reservation.

10. **Brief completeness, FOM comparison, 3D and defer order — WRONG FOM cost claim; remaining scope NEEDS-RESTATEMENT.**

   Error-versus-distance reporting, regime Q1/Q2 and the defer order are present and match the lane brief. **D:237–245,273–278; Briefs/jcp-regimes.md:18–28.**

   The optional FOM timing proposal is not valid as stated. It combines old FOM timing with old **in-range ROM timing** and new-regime errors. Adaptive ROM work can change with the regime; neither cross-job timing nor historical ROM cost establishes an **upper bound** on regime speed-up. **D:239–242.** Measure matched-cost queries together on the same GPU, or report accuracy-only comparisons.

   The 3D phase lacks a frozen qualified configuration list, explicit 65³ query contract, baseline and completed certification plan. D:103 correctly requires an amendment, but that makes this **a 2D preregistration with a deferred 3D specification**, not a ready full-lane design. Also, flipping one overlapping blob does not guarantee a sign-changing field, and `b3d_common`’s positive-family peak routine uses a signed maximum rather than maximum absolute value. **SD/b3d_common.py:135–145.**

Required fixes:

1. Replace clean causal attribution with precise distributional contrasts, or add matched controls that hold the intended nuisance variables fixed; label C1 and C2 as low-ν regimes.
2. Correct bank-versus-head training provenance and pin checkpoint, rotation and training-draw identities.
3. Restate RNG domain separation without claiming mathematical stream disjointness; resolve descriptor-only test64 checking and enforce cohort allowlists.
4. Copy the actual E0 acceptance predicates, specify calibration-transfer checks, and add all required family-aware reference adapters.
5. Define σ and separation-check behavior for rejected/missing references and insufficient N0 coverage.
6. Specify qacc selection, subset, IC, normalization, hashing and reference-routing changes explicitly.
7. State GAL’s qualification exception and apply all validity/Q1/Q2 predicates to every sampled case.
8. Freeze statistical population, label precedence, missing-data behavior and multiplicity scope.
9. Freeze concrete numerical controls and a conservative resource certificate covering tight/refinement paths.
10. Replace the unsupported FOM speed-up bound; complete the 3D amendment before that phase, retaining the registered defer order.