Read-only audit completed. No files changed, scratch created, or cluster commands run.

References below use `D` for [experiments/jcp-robust/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/DESIGN.md). Other lane filenames refer to their named worktrees and `experiments/<lane>/` directories.

1. **CORRECT — E1 matches the pinned qualification population.**  
   `D:65–81` includes precisely the seven qualified Fibonacci configurations per mesh: R128 BE factors 1, 2, 2.5; CN-R factor 1; diagnostic GAL-CN factors 1, 2, 5. FRONT-2D distinguishes front members from `other_qualified`; BE factor 2 and CN-R factor 1 are not front members at every mesh. The two lattice configurations match qualify’s four-member 3D subset (`jcp-qualify/report.md:456`). No qualified Sobol configuration occurs in these pinned populations.  
   **Fix:** No population change needed; explicitly describe this as the **pinned qualify population**, not all configurations subsequently qualified by dims.

2. **CORRECT — Sobol terminology.**  
   `D:68–71` correctly distinguishes SciPy’s LMS plus digital shift from nested-uniform Owen scrambling. This agrees with the [SciPy Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).  
   **Fix:** None; preserve the diagnostic-only designation.

3. **NEEDS-RESTATEMENT — Shift definitions are mathematically correct, but executable notation and floating-point ordering matter.**  
   The generators use `uniform(size=2)` and `uniform(size=3)`, not the literal `uniform(2)`/`uniform(3)` written in `D:86–89`. The latter specify a lower bound, not a vector size. Both generators reduce the unshifted coordinates modulo one **before** adding the shift (`hari_quadrature.py:107–119`; `quad3d/rules.py:99–104`). Algebraically equivalent implementations need not reproduce hashes. Dims correctly delegates to these generators and checks seed-zero regeneration (`rules3d_names.py:47–63`).  
   **Fix:** Write the exact calls and preserve generator operation order; retain byte-for-byte seed-zero checks.

4. **WRONG — E1 “qualification survives redrawing” does not enforce Q1+Q2.**  
   `D:106–108` permits a passing draw without its own-family successor check passing. Its indicator is measured against the independent target (`D:101`), whereas binding Q2 requires successor-rule agreement on own reached states. `D:295–297` computes tight-solve distance only for draw zero and selected extrema. Neither good stationarity nor extreme cross-rule distance bounds solver error on the remaining draws. This violates `COMMON-BRIEF.md:49–60` and `PLAN-v4-amendments.md:16–21`.  
   **Fix:** Require full Q1 and successor Q2 for every draw called qualified, with the cross-family test as an additional gate. Otherwise rename the outcome as a partial robustness diagnostic.

5. **WRONG — Bad-vector admission does not guarantee the registered must-fail outcome.**  
   `D:134–141` admits the fixture using frozen-state advection error, then requires rollout distance or invalidity to flag it. A large relative advection error need not produce a sufficiently large solution error: viscosity, time integration, projection, or a small absolute advection term can suppress it. Dims avoids this implication by checking frozen-state advection first (`dstats.py:159–175`). Its established fixtures also use different sizes/states; those results do not establish these fixtures.  
   **Fix:** Require the independent in-job advection calculation to flag the admitted frozen states, with agreement against the independent CPU calculation. Treat rollout divergence as additional evidence.

6. **NEEDS-RESTATEMENT — Positive gates and blocking precedence are incomplete.**  
   `D:128–149` gives explicit precedence for control failure and reproduction, but not uniformly for point identity, missing arrays, non-finite values, or empty case intersections. “Reproduce … in the scoring metric” must mean distance between fields, not equality of scalar E0 errors. Moreover, qualifying Δt₀ rows cite P1 rather than qualify’s own accuracy runs (`jcp-qualify/report.md:93–95`), so `D:146–147` needs an archive map.  
   **Fix:** Freeze expected case/time counts, archive hashes, finite-value requirements, field-distance predicates, and a single precedence table. Missing evidence must never pass via an empty reduction.

7. **NEEDS-RESTATEMENT — E1 decision thresholds are attainable, but interval interpretation and ties need correction.**  
   `D:112–124` allows all three H-E1a and H-E1c outcomes at 16 draws. However, 0.171 is the **one-sided** 95% Clopper–Pearson upper bound for zero failures; the two-sided 95% interval has upper endpoint approximately 0.206. Including the previously qualified deployed draw also needs a selection-conditioned caveat. Breaking ties by draw index makes seed zero rank first when all results tie, falsely labelling a perfectly insensitive configuration “atypically favourable”; the stated uniform-rank probability then does not apply.  
   **Fix:** Specify one-sided coverage, report a failure bound on the 15 fresh draws, and use tied-rank intervals or a preregistered tie-safe classification. Define invalid-draw and zero-eligible-case handling.

8. **CORRECT — Stage A can reproduce the incumbent training procedure and produce a usable checkpoint.**  
   `smoothtrain.py:40–42,170–177,403–432` preserves the r3a architecture, data pick, optimizer recipe and checkpoint schema. `sep_common.py:281–291` confirms compatibility with extraction. Smooth-bank demonstrated deterministic-compilation equivalence while documenting default-autotuning differences (`report.md:282–290`). Its automatically emitted LS rotation is different, but `D:173–176` correctly replaces that with the deployed head-based procedure.  
   **Fix:** None to this architecture; explicitly discard the smoothtrain LS rotation as a deployment input.

9. **NEEDS-RESTATEMENT — Stage B is faithful in outline but insufficiently specified to reproduce dn256b.**  
   `D:161–175` omits settings essential to the dense extraction. The actual recipe uses `MAX_SNAPS=131072`, `EXTRA_SEED=1000`, `EXTRA_TRAJ=4032`, `SEED0=0`, and `LOOSE=1` (`separable-decoder/cluster/run_dn256b.sbatch:44–62`). Defaults instead select 16,384 states from the canonical population. Dense extraction intentionally has a different code population (`sep_coeff_extract.py:111–137`). The executable head driver is `sep_hfit_run.py`; its `mid` fit is fresh initialization, with key `SEED0+11`, and emits all fitted codes (`sep_hfit_run.py:351–411`).  
   **Fix:** Freeze the complete extraction and head-fit configuration, including state IDs, batch size, learning rate, stopping policy and emitted-code count. Allow the intentional alignment exception only for diagnostics; never use recycled codes for the deployed rotation.

10. **NEEDS-RESTATEMENT — “Full-recipe seed” needs a conditional estimand.**  
    Keeping physical training trajectories fixed is defensible for **algorithmic robustness conditional on the training dataset**. It does not measure training-data sampling robustness. `D:163–165` also calls Stage A randomness “minibatch/point subsampling,” although its value-only training uses all selected states and subsamples spatial points (`smoothtrain.py:255–279`). The fixed 2D state pick and varying 3D spatial subset should be explicitly distinguished.  
    **Fix:** Name the conditional estimand and enumerate each fixed/varying stream. Describe the incumbent as a historical comparator; software differences prevent attributing incumbent-versus-new differences solely to seed.

11. **CORRECT — The proposed 3D bank recipe matches W1024.**  
    `D:200–205` agrees with `train_r1024.json:7–31,47–56,78–80`: fixed physical datasets, varying model/subset seeds, 12,000 steps, no trained heads. Host ordering uses LS coefficients and the 65³ bank (`train2w.py:379–444`). The historical fit selected step 11,000 through bank-validation selection (`report.md:111`).  
    **Fix:** Preserve that checkpoint-selection procedure explicitly; do not silently take the final step.

12. **NEEDS-RESTATEMENT — Incumbent rule recovery is plausible, not established; terminal-rung and no-selection behavior are missing.**  
    Dims supports the proposed R128/R256/R384 choices: Fibonacci 4181 for R128 and Gauss 128 for the larger ranks (`report.md:47,55,71,127`). It does not prove the exact new intersection over BE, CN-R and independent-target checks in `D:178–186`. The final candidate still needs an explicit successor. In 3D, dims distinguishes candidate rungs from additional certification rungs (`DESIGN.md:127–130`); robust’s abbreviated ladder does not. More seriously, dims already reports no qualifying Gauss candidate for W1024 under both requested specs, and for W768 QN-BDF2 (`report.md:294–300`).  
    **Fix:** Register all successor/anchor rungs and their checks, plus “none within ladder,” missing and invalid outcomes. Specify what verdict runs do when no rule is selected. Verify target convergence for each new bank rather than assuming the incumbent’s target remains sufficient.

13. **NEEDS-RESTATEMENT — The selection fallback is fair, but incumbent verdicts must be recomputed on the same universe.**  
    Applying one selection algorithm to every bank (`D:184–186`) is sound. However, the new 2D front excludes GAL and R512 (`D:194–196`), whereas the historical front includes them. Historical membership labels therefore cannot be compared directly with membership in the restricted front. The incumbent job list explicitly budgets selection but does not clearly require its full restricted-front evaluation.  
    **Fix:** Recompute the incumbent’s entire comparison panel under the identical selection, timing and candidate-universe rules; keep historical deployed-front membership separate.

14. **WRONG — E2 omits the requested 3D front replication.**  
    `D:214–219` provides errors, width transitions and qualification of four configurations, but no 3D timing or dominance procedure. These cannot establish a qualified error–cost front. The lane brief explicitly requires fronts at both 1024² and 65³ (`briefs/jcp-robust.md:23–27`).  
    **Fix:** Add a frozen 3D candidate universe, within-job timing and front construction for every bank, including the incumbent, or explicitly register this deliverable as incomplete.

15. **WRONG — E2 can declare vacuous robustness and does not exhaust all label combinations.**  
    At `D:223–228`, three seeds matching an inconclusive/unavailable incumbent satisfy “seed-robust.” Conversely, an inconclusive incumbent followed by definitive new-seed labels has no clear “opposite definitive label.” Mixed pass/fail/inconclusive seeds need explicit precedence. `D:234–235` also leaves the spread ratio undefined when the paired-change denominator is zero.  
    **Fix:** Require a definitive incumbent and three matching definitive replicates for robustness; prioritize demonstrable disagreement, then classify all remaining combinations as inconclusive. Define no-rule and missing-seed states separately. Specify zero-denominator handling. Three seeds support an agreement description, not a strong population-level robustness claim.

16. **CORRECT — The A100 memory restriction is supported for the copied implementation.**  
    `D:243–245` agrees with P1’s measured 4096² process peaks: approximately 96.6 GB in production and 113 GB in the pilot (`jcp-p1-scaling/report.md:204–207`). Those exceed A100-80GB capacity.  
    **Fix:** Qualify the statement as applying to the existing full-bank implementation; it is not an impossibility result for chunked or prefix-retaining implementations.

17. **NEEDS-RESTATEMENT — E5 comparator freezing is sound, but exact keys and case IDs must be pinned.**  
    The actual `FROZEN-SELECTION-1024.json` contains the required comparators. For both subjects, `H|union|joint` and `H|PS|joint` select `ps|N96|0.0125`; the same-mesh CC choices differ: R128 uses `fd|CC|L1024|0.0125|t2`, R384 uses `fd|CC|L1024|0.005|t1` (file lines 108, 168, 378, 14784, 14841, 15039). Thus PS and union are duplicate selections here. `D:248` does not identify the 16 timing cases.  
    **Fix:** Pin the manifest hash and exact selection keys, deduplicate timed subjects, and enumerate the cohort IDs. Label this as performance of frozen H200-selected comparators, not the best comparator on each GPU.

18. **WRONG — E5 changes the inherited timing statistic and ambiguously promotes subset evidence to governing verdicts.**  
    `D:262` defines speed-up using median raw times. Fom2d instead uses a ratio of case medians of within-triplet B/A ratios, with common-valid cases (`c2analyze.py:7–12,27–56`). P1’s default two-mesh result is explicitly “registered three-mesh verdict not reached,” even when its subsidiary procedure passes (`p1stats.py:166–207`). A two-subject 1024² panel also cannot reproduce K2’s all-configurations, largest-mesh verdict.  
    **Fix:** Preserve the normalized C2 estimator. Define a separately named two-mesh flatness test and subset two-times-speed test, with explicit three-way interval rules. Keep the original H-cost-N and K2 scope intact.

19. **NEEDS-RESTATEMENT — Hardware pooling and bootstrap are not fully defined.**  
    `D:260–270` says “jobs pooled” without fixing aggregation across repetitions, cases and jobs, or handling different valid-case sets. Case resampling can estimate workload uncertainty conditional on the measured devices, but cannot establish hardware-population uncertainty from two GPUs. “For each quantity” also appears to include categorical q4, for which a ratio is undefined.  
    **Fix:** Restrict ratio analysis to q1–q3; freeze aggregation order, joint case intersections, invalidity thresholds and undefined-bootstrap handling. Report per-job and pooled results, explicitly conditional on the observed devices.

20. **WRONG — The E5 doubled-work control can be cancelled by genuine hardware dependence, and smk1 cannot execute it as defined.**  
    `D:275–277` produces approximately \(2\phi_{\mathrm{normal}}\), not necessarily 2. If normal H200/A100 B/A is 0.5, doubling gives approximately 1 and the control fails despite a functioning classifier. P1’s existing doubled control instead compares double execution with flanking **same-subject single executions** (`p1time.py:266–294`). Furthermore, H200-only smk1 claims to demonstrate a control requiring A100 data (`D:303`).  
    **Fix:** Validate physical doubled work within each GPU, then test the cross-hardware classifier using a paired synthetic fixture constructed to have a known ratio. Require an A100 smoke for actual cross-device controls. State blocking precedence for both quantity and verdict conclusions.

21. **NEEDS-RESTATEMENT — Training costs are credible; 3D selection costs and smoke coverage are optimistic.**  
    The 2D Stage A allowance is consistent with measured approximately three-hour fits (`smooth-bank/report.md:39–40`), and 3D’s two-hour allowance is close to the measured 1.85-hour job (`wide-bank/report.md:111`). But `D:323` allows only 0.8 hours per complete 3D selection/verdict bank. Dims’s analogous ladder jobs consumed 1.5264 and 2.8592 hours (`runs/GPU-LEDGER.json:77–106`); these are not identical workloads, but make 0.8 hours unsupported. Robust’s smoke tests only a small W256 selection pair, not the expensive W1024 target/tail compilation and memory path. Additional full Q1 checks and 3D fronts increase costs further.  
    **Fix:** Rebuild the estimate from explicit rollout, anchor, compilation and ordering counts; smoke the heaviest evaluation cell. Charge reservations by allocated GPU count for multi-GPU jobs. The 40.1-hour arithmetic is correct, but its feasibility is not yet certified.

22. **CORRECT — Defer order and basic campaign constraints.**  
    `D:327–331` matches the lane brief’s defer order and protected items. `D:12–16,83–84,285–291` correctly preserves E0 ST, matched-S restrictions, development cohorts, per-case-kind eligibility, frozen mode policies and CPU analysis.  
    **Fix:** Retain these provisions while repairing the qualification, front, smoke and statistical gaps above.

**Overall verdict: NEEDS-RESTATEMENT.**