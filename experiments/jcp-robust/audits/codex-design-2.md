Read-only audit completed. No files changed, scratch created, jobs submitted, or cluster commands run.

References use **D** for [experiments/jcp-robust/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/DESIGN.md). Other lane paths are relative to their named worktrees’ `experiments/<lane>/` directories. Items 1–22 correspond to the first audit; items 23 onward identify additional problems in A1.

1. **CORRECT — RESOLVED: E1 population.**  
   A1 explicitly limits the population to pinned qualify results (`D:356–358`). Reading `FRONT-2D.json` confirms seven randomized configurations per mesh: R128 BE factors 1, 2, 2.5; CN-R factor 1; GAL-CN factors 1, 2, 5. The two randomized 3D configurations match qualify’s subset (`jcp-qualify/report.md:456`). No qualified Sobol member appears in either population.  
   **Fix:** None; retain the pinned-population qualification.

2. **CORRECT — RESOLVED: Sobol terminology.**  
   `D:68–71` correctly calls this LMS plus digital shift, not nested-uniform Owen scrambling. The actual 3D generator calls `qmc.Sobol(..., scramble=True)` (`vendor/quad3d/rules.py:93–96`), consistent with [SciPy’s documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).  
   **Fix:** None; preserve its diagnostic-only status.

3. **CORRECT — RESOLVED: Cranley–Patterson generation.**  
   `D:359–362` now delegates to the actual generators and requires seed-zero identity. This matches `vendor/quad2d/vendor/hari_quadrature.py:107–119`, `vendor/quad3d/rules.py:99–104`, and dims `rules3d_names.py:47–63`.  
   **Fix:** None. In implementation, distinguish Fibonacci index \(k\) from point count \(F_k\).

4. **CORRECT — RESOLVED: Full E1 qualification contract.**  
   `D:364–372` adds tight solves for every draw/case, tighter anchors for every draw, successor rollout agreement, and own-state successor \(\rho\). It explicitly withdraws the extrema-only shortcut. This addresses `COMMON-BRIEF.md:49–60` and V4-2.  
   **Fix:** Preserve separate successor and cross-family indicators; they must not share an ambiguously named `rho` field.

5. **CORRECT — RESOLVED: Bad-vector control’s required response.**  
   `D:374–382` now tests the same frozen-state advection discrepancy used to admit the fixture. Admission at \(\rho\ge1\), followed by independent agreement within 1%, implies crossing the 0.116 threshold. Rollout divergence is appropriately optional.  
   **Fix:** Require finite values and a strictly positive target norm explicitly, as dims does in `dstats.py:173–175`.

6. **WRONG — NOT RESOLVED: Reproduction archive map.**  
   `D:387–390` sends every 2D E1 arm to qualify’s `acc*` archives. But `jcp-qualify/configs/acc1024.json:19–68` contains no R128 LSPG BE/CN-R factor-1 runs; `acc256` likewise omits them. Qualify cites P1 for these rows (`report.md:93–95`). At 4096², qualify’s added factor-1 runs cover the complementary val32 cases, not the requested P1 subset. The new missing-file rule therefore guarantees blocked gates for these arms.  
   **Fix:** Map each configuration and case set to its actual P1 or qualify archive, including filenames, array keys, case selection and hashes. Define the reproduction norm and its zero-denominator behavior explicitly.

7. **NEEDS-RESTATEMENT — NOT RESOLVED: E1 statistical edge cases.**  
   The one-sided bounds and fresh-draw caveat are corrected (`D:401–407`). The ranking rule remains incomplete: approximate equality is non-transitive, aggregate errors have no uniquely specified scalar \(\epsilon_{\rm ref}\), and mid-ranks can fall between the old integer bins. Removing every all-tied statistic also leaves an empty “all three statistics” test. The uniform-rank probability in `D:119–120` no longer applies generally with ties or selection conditioning.  
   **Fix:** Specify deterministic tie groups or rank intervals, aggregate uncertainty margins, exhaustive fractional-rank bins, and an explicit all-insensitive outcome. Restrict the \(1/4\) statement to an unconditioned, tie-free reference model.

8. **CORRECT — RESOLVED: Stage A and checkpoint compatibility.**  
   `smoothtrain.py:40–42,170–177,403–432` preserves the relevant r3a recipe and emits the expected `params`, `Z_tr`, and configuration checkpoint. A1 explicitly discards its LS rotation as a deployment input (`D:419–421`).  
   **Fix:** None to this choice of trainer; retain provenance and data-fingerprint checks.

9. **CORRECT — RESOLVED: Stage B recipe specification.**  
   `D:411–420` now freezes dense extraction and the full head-fit call, matching `separable-decoder/cluster/run_dn256b.sbatch:44–62`. Fresh `mid` initialization and emitted fitted codes are supported by `sep_hfit_run.py:351–411`. The head-based rotation matches `make_rotation.py:57–75` on `exp/2026-09-23-burgers-bank-knob`.  
   **Fix:** Retain these settings and record actual stopping steps; see item 30 concerning the wall-time cap.

10. **CORRECT — RESOLVED: Conditional training-data estimand.**  
    `D:409–423` correctly distinguishes fixed physical training data from varying algorithmic streams and corrects the Stage-A minibatch description. Keeping data fixed is justified for this explicitly conditional estimand.  
    **Fix:** Preserve this wording throughout the verdict interpretation; A1.10 currently conflicts with it, as detailed in item 28.

11. **CORRECT — RESOLVED: W1024 training and checkpoint selection.**  
    `D:425–426` preserves validation-selected checkpoints. This agrees with `train3d/train2w.py:193–221`; the seeds, 12,000-step recipe, and absence of heads match `train3d/configs/train_r1024.json:7–31,47–56,78–80`. Ordering remains the training-coefficient construction at `train2w.py:379–444`.  
    **Fix:** None; record selected step and all changed streams.

12. **NEEDS-RESTATEMENT — NOT RESOLVED: Rule-selection outcomes and target certification.**  
    A1 adds terminal successors and explicit “none” outcomes (`D:428–438`). Recovery of the incumbent 2D rules is plausible: dims reports Fibonacci 4181 at R128 and Gauss 128 at R256/R384 (`report.md:47,55,71`). It is not established for this new joint BE/CN-R procedure. More seriously, `D:431–433` allows selection despite a failed target-convergence check, without restricting resulting conclusions.  
    **Fix:** Allow a diagnostic candidate pick after target failure, but block conclusions requiring a converged target. State which ordinary successor-Q2 conclusions remain available independently.

13. **CORRECT — RESOLVED: Like-for-like incumbent comparison.**  
    `D:439–441` now requires the incumbent’s complete selection, verdict, timing and restricted-front runs under the same procedure. Historical front labels remain separate.  
    **Fix:** Preserve this separation. The fallback is scientifically sound once the gate-precedence contradiction in item 24 is repaired.

14. **NEEDS-RESTATEMENT — NOT RESOLVED: 3D front replication.**  
    A1 adds a front, timing workload and dominance rule (`D:443–449`), resolving the outright omission. However, the body selects rules per **width and specification** (`D:207–210`), whereas A1.9 uses “the width’s selected” rules across three specifications, including CN 0.0125, which has no registered selection procedure.  
    **Fix:** Define selection keys as `(bank, width, specification, family)` or explicitly define a joint intersection rule. Specify the source rule for CN 0.0125 and deduplicate overlapping fixed-panel candidates.

15. **CORRECT — RESOLVED: E2 agreement cannot pass through unavailable labels.**  
    `D:451–458` requires definitive agreement and prioritizes definitive disagreement. Three matching unavailable/inconclusive outcomes no longer establish robustness. The basic three-way rule is attainable with three new seeds.  
    **Fix:** Retain this structure, but distinguish “none within ladder” as a definitive *selection* result from unavailable downstream accuracy/front results. Repair the attribution problem in item 28.

16. **CORRECT — RESOLVED: A100 memory claim.**  
    `D:461` appropriately restricts the claim to the existing implementation. P1 records 96.6 GB production peak and approximately 113 GB pilot peak (`jcp-p1-scaling/report.md:204–207`), exceeding A100-80GB capacity.  
    **Fix:** None; do not generalize this to chunked implementations.

17. **WRONG — NOT RESOLVED: The frozen CC selection key selects the wrong comparator.**  
    `D:463` specifies `H|CC|joint`, but the panel promises the **same-mesh** comparator (`D:247–248`). In fom2d’s `FROZEN-SELECTION-1024.json`:
    - R128 `CC|joint` selects **L64**, at lines 48–59; `CC|same_mesh` selects **L1024**, at lines 108–119.
    - R384 `CC|joint` selects **L128**, at lines 14724–14735; `CC|same_mesh` selects **L1024**, at lines 14784–14795.

    Union/PS deduplication is correct.  
    **Fix:** Use `H|CC|same_mesh` for the promised panel, or explicitly change its scientific scope and labels.

18. **WRONG — NOT RESOLVED: P1 estimator is still changed.**  
    The named subset tests correctly avoid replacing full H-cost-N or K2 (`D:469–473`). But `D:466–468` claims unchanged P1 estimators while describing per-case medians. P1’s `h_cost_n` takes the median over **all valid case/repetition entries** per mesh (`p1stats.py:181–188`). Median-of-case-medians can differ.  
    **Fix:** Preserve the inherited statistic for per-job H-cost-N(2). If pooled hardware analysis uses a different estimator, define and name it separately.

19. **NEEDS-RESTATEMENT — NOT RESOLVED: Pooled nonlinear quantities remain ambiguous.**  
    `D:474–478` defines “quantity = median over cases,” which directly fits q1 but does not define q2’s max/min across mesh medians or q3’s ratio of FOM and ROM medians. It is unclear whether ratios are formed before or after job averaging. Those operations do not commute.  
    **Fix:** Write separate formulas for q1, q2 and q3 from per-subject arrays. In each bootstrap replicate, resample case IDs jointly and recompute the full nonlinear statistic. Specify minimum valid repetitions and which subjects/meshes define each common-case intersection.

20. **NEEDS-RESTATEMENT — NOT RESOLVED: E5 controls are improved but incomplete.**  
    The within-GPU doubled/single control removes the original hardware-cancellation error (`D:481–483`). A1’s synthetic classifier fixture still lacks a defined injection level for q2/q3 (`D:484–486`), and its blocking sentence mentions quantity verdicts only.  
    **Fix:** Add explicit fixtures for each statistic and block dependent q4 verdict-agreement conclusions too. See item 29.

21. **NEEDS-RESTATEMENT — NOT RESOLVED: Budget arithmetic works; workload accounting does not yet.**  
    The revised lines sum to 58 GPU-h, and 25% contingency gives 72.5 (`D:491–502`). Raising 3D evaluation to 3 hours per bank is more credible against dims’s 1.5264/2.8592-hour ladder jobs (`GPU-LEDGER.json:77–106`). However, A1.2 gives GAL the same qualification structure, while A1.13 budgets GAL only as “prod + target.” Tight, tighter and successor GAL runs are missing from that count. The Sobol diagnostics are also not explicitly accounted for.  
    **Fix:** Produce an inventory by configuration, draw, case, solver level, successor/target and compile signature. Include diagnostics and controls, then certify the revised total from smokes. Do not treat 72.5 as established feasibility.

22. **CORRECT — RESOLVED: Defer order and core campaign constraints.**  
    `D:327–331,500–502` retains the brief’s defer order and protected work. E0 ST, matched-S restrictions, development cohorts, per-case-kind eligibility, frozen 2D modes, CPU analysis and GPU-hour reservations remain registered (`D:12–16,83–84,285–291`).  
    **Fix:** None to these provisions. The numerical and statistical exceptions identified elsewhere still need repair.

The following are additional A1 problems.

23. **WRONG — Expected numerical failures are conflated with unavailable evidence.**  
    `D:391–398` makes non-finite or partially missing statistics unavailable before applying decision rules. Yet `D:406–407` deliberately assigns \(+\infty\) to invalid draws so they rank worst. Thus a diverging draw can be treated either as evidence of fragility or as grounds to suppress the verdict. The stated precedence favors suppression.  
    **Fix:** Separate infrastructure failure/missing records from a fully recorded numerical failure. A completed invalid draw must count as a qualification failure. Give its accuracy statistic an explicit failed state rather than routing its sentinel through generic finiteness checks.

24. **WRONG — Global positive-gate precedence can disable the registered selection fallback.**  
    `D:397` blocks on failed `G-E2.x` gates. But G-E2.1 explicitly permits incumbent rule mismatch and continues with identical procedure-selected rules (`D:183–186`); G-E2.2 similarly reports disagreement (`D:210–212`). These are not reproduction gates with the same semantics as G-E1.2.  
    **Fix:** Enumerate blocking gates individually. Make incumbent rule recovery a reported anchor comparison with a registered fallback, not a globally blocking positive gate.

25. **NEEDS-RESTATEMENT — All-tied typicality can pass vacuously.**  
    `D:405–406` removes all-tied statistics from typicality, but never specifies what happens when all three are removed. An implementation using `all(...)` on the remaining set returns true.  
    **Fix:** Return “insensitive; typicality not assessed” when none remain. Define the reduced-statistic rule explicitly when only one or two remain.

26. **NEEDS-RESTATEMENT — 3D no-rule fallback is undefined for the upper widths.**  
    `D:437–438` promises the “fixed panel rule below” when no rule is selected. The fixed qualify panel contains only W256 and W512 (`D:217–218`), while W768/W1024 are precisely where dims reports several “none” outcomes (`jcp-dims/report.md:294–301`).  
    **Fix:** Register fallback rules for every width/specification, or state that no fallback run exists for those cells. Keep these diagnostics separate from the gl64 H-acc-r panel.

27. **NEEDS-RESTATEMENT — Front-membership replication needs stable comparison identities.**  
    A1.9 changes actual candidate rules by bank, while A1.10 compares “member vs dominated” labels (`D:443–455`). A literal configuration can disappear because another rule was selected; a recipe slot can persist while representing a different rule. These answer different questions.  
    **Fix:** Define cross-bank identities as registered recipe slots, such as `(width, scheme, step, selected-family)`, and report their realized rules. Keep fixed deployed-configuration membership as a separate comparison. Specify missing-slot behavior.

28. **WRONG — A1 still attributes historical-comparator differences to seeds.**  
    `D:421–423` correctly says incumbent-versus-new differences are not attributable to seed alone. But `D:451–453` labels any opposite pair among all four banks “seed-dependent.” Three identical new-seed outcomes opposite the incumbent would therefore establish seed dependence despite showing no new-seed disagreement.  
    **Fix:** Classify seed dependence among seeds 1–3. Separately classify agreement with the historical incumbent. Alternatively use a combined “recipe/version-sensitive” label without causal seed attribution.

29. **NEEDS-RESTATEMENT — The cross-hardware fixture can cancel for q2/q3, and the smoke cannot validate the full classifier.**  
    `D:484–486` scales “per-case values” by 1.5. Scaling every mesh’s underlying B/A values leaves q2’s max/min unchanged; scaling both FOM and ROM values leaves q3 unchanged. Scaling already-computed q values tests only the final interval classifier, not their estimators.

    Moreover, smk4 has two cases and one repetition (`D:488–489`), below the five-case minimum (`D:476`) and fom2d’s two-valid-repetition requirement (`c2analyze.py:27–43`). It cannot demonstrate the complete E5 analysis path before production.

    **Fix:** Use estimator-specific fixtures: scale q1’s subject; construct known mesh ratios for q2; scale only the FOM numerator for q3. Provide at least five cases and two valid repetitions for a smoke exercising the full pipeline. State which failures block quantity and verdict conclusions.

30. **NEEDS-RESTATEMENT — The “200k-step” head recipe is actually time-capped.**  
    A1 faithfully copies `TIME_CAP=1500` (`D:417–418`), but surrounding descriptions call the fit 200,000 steps. `sep_hfit.py:298–310` can stop earlier and emits `steps_done`. Different GPU performance or contention can therefore change training effort while the design says only seeds change.  
    **Fix:** State the actual stopping rule: 200,000 steps or 1,500 seconds, whichever occurs first. Record steps completed, cap activation and device details. If fixed-step replication is intended, preregister that as a deviation from the historical command.

31. **NEEDS-RESTATEMENT — Seed-spread normalization lacks a bank index.**  
    `D:234–235` defines a numerator across seeds but divides by “median paired change” without specifying incumbent, each seed, or pooled changes. A1 only handles zero denominators (`D:456–457`); it also reports infinity for the undefined \(0/0\) case.  
    **Fix:** Define the denominator’s bank/population explicitly. Report positive-over-zero as infinity and zero-over-zero as undefined/no measured spread or step.

**Overall verdict: NEEDS-RESTATEMENT.**