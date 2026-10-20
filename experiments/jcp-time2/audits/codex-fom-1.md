**NOT CLEAN for submitting `fom1k`.** The LMM mathematics and listed ROM tolerances look correct, but calibration failure handling, accuracy–timing linkage, and audit acceptance need fixes.

Read-only review: no files modified, no jobs submitted, no GPU experiments run. Syntax checks and HEAD parity passed for the reviewed files; the existing manufactured-test certificate matches its recorded sources.

1. **CORRECT — LMM residual, preconditioner, history, and startup.**  
   [fom2.py:38](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fom2.py:38) implements
   \[
   a_0u+a_1u_n+a_2u_{n-1}+\Delta t\bigl(b_0F(u)+b_1F(u_n)\bigr).
   \]
   The DST denominator is correctly \(a_0+\Delta t\,b_0\nu\lambda\); `Fn` uses the current previous state and is conditionally evaluated. History advances correctly. BDF2 starts with one BE step; CNR uses the registered two full BE steps. Output indices give the six prescribed times. The reported local BE identity and convergence orders support this reading, though I did not rerun them.

2. **CORRECT — nonlinear acceptance; WRONG — promised diagnostic records.**  
   [fom2.py:55](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fom2.py:55) follows A2.21: nonfinite or excessive final nonlinear residual fails; convergence on the last Newton iteration passes; linear residual alone does not reject a step.

   However, it retains only trajectory aggregates—`it_sum`, `it_max`, `worst_rel`, `worst_lres`, failure count/index. **Neither per-step nonlinear residuals/Newton counts nor per-Newton linear residuals survive**, contrary to A1.10 and the module description.

   **Fix:** return and persist the diagnostic arrays, with iteration counts/masks. Record the effective schedule and linear tolerance alongside them.

3. **WRONG — failed calibration reference can become an accepted fallback.**  
   [fomrun.py:118](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fomrun.py:118) correctly selects the loosest passing candidate across dev6, using restricted-field discrepancy. But a failed \(10^{-10}\) reference makes every candidate fail and then selects the fallback anyway. The audit repeats this logic and can accept that selection.

   There is also a solver change: calibration reference uses `ltol=1e-12`; fallback evaluation/timing uses `1e-8`.

   **Fix:** require all six tight reference trajectories to verify before accepting calibration. A failed reference means “calibration unresolved,” not ordinary fallback. Store `(ntol, ltol)` together; preferably reuse the verified tight solver pair for fallback. Explicitly name the tight-reference ST error as the denominator.

4. **CORRECT — listed ROM timing subjects match production schedules/tolerances.**  
   [fomrun.py:171](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fomrun.py:171) uses the same model, trust construction, cold initialization, rule data, form transformation, query factory, and schedule as `t2run.py`. Production tolerances match:

   | Form | `gtol` | `tolf` |
   |---|---:|---:|
   | LSPG | \(10^{-3}\) | \(10^{-9}\) |
   | GAL | \(0\) | \(10^{-10}\) |

   A latent generalization bug remains: cached rule data and subject names omit rule identity. Current `fom1k` has one rule per setting, so it is unaffected. **Fix before adding rules:** include rule identity in both keys.

5. **WRONG — ROM warm-up hashes do not establish accuracy–timing equivalence.**  
   [fomrun.py:210](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/fomrun.py:210) checks FOM timings against evaluation outputs, but ROM timings only against warm-up outputs. That establishes repeatability, **not correspondence to an accuracy run**. ROM verification statistics are discarded; a reproducibly failed ROM can pass every timing hash.

   **Fix:** score and verify the same ROM invocations used for timing, retaining fields or coefficients and provenance. Bind their outputs numerically to the pinned `t2run` accuracy artifacts. Cross-job bitwise hashes need not be mandatory, but warm-up self-consistency alone is insufficient.

6. **NEEDS-RESTATEMENT — “best FOM at matched ST error versus best ROM” is not currently computable from this output alone.**  
   `result.json` has FOM errors and both families’ timing arrays, but **no ROM errors or verification records**. An external join is possible only after the linkage above is established.

   Moreover, [make_configs.py:34](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/make_configs.py:34) times ROM factors `{1,2,5}` for fast/acc. It omits the timed `t2run` factors `0.5` and `10`, and the optional wide setting. It therefore cannot promise coverage of whichever ROM selection wins.

   **Fix:** freeze and include the actual selected candidates, or label the result “best among these listed configurations.” Define matching explicitly—for example, minimum median query cost subject to cohort-worst ST error ≤ a fixed threshold. Report that accuracy covers 38 cases while timing covers six development cases. Use this job’s timings for both sides.

7. **WRONG — order check is recorded but never enforced.**  
   [audit_fom.py:85](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:85) copies reported orders into `info`; it checks neither order inventory, failed solves, finite positive differences, nor acceptable rates. A missing order panel or failed CN/BDF2 order test need not prevent `all_pass`.

   **Fix:** register explicit acceptance criteria, require verified order trajectories, and independently recompute rates from saved outputs. The existing certificate covers `t2core.py`, not the new FOM implementation. The supplied L=128 evidence does not replace the required L=256 gate.

8. **WRONG — audit independence is insufficient for its acceptance claim.**  
   [audit_fom.py:35](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/audit_fom.py:35) provides useful independent NumPy rescoring of dev6 and a BE comparison with `engines.make_fom`. But:

   - Calibration discrepancies and verification flags are trusted; empty calibration case lists satisfy `all(...)`.
   - Timing inventory derives from the output’s own `subjects`, allowing an omitted subject and all its records to disappear together.
   - Expected hashes are not independently reconstructed from saved accuracy artifacts.
   - val32 fields are not saved/rescored.
   - `--no-rerun` can still produce an unqualified `all_pass`.
   - BE replay shares the spatial operator and cannot independently establish CN/BDF2 correctness.

   **Fix:** derive inventories from the pinned config, require all six calibration cases, bind evaluated/timed tolerances to calibration, reconstruct verification and hashes from artifacts, save val32 restricted fields, and add rejection tests. Mark skipped independent checks as incomplete.

9. **CORRECT — staging driver selection and principal job controls.**  
   [stage.py:120](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/cluster/stage.py:120) selects `fomrun.py` from this config. Staging checks committed bytes, includes both new runtime modules, requests the GPU partition/A100-80G constraint, exports f64/highest precision, and performs the GPU preflight. Submission uses the lane lock and queue checks.

   **Remaining acceptance fix:** carry reference-contract checks and source/reference identities into the FOM audit. The current ROM manufactured-test certificate is not a FOM correctness certificate.

10. **NEEDS-RESTATEMENT — resources are plausible, not measured for this driver.**  
    Current workload is substantial but reasonable:

    - 384 calibration, 608 evaluation, and 38 reproduction FOM solves.
    - 384 warm-up calls.
    - 1,134 A–B–A triples, including **2,268 repeated baseline solves**.
    - At least **379 seconds of configured timing burn-in**.

    The archived A100-80G `dv1024` lean-tight FOM timings have median **0.445 s**, range **0.348–0.695 s**. This is a scheduling anchor, not a benchmark for the new LMM driver or tighter calibration solves.

    **I would request 6 hours, one A100-80G, 128 GB host memory**, with roughly 1–3 hours as a provisional planning estimate. The persistent 384-column mesh bank is about **2.99 GiB**; six FOM output fields occupy about **48 MiB**. Construction temporaries and compiled workspaces increase the peak, but I see no structural 80-GB memory blocker.

    **Fix:** save progress periodically during calibration/evaluation/timing. Currently calibration saves only at its end, and timing records survive only when the whole timing phase finishes.

The submission blockers are findings **2, 3, 5, 7, and 8**, plus resolving the comparison scope in **6**. The core integrator does not presently look like the problem; the evidence needed to accept its comparison is incomplete.