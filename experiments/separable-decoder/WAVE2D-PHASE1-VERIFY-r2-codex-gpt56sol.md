Phase 1 is earned at N=64 and N=128 for both boundary conditions. I found no blocking numerical defect.

1. V1alg — CORRECT

The block matrices are exactly \(I-\frac{\Delta t}{2}G\) and \(I+\frac{\Delta t}{2}G\) for
\(G=\begin{bmatrix}0&I\\c^2L&-cD\end{bmatrix}\); signs and \((u,v)\) ordering are correct. The block path performs no elimination. The ten-step LU comparison is a valid iterative-tolerance-free empirical certification, with all four cases at \(1.61\text{–}1.80\times10^{-14}\).

The controls are correct: `sqrt(1.01)*c` changes \(a=(c\Delta t/2)^2\) by exactly 1% for reflective BCs, while `-dB` reverses damping consistently for absorbing BCs. The actual full-horizon range is \(3.73\times10^{-12}\) to \(2.96\times10^{-11}\).

2. V1cg — CORRECT

The three-tolerance, full-horizon comparison establishes that the JAX/CG displacement path implements the LU recurrence: all ladders are strictly decreasing and end below \(10^{-8}\). The representative residual is a surrogate solve rather than an actual recorded rollout step, but nothing essential is missing for the recurrence claim. Do not describe it as per-step CG convergence monitoring.

3. Round-2 controls — NEEDS-RESTATEMENT

- F0b zero mode: correct and fires at both resolutions. It does decay like \(h^2\), so its fixed threshold is only certified over the declared N=64/128 scope.
- F0d reflective: correct; the one-sided coefficient change breaks \(ML\) symmetry and fires.
- F1a-form: correct; it removes precisely the boundary-edge contributions and fires.
- F2-spatial: correct; the 1.01c reference produces saturation orders 0.33/0.34 and fires.
- F2-temporal: correct; both BE-order and separation conditions fire.
- F3: correct; slope and coefficient are conjoined, the reflective control fires, and all diagnostics are recorded.
- F4: the damping mutation is correct and the two recorded finite runs fire strongly. However, [`energy_trace`](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_fom_gates.py>) rejects nonfinite arrays before the caller can record their finite prefix or nonfinite tail. It also takes the maximum over the finite prefix, not literally only the last finite value. Thus the advertised overflow-recording path is unreachable; overflow instead aborts safely.

4. F0a/F0b normalization — NEEDS-RESTATEMENT

Numerically acceptable, but not mathematically named correctly. The maximum absolute row sum is \(\|L\|_\infty\) and a Gershgorin spectral-radius bound; it is not generally an upper bound on the Euclidean \(\|L\|_2\) for nonsymmetric \(L_N\). Call it an infinity-norm mesh scale, or use
\(\sqrt{\|L\|_1\|L\|_\infty}\) as a valid 2-norm bound. The separate \(\|\Phi\Lambda\|_F\)-normalized 1% control is genuinely N-independent and fires.

5. Eleven notes — NEEDS-RESTATEMENT

Items 1–4, 6, 8, and 10 have valid reasons. Item 3 is sound because the additional exactly compatible eigenmode datum closes the bump’s merely approximate compatibility.

Items needing qualification:

- 5: fix the norm claim described above.
- 7: the gate is valid, but “an algebraic error would be \(O(\Delta t^2)\)” is not universal; cite the measured mutation controls instead.
- 9: it is a phase-2 claim not substantiated by the scoped files; “coefficient error equals M-norm field error” also requires an explicitly M-orthonormal bank.
- 11: restate F4 overflow behavior as “overflow aborts/fails”; it cannot currently record a nonfinite tail.

The generated table agrees with both JSON files. However, the assertion that every number in the prose is carried by the table is false: examples include the historical 40× ratio, old thresholds, 4000 solves, and \(O(\Delta t^2)\sim10^{-7}\). Those are hand-maintained historical/design numbers under the question’s definition.

6. Phase-1 verdict — CORRECT

All declared numerical gates and applicable controls pass for both BCs at both requested resolutions. The F4 and norm issues above are wording/generalization defects, not blockers for the finite recorded results. A nonblocking stale docstring also remains: [`wav2d_common.py`](</home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_common.py>) still says design r2.

| item | verdict | one-line fix |
|---|---|---|
| 1. V1alg | CORRECT | No numerical fix; describe it as an empirical LU-roundoff certification. |
| 2. V1cg | CORRECT | Only record an actual rollout-step residual if claiming per-step convergence monitoring. |
| 3. Controls | NEEDS-RESTATEMENT | State that F4 overflow aborts, or bypass the inner finite precondition for its diagnostic run. |
| 4. F0a/F0b | NEEDS-RESTATEMENT | Relabel the row sum as an infinity-norm scale or use \(\sqrt{\|L\|_1\|L\|_\infty}\). |
| 5. Notes 1–11 | NEEDS-RESTATEMENT | Qualify items 5, 7, 9, and 11 and remove the “nothing hand-typed” assertion. |
| 6. Phase 1 | CORRECT | No blocking fix; phase 1 is earned at both N and both BCs. |