Overall verdict: **do not implement r1 unchanged.** The quadratic-tensor mathematics is sound, but the positivity justification, gate discipline, cross-resolution training protocol, and headline FOM comparison are not yet sufficient to support the proposed positive conclusion in [B3D-DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/B3D-DESIGN.md:1).

## Findings

1. **CORRECT — the 3D advection is one exact quadratic tensor.** For \(u=Gh\) and fixed backward differences,
   \[
   \Phi^\top[u\odot(D_x^-+D_y^-+D_z^-)u]=h^\top T h=\tfrac12h^\top(T+T^{jk})h,
   \]
   so a single \((M,R,R)\) tensor covers all three axes. This remains exact on the **non-negative** cone: at \(u_c=0\), the branch choice is multiplied by zero.  
   **Fix:** Replace “positive” with “non-negative”; retain the symmetrized \(Q\).

2. **CORRECT — the sine modes are exact eigenvectors.** On the \((N-2)^3\) interior grid with ghost-zero Dirichlet faces, tensor-product DST-I sine modes with \(p,q,r=1,\ldots,N-2\) exactly diagonalize the 7-point Laplacian.  
   **Fix:** State the index ranges and orthonormal normalization of \(S\).

3. **NEEDS-RESTATEMENT — the Helmholtz formula is correct, but the application count is not.** With \(L\Phi=-\Phi\Lambda\), the diffusion Jacobian is \(I+\Delta t\,\nu(-L)\), and
   \[
   \lambda_{pqr}=\frac4{\Delta x^2}\sum_{\alpha=p,q,r}\sin^2\!\frac{\pi\alpha}{2(N-1)}
   \]
   has the correct sign. But applying \(S\,D^{-1}S^\top\) requires a forward and inverse 3D DST: three axis transforms each way, not merely “three matmuls” total.  
   **Fix:** Say “two separable 3D DSTs, each comprising three one-axis transforms,” and require orthonormal \(S\).

4. **WRONG — the unknown/residual layout is internally inconsistent.** Lines 79–81 declare interior-only unknowns and simultaneously boundary residual rows \(R=u\). The DST inverse is exact only for the interior system; “identity on boundary rows” describes a different full-grid system whose interior rows may couple to boundary unknowns.  
   **Fix:** Use interior-only Newton unknowns with fixed zero ghosts, or fully specify and exactly invert the coupled full-grid block system.

5. **NEEDS-RESTATEMENT — non-negativity preservation is plausible, but the M-matrix/Newton claim is false.** A converged nonlinear root admits a discrete negative-minimum proof: at a negative global minimum, forward-upwind advection is non-positive and \(Lu\ge0\), contradicting \(u^{n+1}<0\) when \(u^n\ge0\). The Newton Jacobian is not generically an M-matrix because the reaction term \(\operatorname{diag}(Du)\) can destroy monotonicity, and Newton iterates themselves need not remain non-negative.  
   **Fix:** Replace the M-matrix argument with the negative-minimum proof, and describe F5 as checking the finite-tolerance solver output—not Newton iterates.

6. **WRONG — the opening 3D scaling prediction is numerically false.** From \(32^3\) to \(128^3\), the unknown count rises by \(4^3=64\), about 1.8 orders of magnitude, not three; actual cost also depends on the DST implementation and iteration count.  
   **Fix:** Replace “three orders of magnitude” with “64× more grid unknowns” and leave the cost growth empirical.

7. **NEEDS-RESTATEMENT — peak normalization is sound only at a fixed grid.** The grid maximum is interior for these positive Gaussian mixtures, so the discrete peak is \(A\); however, \(\max_{\rm grid}s\) changes with \(N\), so the self-convergence study starts from slightly different physical initial data. Equal peak and viscosity ranges also do not establish an equal dynamical regime for multi-scale, multi-blob gradients.  
   **Fix:** Normalize using one resolution-independent continuous/reference maximum and claim only matched peak/viscosity ranges.

8. **WRONG — the wall-overlap family creates an \(N\)-dependent one-cell boundary layer.** With \(c=0.15,w=0.20\), an unmasked Gaussian is still \(e^{-0.28125}\approx0.755\) of peak at the nearest wall, then is set abruptly to zero. The decoder’s smooth polynomial boundary mask must approximate an increasingly sharp layer as \(N\) grows—the same family defect recorded by the wave cell.  
   **Fix:** Multiply the Gaussian mixture by a smooth Dirichlet mask before peak normalization, or constrain centers/widths so wall values are negligible.

9. **WRONG — \(B=1\) is not a lifted 2D control.** A spherical 3D blob with a sampled \(c_z\), a 3D boundary mask, and \(z\)-advection/diffusion is not a \(z\)-invariant lift of the 2D problem; a nonzero \(z\)-invariant field is incompatible with the Dirichlet \(z\)-faces.  
   **Fix:** Call it the “single 3D blob subfamily” and add a genuine dimensional-consistency gate.

10. **NEEDS-RESTATEMENT — the finest narrow blobs are poorly resolved at \(N=32\).** For \(w=0.05\), \(w/\Delta x=1.55\) and the FWHM spans only about 3.65 cells. “Errors decrease” is too weak to distinguish convergence from under-resolution.  
    **Fix:** Require a smooth-data order gate, raise \(w_{\min}\) at \(N=32\), or begin the accuracy ladder at a mesh with at least roughly three cells per \(w\).

11. **CORRECT — \(5B+2\) is the generic intrinsic dimension including time.** There are \(4B\) center/width variables, \(B-1\) effective relative-amplitude variables after normalization removes common scaling, and \(A,\nu,t\): \(4B+(B-1)+3=5B+2\). It is only a generic dimension; coincident or near-coincident blobs lower local rank, and blob permutations identify parameters.  
    **Fix:** State “generic maximum dimension” and record overlap/conditioning rather than treating 17 as a hard manifold dimension.

12. **NEEDS-RESTATEMENT — fixed three-blob draws make the stream independent of realized \(B\), but not automatically of cohort size.** Always consuming all three blob records prevents a change in \(B\) from shifting \(A,\nu\). However, prior project failures show that vectorized parameter-array draws with different \(m\) need not have prefix invariance.  
    **Fix:** Generate and persist the full 576-trajectory raw parameter table once, then use exact prefixes at every \(N\).

13. **WRONG — several gates are self-comparisons or have non-discriminating controls.** In the inherited implementation, L compares two calls to the same `lin_of`; D1 shares the same `features` implementation; TB invokes the same builder twice; D3’s \(M=R/2\) control is rank-deficient by shape; F3 incorrectly cites F4 as its control; and a different stall constant need not change STEP/ROLL on the selected state.  
    **Fix:** Use independently assembled reference paths and preselect real-data mutation witnesses whose gate values demonstrably cross the threshold.

14. **WRONG — F1 and F5’s controls fire trivially at \(t=0\).** The displaced-blob F1 control is already asymmetric before evolution, and the negated-blob F5 control is already negative before the solver runs; neither tests the rollout or positivity-preservation machinery.  
    **Fix:** Exclude \(t=0\) from the controls and mutate the axis stencil/solver output on initially admissible data.

15. **WRONG — F2 repeats the forbidden mesh-scaling threshold.** Although normalized by \(\|\Phi\Lambda\|\), cancellation in \(L\Phi\) scales like \(\epsilon\|L\|\sim\epsilon\Delta x^{-2}\), exactly the wave-cell failure. TB’s summation-order discrepancy scales with chunk count/conditioning, and F4 should be judged by backward residual rather than forward error alone.  
    **Fix:** Use backward-error denominators such as \(\|L\|\|\Phi\|+\|\Phi\Lambda\|\), a condition-aware TB bound, and \(\|Hy-v\|/(\|H\|\|y\|+\|v\|)\) for F4.

16. **WRONG — T0 has a hidden bypass and its control is not guaranteed to fire.** The design explicitly permits zero all-positive decoded states and then proceeds, even though 2D had only 0.6–3% and 3D is expected to have fewer. Merely having negative points does not ensure a projected mismatch \(>10^{-8}\); zeros, locally symmetric stencils, or projection cancellation can defeat the control.  
    **Fix:** Make zero testable states a hard stop or separately certify sign-upwind versus backward difference on a constructed strictly positive field and use a deterministic asymmetric sign-changing control.

17. **WRONG — the necessary 3D-versus-2D consistency gate is missing.** Global \(z\)-invariance is impossible under Dirichlet boundaries, but a local exact reduction is available: lift a 2D state with a \(z\)-profile equal to one on three central planes and tapering to zero at the faces; on the middle plane, both \(z\)-advection and \(z\)-Laplacian contributions vanish exactly.  
    **Fix:** Compare the 3D residual and JVP on that middle plane against the actual 2D code, with a mutated \(z\)-index/coefficient control.

18. **WRONG — E1 cannot establish “exactness.”** Equality of aggregate error ratios plus identical stop histograms/attempt counts can pass while the tensor and oracle fields differ materially. A 1% error-ratio tolerance is “oracle-equivalent accuracy,” not exact operator equality.  
    **Fix:** Gate worst per-state field, latent, residual, Jacobian, and \(J^\top r\) discrepancies; reserve “exact” for the fixed-backward/non-negative-cone identity.

19. **WRONG — TQ/TR miss the solver states that can change the trajectory.** TR records only accepted full-arm latents, whereas rejected trial points and their Jacobians determine LM acceptance and damping. The 1D tensor audit instrumented every candidate; this design regresses from that discipline.  
    **Fix:** Record tensor/oracle \(r,J,J^\top r\), sign counts, and decisions at every initial, trial, accepted, and rejected LM candidate.

20. **WRONG — stall-dominated rollouts can still pass every accuracy gate.** The 2D cell stopped all 400/400 steps as `stalled`; identical stall histories carried little evidence. This design has no first-order optimality or censoring gate, so both arms can stop at the same poor point and pass E1.  
    **Fix:** Require normalized \(\|J^\top r\|/(\|J\|\|r\|)\) at every completed step and classify unresolved stalls as incomplete/censored.

21. **WRONG — D2 does not define the “held-out reconstruction floor” consumed by A1.** D2 checks training codes on 64 training states and records the subset/full discrepancy; it does not compute a held-out representation oracle. Moreover, “floor” is incorrect for a finite-budget latent fit, and \(<0.2\) permits a scientifically useless decoder.  
    **Fix:** Add converged multi-start full-grid oracle fits on held-out states, call them an oracle estimate, and impose an absolute usefulness/comparator gate.

22. **WRONG — A1’s causal reading is reversed.** A large ROM/oracle ratio indicates objective, solver, time-stepping, test-space, or IC failure—not “head generalisation.” Head generalisation determines the held-out oracle error itself. Also, a ratio of 3 does not justify “sits on its floor.”  
    **Fix:** Separate held-out oracle quality from ROM/oracle excess and say “within 3×” unless the threshold is tightened.

23. **NEEDS-RESTATEMENT — the size bundle is only partly justified.** \(g_{\rm hidden}=256\) safely avoids an \(R=128\) rank cap, \(m=4M\) is a defensible sampled-control budget, and \(Q\) is 32 MiB. But \(K=32\) is not justified by physical dimension for an auto-decoder, while \(R=128\) and \(M=256\) have no capacity or \(M\)-convergence pilot on this harder family.  
    **Fix:** Predeclare an \(N=32\) \(K/R/M\) pilot or diagnostic ladder and promote the smallest configuration passing held-out oracle, rank, and \(M\)-stability gates.

24. **WRONG — the \(N=128\) training subset changes the objective and its stated count is inaccurate.** \(64^3=262{,}144\), but the \(N=64\) interior has \(62^3=238{,}328\) points. At \(N=128\), training uses only 13.1% of the interior, while the smaller meshes receive full-grid finishing steps; cross-\(N\) accuracy is therefore confounded. D2 merely records the discrepancy.  
    **Fix:** Use the same spatial sampling measure across \(N\), plus a frozen independent stratified/full-grid validation set with a required subset-to-full generalization ratio.

25. **NEEDS-RESTATEMENT — the memory arithmetic is broadly plausible, but the peak budget is incomplete.** The bank is 2.00 GiB, training targets 16.0 GiB, tensor chunk 512 MiB, tensor 32 MiB, and one 51-state decode 816 MiB. Omitted are \(\Phi\) at about 3.82 GiB, duplicate full/interior banks and \(DG\), 6.38 GiB of eight test rollouts, NNLS work arrays, and tens of GiB of training predictions/cotangents. The cluster’s recorded H200 has about 141 GB device memory; `--mem 240G` is host RAM.  
    **Fix:** Add measured host/device peak-memory gates and a compile/training pilot; explicit arguments prevent captured constants but do not reduce activation memory.

26. **NEEDS-RESTATEMENT — 24 hours is plausible as a reservation, not yet justified.** Phase 0 times only the generator; it does not bound 60k training steps, the 8–16 GiB NNLS assembly, two tensor builds, oracle/TQ audits, full-arm compilation, or the 14-rung ladder.  
    **Fix:** Use a complete \(N=64\) timing decomposition to project the \(N=128\) wall time and hard-stop before submission if the upper bound exceeds 24 hours.

27. **WRONG — C1 cannot support the claimed “flat in \(N\)” result.** Each \(N\) runs on a different GPU, C1 has no pass rule, and tensor/full ratios do not measure cross-\(N\) flatness. Iteration counts can also vary even though one residual evaluation has \(N\)-independent dimensions.  
    **Fix:** Either run all three tensor kernels/checkpoints on one GPU or restrict the claim to identical reduced shapes/FLOP counts and separately report hardware-free LM attempts.

28. **WRONG — the Newton–BiCGStab ladder is not sufficient as the sole strong FOM denominator.** It is a legitimate baseline, but the exact Helmholtz inverse makes one-shot or few-shot full-grid residual correction/Picard iteration from cubic history an obvious cheaper competitor already validated elsewhere in this project. Dense axis matmuls must also be compared with an FFT-DST implementation.  
    **Fix:** Add cubic-history exact-Helmholtz defect correction and FFT-DST/Picard variants, then select the fastest accuracy-eligible classical arm.

29. **WRONG — “matched accuracy” is undefined when the ladder does not bracket the ROM.** If every tested FOM rung is substantially more accurate, choosing the cheapest qualifying rung leaves open a still-looser, faster FOM; it proves neither equal-error timing nor a crossover. If no rung is accurate enough, `match=None` is also absent from the decision table.  
    **Fix:** Require an error bracket around every ROM point, extend the ladder with zero-work/history and intermediate tolerances, or report only Pareto dominance—not matched accuracy.

30. **NEEDS-RESTATEMENT — the timing protocol lacks a positive-result uncertainty rule.** Raw repetitions and pairing are good, but three pair repetitions over eight trajectories, with no clustered interval or outlier rule, allow a noisy point estimate just above one to trigger C2. Paired cost and accuracy must also come from the paired invocation itself.  
    **Fix:** Require immediate burn/reburn, retained raw arrays, outlier counts, trajectory-clustered intervals, and a lower confidence bound \(>1\) for a speed-win claim.

31. **WRONG — the decision table is incomplete and can declare an unsupported positive.** It has no rows for phase-2 failure/T0 untestable, mixed E1 across \(N\), incomplete or non-optimal rollouts, no FOM bracket, failed full/subset generalization, unavailable oracle, or C1 not established. Its positive row requires neither useful absolute accuracy nor a strong classical denominator, yet says “exact, flat, and beats the FOM.”  
    **Fix:** Add those outcomes and require direct fidelity, completion/optimality, useful held-out accuracy, same-GPU flatness evidence, a bracketed strongest-FOM comparison, and uncertainty before the positive row.

32. **NEEDS-RESTATEMENT — seed and scope disclaimers do not sufficiently constrain the final claim.** One seed and eight test trajectories are disclosed early, but the positive decision row reads generically; no cross-\(N\) exponent is promised, yet “flat in \(N\)” can still slip through.  
    **Fix:** Force every result-row label to say “seed 0/test seed 1, eight trajectories,” and prohibit exponent/flatness/generalization language without its dedicated gate.

33. **WRONG — several [TENSOR2D-NOTES.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/TENSOR2D-NOTES.md:368) landmines remain operationally ignored.** Specifically: scarce all-positive decoded states, candidate-level undershoot auditing, stall-dominated termination, the distinction between full-quadrature oracle and representation oracle, large accuracy gaps to matched FOM rungs, and single-seed/no-CI limitations. Explicit-argument training and direct-form gate A are correctly carried forward.  
    **Fix:** Promote the unresolved landmines from recorded diagnostics to hard preconditions or explicit negative-result rows.

## Summary table

| item | verdict | one-line fix |
|---|---|---|
| 1. Quadratic tensor | CORRECT | Say non-negative cone; keep one symmetrized \((M,R,R)\) tensor. |
| 2. Sine eigenmodes | CORRECT | State \(p,q,r=1,\ldots,N-2\) and orthonormal DST normalization. |
| 3. DST preconditioner | NEEDS-RESTATEMENT | Specify forward plus inverse 3D DST: six axis transforms total. |
| 4. Boundary layout | WRONG | Choose interior-only unknowns or fully specify the coupled boundary block. |
| 5. Positivity | NEEDS-RESTATEMENT | Use the discrete minimum proof; do not claim an M-matrix Newton iteration. |
| 6. Cost-growth premise | WRONG | Replace “three orders” with “64× more unknowns.” |
| 7. Peak normalization | NEEDS-RESTATEMENT | Use a resolution-independent maximum and narrow the regime claim. |
| 8. Wall overlap | WRONG | Smoothly mask before normalization or keep blobs negligible at walls. |
| 9. “Lifted 2D” control | WRONG | Rename it single-blob 3D and add a true consistency test. |
| 10. Narrow-blob resolution | NEEDS-RESTATEMENT | Enforce a cells-per-width condition or strengthen the convergence gate. |
| 11. Intrinsic dimension | CORRECT | Call \(5B+2\) the generic maximum and record degeneracies. |
| 12. RNG stream | NEEDS-RESTATEMENT | Persist one maximal raw draw and take prefixes at every \(N\). |
| 13. Self-comparison gates | WRONG | Replace shared paths and theorem-by-shape controls with independent mutations. |
| 14. F1/F5 controls | WRONG | Exclude \(t=0\) and mutate evolution on admissible initial data. |
| 15. Mesh-scaled tolerances | WRONG | Use backward-error and condition-aware normalizations. |
| 16. T0 bypass | WRONG | Hard-stop on zero states or certify the branch with constructed fields. |
| 17. 2D/3D consistency | WRONG | Add the compact plateau-in-\(z\) central-plane residual/JVP gate. |
| 18. E1 exactness | WRONG | Gate direct field/latent/operator discrepancies and avoid the word “exact.” |
| 19. Candidate-path audit | WRONG | Instrument every LM trial, rejection, acceptance, residual, and Jacobian. |
| 20. Stall completion | WRONG | Require first-order optimality and treat unresolved stalls as censored. |
| 21. Held-out oracle | WRONG | Add full-grid multistart held-out oracle fits and an accuracy gate. |
| 22. A1 interpretation | WRONG | Separate head oracle quality from ROM-over-oracle excess. |
| 23. \(K,R,M,m\) choices | NEEDS-RESTATEMENT | Run a frozen capacity/test-space pilot before promotion. |
| 24. \(N=128\) subset | WRONG | Use a common sampling measure and gate independent full-grid generalization. |
| 25. Memory | NEEDS-RESTATEMENT | Include \(\Phi\), test truth, duplicate banks, NNLS, and AD peaks in a measured gate. |
| 26. Wall time | NEEDS-RESTATEMENT | Project all phases from a complete \(N=64\) timing decomposition. |
| 27. Flatness | WRONG | Measure on one GPU or claim only \(N\)-independent reduced arithmetic. |
| 28. FOM baseline | WRONG | Add cubic-history Helmholtz correction, Picard, and FFT-DST controls. |
| 29. Accuracy matching | WRONG | Require a bracket or label the result Pareto/dominance-only. |
| 30. Timing inference | NEEDS-RESTATEMENT | Require clustered intervals, outlier counts, balanced burns, and same-invocation accuracy. |
| 31. Decision table | WRONG | Add all missing failure/inconclusive rows and strengthen the positive prerequisites. |
| 32. Seed/exponent scope | NEEDS-RESTATEMENT | Put the one-seed/eight-case scope directly into every conclusion. |
| 33. 2D landmines | WRONG | Turn undershoot, candidate fidelity, censoring, and oracle distinctions into gates. |

The single most important defect is that C2 can certify a headline “FOM win” without an accuracy bracket and without the obvious cubic-history, exact-Helmholtz defect-correction/FFT-DST classical baseline.