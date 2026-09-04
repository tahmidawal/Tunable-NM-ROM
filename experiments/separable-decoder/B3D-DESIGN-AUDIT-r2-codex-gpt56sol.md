Overall verdict: **do not implement r2 unchanged.** It repairs many r1 defects, but the classical comparator is not valid as specified, several gates remain non-discriminating, and the named trainer is hard-wired to 2D.

All `L…` references below point to [B3D-DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/B3D-DESIGN.md:1).

## 1. Disposition of the 33 round-1 items

1. **RESOLVED** — “non-negative” replaces “positive,” and exactness is restricted to the non-negative cone (L14–27).
2. **RESOLVED** — DST-I normalization and mode-index ranges are explicit (L86–89).
3. **RESOLVED** — one forward and one inverse 3D DST, six axis transforms total, are stated (L91–96).
4. **RESOLVED** — the FOM now has interior-only unknowns and fixed zero ghosts (L69–74).
5. **RESOLVED** — the false Newton M-matrix claim is replaced by the negative-minimum root argument (L64–67).
6. **RESOLVED** — the premise now says 64× more unknowns and leaves cost empirical (L21–22).
7. **RESOLVED** — normalization uses one fixed 257-node reference grid (L57–59).
8. **RESOLVED** — the IC is smoothly masked and centers/widths are narrowed (L49–59).
9. **RESOLVED** — \(B=1\) is correctly called a single-blob 3D subfamily, and a dimensional-consistency gate is present (L60, L194), although that gate is defective under item 38 below.
10. **RESOLVED** — \(w_{\min}=0.10\) gives 3.2 cells per width and a nested-grid order gate was added (L49, L59, L193).
11. **PARTIAL** — “generic maximum dimension” and coincident-blob degeneracy are stated, but permutation identification and overlap/conditioning diagnostics remain absent (L60–62).
12. **PARTIAL** — persisted prefix-invariant draws are specified, but “one 576+8-row table” conflicts with the separate test seed and the reduced-\(N=129\) cohort is underspecified (L46–50, L114–116).
13. **PARTIAL** — L and STEP/ROLL are improved and the fake D3 control is removed, but D1 still shares the feature path, TB still repeats the same builder, and several asserted gates have no control (L175–216).
14. **RESOLVED-WRONGLY** — controls are moved after the initial state, but F5’s central-difference control is not guaranteed to fire, and “\(t\ge1\)” is ambiguous when \(T=0.25\) (L177–191).
15. **PARTIAL** — F2 and F4 now use backward-error normalization, but TB still has an unconditional relative threshold rather than a condition/summation-aware bound (L188–190, L216).
16. **RESOLVED-WRONGLY** — T0-truth replaces the bypass, but it tests truth rather than the decoded fields on which the tensor acts and is therefore not the required certification (L218–220, L248–250).
17. **RESOLVED-WRONGLY** — F8 was added, but its stated negative control multiplies an exactly zero middle-plane term and cannot fire (L194).
18. **PARTIAL** — E1 adds direct field and latent comparisons, but residual, Jacobian, and \(J^\top r\) discrepancies remain recorded rather than gated (L220–227).
19. **PARTIAL** — rejected and accepted candidates are now instrumented, but only for two of eight trajectories and with no fidelity threshold (L221).
20. **RESOLVED** — normalized first-order optimality is required and unresolved steps are censored (L168–169, L229).
21. **PARTIAL** — D4 adds a held-out multistart representation fit and an absolute threshold, but its starts, budget, convergence evidence, comparator, and usefulness bar are inadequate (L206).
22. **RESOLVED** — representation quality and ROM-over-oracle excess are separated, and “within 3×” replaces “on its floor” (L232, L255).
23. **RESOLVED-WRONGLY** — a pilot exists, but it selects architecture on the final test cohort and can promote the larger model without requiring that it pass D3, D4, and M-stability (L128–132).
24. **RESOLVED** — every resolution uses 16,384-point draws, no full-grid finishing, and full-grid held-out validation is gated (L134–143).
25. **PARTIAL** — host/device peaks and phase timing are added, but the gate extrapolates from \(N=65\), lacks a host-memory cap, and does not compile the actual \(N=129\) shapes (L197, L241–244).
26. **RESOLVED** — the whole \(N=65\) pipeline is decomposed and a 20-hour projected limit gates the 24-hour job (L197).
27. **PARTIAL** — the three reduced kernels move to one GPU, but C1 has no quantitative flatness rule and excludes the IC/decode costs used by C2 (L171–173, L230).
28. **RESOLVED-WRONGLY** — FFT-DST and defect correction are added, but the requested validated cubic-history correction is replaced by linear extrapolation and an undamped Picard iteration that is not contractive in the declared regime (L95–112).
29. **RESOLVED** — a two-sided accuracy bracket is mandatory and unbracketed results are explicitly Pareto-only (L110–112, L261).
30. **UNRESOLVED** — all-eight-wins plus median \(>1.1\) is not a sampling-uncertainty rule; the reported clustered fifth percentile is not required to exceed one (L231).
31. **PARTIAL** — the decision table is much more complete, but its positive row still relies on defective T0, C1, C2, D4, and classical-baseline prerequisites (L246–263).
32. **RESOLVED** — the one-seed/eight-trajectory scope is explicit in both setup and positive conclusion, and exponent claims are prohibited (L9–10, L35, L254).
33. **PARTIAL** — completion and oracle distinctions become hard gates, but decoded positivity, candidate-path fidelity, and timing uncertainty remain diagnostics rather than preconditions (L218–233, L248–263).

## 2. New round-2 findings

### 34. WRONG — the proposed Picard map is not uniformly convergent

For
\[
R(u)=Hu-u^{n}+\Delta t\,N(u),\qquad H=I+\Delta t\,\nu(-L),
\]
the linearized error iteration is
\[
e_{k+1}=-\Delta t\,H^{-1}J_N(u^\star)e_k.
\]
At a locally constant positive state \(U\), for an equal-wavenumber three-axis Fourier mode with \(s=\sin(\theta/2)\),
\[
q(s)=\frac{6Cs}{1+12rs^2},\qquad
C=\frac{U\Delta t}{\Delta x},\quad r=\frac{\nu\Delta t}{\Delta x^2}.
\]
At \(U=2,\nu=0.01,\Delta t=0.005\), \(q_{\max}\approx1.189\) at \(N=33\) and approximately \(1.225\) at \(N=65,129\). The omitted \(\operatorname{diag}(D^-u)\) part does not restore a general bound. Cell Péclet number alone therefore does not make this a contraction.

**Fix:** use a damped/residual-monotone correction or treat one Helmholtz correction as a fixed-cost heuristic, not a tolerance-convergent solver.

### 35. NEEDS-RESTATEMENT — “extrapolation only” is not a defined zero-work rung

Linear extrapolation requires two previous computed slices; no bootstrap is defined. If the first prediction is \(u^0\) and the rung accepts every predictor without correction, the entire trajectory remains \(u^0\). If a solved \(u^1\) is supplied, it is not zero-work. Extrapolation, copying, boundary enforcement, and output formation also have cost.

**Fix:** call it a “zero nonlinear-correction” rung, define its first step and self-generated history, and charge all predictor/output work.

### 36. CORRECT — the nested-grid and reference-grid construction is coherent

\(N=33,65,129\) gives \(\Delta x=2^{-5},2^{-6},2^{-7}\), and the 257-node normalization grid contains every one of these grids. Thus each discretization restricts the same continuously defined, reference-normalized IC.

**Fix:** no change; retain these node counts and the fixed 257-node normalization.

### 37. NEEDS-RESTATEMENT — using exactly the decoder mask does not make tensor gates tautological, but it biases D4

The identity \(m=\mathrm{bc}\) does not make TA, T0, FOMR, or E1 tautological: the finite bank and nonlinear head must still represent the quotient and dynamics. It does, however, make the eight \(t=0\) D4 states architecture-aligned by construction, contributing 25% of D4’s mean and making the new family easier specifically for this decoder.

**Fix:** disclose the architecture-matched IC and gate/report D4 both overall and with \(t=0\) excluded.

### 38. WRONG — F8 is underspecified and its control is identically inert

The residual identity is exact only on interior \(x,y\) rows. The actual 2D routine returns a full-grid residual with boundary rows \(R=u\) ([2D residual](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/wave2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py:109)). Its JVP agrees only when the tangent is also lifted as \(\delta v(x,y)p(z)\). Since \(p_{k-1}=p_k=p_{k+1}=1\), both the \(z\)-Laplacian and its lifted-tangent JVP are zero; multiplying the \(z\)-Laplacian coefficient by 1.01 changes nothing.

**Fix:** compare only interior 2D rows using a lifted tangent, and mutate an adjacent plateau value, the selected \(z\)-index, or a nonzero \(x/y\) coefficient.

### 39. WRONG — F5’s central-difference outcome is not guaranteed

At cell Péclet 6.25, central convection loses the M-matrix/monotonicity property—so undershoot is possible—but that does not imply this smooth masked Gaussian under backward Euler will reach \(\min u<-10^{-3}\) within 50 steps. Backward Euler and diffusion may suppress the oscillation.

**Fix:** use a prospectively verified admissible witness or a deterministic solver-output/stencil mutation whose negative value is guaranteed.

### 40. NEEDS-RESTATEMENT — F7’s order band is plausible asymptotically, not a reliable three-grid gate

First-order upwinding suggests \(p\to1\), but the observed two-difference order may exceed 1.3 when diffusion dominates or the leading advective error coefficient is small. The IC center, amplitude, viscosity, norm, and exact Richardson formula are also unspecified; the wrong-\(\nu\) control may accidentally remain inside the band.

**Fix:** fully specify the datum and estimator, justify the band with an independent manufactured solution, and use a deterministic coefficient/index mutation as the control.

### 41. WRONG — the capacity pilot leaks the final test cohort and lacks a “neither passes” branch

The pilot selects \(K,R,M\) using D4’s 32 states from test seed 1, then reports the final ROM and speed result on those same eight trajectories. Also, the larger model is promoted automatically even if it fails D3, D4, or M-stability.

**Fix:** perform promotion entirely on the 64 validation trajectories, require every promoted model to pass all three gates, and stop if neither passes before opening test seed 1.

### 42. WRONG — C2 is not an uncertainty-qualified speed-win rule

“All eight trajectory point estimates \(>1\)” and “median \(>1.1\)” constrain the observed cohort but do not quantify timing uncertainty. Five repetitions can still yield a clustered lower bound below one, which the design merely reports.

**Fix:** require the trajectory-clustered 95% lower confidence bound for paired speedup to exceed one, in addition to the cohort-consistency rule.

### 43. NEEDS-RESTATEMENT — C1 measures the reduced kernel, not the C2 end-to-end path

A bank-free run is faithful for the latent residual because \(A\) and \(Q\) already encode the bank, but it excludes the grid-dependent IC fit and 51-state decode charged by C2. “Same eight initial latents” is also ambiguous because independently trained checkpoints use unrelated latent coordinate systems. C1 has no flatness tolerance.

**Fix:** time each checkpoint’s actual 50-step latent path from its corresponding IC latent, predeclare a ratio/interval criterion, and label C1 strictly as kernel-only.

### 44. NEEDS-RESTATEMENT — D4 does not yet certify a useful or converged representation oracle

Eight starts are meaningless without their construction, attempt budget, termination/optimality, and budget-doubling stability. A mean oracle error of 0.05 can hide severe state failures and, combined with “within 3×,” permits a positive cost result at 15% ROM error—far looser than the 2D R=64 cell’s roughly 2.3% best rollout level ([2D accuracy](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/TENSOR2D-NOTES.md:109)). The inherited oracle routine is explicitly finite-budget and does not return an optimality certificate ([oracle LM](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/sep_solvers.py:639)).

**Fix:** define starts/budget/optimality and budget-doubling checks, select on validation, add worst-state and POD/comparator gates, and impose a separate absolute final-ROM usefulness threshold.

### 45. WRONG — T0-truth tests no tensor-arm requirement

Given F5, truth snapshots are non-negative, so sign-upwind equals fixed backward differencing by construction. But the tensor is evaluated on decoded fields, which the 2D evidence says are usually sign-changing. T0-truth neither exercises \(G\), \(DG\), \(Q\), the head, nor the off-cone solver path, yet it is made a headline precondition.

**Fix:** remove T0-truth as a tensor precondition; retain it as a redundant FOM scope check and gate decoded-field/operator fidelity through TA, constructed decoded witnesses, TQ/TR, and E1.

### 46. NEEDS-RESTATEMENT — the 120 GB M1 cap is only conditionally safe

On a nominal 141 GB H200, 120 GB leaves about 21 GB/15% headroom, which is defensible only for an actual compiled \(N=129\) peak including allocator and library workspaces. Extrapolating JAX memory statistics from \(N=65\) does not bound shape-dependent FFT/DST, oracle-Jacobian, compilation, or fragmentation peaks; projected host RSS also has no acceptance threshold.

**Fix:** add an \(N=129\) compile/allocation micro-pilot on the target GPU and explicit device-available and host-RSS headroom gates.

### 47. NEEDS-RESTATEMENT — the immutable cohort specification is internally ambiguous

L46 says one 576+8-row table is drawn from `default_rng(seed)`, whereas L114 assigns seed 0 to train/validation and seed 1 to test. At \(N=129\), the exact training prefix and continued fixed 64-row validation selection are not stated.

**Fix:** specify two immutable files—576 seed-0 train/validation rows and eight seed-1 test rows—and state the exact train-prefix and validation indices at every \(N\).

### 48. WRONG — the “every asserted gate has a negative control” contract is already violated

D3, TB, and F9 have numerical pass thresholds but no negative controls. Two implementations are a cross-check, not a mutation proving the F9 checker can fail. Several other controls—two Newton iterations, continuum eigenvalues, wrong viscosity, and central differencing—merely assume an empirical outcome.

**Fix:** provide deterministic mutations for D3/TB/F9 and prevalidate every empirical control witness before freezing the gate.

### 49. WRONG — `train_autodecoder_v2` cannot train the declared 3D decoder as written

The design names `train_autodecoder_v2` at L134–137, but that trainer calls `sep_common.features` directly in its loss, reconstruction, and code-polish paths ([trainer](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/sep_solvers.py:445)). That feature routine hard-codes the 2D mask \(16x(1-x)y(1-y)\), and the default initializer hard-codes a \(2\times n_{\rm ff}\) Fourier matrix ([2D features](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/sep_common.py:62)). `init_fn` can change the matrix shape but cannot replace the trainer’s 2D boundary factor.

**Fix:** parameterize the trainer with a `features_fn` used consistently in loss/reconstruction/polish, or implement a dedicated explicit-argument 3D trainer.

### 50. NEEDS-RESTATEMENT — “\(t\ge1\)” denotes an empty physical-time set

The experiment ends at \(T=0.25\), so F1 and F5’s \(t\ge1\) is empty if \(t\) has its governing-equation meaning; it evidently intends stored step index \(k\ge1\).

**Fix:** write \(k=1,\ldots,50\), equivalently physical time \(t\ge\Delta t=0.005\).

| item | verdict | one-line fix |
|---|---|---|
| 34. Picard contraction | WRONG | Replace undamped tolerance-Picard with a safeguarded scheme or label one correction as a fixed-cost heuristic. |
| 35. Extrapolation rung | NEEDS-RESTATEMENT | Define bootstrap and self-history, call it zero-correction, and charge predictor/output work. |
| 36. Nested grids/reference | CORRECT | No change; retain 33/65/129 and the fixed nested 257-node normalization. |
| 37. Mask equals decoder BC | NEEDS-RESTATEMENT | Disclose the architecture match and report/gate D4 with \(t=0\) excluded. |
| 38. F8 identity/control | WRONG | Restrict to interior rows and lifted tangents; mutate a nonzero term or plateau/index. |
| 39. F5 central control | WRONG | Use a deterministic or prospectively verified admissible negative witness. |
| 40. F7 order band | NEEDS-RESTATEMENT | Fully specify the datum/estimator and validate the band with an independent MMS. |
| 41. Pilot promotion | WRONG | Select on validation only, require both candidates to pass, and stop if neither does. |
| 42. C2 uncertainty | WRONG | Require the clustered 95% lower speedup bound to exceed one. |
| 43. C1 kernel timing | NEEDS-RESTATEMENT | Measure real corresponding latent paths, add a flatness rule, and keep C1 kernel-only. |
| 44. D4 oracle/usefulness | NEEDS-RESTATEMENT | Specify and certify the multistart solve and add comparator, worst-state, and final-error gates. |
| 45. T0-truth | WRONG | Demote it to an FOM scope check and gate the decoded/operator path instead. |
| 46. M1 memory | NEEDS-RESTATEMENT | Compile the actual \(N=129\) shapes and cap both device and host peaks with explicit headroom. |
| 47. Cohort definition | NEEDS-RESTATEMENT | Freeze separate seed-0 and seed-1 tables with exact train/validation indices. |
| 48. Missing controls | WRONG | Add deterministic negative controls for D3, TB, F9, and every empirical witness. |
| 49. 2D-only trainer | WRONG | Add an explicit 3D `features_fn` throughout the trainer or write a 3D trainer. |
| 50. Time-index ambiguity | NEEDS-RESTATEMENT | Replace \(t\ge1\) with step \(k\ge1\) or \(t\ge0.005\). |

**NO — r2 is not fit to implement as written because its specified `train_autodecoder_v2` path is hard-wired to the 2D boundary feature and therefore cannot train the declared 3D decoder.**