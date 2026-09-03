Overall: the reduced equations are largely correct, but phases 2–3 are not presently certifiable. Several gate defects can create both unearned PASSes and spurious FAILs.

1. **CORRECT — Arm A algebra.**  
   [`ArmA`](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_rom.py:130) implements the stated general and first-step residuals exactly. Its Jacobian is \((B-aA+sC)J_h\). `build_tables` has the correct mass weighting, signs and transposes for \(B,A,C,M_r,K_r,D_r\). No sign, factor, or transpose error found.

2. **NEEDS-RESTATEMENT — Arm C dynamics correct; convergence/energy certification is not.**  
   The Verlet equation, Jacobian, half-stiffness first step, \(dtJ\dot z_0\) kick, and coefficient-space least squares are correct because \(G^\top MG=I\). The central-difference energy formula is also correct, but [`energy_reduced`](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_rom.py:235) returns only interior times; the gates incorrectly treat \(E(dt)\) as \(E(0)\) and \(E(4T-dt)\) as \(E(4T)\). The Newton fallback is not demonstrably “roundoff-level”: it uses \(\|F\|/\|F(z_{\rm predictor})\|\), not a physical backward-error scale. A \(10^{-8}\) fallback is defensible only after replacing that denominator with a term-based residual scale and recording both absolute and normalized residuals.

3. **CORRECT — independent POD controls.**  
   [`pod_verlet`](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_rom.py:302) has the correct damped recurrence and half-kick. `PodCN` has the correct three-level CN recurrence and dynamic-velocity energy update for the frozen \(v_0=0\) setup. Neither calls Arm A nor Arm C; sharing the independently constructed reduced operators is appropriate.

4. **WRONG — W5 is tautological.**  
   With the code’s definitions,
   \[
   \Delta E=v_{\!bar}^{T}M\Delta v+c^2dt\,v_{\!bar}^{T}K\bar u,
   \]
   so substituting the constructed \(R_m\) makes the reported balance identically zero for any \(u\)-history, up to roundoff. It certifies only polarization/algebraic consistency—not the ROM dynamics or momentum equation. Dropping flux produces exactly \(-c\,dt\,v_{\!bar}^{T}MD_Bv_{\!bar}\); that control merely certifies active boundary dissipation. Demote W5 to an accounting diagnostic and separately gate an independently assembled projected momentum residual. Arm C’s promised forced variational balance is also absent.

5. **WRONG — head formulas mostly right, but two required contracts are not implemented.**

   - The auto+vc JVP with free per-row \(\dot z\) is correct.
   - The tangent-residual identity is correct for full-rank \(J_h\), but unpivoted `np.linalg.qr` plus diagonal filtering is not rank-revealing. Use SVD or pivoted QR.
   - Both trajectory-RMS identities are correct.
   - [`sup_latents`](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/wav2d_head.py:44) maps time but passes \(\mu\) through unchanged. Thus it implements \((\tilde\mu,\tilde t)\) only if `mu` is already normalized—without checking that assumption.
   - The training losses are coefficient \(M\)-norm losses with separate trajectory weighting, not design r3’s stated energy-norm loss.

6. **WRONG — D1/G0a controls are invalid; D2 encoding itself is correct.**

   - D2’s `-min(cond) <= -1e-8` encoding is logically correct, and the duplicated-coordinate control should read zero.
   - D2 samples only about 2,048 training codes, rather than all training codes, and Arm A rollout points never receive D2 checks.
   - Shuffling rows does not meaningfully corrupt an autodecoder’s target point cloud: with free codes, the ideal objective is permutation-invariant. Therefore D1’s shuffled-target control need not fire.
   - A corrupted head need not exhibit a large train/held-out *gap*, so G0a’s shuffled-head control also need not fire.
   - The required FiLM comparator is explicitly not recomputed.
   - G0b applies the KE filter before medians, which is correct, but it simultaneously discards nonfinite states, violating the design’s “NaN anywhere is FAIL.” Its random-tangent control checks raw residual \(\ge0.9\), not whether the mutated normalized G0b quantity actually fails.

7. **WRONG — several phase-3 pass rules differ materially from design r3.**

   - W3 uses instantaneous POD-\(K\) projection floors (`podK_T`, `podK_4T`), not the independent POD-\(K\) CN rollout errors. This makes the STOP criterion spuriously strict.
   - Absorbing W3 omits the required \(4T\) oracle-floor and POD-\(K\) tests.
   - W3 chooses the RS with the smallest held-out error, introducing test-set post-selection.
   - Arm A “completion” checks only finite latents—not LM convergence or D2.
   - Arm C’s energy ratio uses \(E(dt)\) as its baseline.
   - W6’s spread formula matches the stated 20% rule, but its control is POD-\(K\) BE at RS 8 versus 40, not the prescribed RS=1 mutation of the checked ROM arm.
   - W4 neither requires W6 success nor asserts shrinking bounds under refinement. Its slope encoding is four times too strict, and `abs(nanmax(slopes))` can miss the most-negative secular drift.
   - W7’s primary comparison is correct. Its control does not assert that the selected projected velocity is nonzero and does not use the prescribed deterministic F3 traveling state.
   - G0c divides a median over four launch times by floors from only destination index 10. It also uses forbidden per-snapshot relative normalization, drops NaNs with `nanmedian`, and treats a genuinely nonfinite wrong-sign divergence as “control did not fire.”
   - The questioned `floor_T` trajectory argument is correct: each call contains one complete trajectory, so an all-zero trajectory-label vector is exactly appropriate.

8. **WRONG — additional certification holes remain.**

   - W0’s alleged independent full-grid path reuses `T["L"]`, the same matrix used to build \(A\), and tests only 32 random states—not captured solves or converged-state backward errors.
   - Arm A accepts any finite LM return, even after arbitrary residual stagnation.
   - D0 bypasses its independent-floor assertion whenever `stride > 1`; at \(N=128\) it can pass without certifying the independent POD subspace. It also compares residual norms rather than the required projection/round-trip fields.
   - `nanmedian`, `nanmax`, and G0b filtering can hide nonfinite trajectories.
   - No problematic absolute mesh-scaling pass threshold was found in these phase-2/3 metrics.
   - `HeadNP` matches `stk2d_head.apply_head` for the frozen `ff=0` architecture, including scale, weight orientation and SiLU derivative.
   - Smoke heads cannot be consumed by a certified run: filename separation and the checked `smoke` metadata both protect that path.

| item | verdict | one-line fix |
|---|---|---|
| 1 | CORRECT | None. |
| 2 | NEEDS-RESTATEMENT | Normalize Newton residuals by a physical backward-error scale and include true \(E(0)\)/endpoint energies. |
| 3 | CORRECT | None for the frozen \(v_0=0\) setup. |
| 4 | WRONG | Demote W5 to an accounting identity and add an independent projected-momentum/variational-balance gate. |
| 5 | WRONG | Use rank-revealing SVD/QR, explicitly normalize/assert \(\mu\), and reconcile training with the frozen energy-norm contract. |
| 6 | WRONG | Replace permutation-invariant shuffled controls, evaluate every D2 point, retain NaNs as failures, and compute FiLM. |
| 7 | WRONG | Rebuild W3 around actual POD-CN rollouts and repair W4/W6/G0c/control semantics before reading results. |
| 8 | WRONG | Make W0/D0 genuinely independent, require solver convergence, and prohibit all NaN-dropping aggregates. |

**Single most important defect:** W3 compares against the POD-\(K\) projection floor rather than the independent POD-\(K\) CN rollout. Because W3 is the STOP gate, the queued jobs can report a spurious structural FAIL even when the implemented ROM beats the required POD-\(K\) dynamical baseline.