# Wave 2D mechanism cell — notes (what ran, what was found, what was retracted)

Branch `exp/2026-09-03-wave2d-mechanism`; design `WAVE2D-DESIGN.md` (r3), its two Codex design
audits, and the code/result verifications `WAVE2D-PHASE1-VERIFY-r*-codex-gpt56sol.md`. **Every
result number in this file is spliced in by `wav2d_tables.py` from `runs/wav2d/*.json`.** The
retraction list below also carries hand-maintained *design* numbers (old thresholds, step counts,
order-of-magnitude estimates); those are not results and are labelled as such by context.
Retractions and design amendments are numbered and never deleted.

## Retractions and amendments (newest last)

Result numbers quoted here are also in the generated table below; the text names the gate, the table
carries the value. Old thresholds and design estimates in this list are hand-maintained.

1. **Retraction 1 (superseded by 4) — V1 threshold 1e-11 → 1e-9.** The u-only recurrence vs a block
   solve read above 1e-11 with CG at 1e-13; the design's threshold ignored iterative-solver
   accumulation over 4000 solves.
2. **Amendment 1 — F2 spatial study on nested grids.** The design's $N\in\{64,128,256\}$ vs 512 are
   not nested ($N-1$ = 63, 127, 255, 511); the study uses $N\in\{33,65,129\}$ vs 257.
3. **Amendment 2 — F2 spatial study on a wall-compatible datum, family blob reported.** The inherited
   family's blobs are chopped to zero at the walls by the hard mask (reflective) or leave
   $\partial_n u_0 \ne 0$ with $v_0 = 0$ (absorbing), so the widest family blob converges at reduced
   order — **a property of the inherited data, not of the scheme** (see the table's "family" rows).
   The gate uses a centred $w=0.1$ bump *and* (after verification round 1) an exactly compatible
   two-eigenmode sum; the family blob's order is reported beside them. Thresholds unchanged.
4. **Amendment 3 — F2 temporal control.** "Backward Euler reads order $1\pm0.3$" was first replaced
   by a separation ratio (BE $\ge 10\times$ worse at the finest step) because BE's fitted order
   saturated on the family blob; verification round 1 pointed out that on the smooth datum BE does
   read order ~1, so the control now requires **both** (order $1\pm0.3$ and separation $\ge10$).
5. **Retraction 2 — F0a/F0b eigenvector-residual normalisation.** $\|L\Phi+\Phi\Lambda\|_F/\|\Phi\Lambda\|_F$
   exceeded 1e-13 at N=128 after passing at N=64: the stencil cancels O(1) values to produce
   O($\lambda$), amplifying roundoff by $\sim 1/(k\pi\Delta x)^2$ — an absolute threshold on a
   mesh-scaling quantity, the failure the design forbids. Now the backward-error form
   $\|L\Phi+\Phi\Lambda\|_F/(\|L\|\,\|\Phi\|_F)$ with $\|L\| = \sqrt{\|L\|_1\|L\|_\infty} \ge \|L\|_2$ (a
   valid 2-norm bound for the non-symmetric $L_N$; verification round 1 rejected "$8/\Delta x^2$ is
   the 2-norm" and round 2 rejected the plain row sum, which is $\|L\|_\infty$); the control keeps the
   $\lambda$-normalised form, which is N-independent.
6. **Retraction 3 (superseded by 4) — V1 threshold 1e-9 → 1e-8 with a two-tolerance ratio.**
   Verification round 1 rejected the ratio as evidence ("$d = d_{\rm alg} + d_{\rm CG}(\tau)$ can give a
   ratio $\ge 10$ with $d_{\rm alg}\ne0$") and found the reference was not a block solve at all but
   the same eliminated algebra with LU — an effective self-comparison. It also caught that the
   "10³–10⁴×" ratio quoted in an earlier version of this note was **hand-typed and wrong** (the
   JSON had 40× for absorbing N=128). Both the gate and the sentence were replaced.
7. **Retraction 4 — V1 rebuilt as V1alg + V1cg (2026-09-03, after verification round 1).**
   V1alg: the u-only recurrence solved by LU vs the **true $2n\times2n$ block CN** solved by LU
   (no elimination, no iterative solver), gated over the **first 10 steps** where LU roundoff is
   ~1e-14 — an empirical LU-roundoff certification whose sensitivity is shown by the mutation
   controls in the table (they read 1e-4…1e-2 over the same 10 steps); the full-horizon LU-vs-LU
   discrepancy (~1e-11, LU roundoff over 4000 solves) is reported, not gated — the 1e-12
   full-horizon threshold I first set was the same mesh/accumulation mistake a fourth time.
   V1cg: the JAX/CG recurrence vs the LU recurrence at CG tolerances 1e-9, 1e-11, 1e-13 must
   decrease monotonically to $\le$ 1e-8, and the achieved CG relative residual at 1e-13 is
   recorded ($\le$ 1e-12). The algebra is now certified tolerance-free.
8. **Amendment 4 — F2 temporal study on the smooth datum.** The family blob's temporal order is
   pre-asymptotic at N=128 (its wall discontinuity excites the high modes whose CN phase error
   dominates at coarse $\Delta t$); gated on the smooth bump, family reported.
9. **Amendment 5 — head training loss (phase 2; not yet exercised by a certified run).** The design's
   "every metric, loss and residual block is in the energy norm" is applied to *metrics*; the
   training loss is the coefficient error with per-trajectory weights, and the vc arm's velocity
   term likewise (`wav2d_head.py` header). Coefficient error equals the $M$-norm field error only
   because the bank is $M$-orthonormal, which gate D0 asserts.
10. **Amendment 6 — F3 pass rule.** "slope $4\pm0.5$ **or** coefficient agreement" could pass an
    absorber with the right rate and a wrong coefficient (verification round 1); now **both**.
    Corroborating diagnostics (plateau two snapshots later, $y$-invariance, mean-field remainder)
    are recorded in the JSON.
11. **Fixed after verification rounds 1–2 (not threshold changes):** the F4 absorbing control no
    longer converts an overflowing anti-damped run into a passing value — it is read from the
    *finite prefix* of the diagnostic run (max growth over the finite steps) and a nonfinite tail is
    recorded, not valued (round 2 found the first version of this path unreachable behind a
    finiteness precondition; the precondition now guards only the main trace); F0b's zero mode, F0d (reflective),
    F1a-form and F2-spatial gained the negative controls they lacked; F0d-spd is also evaluated at
    the design's N=32; V0's control is described honestly (every Laplacian coefficient is perturbed,
    not one).
12. **Retraction 5 — D1 control (2026-09-03, seen in the first N=64 cluster logs).** "A head trained
    on shuffled targets must be worse than POD-K" cannot fire: for the free-code arms a row shuffle
    is not a mutation at all (the codes are per row, so the shuffled problem is the same problem
    with permuted labels), and even for `sup` a smooth K-manifold fitted *per snapshot* by the
    oracle captures about what POD-K does regardless of what it learned. The control is now an
    **untrained (random-init) head**: the trained head's held-out oracle must beat it by ≥ 1.3×
    (a capacity baseline). The shuffled head is still reported. The N=64 numbers that exposed
    this are in the logs of jobs 3225935/3225937 and will be superseded by the re-run.
13. **Retraction 6 — G0a control.** "The shuffled head's held-out–train gap must exceed 0.05"
    cannot fire: a head that learned nothing has no generalisation gap by construction. The control
    is now an **overfit head** (the same arm trained on 4 trajectories), which must show ratio > 1.5
    or gap > 0.05. **Both retractions are of the "control that cannot fail" kind the design forbade
    — they slipped through two design audits and the N=16 smoke (where nothing is learned and every
    head looks alike), and were caught only by real N=64 numbers.**
14. **Finding, not a retraction — the oracle metric is capacity-dominated for auto-decoders.** At
    N=64 reflective the `auto` head's held-out oracle (see the phase-2 table) is essentially the
    08-16 auto-decoder's 0.189, and a head trained on *shuffled* targets reaches the same value:
    with K free parameters per snapshot the oracle projection measures the manifold's dimension,
    not what training put into it. D1's bar (≤ 0.5 × POD-K) is therefore the right kind of bar,
    and G0b (tangent-space residual at the oracle point) is the gate that sees learned structure.

## Phase 1 — FOM gates (both BCs) — generated table

<!-- phase1-table -->
| N | BC | gate | value | pass rule | negative control | verdict |
|---|---|---|---|---|---|---|
| 128 | ref | F0-stencil | 1.82e-16 | ≤ 1e-13 | 2.54e-08 | PASS |
| 128 | ref | F0a | 1.41e-16 | ≤ 1e-13 | 4.70e-03 | PASS |
| 128 | ref | F0d-sym | 0 | ≤ 1e-15 | 1.26e-03 | PASS |
| 128 | ref | F1a | 6.94e-15 | ≤ 1e-10 | 0.598 | PASS |
| 128 | ref | F1a-form | 1.54e-16 | ≤ 1e-14 | 7.89e-03 | PASS |
| 128 | ref | F4 | 0 | ≤ 1e-10 | 0.598 | PASS |
| 128 | ref | V1alg | 1.80e-14 over 10 steps; full horizon 1.04e-11 | ≤ 1e-13 | 1.03e-03 | PASS |
| 128 | ref | V1cg | CG ladder 2.70e-05, 2.40e-08, 3.41e-11 (monotone True); achieved CG resid 6.11e-14 | ≤ 1e-08 | — | PASS |
| 128 | ref | F2-spatial | bump order 2.1892 (errors 0.052, 0.013, 2.52e-03); two-mode order 2.1943; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3335 | PASS |
| 128 | ref | F2-temporal | order 2.0104 (errors 8.14e-04, 2.03e-04, 5.02e-05) | see design | BE order 0.921, separation 575.00x | PASS |
| 128 | ref | V0 | 3.36e-15 | ≤ 1e-13 | 9.46e-06 | PASS |
| 128 | abs | F0-stencil | 2.02e-16 | ≤ 1e-13 | 2.81e-08 | PASS |
| 128 | abs | F0b-zero-mode | 2.93e-18 | ≤ 1e-13 | 6.87e-08 | PASS |
| 128 | abs | F0b | 8.03e-17 | ≤ 1e-13 | 5.25e-03 | PASS |
| 128 | abs | F0d-sym | 0 | ≤ 1e-15 | 0.056 | PASS |
| 128 | abs | F0d-spd | min eig/max(M) = 0.271 | see design | — | PASS |
| 128 | abs | F0c | 2.18e-14 | ≤ 1e-12 | 0.058 | PASS |
| 128 | abs | F1b | 7.98e-14 | ≤ 1e-10 | 3.13e-05 | PASS |
| 128 | abs | F4 | 0 | ≤ 1e-10 | 2.24e+45 | PASS |
| 128 | abs | V1alg | 1.74e-14 over 10 steps; full horizon 3.73e-12 | ≤ 1e-13 | 9.68e-03 | PASS |
| 128 | abs | V1cg | CG ladder 1.55e-05, 5.44e-08, 1.36e-09 (monotone True); achieved CG resid 2.07e-15 | ≤ 1e-08 | — | PASS |
| 128 | abs | F2-spatial | bump order 2.2151 (errors 0.026, 6.02e-03, 1.20e-03); two-mode order 1.8042; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3419 | PASS |
| 128 | abs | F2-temporal | order 2.0107 (errors 3.97e-04, 9.91e-05, 2.45e-05) | see design | BE order 0.957, separation 576.25x | PASS |
| 128 | abs | F5 | -4.51e-15 | ≤ 1e-12 | 2.828 | PASS |
| 128 | abs | F3 | slope 4.0096; fraction/prediction 1.0215, 1.0052, 1.0013, 1.0003 at N [64, 128, 256, 512]; plateau/fraction 1.0000, 1.0000, 1.0000, 1.0000; y-var 5.5e-15, 4.2e-15, 6.0e-15, 6.2e-15 | see design | reflective retains 1.0000 | PASS |
| 64 | ref | F0-stencil | 2.43e-16 | ≤ 1e-13 | 3.33e-08 | PASS |
| 64 | ref | F0a | 1.20e-16 | ≤ 1e-13 | 4.70e-03 | PASS |
| 64 | ref | F0d-sym | 0 | ≤ 1e-15 | 2.55e-03 | PASS |
| 64 | ref | F1a | 2.49e-15 | ≤ 1e-10 | 0.387 | PASS |
| 64 | ref | F1a-form | 1.19e-16 | ≤ 1e-14 | 0.015 | PASS |
| 64 | ref | F4 | 4.44e-16 | ≤ 1e-10 | 0.387 | PASS |
| 64 | ref | V1alg | 1.75e-14 over 10 steps; full horizon 2.96e-11 | ≤ 1e-13 | 2.80e-04 | PASS |
| 64 | ref | V1cg | CG ladder 7.71e-07, 7.49e-07, 1.62e-10 (monotone True); achieved CG resid 3.61e-16 | ≤ 1e-08 | — | PASS |
| 64 | ref | F2-spatial | bump order 2.1892 (errors 0.052, 0.013, 2.52e-03); two-mode order 2.1943; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3335 | PASS |
| 64 | ref | F2-temporal | order 2.0104 (errors 8.09e-04, 2.02e-04, 4.98e-05) | see design | BE order 0.921, separation 576.87x | PASS |
| 64 | ref | V0 | 2.35e-15 | ≤ 1e-13 | 8.09e-06 | PASS |
| 64 | abs | F0-stencil | 2.25e-16 | ≤ 1e-13 | 3.08e-08 | PASS |
| 64 | abs | F0b-zero-mode | 2.90e-18 | ≤ 1e-13 | 2.80e-07 | PASS |
| 64 | abs | F0b | 7.54e-17 | ≤ 1e-13 | 5.25e-03 | PASS |
| 64 | abs | F0d-sym | 0 | ≤ 1e-15 | 0.079 | PASS |
| 64 | abs | F0d-spd | min eig/max(M) = 0.260 | see design | — | PASS |
| 64 | abs | F0c | 8.41e-15 | ≤ 1e-12 | 0.058 | PASS |
| 64 | abs | F1b | 7.01e-14 | ≤ 1e-10 | 1.51e-05 | PASS |
| 64 | abs | F4 | 0 | ≤ 1e-10 | 1.48e+20 | PASS |
| 64 | abs | V1alg | 1.61e-14 over 10 steps; full horizon 1.64e-11 | ≤ 1e-13 | 2.10e-03 | PASS |
| 64 | abs | V1cg | CG ladder 1.96e-06, 6.07e-08, 3.00e-11 (monotone True); achieved CG resid 1.22e-15 | ≤ 1e-08 | — | PASS |
| 64 | abs | F2-spatial | bump order 2.2151 (errors 0.026, 6.02e-03, 1.20e-03); two-mode order 1.8042; N [33, 65, 129] vs 257 | see design | wrong-reference order 0.3419 | PASS |
| 64 | abs | F2-temporal | order 2.0107 (errors 3.99e-04, 9.95e-05, 2.46e-05) | see design | BE order 0.957, separation 576.34x | PASS |
| 64 | abs | F5 | -4.51e-15 | ≤ 1e-12 | 2.828 | PASS |
| 64 | abs | F3 | slope 4.0096; fraction/prediction 1.0215, 1.0052, 1.0013, 1.0003 at N [64, 128, 256, 512]; plateau/fraction 1.0000, 1.0000, 1.0000, 1.0000; y-var 5.5e-15, 4.2e-15, 6.0e-15, 5.4e-15 | see design | reflective retains 1.0000 | PASS |

| N | BC | reported quantity | value |
|---|---|---|---|
| 128 | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | 7.29e-04, 1.58e-05 |
| 128 | abs | family widest blob (w=0.190) spatial order, NOT gated | 1.1920 (errors 8.77e-03, 3.85e-03, 1.68e-03) |
| 128 | ref | family widest blob (w=0.190) spatial order, NOT gated | 0.3479 (errors 0.044, 0.036, 0.027) |
| 128 | ref | V0 energy agreement with the frozen 08-14 FOM | 1.56e-15 |
| 128 | both | provenance | commit 74b49968, backend gpu, jax 0.10.1, matmul highest, wall 97.1 s |
| 64 | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | 1.00e-03, 3.12e-05 |
| 64 | abs | family widest blob (w=0.190) spatial order, NOT gated | 1.1920 (errors 8.77e-03, 3.85e-03, 1.68e-03) |
| 64 | ref | family widest blob (w=0.190) spatial order, NOT gated | 0.3479 (errors 0.044, 0.036, 0.027) |
| 64 | ref | V0 energy agreement with the frozen 08-14 FOM | 4.79e-15 |
| 64 | both | provenance | commit 74b49968, backend gpu, jax 0.10.1, matmul highest, wall 62.4 s |
<!-- /phase1-table -->

**Reading.** F3 is the strongest single result of phase 1: the reflected-energy fraction of an
isolated right-going pulse through the ghost-closed Engquist–Majda face matches the discrete
prediction $|R|^2 = \tan^4(\theta/4) \approx \tfrac{15}{1024}(\Delta x/w)^4$ derived in the r1
audit, within a few percent at every resolution, slope 4.01 — the absorber is exactly what the
design says it is. V0 shows the new reflective FOM reproduces the frozen 08-14 rollout to
roundoff, so the reflective data are bit-comparable with the 08-16 negative.

## Phase 2 — bank and manifold gates (per BC, head arm) — generated table

<!-- phase2-table -->
<!-- /phase2-table -->

## Phase 3 — ROM gates and the decision table — generated table

<!-- phase3-table -->
<!-- /phase3-table -->
