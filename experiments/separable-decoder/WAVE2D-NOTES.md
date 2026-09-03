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
15. **Amendments 7–12 (design r4 section) after the phase-2/3 code verification
    (`WAVE2D-PHASE23-VERIFY-r1-codex-gpt56sol.md`):** W3 comparator = POD-K CN rollout at the same RS and
    the finest complete RS (no held-out selection); W5 restated as independent LSPG optimality (the
    energy identity was a tautology); arm A completion by first-order optimality (an LSPG residual is
    not small at the optimum — my first "residual ≤ 1e-8·scale" rule completed 0/4 rollouts on the
    smoke); W6 control on the linear subspace; W4 with true endpoint energies and the W6/refinement
    preconditions; W0 through the solver's stencil with captured solves; W7 control on a pulse with
    asserted nonzero velocity; G0c with destination floors, trajectory normalisation and NaN=FAIL; D0
    independent path on the full matrix with the near-degenerate-tail rule; D2 at all codes; G0b
    NaN=FAIL.
16. **Retraction 7 — arm A completion / W5 thresholds 1e-6 → 1e-4 (2026-09-03, from the first
    corrected N=64 absorbing run, `runs/wav2d/p2fix_n64_abs`).** With the relative-gradient
    completion criterion at 1e-6, arm A lost 4/16 to 16/16 trajectories at RS 20 and 40 while every
    "incomplete" step had stopped at a relative gradient of 1e-6 to 5e-6 with healthy conditioning:
    that is LM's stall level when the residual is tiny at fine time steps, and it is first-order
    optimal for any practical purpose (the perturbed-latent control reads 0.48). The 1e-6 was set by
    guess. Both the completion rule and W5 now use 1e-4 (control threshold 1e-2). The affected
    W3-A / W6-A verdicts in that run are superseded by the re-run.
17. **Retraction 8 — W6 control threshold 20% → 10%.** The first-order POD-K BE control's excess
    differed by 17–18% between RS 8 and 40 on the absorbing N=64 runs (and by 439% on the
    reflective); the 20% bar was set by guess. The control's purpose is to show the check sees
    time-step dependence at all; 10% does that.

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

## Reading of phases 2–3 (rewritten after the Codex verification `WAVE2D-RESULTS-VERIFY-r1-codex-gpt56sol.md`; every result number is in the generated tables below)

**Provenance caveat first.** The four re-runs (`p2fix_*`) were launched with the driver at commit
003d138, i.e. **before retractions 7–8**: in their JSONs arm A "completes" only at a relative
gradient ≤ 1e-6 (hence the INCOMPLETE arm A rows at RS 20/40, all of which stopped at gradients
1e-6–5e-6 with healthy conditioning) and the W6 controls are stored against the 20% bar (they read
17–18% on absorbing and 439–636% on reflective). Neither changes a W3 verdict: W3-A fails on the error
ratio at the finest complete RS wherever it fails, and the one W3-A pass (absorbing `sup`, N=64 and
128) is read at RS=8. The tables are what the code that produced them computed; the retractions are
recorded, not applied retroactively.

**What was certified before any verdict.** Both FOMs (phase 1, two verification rounds); the bank
($G^\top MG = I$ to 1e-14; independent-SVD / ARPACK floors agree to 2e-13 at both N); the Petrov
tables ($A = -\Lambda B$ to 1e-15 for both closures, W1); the table residual against the solver's
stencil at random states and captured solves (W0, ≤1e-15); arm C on a linear head against an
independent POD-K Verlet (W7, ≤2e-12); the POD-K CN control's reflective energy (W2, deviation
2e-11–5e-11); arm A first-order optimal through the independent path (W5). Controls fired except
where the tables say otherwise (the N=128 reflective `sup` G0b control at 1.179 against a 1.2 bar; the
absorbing W6 controls against the pre-retraction 20% bar).

**Manifold gates (phase 2).** 11 of 12 (head, BC, N) cells pass G0; the exception is reflective
`sup` at N=128, which fails G0b *through its control* (its own tangent residual, 0.741, is better
than POD-K's 0.825). Tangent-space velocity residuals: reflective heads 0.63–0.74 against POD-K's
0.69–0.83; absorbing heads 0.23–0.30 against 0.68–0.79. The reflective oracle is 2.6–4× the bank
ceiling (POD-64, 0.064–0.073); `auto` at N=64 reads 0.186, next to the 08-16 auto-decoder's 0.189
(a lab-log number, not recomputed here); `auto+vc` differs from `auto` by less than the spread
between N=64 and 128 in every phase-2 metric (a single seed; no uncertainty estimate). D1 (oracle ≤
0.5 × POD-K) fails on every reflective head and passes on every absorbing head.

**The stop gate (phase 3).** Reflective W3 **fails for every head, both arms, both N** (incomplete
rollouts count as failures). At the W3-selected RS (finest complete): arm A at 2.2–5.0× its floor with
the dynamic-velocity energy ratio at $T$ of 6.5–19.1 (4.5–4.7 at RS=8); on `auto` at N=64 it reads
0.888 at RS=8 and 0.932 at the selected RS=20, beside the 08-16 record's 0.878 (lab-log number).
Arm C, where 16/16 complete: 2.2–5.7× its floor; at N=128 `sup` it is *better* than arm A (2.23×
vs 2.59×). Arm C's reduced-energy ratio at $4T$ spans 0.996–1.002 over the completed rows, and W4
(bounded deviation, no secular trend, refinement) passes on `auto+vc` at both N and on `auto` at
N=128, fails on `auto` at N=64 by 3.4% (1.034 against 1.0) and fails on `sup` at N=128 with a 116%
deviation and a secular slope (the `sup` manifold is defined on the $(\mu,t)$ box; arm C leaves it
and at N=64 stalls on 1–7 of 16 trajectories at $10^{-2}$–$10^{-1}$ of the term scale). For
`auto`/`auto+vc` the deviation shrinks under refinement (0.157 → 0.014 → 0.005 for `auto` at N=128),
as a modified-energy bound should.

Absorbing W3 **passes on `sup` at both N for both arms** (arm C 1.29× / 1.27× floor at $T$, 0.25–0.28×
the POD-K CN rollout, 16/16 at every RS); `auto` / `auto+vc` fail the floor ratio at 2.4–3.3× at $T$
(4T ratios up to 4.8×) while beating POD-K's rollout by 2.2–3×.

**Verdict against the predeclared decision table (design r3/r4).** The design's row "G0 pass +
reflective W3 fail + W4 pass, all heads" is **not met as a bundle**: it holds for `auto+vc` at both N
individually; `auto` at N=64 fails W4 by 3.4% (which the design reads as an implementation item to
resolve, not as a result); `sup` fails W4 at N=128 and G0 at N=128. The curvature alternative the
design lists is not excluded by any gate here. So the formal outcome for the structural-vs-manifold
question is **INCONCLUSIVE**, with these facts standing:

1. **Energy loss is not necessary for the accuracy failure**: `auto+vc` passes W4 (energy deviation
   ≤ 6e-3 over $4T$, no secular trend, shrinking under refinement) and fails W3 at both N with the
   same error as the energy-losing arm A. The 08-16 explanation ("the recurrence destroys energy on a
   nonlinear manifold, not fixable by tuning") is refuted as a *mechanism*; its verdict stands.
2. **Hypothesis, consistent with every number here but not established by the design's own row**:
   the accuracy failure tracks the tangent-space quality of the manifold (reflective heads no better
   than POD-K, absorbing heads 2–3× better, and only absorbing `sup` passes). G0b was the gate that
   sees it, with a bar too lenient to predict W3; a single seed, no uncertainty, no curvature
   exclusion — it is a hypothesis for the next cell, not a finding of this one.
3. `auto+vc` (velocity-consistency loss) does not separate from `auto` in any metric here.

**Consequences.** Phase 4 (the cost ladder) is **not run** — the design gates it on a reflective W3
pass ("PASS for at least one (head, ROM arm) pair on the reflective BC or the ladder is not run").
`wav2d_ladder.py` exists and is smoke-tested; running it on the absorbing `sup` arm (the one
configuration that passes) is a decision for the user.

**Predictions scored.** G0: `auto` predicted FAIL — passed (11/12 overall; the one failure is `sup`
at N=128 through its control); `sup`/`auto+vc` predicted PASS — right except `sup` at N=128.
Reflective W3: `auto` FAIL predicted — right; `sup`/`auto+vc` PASS predicted — wrong. Absorbing
≤ 2× floor: `sup` right (1.27–1.29×), `auto`/`auto+vc` wrong (2.4–3.3× at $T$). Arm C energy
bounded: right on `auto`/`auto+vc` (W4 pass at N=128, 3.4% miss for `auto` at N=64), wrong on
`sup` (116% at N=128).

## Phase 2 — bank and manifold gates (per BC, head arm) — generated table

<!-- phase2-table -->
| N | BC | head | K | params | final loss | D0 | D1 held-out/POD-K (ctrl shuffled) | D2 min cond (ctrl dup.) | G0a ratio, gap (ctrl) | G0b tangent/POD-K (ctrl random) | G0 | predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | abs | sup | 6 | 42560 | 9.69e-03 | PASS (orth 1.11e-14, floor 0.0250, sigma_R/sigma_1 0.016) | 0.3107 (6.493) PASS | 2.95e-03 (0) PASS | 0.7326, -0.0535 (7.8544) PASS | 0.3247 (1.2232) PASS | PASS | PASS |
| 128 | abs | auto | 8 | 42944 | 6.16e-03 | PASS (orth 1.11e-14, floor 0.0250, sigma_R/sigma_1 0.016) | 0.2118 (13.076) PASS | 0.064 (0) PASS | 0.9984, -1.179e-04 (16.6938) PASS | 0.4161 (1.3953) PASS | PASS | FAIL |
| 128 | abs | auto+vc | 8 | 42944 | 0.021 | PASS (orth 1.11e-14, floor 0.0250, sigma_R/sigma_1 0.016) | 0.2152 (12.868) PASS | 0.056 (0) PASS | 0.9692, -2.334e-03 (17.2197) PASS | 0.3418 (1.3953) PASS | PASS | PASS |
| 128 | ref | sup | 6 | 42560 | 0.037 | PASS (orth 1.04e-14, floor 0.0767, sigma_R/sigma_1 0.021) | 0.6846 (3.677) FAIL | 3.71e-03 (0) PASS | 0.9572, -0.0117 (4.6852) PASS | 0.8981 (1.1787) FAIL | FAIL | PASS |
| 128 | ref | auto | 8 | 42944 | 0.032 | PASS (orth 1.04e-14, floor 0.0767, sigma_R/sigma_1 0.021) | 0.6345 (4.852) FAIL | 0.082 (0) PASS | 1.0789, 0.0142 (10.3949) PASS | 0.9553 (1.2379) PASS | PASS | FAIL |
| 128 | ref | auto+vc | 8 | 42944 | 0.161 | PASS (orth 1.04e-14, floor 0.0767, sigma_R/sigma_1 0.021) | 0.6258 (4.921) FAIL | 0.104 (0) PASS | 1.0718, 0.0128 (10.3677) PASS | 0.9358 (1.2379) PASS | PASS | PASS |
| 64 | abs | sup | 6 | 42560 | 9.99e-03 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2655 (7.640) PASS | 2.49e-03 (0) PASS | 0.8209, -0.0273 (10.9204) PASS | 0.3172 (1.2178) PASS | PASS | PASS |
| 64 | abs | auto | 8 | 42944 | 6.78e-03 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2183 (12.623) PASS | 0.082 (0) PASS | 0.9872, -9.682e-04 (16.5866) PASS | 0.4368 (1.3750) PASS | PASS | FAIL |
| 64 | abs | auto+vc | 8 | 42944 | 0.023 | PASS (orth 1.05e-14, floor 0.0256, sigma_R/sigma_1 0.017) | 0.2222 (12.422) PASS | 0.095 (0) PASS | 0.9460, -4.332e-03 (12.7449) PASS | 0.3565 (1.3750) PASS | PASS | PASS |
| 64 | ref | sup | 6 | 42560 | 0.036 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6492 (3.912) FAIL | 3.66e-03 (0) PASS | 0.8134, -0.0564 (6.1720) PASS | 0.8594 (1.2799) PASS | PASS | PASS |
| 64 | ref | auto | 8 | 42944 | 0.032 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6156 (5.080) FAIL | 0.077 (0) PASS | 1.1192, 0.0198 (10.5424) PASS | 0.9531 (1.3883) PASS | PASS | FAIL |
| 64 | ref | auto+vc | 8 | 42944 | 0.159 | PASS (orth 1.07e-14, floor 0.0693, sigma_R/sigma_1 0.021) | 0.6173 (5.065) FAIL | 0.095 (0) PASS | 1.1044, 0.0177 (10.5143) PASS | 0.9150 (1.3883) PASS | PASS | PASS |

| N | BC | head | held-out oracle median | train oracle median | POD-K median | POD-R ceiling median | G0b tangent median / POD-K median (n states) |
|---|---|---|---|---|---|---|---|
| 128 | abs | sup | 0.14647 | 0.19993 | 0.47135 | 0.01743 | 0.25511 / 0.78562 (788) |
| 128 | abs | auto | 0.07232 | 0.07243 | 0.34141 | 0.01743 | 0.28232 / 0.67847 (788) |
| 128 | abs | auto+vc | 0.07347 | 0.07580 | 0.34141 | 0.01743 | 0.23188 / 0.67847 (788) |
| 128 | ref | sup | 0.26083 | 0.27249 | 0.38102 | 0.07293 | 0.74116 / 0.82529 (796) |
| 128 | ref | auto | 0.19377 | 0.17960 | 0.30540 | 0.07293 | 0.74147 / 0.77618 (796) |
| 128 | ref | auto+vc | 0.19113 | 0.17833 | 0.30540 | 0.07293 | 0.72639 / 0.77618 (796) |
| 64 | abs | sup | 0.12524 | 0.15256 | 0.47162 | 0.01771 | 0.25035 / 0.78936 (788) |
| 64 | abs | auto | 0.07464 | 0.07561 | 0.34186 | 0.01771 | 0.30094 / 0.68898 (788) |
| 64 | abs | auto+vc | 0.07596 | 0.08029 | 0.34186 | 0.01771 | 0.24565 / 0.68898 (788) |
| 64 | ref | sup | 0.24580 | 0.30217 | 0.37862 | 0.06385 | 0.64902 / 0.75518 (796) |
| 64 | ref | auto | 0.18615 | 0.16633 | 0.30239 | 0.06385 | 0.65304 / 0.68520 (796) |
| 64 | ref | auto+vc | 0.18667 | 0.16902 | 0.30239 | 0.06385 | 0.62695 / 0.68520 (796) |
<!-- /phase2-table -->

## Phase 3 — ROM gates and the decision table — generated table

<!-- phase3-table -->
| N | BC | head | arm | RS | complete | err_T median | err_4T median | oracle floor T / 4T | POD-K T / 4T | same-dt FOM | energy ratio T (Er arm C / dyn arm A) | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | abs | sup | A | 8 | 16/16 | 0.15415 | 0.25538 | 0.14647 / 0.26794 | 0.47135 / 0.43953 | 1.75e-03 | 0.86613 | 4.03 |
| 128 | abs | sup | A | 20 | 12/16 | INCOMPLETE | | 0.14647 / 0.26794 | 0.47135 / 0.43953 | | | |
| 128 | abs | sup | A | 40 | 8/16 | INCOMPLETE | | 0.14647 / 0.26794 | 0.47135 / 0.43953 | | | |
| 128 | abs | sup | C | 8 | 16/16 | 0.18626 | 0.32722 | 0.14647 / 0.26794 | 0.47135 / 0.43953 | 1.75e-03 | 7.7847e-03 | 3.08 |
| 128 | abs | sup | C | 20 | 16/16 | 0.18634 | 0.33161 | 0.14647 / 0.26794 | 0.47135 / 0.43953 | 3.44e-04 | 7.7805e-03 | 2.97 |
| 128 | abs | sup | C | 40 | 16/16 | 0.18635 | 0.33305 | 0.14647 / 0.26794 | 0.47135 / 0.43953 | 8.51e-05 | 7.7799e-03 | 2.79 |
| 128 | abs | auto | A | 8 | 16/16 | 0.21017 | 0.34212 | 0.07232 / 0.07020 | 0.34141 / 0.32073 | 1.75e-03 | 1.46126 | 4.17 |
| 128 | abs | auto | A | 20 | 9/16 | INCOMPLETE | | 0.07232 / 0.07020 | 0.34141 / 0.32073 | | | |
| 128 | abs | auto | A | 40 | 2/16 | INCOMPLETE | | 0.07232 / 0.07020 | 0.34141 / 0.32073 | | | |
| 128 | abs | auto | C | 8 | 16/16 | 0.21066 | 0.34652 | 0.07232 / 0.07020 | 0.34141 / 0.32073 | 1.75e-03 | 0.01225 | 2.94 |
| 128 | abs | auto | C | 20 | 16/16 | 0.21116 | 0.34727 | 0.07232 / 0.07020 | 0.34141 / 0.32073 | 3.44e-04 | 0.01228 | 2.71 |
| 128 | abs | auto | C | 40 | 16/16 | 0.21124 | 0.34737 | 0.07232 / 0.07020 | 0.34141 / 0.32073 | 8.51e-05 | 0.01228 | 2.60 |
| 128 | abs | auto+vc | A | 8 | 16/16 | 0.13934 | 0.25348 | 0.07347 / 0.07695 | 0.34141 / 0.32073 | 1.75e-03 | 1.34172 | 4.19 |
| 128 | abs | auto+vc | A | 20 | 9/16 | INCOMPLETE | | 0.07347 / 0.07695 | 0.34141 / 0.32073 | | | |
| 128 | abs | auto+vc | A | 40 | 0/16 | INCOMPLETE | | 0.07347 / 0.07695 | 0.34141 / 0.32073 | | | |
| 128 | abs | auto+vc | C | 8 | 16/16 | 0.17647 | 0.31235 | 0.07347 / 0.07695 | 0.34141 / 0.32073 | 1.75e-03 | 9.6658e-03 | 2.96 |
| 128 | abs | auto+vc | C | 20 | 16/16 | 0.17644 | 0.31241 | 0.07347 / 0.07695 | 0.34141 / 0.32073 | 3.44e-04 | 9.6593e-03 | 2.66 |
| 128 | abs | auto+vc | C | 40 | 16/16 | 0.17644 | 0.31242 | 0.07347 / 0.07695 | 0.34141 / 0.32073 | 8.51e-05 | 9.6584e-03 | 2.52 |
| 128 | ref | sup | A | 8 | 16/16 | 0.63485 | 0.92817 | 0.26083 / 0.41214 | 0.38102 / 0.38335 | 0.022 | 4.69330 | 5.23 |
| 128 | ref | sup | A | 20 | 16/16 | 0.67544 | 0.96855 | 0.26083 / 0.41214 | 0.38102 / 0.38335 | 3.92e-03 | 7.10722 | 5.07 |
| 128 | ref | sup | A | 40 | 13/16 | INCOMPLETE | | 0.26083 / 0.41214 | 0.38102 / 0.38335 | | | |
| 128 | ref | sup | C | 8 | 9/16 | INCOMPLETE | | 0.26083 / 0.41214 | 0.38102 / 0.38335 | | | |
| 128 | ref | sup | C | 20 | 14/16 | INCOMPLETE | | 0.26083 / 0.41214 | 0.38102 / 0.38335 | | | |
| 128 | ref | sup | C | 40 | 16/16 | 0.58239 | 1.05074 | 0.26083 / 0.41214 | 0.38102 / 0.38335 | 9.34e-04 | 1.00001 | 3.02 |
| 128 | ref | auto | A | 8 | 16/16 | 0.90766 | 0.99355 | 0.19377 / 0.28478 | 0.30540 / 0.30413 | 0.022 | 8.59693 | 5.26 |
| 128 | ref | auto | A | 20 | 16/16 | 0.89435 | 1.01246 | 0.19377 / 0.28478 | 0.30540 / 0.30413 | 3.92e-03 | 13.25506 | 5.16 |
| 128 | ref | auto | A | 40 | 12/16 | INCOMPLETE | | 0.19377 / 0.28478 | 0.30540 / 0.30413 | | | |
| 128 | ref | auto | C | 8 | 16/16 | 1.01364 | 1.19119 | 0.19377 / 0.28478 | 0.30540 / 0.30413 | 0.022 | 1.00026 | 3.75 |
| 128 | ref | auto | C | 20 | 16/16 | 1.04994 | 1.14335 | 0.19377 / 0.28478 | 0.30540 / 0.30413 | 3.92e-03 | 1.00004 | 3.03 |
| 128 | ref | auto | C | 40 | 16/16 | 1.03511 | 1.18184 | 0.19377 / 0.28478 | 0.30540 / 0.30413 | 9.34e-04 | 1.00001 | 3.00 |
| 128 | ref | auto+vc | A | 8 | 16/16 | 0.89521 | 0.99128 | 0.19113 / 0.29114 | 0.30540 / 0.30413 | 0.022 | 11.31343 | 5.34 |
| 128 | ref | auto+vc | A | 20 | 16/16 | 0.89337 | 1.02138 | 0.19113 / 0.29114 | 0.30540 / 0.30413 | 3.92e-03 | 19.08587 | 5.16 |
| 128 | ref | auto+vc | A | 40 | 12/16 | INCOMPLETE | | 0.19113 / 0.29114 | 0.30540 / 0.30413 | | | |
| 128 | ref | auto+vc | C | 8 | 16/16 | 0.98564 | 1.17401 | 0.19113 / 0.29114 | 0.30540 / 0.30413 | 0.022 | 1.00068 | 3.78 |
| 128 | ref | auto+vc | C | 20 | 16/16 | 0.96422 | 1.21789 | 0.19113 / 0.29114 | 0.30540 / 0.30413 | 3.92e-03 | 1.00013 | 3.02 |
| 128 | ref | auto+vc | C | 40 | 16/16 | 0.96175 | 1.21764 | 0.19113 / 0.29114 | 0.30540 / 0.30413 | 9.34e-04 | 1.00003 | 3.00 |
| 64 | abs | sup | A | 8 | 16/16 | 0.11987 | 0.17214 | 0.12524 / 0.29200 | 0.47162 / 0.43998 | 1.27e-03 | 0.87108 | 3.98 |
| 64 | abs | sup | A | 20 | 12/16 | INCOMPLETE | | 0.12524 / 0.29200 | 0.47162 / 0.43998 | | | |
| 64 | abs | sup | A | 40 | 8/16 | INCOMPLETE | | 0.12524 / 0.29200 | 0.47162 / 0.43998 | | | |
| 64 | abs | sup | C | 8 | 16/16 | 0.16194 | 0.24587 | 0.12524 / 0.29200 | 0.47162 / 0.43998 | 1.27e-03 | 8.4685e-03 | 3.11 |
| 64 | abs | sup | C | 20 | 16/16 | 0.16201 | 0.25833 | 0.12524 / 0.29200 | 0.47162 / 0.43998 | 2.15e-04 | 9.2726e-03 | 2.92 |
| 64 | abs | sup | C | 40 | 16/16 | 0.16203 | 0.28005 | 0.12524 / 0.29200 | 0.47162 / 0.43998 | 4.32e-05 | 7.6132e-03 | 2.73 |
| 64 | abs | auto | A | 8 | 16/16 | 0.21628 | 0.25206 | 0.07464 / 0.07591 | 0.34186 / 0.32112 | 1.27e-03 | 1.61974 | 4.27 |
| 64 | abs | auto | A | 20 | 7/16 | INCOMPLETE | | 0.07464 / 0.07591 | 0.34186 / 0.32112 | | | |
| 64 | abs | auto | A | 40 | 1/16 | INCOMPLETE | | 0.07464 / 0.07591 | 0.34186 / 0.32112 | | | |
| 64 | abs | auto | C | 8 | 16/16 | 0.24517 | 0.28709 | 0.07464 / 0.07591 | 0.34186 / 0.32112 | 1.27e-03 | 9.2075e-03 | 2.81 |
| 64 | abs | auto | C | 20 | 16/16 | 0.24574 | 0.28708 | 0.07464 / 0.07591 | 0.34186 / 0.32112 | 2.15e-04 | 9.2142e-03 | 2.64 |
| 64 | abs | auto | C | 40 | 16/16 | 0.24581 | 0.28708 | 0.07464 / 0.07591 | 0.34186 / 0.32112 | 4.32e-05 | 9.2153e-03 | 2.50 |
| 64 | abs | auto+vc | A | 8 | 16/16 | 0.18140 | 0.36491 | 0.07596 / 0.07822 | 0.34186 / 0.32112 | 1.27e-03 | 1.70118 | 4.15 |
| 64 | abs | auto+vc | A | 20 | 8/16 | INCOMPLETE | | 0.07596 / 0.07822 | 0.34186 / 0.32112 | | | |
| 64 | abs | auto+vc | A | 40 | 0/16 | INCOMPLETE | | 0.07596 / 0.07822 | 0.34186 / 0.32112 | | | |
| 64 | abs | auto+vc | C | 8 | 16/16 | 0.22679 | 0.37908 | 0.07596 / 0.07822 | 0.34186 / 0.32112 | 1.27e-03 | 7.3586e-03 | 2.95 |
| 64 | abs | auto+vc | C | 20 | 16/16 | 0.22657 | 0.37895 | 0.07596 / 0.07822 | 0.34186 / 0.32112 | 2.15e-04 | 7.3732e-03 | 2.65 |
| 64 | abs | auto+vc | C | 40 | 16/16 | 0.22654 | 0.37894 | 0.07596 / 0.07822 | 0.34186 / 0.32112 | 4.32e-05 | 7.3753e-03 | 2.47 |
| 64 | ref | sup | A | 8 | 16/16 | 0.47627 | 0.91472 | 0.24580 / 0.40519 | 0.37862 / 0.38020 | 8.54e-03 | 4.48395 | 5.33 |
| 64 | ref | sup | A | 20 | 16/16 | 0.54959 | 0.94539 | 0.24580 / 0.40519 | 0.37862 / 0.38020 | 1.31e-03 | 6.45248 | 5.21 |
| 64 | ref | sup | A | 40 | 13/16 | INCOMPLETE | | 0.24580 / 0.40519 | 0.37862 / 0.38020 | | | |
| 64 | ref | sup | C | 8 | 9/16 | INCOMPLETE | | 0.24580 / 0.40519 | 0.37862 / 0.38020 | | | |
| 64 | ref | sup | C | 20 | 14/16 | INCOMPLETE | | 0.24580 / 0.40519 | 0.37862 / 0.38020 | | | |
| 64 | ref | sup | C | 40 | 15/16 | INCOMPLETE | | 0.24580 / 0.40519 | 0.37862 / 0.38020 | | | |
| 64 | ref | auto | A | 8 | 16/16 | 0.88750 | 0.98370 | 0.18615 / 0.28113 | 0.30239 / 0.29938 | 8.54e-03 | 6.95309 | 5.28 |
| 64 | ref | auto | A | 20 | 16/16 | 0.93210 | 1.03861 | 0.18615 / 0.28113 | 0.30239 / 0.29938 | 1.31e-03 | 13.92916 | 5.40 |
| 64 | ref | auto | A | 40 | 14/16 | INCOMPLETE | | 0.18615 / 0.28113 | 0.30239 / 0.29938 | | | |
| 64 | ref | auto | C | 8 | 16/16 | 1.01234 | 1.16406 | 0.18615 / 0.28113 | 0.30239 / 0.29938 | 8.54e-03 | 1.00066 | 3.76 |
| 64 | ref | auto | C | 20 | 16/16 | 1.06902 | 1.16886 | 0.18615 / 0.28113 | 0.30239 / 0.29938 | 1.31e-03 | 1.00015 | 3.04 |
| 64 | ref | auto | C | 40 | 16/16 | 1.06406 | 1.18581 | 0.18615 / 0.28113 | 0.30239 / 0.29938 | 2.63e-04 | 1.00002 | 3.00 |
| 64 | ref | auto+vc | A | 8 | 16/16 | 0.82117 | 0.97719 | 0.18667 / 0.29347 | 0.30239 / 0.29938 | 8.54e-03 | 7.93679 | 5.41 |
| 64 | ref | auto+vc | A | 20 | 16/16 | 0.84681 | 1.00260 | 0.18667 / 0.29347 | 0.30239 / 0.29938 | 1.31e-03 | 14.69816 | 5.42 |
| 64 | ref | auto+vc | A | 40 | 13/16 | INCOMPLETE | | 0.18667 / 0.29347 | 0.30239 / 0.29938 | | | |
| 64 | ref | auto+vc | C | 8 | 16/16 | 0.93151 | 1.21585 | 0.18667 / 0.29347 | 0.30239 / 0.29938 | 8.54e-03 | 1.00075 | 3.77 |
| 64 | ref | auto+vc | C | 20 | 16/16 | 0.92810 | 1.19005 | 0.18667 / 0.29347 | 0.30239 / 0.29938 | 1.31e-03 | 1.00008 | 3.01 |
| 64 | ref | auto+vc | C | 40 | 16/16 | 0.92778 | 1.20259 | 0.18667 / 0.29347 | 0.30239 / 0.29938 | 2.63e-04 | 1.00002 | 3.00 |

| N | BC | head | gate | value | threshold | control | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 128 | abs | — | W1 | 3.41e-15 | 1.00e-12 | 0.392 | PASS | ||A + Lambda B||/||A|| with the M-weighted tables; control: unweighted Phi^T L G (fires for 'abs' where L_N is not Euclidean-symmetric; identical for 'ref', rep |
| 128 | abs | — | W1-Cterm | -0.036 | -0.010 | — | PASS | -(|| (c dt/2) C dh || / || B dh ||) on a face-supported bump at RS=8: the damping term must be >= 1e-2 of the mass term (value 3.630e-02); this is what kills th |
| 128 | abs | — | W2 POD-6 | 0.656 | — | — | PASS | error 0.65585 vs floor 0.47135, energy ratio 3.603867e-03 |
| 128 | abs | — | W2 POD-8 | 0.541 | — | — | PASS | error 0.54081 vs floor 0.34141, energy ratio 5.726559e-03 |
| 128 | abs | — | W2 POD-64 | 0.027 | — | — | PASS | error 0.02730 vs floor 0.01743, energy ratio 1.539390e-04 |
| 128 | abs | sup | W0 | 8.56e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 8.6e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 5.8e-17); control: B p |
| 128 | abs | sup | W0-grad | 1.31e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 128 | abs | sup | W7 | 3.45e-13 | 1.00e-11 | 0.561 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.130; control: traveling pulse (|qv0| = 1.09e+00) with zdot_0 dropped |
| 128 | abs | sup | W6-C | 3.55e-04 | 0.200 | 0.184 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.05037), (20, 0.05024), (40, 0.05023)]; control (AMENDMENT 8: a first-order integrator on the linear  |
| 128 | abs | sup | W6-A | nan | 0.200 | 0.184 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.01507), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 128 | abs | sup | W3-A | 0.702 | 1.000 | 482.558 | PASS | arm A RS=8 (finest complete): err_T/floor 1.052 (<=1.5), err_4T/floor 0.953 (<=1.5), err_T/POD-K-CN 0.235 (<=0.5), err_4T/POD-K-CN 0.336 (<=0.5), pre-exit err/floor (E0-normalised) 0.539 (<=1.5); post |
| 128 | abs | sup | W3-C | 0.877 | 1.000 | 2.59e+08 | PASS | arm C RS=40 (finest complete): err_T/floor 1.272 (<=1.5), err_4T/floor 1.243 (<=1.5), err_T/POD-K-CN 0.284 (<=0.5), err_4T/POD-K-CN 0.439 (<=0.5), pre-exit err/floor (E0-normalised) 0.563 (<=1.5); pos |
| 128 | abs | sup | G0c | 0.492 | 0.500 | 2.508 | PASS | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00021), (2, 0.00129), (5, 0.01293), (10, 0.03 |
| 128 | abs | auto | W0 | 6.99e-16 | 1.00e-12 | 9.99e-09 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 7.0e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 6.6e-17); control: B p |
| 128 | abs | auto | W0-grad | 4.35e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 128 | abs | auto | W7 | 1.81e-12 | 1.00e-11 | 0.569 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.130; control: traveling pulse (|qv0| = 1.32e+00) with zdot_0 dropped |
| 128 | abs | auto | W6-C | 2.81e-04 | 0.200 | 0.170 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.1327), (20, 0.13296), (40, 0.133)]; control (AMENDMENT 8: a first-order integrator on the linear POD |
| 128 | abs | auto | W6-A | nan | 0.200 | 0.170 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.13389), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 128 | abs | auto | W3-A | 3.249 | 1.000 | 3.00e+03 | FAIL | arm A RS=8 (finest complete): err_T/floor 2.906 (<=1.5), err_4T/floor 4.874 (<=1.5), err_T/POD-K-CN 0.389 (<=0.5), err_4T/POD-K-CN 0.529 (<=0.5), pre-exit err/floor (E0-normalised) 1.531 (<=1.5); post |
| 128 | abs | auto | W3-C | 3.299 | 1.000 | 5.20e+11 | FAIL | arm C RS=40 (finest complete): err_T/floor 2.921 (<=1.5), err_4T/floor 4.948 (<=1.5), err_T/POD-K-CN 0.391 (<=0.5), err_4T/POD-K-CN 0.537 (<=0.5), pre-exit err/floor (E0-normalised) 1.918 (<=1.5); pos |
| 128 | abs | auto | G0c | 0.478 | 0.500 | 4.985 | PASS | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00034), (2, 0.00203), (5, 0.01328), (10, 0.03 |
| 128 | abs | auto+vc | W0 | 6.33e-16 | 1.00e-12 | 9.99e-09 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 6.3e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 4.8e-17); control: B p |
| 128 | abs | auto+vc | W0-grad | 4.60e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 128 | abs | auto+vc | W7 | 1.81e-12 | 1.00e-11 | 0.569 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.130; control: traveling pulse (|qv0| = 1.32e+00) with zdot_0 dropped |
| 128 | abs | auto+vc | W6-C | 4.45e-05 | 0.200 | 0.170 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.08961), (20, 0.08958), (40, 0.08958)]; control (AMENDMENT 8: a first-order integrator on the linear  |
| 128 | abs | auto+vc | W6-A | nan | 0.200 | 0.170 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.06221), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 128 | abs | auto+vc | W3-A | 2.196 | 1.000 | 251.664 | FAIL | arm A RS=8 (finest complete): err_T/floor 1.897 (<=1.5), err_4T/floor 3.294 (<=1.5), err_T/POD-K-CN 0.258 (<=0.5), err_4T/POD-K-CN 0.392 (<=0.5), pre-exit err/floor (E0-normalised) 1.433 (<=1.5); post |
| 128 | abs | auto+vc | W3-C | 2.707 | 1.000 | 9.36e+10 | FAIL | arm C RS=40 (finest complete): err_T/floor 2.402 (<=1.5), err_4T/floor 4.060 (<=1.5), err_T/POD-K-CN 0.326 (<=0.5), err_4T/POD-K-CN 0.483 (<=0.5), pre-exit err/floor (E0-normalised) 1.480 (<=1.5); pos |
| 128 | abs | auto+vc | G0c | 0.510 | 0.500 | 8.542 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00024), (2, 0.00138), (5, 0.01297), (10, 0.03 |
| 128 | ref | — | W1 | 2.80e-15 | 1.00e-12 | 2.78e-15 | PASS | ||A + Lambda B||/||A|| with the M-weighted tables; control: unweighted Phi^T L G (fires for 'abs' where L_N is not Euclidean-symmetric; identical for 'ref', rep |
| 128 | ref | — | W2 POD-6 | 4.22e-11 | 1.00e-09 | -0.926 | PASS | error 0.38173 vs floor 0.38102, energy ratio 1.0000000 |
| 128 | ref | — | W2 POD-8 | 4.30e-11 | 1.00e-09 | -0.911 | PASS | error 0.30594 vs floor 0.30540, energy ratio 1.0000000 |
| 128 | ref | — | W2 POD-64 | 3.57e-11 | 1.00e-09 | -0.865 | PASS | error 0.07791 vs floor 0.07293, energy ratio 1.0000000 |
| 128 | ref | sup | W0 | 8.56e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 8.6e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 5.3e-17); control: B p |
| 128 | ref | sup | W0-grad | 2.12e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 128 | ref | sup | W7 | 1.01e-12 | 1.00e-11 | 0.484 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.190; control: traveling pulse (|qv0| = 1.10e+00) with zdot_0 dropped |
| 128 | ref | sup | W6-C | nan | 0.200 | 4.482 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, nan), (20, nan), (40, 0.35516)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 128 | ref | sup | W6-A | nan | 0.200 | 4.482 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.28542), (20, 0.32352), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD- |
| 128 | ref | sup | W4 | 116.383 | 1.000 | 0.884 | FAIL | arm C reflective at RS=40: max |E_r/E_r0 - 1| over 4T (max over 16) 1.16e+00 (<= 1e-2), max |secular slope| 9.71e-03/T (<= 1e-3), bound at RS=8 nan (must not grow under refinement: False), W6-C passed |
| 128 | ref | sup | W3-A | 5.030 | 1.000 | 4.20e+05 | FAIL | arm A RS=20 (finest complete): err_T/floor 2.590 (<=1.5), err_4T/floor 2.350 (<=1.5), err_T/POD-K-CN 1.769 (<=0.5), err_4T/POD-K-CN 2.515 (<=0.5), energy ratio at T median 7.1072 (in [0.9,1.1]: False) |
| 128 | ref | sup | W3-C | 5.457 | 1.000 | 1.77e+09 | FAIL | arm C RS=40 (finest complete): err_T/floor 2.233 (<=1.5), err_4T/floor 2.549 (<=1.5), err_T/POD-K-CN 1.526 (<=0.5), err_4T/POD-K-CN 2.728 (<=0.5), energy ratio at T median 1.0000 (in [0.9,1.1]: True); |
| 128 | ref | sup | G0c | 0.372 | 0.500 | 5.147 | PASS | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00026), (2, 0.00116), (5, 0.01505), (10, 0.07 |
| 128 | ref | auto | W0 | 7.75e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 7.7e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 8.9e-17); control: B p |
| 128 | ref | auto | W0-grad | 7.66e-10 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 128 | ref | auto | W7 | 1.02e-12 | 1.00e-11 | 0.485 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.190; control: traveling pulse (|qv0| = 1.27e+00) with zdot_0 dropped |
| 128 | ref | auto | W6-C | 0.042 | 0.200 | 6.280 | PASS | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.78633), (20, 0.86678), (40, 0.83071)]; control (AMENDMENT 8: a first-order integrator on the linear  |
| 128 | ref | auto | W6-A | nan | 0.200 | 6.280 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.69309), (20, 0.70641), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD- |
| 128 | ref | auto | W4 | 0.497 | 1.000 | 0.865 | PASS | arm C reflective at RS=40: max |E_r/E_r0 - 1| over 4T (max over 16) 4.97e-03 (<= 1e-2), max |secular slope| 1.23e-05/T (<= 1e-3), bound at RS=8 1.57e-01 (must not grow under refinement: True), W6-C pa |
| 128 | ref | auto | W3-A | 6.616 | 1.000 | 3.44e+03 | FAIL | arm A RS=20 (finest complete): err_T/floor 4.616 (<=1.5), err_4T/floor 3.555 (<=1.5), err_T/POD-K-CN 2.923 (<=0.5), err_4T/POD-K-CN 3.308 (<=0.5), energy ratio at T median 13.2551 (in [0.9,1.1]: False |
| 128 | ref | auto | W3-C | 7.723 | 1.000 | 1.73e+10 | FAIL | arm C RS=40 (finest complete): err_T/floor 5.342 (<=1.5), err_4T/floor 4.150 (<=1.5), err_T/POD-K-CN 3.383 (<=0.5), err_4T/POD-K-CN 3.861 (<=0.5), energy ratio at T median 1.0000 (in [0.9,1.1]: True); |
| 128 | ref | auto | G0c | 2.374 | 0.500 | 8.986 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.0025), (2, 0.01704), (5, 0.15373), (10, 0.439 |
| 128 | ref | auto+vc | W0 | 9.43e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 9.4e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 4.5e-17); control: B p |
| 128 | ref | auto+vc | W0-grad | 9.34e-10 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 128 | ref | auto+vc | W7 | 1.02e-12 | 1.00e-11 | 0.485 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.190; control: traveling pulse (|qv0| = 1.27e+00) with zdot_0 dropped |
| 128 | ref | auto+vc | W6-C | 3.46e-03 | 0.200 | 6.280 | PASS | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.77222), (20, 0.72721), (40, 0.7247)]; control (AMENDMENT 8: a first-order integrator on the linear P |
| 128 | ref | auto+vc | W6-A | nan | 0.200 | 6.280 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.62464), (20, 0.66215), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD- |
| 128 | ref | auto+vc | W4 | 0.607 | 1.000 | 0.814 | PASS | arm C reflective at RS=40: max |E_r/E_r0 - 1| over 4T (max over 16) 6.07e-03 (<= 1e-2), max |secular slope| 1.11e-05/T (<= 1e-3), bound at RS=8 7.94e-02 (must not grow under refinement: True), W6-C pa |
| 128 | ref | auto+vc | W3-A | 6.675 | 1.000 | 5.35e+08 | FAIL | arm A RS=20 (finest complete): err_T/floor 4.674 (<=1.5), err_4T/floor 3.508 (<=1.5), err_T/POD-K-CN 2.920 (<=0.5), err_4T/POD-K-CN 3.337 (<=0.5), energy ratio at T median 19.0859 (in [0.9,1.1]: False |
| 128 | ref | auto+vc | W3-C | 7.957 | 1.000 | 1.03e+12 | FAIL | arm C RS=40 (finest complete): err_T/floor 5.032 (<=1.5), err_4T/floor 4.182 (<=1.5), err_T/POD-K-CN 3.144 (<=0.5), err_4T/POD-K-CN 3.978 (<=0.5), energy ratio at T median 1.0000 (in [0.9,1.1]: True); |
| 128 | ref | auto+vc | G0c | 1.490 | 0.500 | 9.406 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00209), (2, 0.01026), (5, 0.08504), (10, 0.26 |
| 64 | abs | — | W1 | 1.23e-15 | 1.00e-12 | 0.383 | PASS | ||A + Lambda B||/||A|| with the M-weighted tables; control: unweighted Phi^T L G (fires for 'abs' where L_N is not Euclidean-symmetric; identical for 'ref', rep |
| 64 | abs | — | W1-Cterm | -0.036 | -0.010 | — | PASS | -(|| (c dt/2) C dh || / || B dh ||) on a face-supported bump at RS=8: the damping term must be >= 1e-2 of the mass term (value 3.619e-02); this is what kills th |
| 64 | abs | — | W2 POD-6 | 0.656 | — | — | PASS | error 0.65601 vs floor 0.47162, energy ratio 3.633798e-03 |
| 64 | abs | — | W2 POD-8 | 0.541 | — | — | PASS | error 0.54115 vs floor 0.34186, energy ratio 5.789599e-03 |
| 64 | abs | — | W2 POD-64 | 0.027 | — | — | PASS | error 0.02740 vs floor 0.01771, energy ratio 1.522928e-04 |
| 64 | abs | sup | W0 | 1.11e-15 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 1.1e-15) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 7.4e-17); control: B p |
| 64 | abs | sup | W0-grad | 1.15e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 64 | abs | sup | W7 | 3.95e-13 | 1.00e-11 | 0.561 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.129; control: traveling pulse (|qv0| = 1.09e+00) with zdot_0 dropped |
| 64 | abs | sup | W6-C | 4.53e-05 | 0.200 | 0.184 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.03792), (20, 0.0379), (40, 0.0379)]; control (AMENDMENT 8: a first-order integrator on the linear PO |
| 64 | abs | sup | W6-A | nan | 0.200 | 0.184 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.02119), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 64 | abs | sup | W3-A | 0.638 | 1.000 | 256.125 | PASS | arm A RS=8 (finest complete): err_T/floor 0.957 (<=1.5), err_4T/floor 0.590 (<=1.5), err_T/POD-K-CN 0.183 (<=0.5), err_4T/POD-K-CN 0.227 (<=0.5), pre-exit err/floor (E0-normalised) 0.467 (<=1.5); post |
| 64 | abs | sup | W3-C | 0.863 | 1.000 | 4.95e+09 | PASS | arm C RS=40 (finest complete): err_T/floor 1.294 (<=1.5), err_4T/floor 0.959 (<=1.5), err_T/POD-K-CN 0.247 (<=0.5), err_4T/POD-K-CN 0.369 (<=0.5), pre-exit err/floor (E0-normalised) 0.527 (<=1.5); pos |
| 64 | abs | sup | G0c | 0.412 | 0.500 | 3.064 | PASS | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00017), (2, 0.00121), (5, 0.00866), (10, 0.02 |
| 64 | abs | auto | W0 | 8.03e-16 | 1.00e-12 | 9.99e-09 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 8.0e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 7.8e-17); control: B p |
| 64 | abs | auto | W0-grad | 4.73e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 64 | abs | auto | W7 | 1.44e-12 | 1.00e-11 | 0.569 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.129; control: traveling pulse (|qv0| = 1.32e+00) with zdot_0 dropped |
| 64 | abs | auto | W6-C | 5.69e-06 | 0.200 | 0.170 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.16281), (20, 0.1628), (40, 0.1628)]; control (AMENDMENT 8: a first-order integrator on the linear PO |
| 64 | abs | auto | W6-A | nan | 0.200 | 0.170 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.12901), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 64 | abs | auto | W3-A | 2.214 | 1.000 | 127.312 | FAIL | arm A RS=8 (finest complete): err_T/floor 2.898 (<=1.5), err_4T/floor 3.321 (<=1.5), err_T/POD-K-CN 0.400 (<=0.5), err_4T/POD-K-CN 0.389 (<=0.5), pre-exit err/floor (E0-normalised) 1.518 (<=1.5); post |
| 64 | abs | auto | W3-C | 2.521 | 1.000 | 1.57e+11 | FAIL | arm C RS=40 (finest complete): err_T/floor 3.293 (<=1.5), err_4T/floor 3.782 (<=1.5), err_T/POD-K-CN 0.454 (<=0.5), err_4T/POD-K-CN 0.443 (<=0.5), pre-exit err/floor (E0-normalised) 1.793 (<=1.5); pos |
| 64 | abs | auto | G0c | 0.454 | 0.500 | 6.009 | PASS | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00037), (2, 0.00253), (5, 0.01781), (10, 0.03 |
| 64 | abs | auto+vc | W0 | 7.75e-16 | 1.00e-12 | 9.99e-09 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 7.7e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 7.3e-17); control: B p |
| 64 | abs | auto+vc | W0-grad | 3.96e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 64 | abs | auto+vc | W7 | 1.44e-12 | 1.00e-11 | 0.569 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.129; control: traveling pulse (|qv0| = 1.32e+00) with zdot_0 dropped |
| 64 | abs | auto+vc | W6-C | 8.79e-05 | 0.200 | 0.170 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.13996), (20, 0.13987), (40, 0.13986)]; control (AMENDMENT 8: a first-order integrator on the linear  |
| 64 | abs | auto+vc | W6-A | nan | 0.200 | 0.170 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.07952), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K su |
| 64 | abs | auto+vc | W3-A | 3.110 | 1.000 | 6.73e+03 | FAIL | arm A RS=8 (finest complete): err_T/floor 2.388 (<=1.5), err_4T/floor 4.665 (<=1.5), err_T/POD-K-CN 0.335 (<=0.5), err_4T/POD-K-CN 0.564 (<=0.5), pre-exit err/floor (E0-normalised) 1.535 (<=1.5); post |
| 64 | abs | auto+vc | W3-C | 3.230 | 1.000 | 6.85e+11 | FAIL | arm C RS=40 (finest complete): err_T/floor 2.982 (<=1.5), err_4T/floor 4.845 (<=1.5), err_T/POD-K-CN 0.419 (<=0.5), err_4T/POD-K-CN 0.585 (<=0.5), pre-exit err/floor (E0-normalised) 1.856 (<=1.5); pos |
| 64 | abs | auto+vc | G0c | 0.582 | 0.500 | 6.650 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00028), (2, 0.00205), (5, 0.01477), (10, 0.04 |
| 64 | ref | — | W1 | 1.06e-15 | 1.00e-12 | 1.09e-15 | PASS | ||A + Lambda B||/||A|| with the M-weighted tables; control: unweighted Phi^T L G (fires for 'abs' where L_N is not Euclidean-symmetric; identical for 'ref', rep |
| 64 | ref | — | W2 POD-6 | 3.71e-11 | 1.00e-09 | -0.926 | PASS | error 0.37931 vs floor 0.37862, energy ratio 1.0000000 |
| 64 | ref | — | W2 POD-8 | 1.65e-11 | 1.00e-09 | -0.911 | PASS | error 0.30293 vs floor 0.30239, energy ratio 1.0000000 |
| 64 | ref | — | W2 POD-64 | 4.99e-11 | 1.00e-09 | -0.873 | PASS | error 0.06519 vs floor 0.06385, energy ratio 1.0000000 |
| 64 | ref | sup | W0 | 7.11e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 7.1e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 8.7e-17); control: B p |
| 64 | ref | sup | W0-grad | 1.53e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 64 | ref | sup | W7 | 2.40e-13 | 1.00e-11 | 0.484 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.184; control: traveling pulse (|qv0| = 1.10e+00) with zdot_0 dropped |
| 64 | ref | sup | W6-C | nan | 0.200 | 4.391 | FAIL | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, nan), (20, nan), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD-K subspa |
| 64 | ref | sup | W6-A | nan | 0.200 | 4.391 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.27066), (20, 0.34147), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD- |
| 64 | ref | sup | W4 | nan | — | — | FAIL | arm C rollouts incomplete |
| 64 | ref | sup | W3-A | 4.953 | 1.000 | 785.632 | FAIL | arm A RS=20 (finest complete): err_T/floor 2.236 (<=1.5), err_4T/floor 2.333 (<=1.5), err_T/POD-K-CN 1.449 (<=0.5), err_4T/POD-K-CN 2.476 (<=0.5), energy ratio at T median 6.4525 (in [0.9,1.1]: False) |
| 64 | ref | sup | W3-C | nan | — | — | FAIL | arm C: no RS with 16/16 complete |
| 64 | ref | sup | G0c | nan | 0.500 | 5.029 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, nan), (2, nan), (5, nan), (10, nan)]; floor per |
| 64 | ref | auto | W0 | 8.94e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 8.9e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 8.7e-17); control: B p |
| 64 | ref | auto | W0-grad | 9.67e-10 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 64 | ref | auto | W7 | 2.57e-13 | 1.00e-11 | 0.485 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.184; control: traveling pulse (|qv0| = 1.27e+00) with zdot_0 dropped |
| 64 | ref | auto | W6-C | 1.65e-05 | 0.200 | 6.358 | PASS | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.79117), (20, 0.79749), (40, 0.7975)]; control (AMENDMENT 8: a first-order integrator on the linear P |
| 64 | ref | auto | W6-A | nan | 0.200 | 6.358 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.68198), (20, 0.71833), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD- |
| 64 | ref | auto | W4 | 1.034 | 1.000 | 0.814 | FAIL | arm C reflective at RS=40: max |E_r/E_r0 - 1| over 4T (max over 16) 1.03e-02 (<= 1e-2), max |secular slope| 8.87e-06/T (<= 1e-3), bound at RS=8 1.10e-01 (must not grow under refinement: True), W6-C pa |
| 64 | ref | auto | W3-A | 6.907 | 1.000 | 2.86e+03 | FAIL | arm A RS=20 (finest complete): err_T/floor 5.007 (<=1.5), err_4T/floor 3.694 (<=1.5), err_T/POD-K-CN 3.077 (<=0.5), err_4T/POD-K-CN 3.454 (<=0.5), energy ratio at T median 13.9292 (in [0.9,1.1]: False |
| 64 | ref | auto | W3-C | 7.886 | 1.000 | 2.66e+09 | FAIL | arm C RS=40 (finest complete): err_T/floor 5.716 (<=1.5), err_4T/floor 4.218 (<=1.5), err_T/POD-K-CN 3.513 (<=0.5), err_4T/POD-K-CN 3.943 (<=0.5), energy ratio at T median 1.0000 (in [0.9,1.1]: True); |
| 64 | ref | auto | G0c | 1.808 | 0.500 | 7.379 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00215), (2, 0.01012), (5, 0.11874), (10, 0.32 |
| 64 | ref | auto+vc | W0 | 7.60e-16 | 1.00e-12 | 1.00e-08 | PASS | Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, 7.6e-16) and 23 captured arm-A solves (backward-error normalised by ||B h_n||, 9.2e-17); control: B p |
| 64 | ref | auto+vc | W0-grad | 2.12e-09 | 1.00e-07 | — | PASS | analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path) |
| 64 | ref | auto+vc | W7 | 2.57e-13 | 1.00e-11 | 0.485 | PASS | arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS=8, complete=True; CFL c dt sqrt(lam_max) = 0.184; control: traveling pulse (|qv0| = 1.27e+00) with zdot_0 dropped |
| 64 | ref | auto+vc | W6-C | 3.49e-03 | 0.200 | 6.358 | PASS | arm C: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.69972), (20, 0.74006), (40, 0.74265)]; control (AMENDMENT 8: a first-order integrator on the linear  |
| 64 | ref | auto+vc | W6-A | nan | 0.200 | 6.358 | FAIL | arm A: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS [(8, 0.58846), (20, 0.65826), (40, nan)]; control (AMENDMENT 8: a first-order integrator on the linear POD- |
| 64 | ref | auto+vc | W4 | 0.545 | 1.000 | 0.786 | PASS | arm C reflective at RS=40: max |E_r/E_r0 - 1| over 4T (max over 16) 5.45e-03 (<= 1e-2), max |secular slope| 1.37e-05/T (<= 1e-3), bound at RS=8 1.65e-01 (must not grow under refinement: True), W6-C pa |
| 64 | ref | auto+vc | W3-A | 6.668 | 1.000 | 6.95e+03 | FAIL | arm A RS=20 (finest complete): err_T/floor 4.536 (<=1.5), err_4T/floor 3.416 (<=1.5), err_T/POD-K-CN 2.795 (<=0.5), err_4T/POD-K-CN 3.334 (<=0.5), energy ratio at T median 14.6982 (in [0.9,1.1]: False |
| 64 | ref | auto+vc | W3-C | 7.998 | 1.000 | 5.91e+09 | FAIL | arm C RS=40 (finest complete): err_T/floor 4.970 (<=1.5), err_4T/floor 4.098 (<=1.5), err_T/POD-K-CN 3.063 (<=0.5), err_4T/POD-K-CN 3.999 (<=0.5), energy ratio at T median 1.0000 (in [0.9,1.1]: True); |
| 64 | ref | auto+vc | G0c | 1.447 | 0.500 | 9.671 | FAIL | stepdiag from oracle starts (arm C, RS=20, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; excess per H [(1, 0.00183), (2, 0.01271), (5, 0.06249), (10, 0.25 |

| N | BC | head | G0 (phase 2) | predicted G0 | W3 arm A | W3 arm C | reading |
|---|---|---|---|---|---|---|---|
| 128 | abs | sup | PASS | PASS | PASS | PASS | absorbing: dissipative comparator, not used for the causal verdict |
| 128 | abs | auto | PASS | FAIL | FAIL | FAIL | absorbing: dissipative comparator, not used for the causal verdict |
| 128 | abs | auto+vc | PASS | PASS | FAIL | FAIL | absorbing: dissipative comparator, not used for the causal verdict |
| 128 | ref | sup | FAIL | PASS | FAIL | FAIL | G0 fail + reflective fail: consistent with the manifold-quality diagnosis (not alone decisive) |
| 128 | ref | auto | PASS | FAIL | FAIL | FAIL | G0 pass + reflective arm C fail: INCONCLUSIVE (structural not refuted on this head; check W4/W6/D2) |
| 128 | ref | auto+vc | PASS | PASS | FAIL | FAIL | G0 pass + reflective arm C fail: INCONCLUSIVE (structural not refuted on this head; check W4/W6/D2) |
| 64 | abs | sup | PASS | PASS | PASS | PASS | absorbing: dissipative comparator, not used for the causal verdict |
| 64 | abs | auto | PASS | FAIL | FAIL | FAIL | absorbing: dissipative comparator, not used for the causal verdict |
| 64 | abs | auto+vc | PASS | PASS | FAIL | FAIL | absorbing: dissipative comparator, not used for the causal verdict |
| 64 | ref | sup | PASS | PASS | FAIL | FAIL | G0 pass + reflective arm C fail: INCONCLUSIVE (structural not refuted on this head; check W4/W6/D2) |
| 64 | ref | auto | PASS | FAIL | FAIL | FAIL | G0 pass + reflective arm C fail: INCONCLUSIVE (structural not refuted on this head; check W4/W6/D2) |
| 64 | ref | auto+vc | PASS | PASS | FAIL | FAIL | G0 pass + reflective arm C fail: INCONCLUSIVE (structural not refuted on this head; check W4/W6/D2) |
<!-- /phase3-table -->
