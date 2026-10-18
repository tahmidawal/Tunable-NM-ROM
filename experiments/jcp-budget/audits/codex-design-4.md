**Overall verdict: READY.** No blocking design defects remain after A3. All six required fixes from audit 3 are resolved at the pre-registration stage. Real-size smoke demonstrations and resource certificates remain execution gates, not established results.

Read-only audit: no files changed, jobs submitted, arrays copied, or scratch created.

Below, **D** denotes [jcp-budget/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/DESIGN.md); **Q**, **C**, and **E** denote the supplied qualify, combined, and references experiment directories. Line citations include the applicable amendments.

1. **Lane coverage — CORRECT.**

   A3 adds M to the 2D candidate contest and measures all five M pairs, completing pairwise coverage of the six trajectory ingredients. It separately measures ingredient–ideal-dynamics cross terms; floor interactions vanish by orthogonality. B7 beyond 0.25, new references, long-horizon Q1/Q2, spectral cross-checks, all requested B8 diagnostics, and the manufactured controls are present. The defer order and never-dropped core match the lane brief. Evidence: D:695–708, 229–242, 260–305, 367–370, 428–430.

   **3D F×S:** A3.1 is mathematically defensible **for the frozen solver definitions**. The four-corner contrast requires a GAL/deployed-solver trajectory, but the deployed FS/QN algorithms explicitly construct the weighted LSPG residual and its gradient. That corner has no definition in this configuration domain. Evidence: Q/fs3.py:66–94; Q/qn3.py:91–112. This restriction also appears in C/DESIGN.md:424–431.

   This is not a claim that no GAL fixed-work algorithm could ever be constructed. Such an extension would define a new solver policy and a new estimand. The registered tight-end F×Q and F×T measurements are legitimate; F×S must remain “undefined,” never “zero” or “additive.”

2. **B2 mathematics — CORRECT.**

   Möbius inversion and vector telescoping are exact. After A3, the full factorial remains explicitly the **five-factor** Q/T/F/S/I experiment; M has its main effect and pairwise corners, not an asserted six-factor full-factorial closure. Evidence: D:179–182, 696–699.

   The floor identity holds because \(\Pi\) is the orthogonal projector onto the bank sampled at the scoring interior: \(255^2\) nodes in 2D and \(63^3\) in 3D. Adding zero walls does not change the norm, and every term uses the same \(n_0\). C/budget2d.py:219–231 implements this convention. The identity does not extend to independently maximized components.

   A1 distinguishes the GAL-member endpoint and the 3D closure path correctly. A3 replaces the misidentified dynamics interaction with \(2\langle I_x,u_\star-\Pi u_{\rm ref}\rangle\). \(U_\star\) remains an indicator, not a bound. Evidence: D:432–441, 700–703.

3. **Ingredient switches — CORRECT.**

   These are logical package switches, not claims that only one scalar parameter changes. Form includes its stopping policy; time includes scheme, step, startup and weights. Those semantics are explicit, with a separate tight-form comparison. Evidence: D:443–449, 670–672.

   A3’s streamed Householder TSQR avoids local inverses, tolerates rank-deficient individual blocks, and rejects unacceptable global conditioning or orthogonality. The blocked bank permits full-grid LS at \(4096^2\) without assembling another full bank. Its memory and runtime still require the pilot. Evidence: D:719–724; Q/p1common.py:137–153.

   Preserving the initialization-derived stopping scale isolates the IC replacement; Q/p1core.py:346–350 confirms that scale exists. The 3D switch is structurally zero: Q/s3common.py:135–143 calls `TB.project`, whose triangular solves implement mesh LS (Q/vendor/quad3d/vendor/burgers3d-retry/tables.py:120–124). Projection **error** is not zero.

4. **Decision rules — CORRECT.**

   Completeness, minimum populations, ties, resolution floors, common comparison populations, zero denominators and inconclusive outcomes are specified. Missing Q now prevents unrestricted dominance; failed ideal-anchor quadrature verification invalidates dependent verdicts. Evidence: D:460–479, 639–650, 710–714.

   Growth, sampled-time stability and sampling adequacy can pass or fail. The sampling claim is explicitly relative to the fine route, not a guarantee against every finer feature. No remaining production threshold was found that is structurally forced to pass or fail.

5. **Reference generation — CORRECT at design stage.**

   The fine-lattice strides are correct: 2/4/8 for 2D \(L=2048/4096/8192\), and 1/2/4 for 3D \(L=128/256/512\). Interior indexing is \(js-1\), matching E/refcore.py:225–236.

   Horizon generalization must cover buffers, stage counts, clipping, reshaping, metadata and judge loops; changing `schedule(horizon=...)` alone is insufficient. The registered reproduction gates appropriately constrain this implementation.

   G-R1/G-R2 bars are acceptance requirements, not guaranteed cross-GPU reproducibility. Failed or unevaluable matching cannot produce ranked long-horizon results. Restricting refinement to \(T=0.5\) is acceptable with terminal non-rankable failures at \(T=1\). Window eligibility and matched reference realizations are specified. Evidence: D:481–503, 619–637.

   A3 adds the missing val32/19 FD ladder at \(T=1\). The selected spectral cases belong to E0’s validation roster, confirmed by E/refstore.py:40–41.

6. **Must-fail controls — CORRECT as pre-registered controls; empirical demonstrations pending.**

   - **C-A:** The alias vanishes mathematically at \(x=i/256\). The uncertainty-aware amplitude guarantees the fine-route rejection inequality by the reverse triangle inequality. A3 correctly separates raw detection from classification of resolved cases.
   - **C-F:** Orthogonalization against both the bank and existing floor residual establishes the Pythagorean increase; unchanged \(t_0\), uncertainty-aware amplitude and seed fallback address cancellation and degeneracy.
   - **C-E:** The raw discrete-balance formula is correct; zero and scaled-reference fixtures test the diagnostic without relying on an arbitrary tenfold increase.
   - **C-I:** The direct decoded perturbed-minus-unperturbed IC distance is well-defined.
   - **C-S, C-T, C-Q:** Explicit dimension-specific fixtures and comparators are provided. Budget zero need not freeze the trajectory; Gauss32’s changed comparator is acknowledged. Their thresholds require smoke evidence.
   - **C-R1:** The first-order ladder must specifically fail order with prerequisites satisfied. E0’s previous first-order failures support the fixture but do not prove its \(T=0.5\) behavior.
   - **C-R2/C-score:** Identity rejection is a valid guaranteed plumbing control when exercised through the actual scorer.

   Evidence: D:505–540, 603–617, 732–735. A failed demonstration withholds the protected verdict.

7. **Scoring contract — CORRECT.**

   Development/validation populations, ST primary scoring, restricted S use, per-case-kind eligibility, uncertainty ties, inherited qualification/post-hoc labels and diagnostic GAL treatment are retained. Cross-check status controls whether comparisons receive verdicts. Evidence: D:65–98, 545–552, 623–637.

   The FRONT-2D and EXPORT SHA256 values match the pinned values in D:578–580.

8. **Compute, memory and storage — CORRECT in its blocking safeguards.**

   The lane/campaign caps, outstanding time-limit reservations, sequential smoke allowance, production release, cluster reference retention and checksum collection are specified. I accept A2.7’s coordinator instruction as given.

   The memory pilot covers the combined \(4096^2\) bank/LS/fine-decode path and large 2D/3D quadrature-table construction peaks. Estimates do not substitute for those measurements. Evidence: D:557–571, 674–691.

   **Non-blocking arithmetic correction:** applying the stated 15% increase to b1024/b256/b4096 and adding 0.4 GPU-h gives **35.1 GPU-h**, or **43.875 with contingency**, rather than approximately 34.5/43. It still fits 45; certified reservations govern submissions.

9. **Other validity issues — CORRECT; no additional blocker found.**

   A3.5 fixes the incompatible universal `stats.nfail` requirement: check-rule replays retain their solver, with adaptive convergence checks or FS/QN validity plus fresh stationarity certification. This matches the distinction in Q/s3common.py:223–235 and Q/s3ctx.py:155–159. No substitution of adaptive LM is needed to manufacture a quadrature comparison.

Audit-3 required-fix closure:

| Required fix | Round-4 result |
|---|---|
| 1. Interaction coverage and actual ideal-dynamics cross term | Resolved by A3.1; 3D F×S undefined within the frozen domain |
| 2. Missing Q and unverified-anchor propagation | Resolved by A3.2 |
| 3. Missing \(T=1\) FD case and cost | Resolved by A3.3 |
| 4. Stable QR application and failure handling | Resolved by A3.4 |
| 5. Solver-specific check-rule acceptance | Resolved by A3.5 |
| 6. C-A classifier applicability | Resolved by A3.6 |

**Overall verdict: READY.**

Required fixes:

1. None.