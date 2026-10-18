**Overall verdict: NEEDS-RESTATEMENT.** The design covers most of the requested scope, and its core projection and Möbius identities are sound. Blocking problems remain in the manufactured controls, factorial endpoints, qualification rules, and resource plan.

This audit was read-only: no files changed, jobs submitted, or arrays copied. Control outcomes below distinguish mathematical guarantees from behavior that still requires a smoke.

Evidence shorthand: **D** = [audited DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/DESIGN.md); **Q**, **C**, **E** denote the supplied `jcp-qualify`, `jcp-combined`, and `jcp-references` experiment directories.

1. **Lane coverage — NEEDS-RESTATEMENT**

   Most obligations are explicitly present: front configurations, new long-horizon references, Q1, spectral cross-checks, physical diagnostics, finer sampling, controls, and the prescribed defer order. The listed 2D front members and four 3D configurations match the supplied JSONs. The FRONT-2D checksum also matches.

   Three gaps prevent full coverage:

   - **“Two largest components” is narrowed to “two largest main effects.”** D:206–217 includes floor, ideal dynamics and interactions among candidate components, but only pairs trajectory ingredients. In 3D, form sensitivity is measured only at the far corner, so an interaction involving form cannot be evaluated even if form is among the largest sensitivities. Specify the intended eligible ingredients and provide a conditional form-pair panel.
   - **Form is not test-space variation.** D:167 explicitly leaves test functions unchanged. This is a legitimate form sensitivity, but does not independently measure test-space error. State the scope and relationship to jcp-dims/B6.
   - **Deferral can remove essential endpoints.** D:367–369 permits dropping corners beyond pairwise, yet B2 still requires the all-switched ideal and its uncertainty replays. Exempt these mandatory anchor runs from deferral.

   The main horizons, diagnostics and never-dropped items otherwise agree with the lane brief.

2. **B2 mathematics — NEEDS-RESTATEMENT**

   **Correct:** D:179–182 defines Möbius effects correctly:
   \[
   I_S=\sum_{T\subseteq S}(-1)^{|S|-|T|}u_T,\qquad
   u_{\mathrm{all}}-u_\varnothing=\sum_{S\ne\varnothing}I_S.
   \]
   This is an exact vector identity, not an additive decomposition of norms.

   **Correct with the stated norm:** the per-time Pythagorean identity holds for every trajectory in the same sampled bank span. In 2D, projecting on the \(255^2\) interior nodes of the \(257^2\) lattice is equivalent to using the whole lattice when both reference and bank have zero walls. In 3D the campaign lattice already contains only \(63^3\) interior nodes. All three terms must use the same \(n_0\). D:192–202 states that requirement; C/budget2d.py:223–226 implements the corresponding predecessor calculation.

   **Wrong endpoint association:** for a GAL-deployed member, the all-switched corner is **LSPG**, whereas \(u_\star\) remains GAL. Therefore the full Möbius sum closes to \(u_{\mathrm{all}}-u_a\), not to the \(D_a=\|u_a-u_\star\|\) named in D:222–223. The analogous 3D factorial ends at adaptive LSPG; GAL is an additional corner. Report factorial closure and distance to the common GAL ideal separately.

   \(U_\star\) has the predecessor’s sensible replay structure, but is an **uncertainty indicator**, not an error bound for every corner. D:208–209 also exempts ideal-dynamics error from the \(U_\star\) test even though it depends on \(u_\star\). State its time aggregation and apply both reference and anchor resolution checks where appropriate.

3. **One-ingredient switches — NEEDS-RESTATEMENT**

   - **Form:** switching equations and form-specific production tolerances measures a *form-plus-stopping-policy sensitivity*. It does not isolate the equation change. The predecessor deliberately compared forms at tight tolerances: C/DESIGN.md:425–431. Define the new estimand explicitly or restore that controlled comparison.
   - **Time:** the declared package is legitimate, but should be called a temporal-discretisation package sensitivity, not pure time-step error. In 3D include changes to adaptive startup behavior, as the predecessor does.
   - **2D IC:** full-grid LS is feasible with the existing blocked bank; it need not require a dense pseudoinverse. However, the implementation must specify blocked Gram/RHS accumulation or a stable blocked QR strategy, conditioning checks, and memory lifetime. Q/p1common.py:137–153 and vendor/quad2d/vendor/bkfast.py:38–69 expose the necessary blocks.
   - **IC/tolerance coupling:** Q/p1core.py:346–350 returns both coefficients and a quadrature-derived scale; line 192 uses that scale in the LSPG residual stopping tolerance. Changing initialization must explicitly preserve that scale or declare the stopping-scale change as part of the IC package.
   - **3D IC zero:** correct for the deployed mesh path. Q/s3common.py:135–143 calls `TB.project`; vendor/quad3d/vendor/burgers3d-retry/tables.py:120–124 solves the mesh normal equations. This means the **initialization switch** is zero, not that IC projection error is zero.
   - **3D form:** the far-corner comparison is valid context-specific sensitivity, but is not a deployed-end main effect and supplies no form interactions.

4. **Decision rules — NEEDS-RESTATEMENT**

   The thresholds are mostly falsifiable, but important outcomes remain undefined:

   - **Largest/dominant:** D:210–212 has no explicit treatment of tied largest components. The general \(2\varepsilon_{\rm ref}\) rule does not specify how to rank reference-free effects against reference-dependent components.
   - **Missing/invalid corners:** no explicit B2 rule prevents a failed corner from becoming a large “ingredient effect,” or a NaN from falling through to “unresolved.” The predecessor requires every constituent trajectory to be valid and otherwise marks the contrast unavailable: C/DESIGN.md:555–558, 843–846.
   - **Mesh invariance:** three “no single dominant” labels currently produce “same.” That is agreement in inconclusiveness, not evidence that the dominant ingredient is invariant.
   - **Growth:** define zero/reference-limited denominators and use the intersection of window-eligible cases. Otherwise the ratio can be undefined or based on unresolved errors.
   - **Sampling:** define zero-denominator behavior and an uncertainty-aware rule when errors are comparable to reference uncertainty. A 10% ratio discrepancy can otherwise be driven entirely by unresolved reference error.
   - **Ranking:** the global rank condition in D:299–302 can mark every configuration inadequate because one unrelated pair reverses. State whether that is intended, or separate individual adequacy from front-ranking adequacy.

   Stability is a well-defined sampled diagnostic, but its claim should say **at recorded output times**. It cannot exclude transient energy growth between outputs. The pair-label thresholds themselves are non-degenerate; missing-data and zero-effect conventions still need freezing.

5. **Reference generation — NEEDS-RESTATEMENT**

   **Strides are correct.**

   | Route | Levels | Strides |
   |---|---|---|
   | \(1025^2\), including walls | 2048 / 4096 / 8192 | 2 / 4 / 8 |
   | \(127^3\), interior | 128 / 256 / 512 | 1 / 2 / 4 |

   For 3D interior arrays, use indices \(js-1\), not a naïve zero-offset slice. E/refcore.py:225–236 already implements that convention. Refinement levels also divide correctly: strides 12 and 6.

   Horizon generalization is reasonable, but must include output buffers, clipping indices, stage counts, metric identifiers, canonical IC sampling, and all judge/cross-check loops. E/refcore.py:155–165, 214–236 and E/verdict.py:95–132 contain hard-coded assumptions.

   Other issues:

   - **Fine-route acceptance is underspecified.** Computing \(d_{12},d_{23}\) alone does not establish direction consistency. E/verdict.py:103–104 uses the difference vectors. Preserve sufficient evidence and judge both routes independently.
   - **G-R1/G-R2:** \(10^{-10}\) cross-GPU agreement is not established by saying rounding is allowed. It may pass, but needs evidence or a declared investigation/equivalence procedure. G-R2 must match identical levels **and tolerance/retry purposes**; longer runs can trigger a tight retry that changes their earlier outputs. E/refjob.py:137–153 shows this mechanism.
   - **Refinement only at \(T=0.5\):** acceptable as a disclosed resource restriction, not a mathematical justification. Failed \(T=1\) references must remain unrankable.
   - **Window eligibility:** D:141 makes acceptance horizon-wide, despite reporting window outcomes. Clarify whether a late \(W_3\) failure invalidates \(W_2\), and which reference version is used when both horizons exist.
   - **Spectral cases:** legitimate validation cases and genuine subsets of E0’s cross-check roster; see E/refstore.py:40–41.
   - **Spectral procedure:** D:150–151 abbreviates E0 too far. E/verdict.py:182–214 requires spatial qualification, a finest-grid half-time-step run, stability/gates, and \(s_x+s_t\le\varepsilon/2\). Explicitly inventory these runs and retain their acceptance logic.
   - **Failed cross-checks:** treating them as a label only weakens E0’s eligibility contract. E/refstore.py:132–141 requires dimension validation and passing cross-check status for designated cases. Preserve that distinction or explicitly register a different, provisional status.

6. **Must-fail controls — WRONG as a complete specification**

   | Control | Verdict | Finding |
   |---|---|---|
   | **C-A** | NEEDS-RESTATEMENT | The sine product vanishes mathematically at \(x=i/256\), and the fine lattice resolves it. But the chosen amplitude does **not guarantee** \(s>1.10\): the injected field can cancel existing fine-route error. Near orthogonality and \(s\approx1\) would give approximately \(\sqrt2\), which is plausible, not guaranteed. Freeze a construction/sign/amplitude with a provable lower bound, keeping the normalizer fixed. Specify the 3D analogue. |
   | **C-E** | WRONG | A 2% field scaling cannot guarantee a tenfold increase over an arbitrary existing balance defect. It can even reduce a signed defect. Use a manufactured energy change with a known expected diagnostic response. |
   | **C-F** | WRONG | Orthogonality to the bank does not imply orthogonality to the original projection residual. The asserted formula omits a cross term; see below. The median-based amplitude also does not guarantee dominance on 90% of cases. |
   | **C-I** | NEEDS-RESTATEMENT | A 1% coefficient perturbation need not retain a \(10^{-3}\) field effect at \(t_1\) on every case; dissipation and cancellation with the original IC-fit error can defeat it. Verify the known perturbation at \(t_0\), then separately assess propagation. |
   | **C-S** | NEEDS-RESTATEMENT | Budget zero is a credible failure fixture. With the inherited predictor and zero updates from the start, the trajectory remains at its initial state. But the distance threshold needs a named fixture and a verified tight comparator; do not require the intentionally invalid run to satisfy ordinary B2 validity. |
   | **C-T** | NEEDS-RESTATEMENT | Credible prior evidence: C/report.md:244–252 shows BE temporal sensitivities well above \(10^{-3}\). Freeze the fixture and valid comparison population; that evidence is not a guarantee for every new setting. |
   | **C-Q** | NEEDS-RESTATEMENT | The inherited R384 Gauss32 fixture establishes failure on **at least one case**, not “most.” Q/DESIGN.md:318 uses that weaker criterion. The proposed stronger predicate needs separate evidence. |
   | **C-R1** | NEEDS-RESTATEMENT | First-order rejection is well supported by E/report.md:103–106. But at \(T=0.5\), solver/floor/Péclet failure can precede `order-fail` in the judge. Require valid prerequisite checks and an explicit order-check failure, rather than assuming the final outcome string. |
   | **C-R2** | NEEDS-RESTATEMENT | A shifted reference-versus-ROM comparison is not a G-L1 reproduction test: correctly aligned ROM and reference already differ. Mutate the time indexing in an independently reproducible scoring path and compare against its aligned result. |
   | **C-score** | NEEDS-RESTATEMENT | Specify actual cases/times and availability of S. S is not generated for long horizons, and arbitrary swaps need not change a scalar maximum error by the proposed threshold. |

   For C-F, with \(r=(I-\Pi)u_{\rm ref}\) and unit \(v\perp\operatorname{range}(\Pi)\),
   \[
   \|(I-\Pi)(u_{\rm ref}+\eta v)\|^2
   =\|r\|^2+2\eta\langle r,v\rangle+\eta^2.
   \]
   Orthogonalize against **both the bank and the original residual**, or retain the cross term in the expected answer. Define physical versus normalized amplitude and keep \(t_0\)/normalization unchanged. “Flip to floor” also needs an exception when floor already wins.

   Evidence: D:314–330. Calling every non-discriminating empirical control “a bug” at D:333–335 is incorrect; it can instead mean the proposed fixture is unsuitable.

7. **Scoring contract — NEEDS-RESTATEMENT**

   Correct: development/validation-only populations, ST primary, no production S comparisons, carried post-hoc labels, per-case E0 eligibility for \(W_1\), and separate GAL diagnostic rows. D:65–98 supports these choices.

   Required corrections:

   - Preserve the reference eligibility distinctions described in item 5.
   - Apply the tie policy explicitly to component selection and aggregate front-order comparisons.
   - **Complete long-horizon Q2:** D:231–232 schedules \(q^{++}\), but D:239–242 does not require \(q^+\)-versus-\(q^{++}\) convergence or valid successor/check trajectories. Q/DESIGN.md:219–224 does, and returns “unverified” when convergence is not established.
   - Label the untested population accurately: Q2 on 16/38 or 8/16 cases is not individual Q2 evidence for every case.
   - Name the inherited GAL residual-certification exception explicitly; it is different from literal LSPG stationarity certification, though D:230 correctly points to qualify’s implementation.

8. **Compute and storage — NEEDS-RESTATEMENT**

   The 3-job lane cap, 8-job campaign cap and 45 GPU-hour cap are correctly stated. The table totals **29 GPU-hours**, or **36.25** with 25% contingency. These are estimates, not certificates.

   **Immediate conflict:** `smk1` plus `smk2` allocates up to **3.5 GPU-hours**, exceeding the coordinator’s current **2 GPU-hour aggregate smoke allowance**. D:349–350 also samples fewer cases than the dev6/cases-0–3 control predicates require. State the production dependency on C3 reporting explicitly.

   Nominal f64 storage, before temporaries:

   | Object | Approximate GiB |
   |---|---:|
   | \(4095^2\times512\) bank | 63.97 |
   | \(4095^2\times384\) retained prefix | 47.98 |
   | 2D Gauss320 tables, \(r=384,M=1536\) | 1.76 |
   | 2D Gauss640 tables, same rank | 7.03 |
   | 3D Gauss56 tables, \(r=512,M\approx2048\) | 4.02 |
   | 3D Gauss64 tables, same rank | 6.00 |

   These use \(8m(2r+M)\) table bytes; additional projection tables, construction intermediates and cached rules increase peak memory. The previous 96.6 GB pilot does not automatically cover Gauss640, LS initialization, fine decoding and new caches. D:363–365 needs a documented peak bound or an actual combined-path pilot.

   Reference output buffers also grow materially: E/refcore.py:164 retains full-grid snapshots. At \(L=8192,T=1\), 21 f64 snapshots alone occupy roughly **10.5 GiB**; the \(512^3,T=0.5\) output buffer is roughly **10.9 GiB**. Include these in certificates.

   Keep cluster fine references as permitted by COMMON-BRIEF §5, but define their retention/manifest policy separately from deletion of disposable job directories. The ledger should reserve **all outstanding jobs**, not merely compare each new job against completed spend. Local free space was approximately **130 GiB**; no scratch was used.

9. **Other ways numbers or verdicts could be wrong — WRONG / NEEDS-RESTATEMENT**

   - **3D physical formulas are missing.** D:276–277 hard-codes \(h^2\) and \(h=1/1024\). For 3D use volume weight \(h^3\), \(h=1/128\), and three gradient components. Specify boundary stencils.
   - **Balance-error cancellation is overstated.** D:280 says the common temporal quadrature error cancels to leading order. Its leading coefficient depends on each trajectory’s dissipation history; equal output times alone do not ensure cancellation. Call this a difference of discrete balance defects.
   - **Converged quadrature is checked too narrowly.** G-B4 checks only the Q-only corner and ideal. Mixed corners can reach different states and have different quadrature error. C/DESIGN.md:560–568 explicitly checks every converged-rule-dependent trajectory type and limits claims to checked populations.
   - **No independent factorial truth check:** G-B2 is algebraically guaranteed when all effects are calculated from the same corner array. It detects assembly errors, but cannot prove that corner switches changed the intended ingredients. Add explicit configuration/unchanged-state assertions.
   - **Sampling adequacy is relative to the chosen finer route.** A \(1025^2\) route still subsamples the \(4096^2\) field. It cannot establish absence of all unresolved fine features. D:264 discloses the stride, but conclusions must preserve that limitation.
   - **Pins need completion:** the abbreviated FRONT hash matches, but the EXPORT hash is only described as pinned. Record complete hashes and distinguish artifact-producing commit from later source-copy commit.

**Overall verdict: NEEDS-RESTATEMENT. Required fixes:**

1. Repair C-F’s mathematics and replace C-E with a guaranteed manufactured response; make C-A’s amplitude/sign construction deterministic and discriminating.
2. Freeze named fixtures and prerequisite checks for every empirical control, including explicit 3D variants and smoke populations.
3. Separate factorial endpoints from the common GAL ideal; preserve ideal and uncertainty runs under deferral.
4. Define switch semantics, IC stopping-scale handling, and a blocked full-grid LS implementation.
5. Complete B2 validity, uncertainty, tie, missing-data and denominator rules; resolve the 3D form-interaction coverage gap.
6. Restore successor convergence and validity requirements for long-horizon Q2; check all relevant B2 quadrature-corner types.
7. Specify route/window reference eligibility, full spectral checks, replay-attempt matching, and cross-GPU reproduction handling.
8. Correct 3D diagnostics and qualify the balance-cancellation and sampling-adequacy claims.
9. Reconcile the aggregate smoke cap and C3 dependency; certify combined memory peaks, outstanding-job reservations, and reference retention.