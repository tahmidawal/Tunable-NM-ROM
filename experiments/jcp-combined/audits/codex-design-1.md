**Overall: the design needs amendments before implementation.** The LMM algebra is sound, but several classifications and hypotheses cannot support their stated conclusions. I reviewed the [pre-registration](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md) against the named C1/C2 sources and vendored code. No files were modified.

1. **CORRECT — LMM residual and simplified Jacobian (§2.1).**  
   For fixed history, viscosity and step size,
   \[
   r=S\mathcal R,\qquad
   \partial_c r
   =\operatorname{diag}(S)\left[(\alpha_0+\beta_0h\nu\Lambda)A+\beta_0hJ_u(c)\right]
   =A+\beta_0h\operatorname{diag}(S)J_u(c).
   \]
   The coefficients and BE startup conventions match `t2core.py`. These are full BE startup steps, as explicitly specified, rather than the usual half-step Rannacher construction.

2. **NEEDS-RESTATEMENT — The quadratic identity is correct; “\(F(c_n)\) costs nothing extra” is not.**  
   The deployed off-mesh contraction satisfies
   \[
   N(c)=P^\mathsf T[(Bc)\odot(Dc)],\quad
   J_u(c)=P^\mathsf T[\operatorname{diag}(Dc)B+\operatorname{diag}(Bc)D],
   \]
   so \(J_u\) is linear and \(N(c)=J_u(c)c/2\). However, using a cached Jacobian still requires a matrix-vector product and diffusion arithmetic unless these values are cached too. The dense sign-upwind operator is expressly excluded from linear-Jacobian extrapolation.

   **Fix:** say “no additional Jacobian contraction”; specify whether accepted \(N(c_n)\), \(Ac_n\), and \(F(c_n)\) are cached and charge their computation to timing.

3. **NEEDS-RESTATEMENT — Predictor correctness does not guarantee fixed-sweep accuracy.**  
   Extrapolating \(J_u\) alongside the coefficient history is algebraically exact for these quadratic rules. Choosing the lowest-residual candidate among previous, linear and quadratic predictions is also consistent with the vendor. But the selected candidate need not be the quadratic predictor; rejected sweeps and carried damping can materially affect accuracy.

   **Fix:** retain the predictor, but record candidate selections, rejected sweeps and damping histories. Treat second-order accuracy of the resulting algorithm as a measured property, not a consequence of the predictor formula.

4. **WRONG — The adaptive-start schedule is undefined at the largest 3D step.**  
   At \(h=5h_0=0.05\), the entire trajectory has five steps. J4’s \(a=6\) exceeds that count. Copying the vendor’s two scans literally executes six adaptive steps, exceeding the requested horizon. Even after clipping, this candidate is entirely adaptive.

   **Fix:** define
   \[
   a_{\rm eff}=\min(N_{\rm steps},\max(a,n_{\rm startup})),
   \]
   specify \(a\) for every W prefix and scheme, and report the adaptive/fixed step counts. Label entirely adaptive candidates accordingly. Preserve \(a=6\) for the BE reproduction at \(h_0\).

5. **NEEDS-RESTATEMENT — BE reduction is algebraic; G-FS1 demands stronger numerical identity.**  
   BE coefficients recover the vendor equation exactly in real arithmetic. A generalized implementation can nevertheless reorder operations and change a predictor choice or acceptance decision. C2’s amended generic/vendor gate used \(10^{-8}\) trajectory agreement, not universal \(10^{-12}\) agreement with identical exits.

   **Fix:** preserve the vendor BE arithmetic path if exact exit/iteration parity is required. Otherwise separate residual/Jacobian identity from trajectory equivalence, with justified field and coefficient tolerances. Use a named nonstationary case for the CN negative control; CN need not differ materially on every possible case.

6. **WRONG — The proposed candidates cannot inherit “second-order” status from C2.**  
   C2 amendment A1.2 explicitly predicts generically first-order asymptotic convergence of the weighted LSPG CN/BDF2 families, with apparent second-order ranges possible. Its real-data order findings do not transfer automatically to wider prefixes, different rules, or fixed-sweep solvers. Section 3.3 even calls LSPG-CN a “representative second-order arm,” contradicting §2.

   **Fix:** call them CN/CN-R/BDF2-family candidates. Add tight, solve-resolved self-convergence tests on each new setting before claiming second order. Alternatively scope H1–H6 to measured accuracy and cost, without asserting formal order.

7. **NEEDS-RESTATEMENT — G-FS2 is an agreement test, not an order test, and its must-fail control is unproven.**  
   Agreement with adaptive LSPG at one finest step does not establish convergence order. Conversely, one Gauss–Newton sweep from the previous state can sometimes be sufficiently accurate; it is not mathematically guaranteed to miss the proposed bar by \(10\times\). “Time error” also needs an explicit oracle and norm.

   **Fix:** test a step-refinement sequence, state the oracle and normalization, and measure fixed-versus-adaptive agreement separately from observed order. Use deterministic implementation mutations—wrong history, wrong startup, stale \(F_n\), wrong Jacobian weight—as mandatory negative controls. Retain the previous-state predictor as an empirical comparison unless its failure is established before production.

8. **WRONG — Some “one-ingredient” budget ablations are not defined (§5).**  
   For a fixed-sweep LSPG arm, \(a[f\to\mathrm{GAL}]\) requires a fixed-sweep GAL algorithm, which is not specified. Substituting adaptive GAL changes both form and solver. Similarly, \(\star[\mathrm{tight}\to\sigma]\) puts the deployed fixed-sweep solver onto the ideal GAL model without defining that combination. In 2D, “production tolerance” also means different stopping tests for GAL and LSPG.

   **Fix:** define the full form × solver cross-product, including startup, predictor, trust and stopping semantics; or explicitly report a joint form-and-solver change. Keep initial coefficients identical across every ablation. Rename \(\delta_{\rm test}\) to **form sensitivity**: it does not vary \(M\) or the sine-test space.

9. **NEEDS-RESTATEMENT — The time ablation is an algorithm-package change.**  
   Replacing \((s,h)\) changes scheme, step, LSPG weights, startup duration and—for fixed sweeps—the fraction of adaptively solved steps. That is a legitimate operational comparison, but not an isolated truncation-error measurement.

   **Fix:** label it “temporal-discretization package sensitivity.” Add a within-scheme refinement comparison with negligible solver error when making a specific time-error attribution.

10. **CORRECT — The Pythagorean identity holds pointwise in time under the stated projection.**  
    With the same node matrix \(G_X\), orthogonal projector onto its column space, and common denominator \(n_0=\|u_0(X)\|\),
    \[
    \frac{\|u_\star-u_{\rm ref}\|^2}{n_0^2}
    =
    \frac{\|(I-\Pi_X)u_{\rm ref}\|^2}{n_0^2}
    +
    \frac{\|u_\star-\Pi_Xu_{\rm ref}\|^2}{n_0^2}.
    \]
    This requires the decoder and projection to use exactly the same basis, coordinates, boundary convention and inner product. A numerically truncated projector must also retain the span containing \(u_\star\).

11. **WRONG — That identity does not hold for independently time-maximized components.**  
    Section 5 first defines all quantities as maxima over time, then reuses those symbols in the identity. In general,
    \[
    \max_t[p(t)^2+d(t)^2]\ne(\max_t p(t))^2+(\max_t d(t))^2.
    \]
    The promised stacked budget bars are additionally invalid for the nonorthogonal ablation distances.

    **Fix:** use explicit per-time notation. Check B0 per time and, if desired, display both orthogonal components at the same maximizing time. Plot ablation distances as grouped points/bars, not additive contributions to total error.

12. **NEEDS-RESTATEMENT — Nothing is an algebraic remainder, but “reduced-dynamics error” remains a composite discrepancy.**  
    \(e_{\rm dyn}\) is directly measured, so it is not manufactured by subtraction. It nevertheless contains effects of initialization, finite mesh operators, finite test space, finite anchor step, residual quadrature error and solver error. Calling its magnitude “what limits next” overstates causal identification.

    **Fix:** call it “ideal-ROM versus projected-reference discrepancy.” Report initial-fit error separately, qualify remaining anchor errors, and describe the largest component as the largest measured sensitivity/discrepancy—not necessarily the intervention that will most improve accuracy.

13. **NEEDS-RESTATEMENT — The proposed interaction diagnostics are descriptive, not an interaction decomposition.**  
    The closure ratios respond to vector alignment even in an additive model. The triangle inequality is a useful consistency check but cannot establish absence or presence of interactions. Endpoint sensitivity differences can reveal context dependence, but omit the temporal ingredient and can become unstable near zero.

    **Fix:** retain these as diagnostics; define zero-denominator handling. For actual interaction evidence, save signed per-time changes and evaluate selected mixed differences,
    \[
    I_{ij}=u_{ij}-u_i-u_j+u_a,
    \]
    particularly time × quadrature and time × solver.

14. **WRONG — “Time-converged ideal” and “converged budget rule” lack sufficient qualification.**  
    C2’s accepted anchor was at \(h_0/16\), with an \(h_0/8\) comparison and solve-sensitivity checks. This design steps back to \(h_0/8\). It specifies no equivalent 2D convergence gate, and only checks the 3D anchor on 16 of 64 cases. The Gauss \(320^2/640^2\) check covers certain deployed-end ablations, not necessarily the ideal or reverse-end trajectories. If \(320^2\) fails, simply promoting \(640^2\) does not verify it.

    **Fix:** qualify anchors using temporal refinement and tight/tighter solve checks on their actual rule and form. Include ideal/reverse-end runs in quadrature checks. Pre-register a checked successor and an unresolved outcome. Restrict convergence claims to checked cases, or expand checks before cohort-wide verdicts.

15. **WRONG — J4 classifications are not causally valid as written (§3.3).**  
    Passing adaptive \(d(56,64)\) establishes only that this pair agrees under adaptive BE. It does not show C1’s failure was an inadequate \(48^3\) check. Category B identifies solver-dependent sensitivity, but “not quadrature” is too strong: quadrature changes can trigger solver-branch changes. Nonfinite/unverified adaptive runs have no classification. The \(64/80\) fallback examines only 16 cases while authorizing selection on 64.

    **Fix:** classify quadrature agreement and solver sensitivity as separate axes, including mixed and unresolved outcomes. At every rule, measure fixed-versus-adaptive distance for the **same scheme and step**, including adaptive CN at \(2.5h_0\). Check quadrature operators on common coefficient states to distinguish operator differences from rollout amplification. Require an eligible convergence check on the population for which selection is authorized.

16. **NEEDS-RESTATEMENT — Two-solver rule selection is useful but narrowly scoped.**  
    Intersecting the BE and CN admissible-rule sets is sensible. It does not certify CN-R, BDF2, \(K=2\), other steps, or the GAL anchor. Worse, the “converged” reference is certified under adaptive BE, although selection compares fixed-sweep trajectories. The faster-family rule is also ambiguous when one family wins under BE and the other under CN.

    **Fix:** require eligible, checked reference trajectories under both selection solvers. Define a deterministic family-cost objective. Verify quadrature sensitivity for every finalist under its actual solver/scheme/step, with a frozen fallback for failure and for no admissible family. Apply equivalent finalist checks to inherited 2D BE-selected rules.

17. **NEEDS-RESTATEMENT — ST–TX is a useful temporal sensitivity indicator, not E0’s verified reference uncertainty.**  
    Algebraically,
    \[
    \mathrm{TX}-\mathrm{ST}=(\mathrm{ST}-\mathrm S)/15.
    \]
    This estimates the leading ST temporal error only under a valid first-order asymptotic expansion. Two levels separated by a factor of 16 do not verify that expansion, estimate TX’s remaining error, or control spatial error. Thus \(2\varepsilon_{\rm ref}\) is not a certified resolution limit here.

    **Fix:** call it a heuristic temporal tie threshold and retain the provisional interpretation. Use E0’s verified uncertainty when available. Apply the chosen tie policy consistently to scheme selection, Pareto dominance and H1/H2/H5/H6; currently it is explicit mainly for H1’s paired counts.

18. **WRONG — H3’s 50% criterion is tautological.**  
    At one common time, with all vectors normalized consistently,
    \[
    e_{\rm BE}^2=e_\infty^2+\|\delta_{\rm BE}\|^2
        +2\langle\delta_{\rm BE},\varepsilon_\infty\rangle.
    \]
    Therefore
    \[
    e_\infty^2-e_{\rm BE}^2+\|\delta_{\rm BE}\|^2
       =-2\langle\delta_{\rm BE},\varepsilon_\infty\rangle.
    \]
    Condition (c) compares the cross term with half itself. When (a) holds, it automatically passes.

    **Fix:** define one common evaluation time explicitly and avoid mixing independently maximized errors. Replace (c), for example, with
    \[
    e_\infty^2-e_{\rm BE}^2\ge0.5\|\delta_{\rm BE}\|^2,
    \]
    equivalently a cancellation cross term at least \(1.5\|\delta_{\rm BE}\|^2\). Keep the angle and CN-to-anchor tests. Also schedule the deployed-rule 2D anchor needed by H3; the converged-rule budget ideal is not that anchor.

19. **WRONG — Scheme selection and H1/H2 leave consequential cases undefined.**  
    Selection first chooses a potentially different winning scheme per rung, then selects a global scheme. H1 subsequently asks for the global scheme “at each dial setting’s accurate-equal-cost arm,” which may belong to another scheme. Missing eligible candidates, exact ties between LSPG schemes, and reference-limited rungs are unspecified. “At least three of four rungs” does not define whether H2 means a subsequence or adjacent improvements.

    **Fix:** first define the best eligible step for each **scheme × setting**, then choose the global scheme and use that scheme’s stored arms. Specify missing-candidate and complete tie rules. Define H2 as either all three adjacent transitions or a named three-rung subsequence, with uncertainty-aware comparisons. State that the dial jointly changes \(R'\), \(M\), rule and possibly step—not bank width alone.

20. **WRONG — H4/H6 can outrun their anchor evidence and budget coverage.**  
    H4 ranks all 64 cases using anchors refined on only 16, with no unresolved policy. Its fixed-versus-tight threshold is an absolute allowance that may be material relative to the claimed \(10\times\) accuracy gain. H6 says “with the converged rule,” whereas accurate-equal-cost selection uses deployed-rule costs; a converged-rule rerun is a different arm. Furthermore, selected \(K=2\) or \(5h_0/3\) arms have no scheduled budget, and complete ideal-based budgets exist only for the first 32 cases.

    **Fix:** define anchor-resolution eligibility and verify adaptive comparators. Relate allowable solver discrepancy to the claimed improvement as well as an absolute cap. Distinguish selected deployed performance from converged-rule diagnostics. Budget every selected arm, or limit conclusions explicitly to the prespecified diagnostic arms and 32-case subset.

21. **NEEDS-RESTATEMENT — Gates need distinct implementation, numerical and scientific failure semantics.**  
    G0, B0, B1, B4, timing determinism and score reproduction are useful. However:
    - G1’s historical \(10^{-6}\) agreement is a diagnostic, not proof of correctness; C2 explicitly amended it to be non-gating across jobs.
    - G-score’s relative-only tolerance becomes ill-defined near zero.
    - B2/B3 and CN-versus-BE controls are empirical expectations, not universally necessary truths.
    - The vendor fixed-sweep phase does not independently label a damping-limit exit; its `reason=3` tracks nonfinite residuals there.
    - B0 lacks a specified negative control.

    **Fix:** retain strict same-job implementation checks, separate historical reproduction investigations, use mixed absolute/relative scoring tolerances, and define the action for each failed empirical control. Record damping saturation explicitly. Add deliberate projection/coordinate/normalization mutations for B0/G-score. Spell out inherited tight tolerances: C2 ultimately used LSPG \(10^{-7}\)/\(10^{-8}\), after tighter values proved unattainable.

22. **NEEDS-RESTATEMENT — Arbitrary-coordinate ROM rescoring is feasible, but the scoring contract is incomplete.**  
    Both implementations support it: 2D evaluates features then applies `T[:, :Rp]`; 3D’s `feature_rows` does likewise. In 2D, the physical initial field is an unwindowed Gaussian with boundary rows/columns explicitly zeroed. Cold initialization samples the supplied mesh field by bilinear interpolation at Gauss points. The scoring denominator is the physical initial field on shared nodes, not the reconstructed \(W_0\), cold-fit scale, or full-mesh norm.

    **Fix:** pin bank/rotation hashes, case parameters, coordinate ordering, boundaries, times and normalization rules in the registry. Preserve original coefficients and cold initialization when rescoring. At new nodes, evaluate the physical initial condition consistently; do not substitute a boundary-envelope Gaussian or decoded initial state. Chunk decoding to avoid materializing enormous lattice-by-rank matrices.

23. **WRONG — Coefficients alone cannot “recompute every table” against arbitrary future references.**  
    They suffice for ROM fields on new coordinates at saved times. They do not suffice for FOM fields on a new lattice, unsaved times, or new sealed cases. Recomputing H5 requires FOM rescoring too.

    **Fix:** save the necessary FOM fields with a defined audited restriction/interpolation path, or explicitly require FOM reruns. Limit coefficient-only rescoring to the same cases and saved times. Preserve frozen selections when references change; distinguish rescoring from a new development selection.

24. **NEEDS-RESTATEMENT — Timing structure is good, but cross-job and cross-setting claims need tightening.**  
    A common 2D baseline supports a common cost axis. Per-setting 3D baselines support H4 within each setting, but their ratios cannot alone compare absolute cost across widths. Cross-GPU coefficient agreement at \(10^{-10}\) is stronger than necessary and can fail through harmless solver-branch changes. Re-evaluating only six timing cases does not reproduce the 38-case accuracy qualification inside the timing job.

    **Fix:** save all invocation arrays and define aggregation of case/repetition ratios. Retain absolute 3D times or add a common cross-setting panel. Use same-job exact determinism plus a justified field-distance cross-job check. For H5’s same-job accuracy claim, evaluate all comparison cases in `c2dtime`, as C2 A9.3 required. Explicitly time identical end-to-end work: initialization, stepping, diagnostics and six-field decoding. Define 3D randomization, burn-in and drift handling as explicitly as 2D.

25. **NEEDS-RESTATEMENT — H5 is defensible only as “best among the listed calibrated configurations.”**  
    Its matching rule is falsifiable and avoids importing C2’s printed speedup across hardware. But the strongest combined ROM may have no matching FOM in the prescribed ladder, and the candidate set of “C2 acc arms” is not enumerated. Undamped FOM CN also had an order-gate exception in C2.

    **Fix:** freeze the exact historical comparator inventory; report unmatched arms as unmatched rather than extrapolating. Carry forward FOM calibration, achieved-residual verification and the CN qualification. Apply the reference-resolution policy to matching decisions, and label the resulting claim as a development-cohort comparison on this fixed mesh and ladder.

26. **NEEDS-RESTATEMENT — The 17 GPU-hour estimate is plausible as a placeholder, not substantiated.**  
    Measured precedents are substantial: C1 J1 took 2.28 A100-hours and J4 2.64 H200-hours; C2’s accurate 2D panel took 1.18 A100-hours, FOM panel 1.15, and its two-setting, 16-case 3D panel 0.40 H200-hours. The proposed work adds fine-step ideals, both-end ablations, tight solves at large rules and much broader cohorts. The largest uncertainty is the expensive converged-rule/fine-step combinations—not deployed fixed-sweep query time. Necessary fixes above add work too.

    **Fix:** generate a deduplicated run inventory with per-arm counts and steps. Estimate setup, compilation, adaptive iterations, rule contractions, decoding and timing separately from real-size smoke measurements. Include conditional \(80^3\) escalation and memory/cache management. The job split is reasonable, but `c3dmain` needs a measured wall-time and memory gate before submission.

27. **CORRECT — Current cohort use avoids new test-cohort leakage, with limited claim scope.**  
    Restricting selection to dev/validation and excluding the historical tests is consistent with the lane rules. These are nevertheless reused development cohorts, not independent confirmation. A local `FROZEN-SELECTION.json` does not replace the campaign’s global sealing rule. This lane can answer a provisional combined-method development question; it cannot by itself complete E3’s strong-reference, sealed-cohort, multi-mesh claim.

**Overall verdict: WRONG as a ready-to-run pre-registration; amend and re-audit.** The highest-priority fixes are the undefined budget ablations, H3’s tautological criterion, the J4 causal classification, anchor/order qualification, the five-step startup edge case, and complete specification of selection and finalist budgets.