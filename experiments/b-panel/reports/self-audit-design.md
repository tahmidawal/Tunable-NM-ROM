# Self-audit of DESIGN.md (Codex unavailable, see DESIGN.md §A1)

Answers to the eight questions that were put to Codex, written by the lane itself before the first
job. This is not an independent audit and is labelled as such.

1. **Two readers, two numbers?** Every reported column is defined in §4 in terms of saved fields
   and the same-job `fft_tight`, and the audit computes them; the only reader-chosen quantity was
   the convergence rule, now fixed in §5 with both variants reported. Remaining ambiguity: "median
   GPU ms" is the median over all (case, rep) invocations (18), not the median of per-case
   medians; the audit records both.
2. **Convergence rule.** The residual floor $10^{-10}$ sits eight orders above the observed
   $10^{-18}$–$10^{-20}$ and six below any converged-by-gradient initial fit seen ($8\times10^{-3}$
   relative at $q=0$), so it separates the two populations. It cannot admit a genuinely
   unconverged solve: a solve that stalls has a residual far above $10^{-10}$ of the input norm.
3. **Transferred rules.** Fit and certification trajectories are disjoint by index (8 vs 4) and
   both disjoint from the six cases; the support is fixed before either is seen. The risk is that
   the transferred support is simply wrong at the finer mesh and the rule certifies at neither
   tier; that outcome is reported as "uncertified" and the arm excluded from the admissible set.
4. **Non-dominance.** A tie on both axes leaves both points in the set; uncertified `eqxfer`,
   unconverged reduced subjects and a parity-failing `fast` are excluded from the admissible set
   but still shown in the all-subjects set. No subject is excluded by anything measured after the
   fact except these pre-registered flags.
5. **Second-tier $10^{-3}$.** qtd02 measured $10^{-9}$–$3\times10^{-9}$ against cclad01 for
   $q\ge64$ with refit directions; this lane loads qtd02's own $C$, so the first tier should hold
   on every dense rung. The second tier exists for the EQ arms whose rules were certified under
   qrg304's refit $C$; a miss there would be a real finding and is recorded as such.
6. **Memory.** Persistent device memory is the operator data: dense $\Phi$ for the ladder 1.23 GB,
   the fidelity arm 0.13 GB, POD $\Phi$ + $V$ 2.6 GB, free bank 0.53 GB, EQ stencils ~0.2 GB,
   $G$ 0.27 GB, cold starts and executables. On an 80 GB card with a 72 GB pool this is far from
   binding; the unknown is executable-resident memory, which the OOM-drop order protects
   against in the priority the paper needs.
7. **Left to the report.** The rho bar (0.116) and the parity bar ($10^{-12}$) are inherited and
   fixed; the figure's axes are fixed (log–log, GPU ms vs worst evolved %). Nothing else is chosen
   after seeing data.
8. **Parent characterisation.** btq201's POD rows: verified from its `result.json` (every step
   exits on reason 4; initial fit reason 2 at relative residual $\le10^{-18}$). qrg304's
   certification: verified from its `rule_choice` (primary at $q\le64$, secondary at 128/256).

---

## Post-job addendum, 2026-09-17, after job 1 (`bpn101`, 3780638)

Codex is still unavailable (DESIGN.md §A1). In its place the lane ran a **second, independent
code path** over the same saved fields — `checks/recheck_headline.py`, which imports neither
`audit_panel.py` nor the driver nor JAX — and compared it against the audit arm by arm.

- **Worst relative difference between the audit and the independent re-derivation: 0.0** over
  every arm and every reported metric (`checks/bpn101-recheck.json`).
- Both paths independently return the same non-dominated set and the same two headline claims.

What this does and does not cover. It covers arithmetic and the frontier logic: no reported
number is a transcription or a one-sided bug. It does **not** cover design judgement — whether
the convergence rule, the admissibility rule or the subject list are the right ones is exactly
what an independent model was meant to challenge, and that challenge did not happen. The three
judgements most exposed, stated so a reader can attack them:

1. **§5's convergence rule admits an initial fit whose residual is at round-off.** Job 1 shows
   this changes the flag for POD-32/64/128/256/512 and the free bank and for nothing else, and
   it changes no error or cost. The strict column is printed beside it throughout.
2. **`fft_tight` is both a subject and the reference**, so it sits at exactly zero same-grid
   error and is trivially on the same-grid frontier. The report now says so in a generated
   paragraph and points at the reference-metric column instead.
3. **The `reduced_only` frontier (§A4) was added after seeing the data.** It changes no
   pre-registered criterion, and it is labelled post-hoc everywhere.
