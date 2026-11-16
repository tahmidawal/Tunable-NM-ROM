# Protocol: why the 2D sealed Burgers quadrature rules fail against the S reference (post-hoc diagnosis)

Written 2026-11-15 before pulling any per-case sealed scores. **This is a post-hoc diagnosis of an
already opened and closed cohort.** It re-runs no model on sealed cases, changes no frozen decision
(F1, `frozen_decisions.py`), and makes no prospective claim.

## What had already been seen when this protocol was written
The public closeout report (counts and PASS/FAIL per rule), and the aggregate arm statistics of
`results/sealed2d256-decision.json` (S and ST median `T` / worst `W`, eps bands, suffix verdicts). In that
file the S failure of fib987 and gauss45/gauss50 is on the median (`T`) band; the worst (`W`) band passes.
The E0 report (`exp/2026-10-06-jcp-references:experiments/jcp-references/report.md`) states:
$R_{ST}$ = central differences + Crank-Nicolson, $\Delta t\propto h$; $R_S$ = the same space discretisation with
backward Euler at the reduced model's $\Delta t_0=0.005$; 2D S `unresolved-floor` arises for high-viscosity cases.

## Inputs (saved outputs only)
1. `experiments/jcp-sealed-closeout/results/sealed{2d256..4096,3d65..257}-decision.json` (aggregate, committed).
2. The per-case raw result JSON of each panel job on the cluster (namespace
   `/cluster/tufts/paralab/tawal01/jcp-sealed-closeout-20261115/` or its cold archive), read in place;
   only reduced per-case scalars (per case, kind, arm: `e`, `e_t`, `eps_ref`, eligibility, outcome,
   `dist_conv` where present) are copied local. No fields.
3. The reference status rows used by the frozen scorer (outcome, `eps_ref`, `n0`), as embedded in the raw result.

## Decomposition (per mesh, per rule in the failing suffix, per statistic T=median, W=worst)
- $A_k$ = anchor (dense rule `conv_arm`) statistic against reference $k\in\{S,ST\}$; $X_k$ = subject statistic.
- Rule term: $\Delta_k = X_k - A_k$ (absolute, %) and $r_k=\Delta_k/A_k$.
- Margin term: allowed band $0.05\,A_k$ adjusted by the eps bands exactly as `band_interval` does
  (PASS iff $X_k^{lo}\ge0.95A_k^{hi}$ and $X_k^{hi}\le1.05A_k^{lo}$).
- Reference term: $D=A_{ST}-A_S$ (how much of the anchor error is the S-vs-ST reference difference), and per case
  $|e_{ST}-e_S|$ for the anchor, which is bounded by $\lVert R_S-R_{ST}\rVert/n_0$.
- Uncertainty term: median and max eps_ref over eligible cases; the share of the band width it contributes.
- A failure is attributed to the term that, if replaced by its ST counterpart, flips the verdict
  (counterfactuals: (a) S subject/anchor Δ with ST margin; (b) ST Δ with S margin; (c) eps set to 0; (d) S evaluated on
  the ST-eligible set vs the S-eligible set, and both on the intersection).

## Hypotheses tested
- H1 (relative margin on a time-error-free target): against S the anchor error excludes the shared backward-Euler
  time error, so $A_S\ll A_{ST}$, the 5% band shrinks in absolute terms, and the unchanged quadrature perturbation exceeds it.
- H2 (S reference time error moves the dense anchor): predicts $A_S$ larger, not smaller, than the quadrature-free target.
- H3 (S uncertainty too small): eps_ref(S) negligible vs band, so setting eps to 0 does not change verdicts; tested by (c).
- H4 (eligibility filtering): verdicts change when evaluated on the common eligible set; tested by (d).
- H5 (continuous advection + discrete diffusion target mismatch): the bank's continuum/discrete mix means neither
  reference is its exact target; examined via 3D vs 2D contrast and mesh trend of $A_S$, labelled Hypothesis unless measured.

## Controlled test (only if needed)
Only on development/validation cohorts (never sealed): e.g. reference time-step refinement. GPU cap 1 job.

## Stopping rule
Stop when every 2D S failure (fib987 x5 meshes, gauss45 x3 meshes) is attributed to a term by the counterfactuals,
and the 3D pass and the 107/123 eligibility gap are explained from saved data. No decision is changed.
