# Rebuttal manuscript revision, 2026-11-13

The updated manuscript builds to 38 pages. Its conclusion ends on page
10, within the discussion-phase main-text limit.
The source ZIP supplied by the author is preserved under `manuscript_tools/original`;
its `main.tex` exactly matches `submitted.tex`, which remains unchanged.
The author's `todonotes` and `soul` preamble additions are retained.

## Changes from the uploaded source

- Incorporated the existing rebuttal revision: harder interacting-vortex Navier–Stokes,
  cylinder flow, matched solvers, neural-field and matched-data operator comparisons,
  ablations, loss definitions, offline costs and supporting figures.
- Included the completed cylinder CPU AMG, Navier–Stokes batched-throughput and
  reduced-solve scaling studies. No pending experiment is presented as a result.
- Added main-text Figure 2 from saved fresh-cohort JSONs.
  Its generator retains all error seeds and uses only the two globally admitted
  timing seeds; no new solver or training run was performed.
- Clarified the interacting-vortex family, P4/P3 cylinder truth versus P2/P1
  export mesh, and the separate accuracy-matched finite-element comparator.
- Removed the stale statement that multigrid was untested; distinguish measured
  CPU AMG from the pending GPU sparse-direct/AMG comparisons.
- Corrected Eq. (22) against the preserved heat training code:
  the mean term uses weighted POD-mode projection errors, and the tail term
  uses a snapshot minibatch. Exact thin-QR elimination is unchanged.
- Clarified total-cost amortisation and identified the heat modal-comparator arm.
- Qualified scaling: the reduced computation is distinct from mesh-dependent
  projection/output, and cross-allocation wall times are not paired speedups.
- Kept CROM/CoLoRA re-implementation, budget, failed-timing, transfer and
  extrapolation disclosures. Older cylinder field plots retain their explicit
  earlier-development-bank caption and do not depict the held-out headline bank.
- Following the author's presentation choice, neural operators are compared
  by error only. Their inference times and relative-speed claims are removed
  from the main text, historical/matched-data appendix tables and cylinder/NS
  descriptions; the cost plot now shows ROMs and numerical solvers only.
  All prediction-error values and training-budget disclosures are retained.
  Numerical-solver and reduced-model timing comparisons remain in place.

## Build and upload

Run `bash build.sh` in this directory. The source archive contains all referenced
styles and figures, with `main.tex` at its root. The compiled PDF is supplied separately.
Changes relative to the original submission remain marked in blue; moved/deleted
passages are described in the existing revision footnote.

Sparse-solver and equal-median-error operator experiments remain pending.
Overleaf has not been edited or uploaded by this session.
