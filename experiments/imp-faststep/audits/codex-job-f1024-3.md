**DO NOT SUBMIT.**

- One evidence-production blocker remains: staged `fsrun.py:464` saves LSPG refinement states, fresh gamma and outputs, but discards `exits`, `step_reason` and `step_gn`.
- The validity predicate explicitly uses `exits[3]`; complete production validity also checks recorded stationarity finiteness. Saved trajectories and fresh gamma cannot establish the solver’s recorded exit reasons.
- Consequently, the results audit still cannot independently reconstruct complete LSPG refinement validity under A4.1. Persist those diagnostics for both refinements before restaging.
- POD refinement persistence and stage-finiteness repairs pass.
- Staged manifest checksums pass; projected 7,108.77 seconds fits the 7,860-second driver timeout.
- Budget admission passes: 0.63 + 2.25 = 2.88 GPU-hours < 4.
- Deferred offline-checker fixes and pricing conservatism are not blockers in this verdict.
- No files changed, SSH used, or jobs submitted.