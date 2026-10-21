# g2d / fomdt — Burgers 2D Newton–BiCGStab at intermediate time steps (solver-only timing)

Coordinator task, user-approved, 2026-09-25. Written before any job. Goal: fill the gap between ~0.05 % (Δt 0.005)
and ~1.6 % (Δt 0.01) on the FOM cost curve.

- Meshes 256², 512², 1024², 2048² on A100-80GB. 4096² on an H200 if one starts within ~1 h of submission,
  otherwise an A100-80GB (the job records which). One job per mesh, all run in parallel.
- Cohort, truth and metric are the Table 1 / burgers-bank-knob ones:
  - dev6 (hash `108f12dc…` asserted);
  - same-grid `fft_tight` truth (Δt 0.005, ntol 1e-6, ltol 1e-8);
  - per case, the max over evolved times of ||u−u_ref||/||u_0||; worst / mean / median over cases.
- Settings: `engines.make_fom(L, Δt, target=L)`, the "lean" implementation that bank-knob timed, with
  Δt ∈ {1/100, 1/120, 1/140, 1/160, 1/200} × (ntol, ltol) ∈ {(1e-2,1e-2), (3e-3,3e-3), (1e-3,1e-3)}. That is 15
  settings. The Table 1 anchors `lean_nt3e-3_l3e-3_dt005` and `lean_nt1e-2_l1e-2_dt01` are members. Every Δt
  divides 0.05 exactly (the factory asserts it).
- Timing is bank-knob's protocol in default compile mode (bank-knob did not time graphs mode):
  - main/slow phases split at 1 s;
  - 5 repetitions, a fresh random order per (repetition, case), seed 20260923;
  - 0.25 s GPU burn before each invocation, cooldown after slow ones;
  - input device_put and synchronised; median GPU ms over all timed invocations;
  - every timed output hash compared with the quick run.
- Check: in every job the anchors' worst errors must equal the bk<n>b summaries to the printed digits. Mismatches
  are reported, never adjusted.
- Code: `scripts/fomdt.py` (bankknob.py's cohort, truth and timing, reduced to the FOM). No NM-ROM, no operators.
  Output `results.json` (`scripts/summarize_fomdt.py`).
