### burgers-heldout — closing pass on bh5 (eqcert audit actually run; one retraction)

Branch `exp/2026-09-21-burgers-heldout`. No new jobs (8/8 used, namespace `bheld_20260921` empty, verified by `ssh
tufts-login ls`). bh5 = job 4153483 (H200, source commit 7912b033) re-timed the INCUMBENT model's paper settings at
$4096^2$ with quadrature rule `lat64` j=1 (first step on the exact residual) beside j=0, dev6 + hold64, 5 reps.

**Retracted from the previous bh5 commit (0e20e20c).** (i) `checks/bh5-eqcert-summary.json` was cited in the report,
the HANDOFF and lab-log milestone 3 but did **not exist**: that audit run had been OOM-killed (`audit_eqcert.py` builds
the full $4096^2$ bank matrix $G$, ~69 GB in f64, so it needs `JAXRUN_MAX=96G`, not the 36 GB default). It has now been
run and committed (one memory-only change to the copied `audit_eqcert.py`: `bank_np` fills a preallocated array instead
of concatenating chunks; chunk-invariance checked in `checks/bank_np_chunk_check.txt`). (ii) `audit_bh.py` marked the j=0 rows `cert True` in `tables.generated.md` / `summary.json` because
it read only the 5-draw status and ignored the failed confirmation draw; the prose was right, the table was not. Fixed
(certified = 5/5 draws **and** confirmation draw) and both regenerated; the j=0 rows now read `cert False`.

**bh5 numbers (unchanged by the fix).** Paper arm q256/M1088 `lat64` g1e-2, dev6: 0.604 % @ 107.0 ms, FOM
`lean_nt3e-3_l3e-3_dt005` 523.8 ms / 0.050 % → 4.90× (hb4k04: 0.604 % @ 107.5 ms → 4.87×). hold64: 1.331 % @ 99.6 ms,
FOM 536.8 ms / 0.138 % → 5.39× (hb4kh64: 1.331 % @ 100.0 ms → 5.38×). So the incumbent's 4096² numbers reproduce.
With j=1 the error is identical (0.604 / 1.331 %) but the query costs 5308 ms (dev6) / 6558 ms (hold64) — 0.10× / 0.08×
the FOM — because the driver's exact first step evaluates the dense full-grid residual against all $M=1088$ tests.

**Certificate at $4096^2$** (population `params_draw(20260921,56)`, 5 × 8-trajectory draws + a 16-trajectory
confirmation draw, bar 0.116, ρ over $k \ge j$), recomputed independently in NumPy from the saved `population_q*.npz` /
`deployed_*.npz` and by `eqcert/audit_eqcert.py`:

- j=0: draws 5/5 (held-out ρ 0.0717, deployed 0.1072) but **confirmation FAILS** (held-out 0.1173 > 0.116, deployed
  0.1084) → **not certified at $4096^2$**.
- j=1: draws 5/5 (0.0371 / 0.0313) and confirmation passes (0.0513 / 0.0486) → certified, with margin.
- q0 `scaled`: certified (≤ 0.028 everywhere). Control `bad0` fails (0.529 / 0.579), as required.
- Diagnostic: at $k \ge 2$ the confirmation draw would sit at ρ 0.0147, so the miss is concentrated in the first one or
  two steps after the initial condition.
- The NumPy audit reproduces every status, matches ρ on 79 spot-check states to $2.7\times10^{-9}$ relative and the
  full-grid errors to $1.4\times10^{-18}$; verdict `certified_rule_exists: false`. Its only failed gate is the
  restricted-proxy gate (0.060), which fails in every $4096^2$ job of this lane.

**Consequence for the paper.** The rule behind the 4096² Burgers headline (`lat64`, j=0) is *uncertified at that mesh*
under the eqcert procedure — a thin miss on the confirmation draw only, consistent with burgers-eqcert's 2048² result
(0.1156 vs 0.116). The certified alternative, j=1, is ~50× slower, so there is no certified 4096² Burgers speed row.
Label the 4096² Burgers rows accordingly.

**Side observation (not a retraction).** The paper's fast-row speedups at $4096^2$ (12.9× dev6, 13.2× hold64) are taken
against the *accurate* row's FOM, which is the table's convention. Under the campaign protocol's per-row rule (fastest
tested FOM at least as accurate as that row) bh5's wider FOM grid gives 10.10× on both cohorts, the binding FOM being
`lean_nt1e-3_l1e-3_dt01` (406 ms, 1.66 % dev6 / 401 ms, 2.01 % hold64) — a setting hb4kh64 never ran.

**Lane verdict (bh1–bh5), unchanged.** The improved bank halves the projection floor (sel32 at 256²: 0.513 % → 0.183 %;
hold64 at 4096²: 0.295 %) but no certified arm reaches ≤ 1 % held-out error at ≥ 5×: the certified q256/M1088 rung is
1.601 % (hold64, 6.70×) / 2.407 % (fresh64, 5.48×), and the sub-0.5 % arms (q512/M2112 `lat128`) are uncertified and run
at ~2.1× the FOM. The binding term is the correction subspace and weak test space, not the bank.

Artifacts: `experiments/burgers-heldout/checks/bh5-summary.json`, `checks/bh5-eqcert-summary.json`,
`checks/bh5-audit.txt`, `checks/bh5-eqcert-audit.txt`; report `experiments/burgers-heldout/reports/
2026-09-21-burgers-heldout.md` + `reports/tables.generated.md` (new paired j=0/j=1 block) + `reports/summary.json`.
Nothing left running; lane closed.
