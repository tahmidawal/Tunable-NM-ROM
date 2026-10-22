**NOT READY for the registered real-size smoke at HEAD `6669c9944`.** The E5 mathematics checks out; acceptance, configuration, and evidence-validation defects remain.

Read-only audit completed: no files changed, no jobs submitted, no GPU code run, no LAB-LOG writes. Small CPU checks used synthetic arrays. The pre-existing `HEARTBEAT.json` modification was unchanged.

Audit-1 disposition:

| Item | Status | Findings after fixes |
|---|---|---|
| 1. Tables/provenance | **FIXED / NEW ISSUE** | Fitted fingerprints and point/weight hashes are refreshed; duplicated metadata and mass-rule documentation corrected. New: fingerprints exclude `T`, although E5 consumes `T`; NaN fixtures use a constant identity. |
| 2. Certificate | **NOT FIXED completely** | Solver and certifier now handle exact zero correctly. `hjob_main.py:144` `_np_eta()` still divides unconditionally, yielding NaN for zero/zero. |
| 3. Q1/tolerances/populations | **NOT FIXED completely** | A7/E5R now authorize the implemented tolerances. Empty dictionaries fail; production Q1 case IDs are checked. Required step counts, anchor populations, Q2 populations, and array shapes remain unchecked. CPU reproduction: one case with empty `eta` and `eta_rec` still **passes Q1**. |
| 4. Q2 | **FIXED** | A7 precedence and separate binding/projected parent downgrades are implemented. |
| 5. Fitting | **FIXED / NEW ISSUE** | All-dropped handling, finite-system checks, test64 leakage checks, validity-only replacement, saved snapshots and row hashes are corrected. Per-fit `stepper_sha256` hashes `gstep.py`, not the executing `e5/fscore.py`; bind both. |
| 6. Controls/acceptance | **NOT FIXED** | C2 validity, saved C3 states, C5 repetitions, required controls, P4 fixture/dev coverage and P5 coverage improved. Acceptance still trusts several summary predicates; scoring does not gate qualification on accepted controls. A new disagreement-overwrite defect is described below. |
| 7. Timing | **NOT FIXED completely** | Actual timed B outputs are compared with scored outputs; evolution parity, repetitions and descriptive ratios are retained. Defaults are 5/3, but **`configs/smk1.json:4` explicitly overrides them to 3/2**. Timing population/settings and actual E5-table identities are not enforced offline. |
| 8. Offline scoring | **NOT FIXED** | Distances, projected rho, labels, E0 pin and decoder identity checks improved; invalid-reference crash corrected. Population/shape checks remain incomplete; F3 unresolved precedence still fails; control acceptance remains disconnected. |
| 9. Resource certificate | **NOT FIXED** | Complete-smoke and phase checks, config hash and expanded candidate projection added. Missing device/source/evidence/ledger binding, counter fallback and full ordered deferrals remain. Controls are omitted from projected phases; unavailable counters cause an empty `max()` failure. |
| 10. Staging/submission | **NOT FIXED completely** | Lane-cap check moved inside the shared lock; staging requires production certificates. A `smk*` name bypasses production checks without validating smoke configuration. Submission does not revalidate acceptance/certificate bindings. `submit.sh:16` still invokes forbidden bare `python3`. |

**E5 adapter assessment**

- **Bank coordinates are correct.** With \(\mathcal M=LL^\mathsf T\), the code diagonalizes \(L^{-1}\mathcal K L^{-\mathsf T}\), takes \(V=L^{-\mathsf T}W\), and obtains \(V^\mathsf T\mathcal MV=I\). Thus \(c=Vz\), \(P_z=WG_qV\), and both advection and diffusion have the correct negative signs.
- Synthetic CPU verification gave relative RHS disagreement **\(2.44\times10^{-16}\)** and mass-orthogonality error **\(3.33\times10^{-16}\)**.
- **Sine construction is correct:** `fscore.precontract()` receives the weighted tested rows and a shared diffusion basis. Both copies of `fscore.py` match the recorded SHA256.
- **Output mapping is correct:** E5 already returns `W` in coefficient coordinates; the adapter transforms saved `Z` once and maps reason 3→2. Synthetic output/state parity was exact; G-cert passed with maximum eta difference **\(2.86\times10^{-16}\)**.
- `tolratio*tolf` recovers the Gram-relative residual at ordinary scales. Strictly, the denominator includes E5’s `1e-300` regularizer, so it is not an exact implementation of the zero-scale three-case convention. Exact zero trajectories work.
- **P2 is appropriately wired**, but compares outputs without requiring both tight runs to finish by tolerance. Add validity/tolerance checks and explicit E5 eigen/map gates; currently table checks are recorded, not enforced.

**Required fixes before accepting the real-size smoke**

1. **Repair smoke acceptance.** At [smoke_accept.py:70](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/smoke_accept.py:70), `{**ana['q1'], **ana['q2']}` overwrites Q1 records sharing an arm name. CPU reproduction: a Q1 disagreement for R plus a clean Q2 record produces an empty disagreement set. Preserve both namespaces, reject G-cert disagreement explicitly, and bind acceptance to the exact run/config/source.

2. **Restore registered timing to 5 repetitions / 3 burn-ins in `smk1.json`.** Enforce these settings in acceptance rather than relying on defaults.

3. **Harden the E5 cache and table evidence.** [Job.tab():138](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hjob.py:138) keys only on `P` identity. Holding `P` prevents Python ID reuse, but does not identify `Gq`, `Gs`, or `T`. Changing `Gq` while sharing `P` returned the identical stale table in a CPU reproduction. Current constructors generally allocate fresh `P`; this is a latent correctness defect, not evidence that existing arms collide. Key all consumed arrays, fingerprint the actual E5 tables, and evict their cache entries in `drop()`; currently dropped tables remain retained.

4. **Enforce structural evidence checks.** Require 51 states, 50 residual/reason entries, six outputs, the exact dev anchor IDs, and the registered Q2 population. Require both P2 runs to be valid and tolerance-converged; reject failed eigen/map checks.

5. **Use the absolute Python path in submission**, and validate smoke configuration/reservation independently of the attempt-name prefix.

**Additional production blockers, non-blocking for collecting diagnostic smoke data**

- [hscore.py:376](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/hscore.py:376) still treats demonstrated Q1 failure as resolved even with invalid Q2 checks. Six such candidates reproduce **“fired (tested failure)”**, contrary to A2.3’s unresolved precedence.
- Qualification must consume accepted control evidence. Timing eligibility must require full case coverage, registered repetitions/burn-ins, and verified table identities.
- Complete resource/device/source binding, mandatory-work accounting, memory-counter fallback and ordered deferrals before production submission.
- Finish zero-scale NumPy handling and explicit E5 provenance. F2+F3’s comparison with F2(m*) currently adds accuracy only, without a corresponding cost comparison.

**Timing callables themselves pass inspection:** tables and initial coefficients are prepared before timing; compiled fused/evolution functions receive large arrays as explicit arguments. The Python lambdas capture arguments, but do not create large JIT closures. No table construction occurs inside the timed calls.