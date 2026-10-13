**Overall verdict: NEEDS-FIX — do not accept production results yet.** I found one reproducible runtime blocker and several paths that can accept invalid evidence. No files were modified, including the lab log. Validation was read-only inspection plus small CPU probes; I did not run production GPU workloads.

1. **BLOCKER / WRONG — Adaptive 3D LSPG crashes immediately.**  
   [c3common.py:199](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3common.py:199) passes raw rule data to LSPG. That data lacks `hmask`, which inherited `t2core.make_evolve` accesses unconditionally. A minimal call reproduced **`KeyError: 'hmask'`**. This breaks grid mode and the large-width escalation diagnostics.

   **Fix:** Prepare `T2.form_data` for **both** forms, once per setting/rule, and pass the prepared dictionaries to the queries.

2. **MAJOR / WRONG — Escalation can accept wholly non-stationary trajectories.**  
   [c3common.py:211](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3common.py:211) documents the non-stationarity restriction but checks only finite coefficients, reason 3 and saturation. I supplied 25 reason-zero steps: `verified()` returned **True**.

   `c3run.pair()` uses these flags to accept a converged rule immediately. Deferring pooled eligibility to offline selection does not repair that decision. A1.7 requires eligible runs, including the ≤1% pooled non-stationary condition.

   **Fix:** Separate per-run validity from pooled eligibility; enforce both when accepting each rule pair under each selection solver. Preserve the actual numerator and denominator.

3. **MAJOR / WRONG — The manufactured BE “equivalence” gate does not test solver equivalence.**  
   [test_fs.py:125](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/test_fs.py:125) compares two algebraically equivalent residual expressions evaluated on the *same supplied trajectory*. It never independently executes the vendor predictor, adaptive start or sweep.

   Replacing `fs_run` in memory with a **constant trajectory** still passed, with discrepancy **\(1.27\times10^{-16}\)**. Thus the docstring’s “trajectory recomputation” and “bitwise” claims are false.

   The other tests are not all vacuous: gate (b) compares distinct implementations, and the recorded mutations exceed their thresholds. However, the mutation step check compares against a SciPy minimizer, rather than checking the implemented residual against A2.2’s independent residual expectation; SciPy success is also unchecked. Test failure still exits with status zero.

   **Fix:** Execute an independent vendor trajectory and compare states, iterations and exits; add a deliberately broken BE solver control. Directly compare residual/Jacobian evaluations for the mutation fixtures, check optimizer success, and exit nonzero on failure. Regenerate the certificate.

4. **MAJOR / NEEDS-FIX — G-FS1 and G1a are measurements, not enforced gates.**  
   [c3run.py:270](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:270) records coefficient and step comparisons but continues regardless. Its coefficient statistic uses one global maximum denominator, potentially hiding a bad individual case. There is no A2.1 approved-equivalent path, eligibility comparison, verified-exit-count comparison or named CN negative control.

   [c2grid.py:133](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2grid.py:133) likewise records G1a without enforcing acceptance.

   **Fix:** Implement explicit per-case acceptance, including the exact/approved-equivalent distinction and required controls. Failed implementation gates must prevent result acceptance, not merely appear alongside `complete=true`.

5. **MAJOR / NEEDS-FIX — Timing hashes and drift are recorded but never checked.**  
   Both timing implementations save actual and expected hashes without comparing them. Missing expectations obtained through `.get()` also remain acceptable. Both save drift without applying A2.11’s inherited 10% gate.

   The indexing itself is correct: 2D uses actual case indices consistently, and 3D maps timing positions back to `tcases`. Nevertheless, a mismatching invocation currently remains in the timing evidence.

   **Fix:** Require nonmissing expected hashes and equality for **both flanking baselines and every subject invocation**. Mark failing panels `timing-invalid` using the specified drift criterion; make those panels unusable for selection.

6. **MAJOR / NEEDS-FIX — The 3D per-setting timing panel omits adaptive arms; GAL preparation is inside each call.**  
   [c3run.py:279](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:279) includes only fixed-sweep arms and vendor BE. A2.11 explicitly includes adaptive arms.

   Meanwhile, `Solvers.run()` performs `T2.form_data(..., 'GAL')` on every GAL invocation, repeating QR and matrix preparation. Merely adding GAL to timing would charge this repeated setup to every query, unlike the prepared 2D path.

   **Fix:** Cache form data outside query timing, then include the required adaptive subjects. Keep projection, stepping, diagnostics and six-field decoding inside timing.

7. **MAJOR / NEEDS-FIX — FOM storage violates A2.10.**  
   [c2grid.py:214](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2grid.py:214) saves lossless restricted fields, which is correct, but never archives the full-mesh float32 fields. It also omits the serialization-error bound, archive gate and explicit coordinate metadata.

   Consequently, the promised rescoring on other mesh-subset lattices cannot be performed from these outputs. References to “A4.2” in the docstring do not authorize this change: the supplied DESIGN ends at A3.

   **Fix:** Stream each full-mesh float32 archive to disk, retain float64 shared-node fields, and record mesh/coordinates plus the prescribed float32 error bound and acceptance.

8. **MAJOR / NEEDS-FIX — Escalation diagnostics do not implement the registered classification.**  
   [c3run.py:214](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:214) emits per-arm booleans and a single overall status. It does not emit A2.5’s separate `selection_pair_status` and `adaptive_diagnostic_status`, distinguishing **fail** from **unresolved**.

   Additional deviations:
   - `diagnostic=big` omits adaptive diagnostics at W256/W512 despite A1.7’s four-arm inventory.
   - Fixed-versus-adaptive sensitivities are not computed.
   - The operator population switches to the `gl80` trajectory when escalation occurs; A1.7 pins the `gl64` adaptive-BE states.
   - `nanmax`/`nanmedian` silently discard zero-denominator states rather than explicitly reporting their undefined status.

   **Fix:** Implement the two status axes and required sensitivities, restore the prescribed diagnostic inventory and operator population, and retain undefined-state counts/statuses.

9. **MAJOR / NEEDS-FIX — Required per-step evidence is discarded.**  
   [c3common.py:222](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3common.py:222) drops fixed-sweep `gn`, `rn` and damping history even though `fs3` computes them. Adaptive records discard the inherited per-step verification arrays. The driver then deletes the output.

   A1.2 explicitly requires stationarity ratios and residual norms. Aggregate counts cannot reconstruct them later.

   **Fix:** Persist these arrays, together with `a_eff`, actual step count and unambiguous exit semantics. Retain adaptive per-step verification evidence as well.

10. **BLOCKER / NEEDS-FIX — The mandatory submission resource gate is absent.**  
    The drivers’ `--dry-run` paths print partial summaries, not A2.12’s complete inventory. They omit timing invocations, decodes, relevant conditional successors and other work. No measured peak-memory/time certificate is required by [stage.py:39](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/cluster/stage.py:39) or submission.

    There is also no cluster 3D smoke configuration exercising the complete required path. `smk2d` alone cannot validate this lane.

    **Fix:** Generate the complete conditional inventory, measure representative production shapes and compilation costs, and reject submission unless the 1.5× wall-time projection and peak-memory estimate fit the requested allocation.

11. **MAJOR / NEEDS-FIX — Large-rule memory accumulates; feasibility is unestablished.**  
    [c3run.py:199](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:199) keeps every constructed rule in `datas` until the setting finishes.

    For R′=1024, complete \(M=4097\), **B+D+P alone** occupy approximately:

    | Rule | GiB |
    |---|---:|
    | \(48^3\) | 5.063 |
    | \(56^3\) | 8.040 |
    | \(64^3\) | 12.002 |
    | \(80^3\) | 23.441 |

    That is **48.546 GiB** before mesh banks, scoring banks, ladder rules, construction temporaries, Jacobian temporaries and executables. The staging default is A100-80G, although c3desc is registered for H200.

    `live_G65` additionally retains every setting’s scoring bank, and each rollout converts it to NumPy again. At mesh 65 these duplicate mesh-node evaluations. Static scheme/step/K signatures, rule shapes and `unroll=True` also multiply compilation work. I have **not** measured an OOM; the current code cannot establish that the jobs fit.

    **Fix:** Bound rule residency, rebuild or reload rules for timing, reuse bank prefixes/scoring arrays, and measure peak memory and compilation time on the intended GPU. Gate GPU selection accordingly.

12. **MINOR / NEEDS-FIX — The common 3D panel uses the wrong random seed.**  
    [c3run.py:165](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:165) seeds every panel with `seed + len(rep['timing'])`. After five setting panels, the common panel uses **20261018**, not A2.11’s **20261013**. Its effective seed is not saved.

    **Fix:** Pass explicit panel seeds, use the registered common-panel seed, and persist each effective seed.

The following checks were **OK** on inspection:

- **`fs3` mathematics:** Residual, simplified Jacobian, history advancement, BE startup, cached-Jacobian predictor and configured `a_eff` values are consistent. All four mutation implementations have the intended A2.2 meaning, including the unsimplified derivative for wrong weights.
- **BE arithmetic:** The residual branch directly calls vendor `_lm_step_parts`; sweep, predictor, initial damping, adaptive budget and tolerance match the vendor at `dt0`, K=1. This supports arithmetic equivalence, but does not replace a functioning trajectory gate.
- **3D distance:** The einsum is correct. The caller supplies `Lt=L.T`, so it computes \(\Delta C L\), whose squared norm is \(\Delta C\,LL^\mathsf T\Delta C^\mathsf T\).
- **Escalation direction:** The code correctly tries `gl56` against `gl64`, then `gl64` against `gl80`, using the two fixed selection arms without an adaptive veto. Eligibility remains defective as finding 2 explains.
- **2D runs and metrics:** Production inventory, four anchor runs, ST/S/TX formula, physical-initial-field normalization and exclusion of \(t=0\) from maxima align with the design. At L=1024, stride four produces 257² nodes.
- **Schedules:** All configured steps divide 0.05, including factors 5/3 and 5/2. Production configs match the generator. The 3D common panel includes every configured potentially selectable fixed-sweep arm.
- **FOM calibration:** The tight pair, candidate linear tolerance, per-case 1%-of-tight-ST-error criterion, all-dev6 requirement and tight fallback are implemented correctly. Order fields and evaluation residual statistics are collected.
- **Staging:** Import inspection found no missing repository modules for either production driver. Certificate source hashes match. Reference copying, manifest checking, locked three-job lane cap, GPU preflight and checksum-before-deletion collection are structurally sound. Shell syntax checks passed. The copied 2D package cannot independently rerun `test_fs.py` without its 3D vendor dependencies, but the 2D driver does not import that test.

**Planned work, not counted as defects:** budget jobs and finalist/fallback qualification; independent scoring and reference registry; frozen selection and hypothesis evaluation; timing aggregation and matched H5 ratios; historical reproduction diagnostics; budget controls and uncertainty analyses; generated report and glossary. Those later stages must consume qualified evidence rather than treating `COMPLETE` as acceptance.

**Overall verdict: NEEDS-FIX.** Repair the adaptive crash, false acceptance paths and resource gate before production submission; regenerate the manufactured certificate after fixing its BE test.