**NOT READY to run `configs/smk1.json`.** The staged ladder job is missing an unconditional import and will fail before reaching its gates. Several validation paths can also report matches despite failed checks.

Audited committed code at `7081bc8440bed8f8f0b3bbf189cda1087222688a`. No files changed, no cluster commands run, no scratch files created. Additional untracked implementation files appeared during the audit; they are outside this verdict.

Paths below are relative to `experiments/jcp-fom2d/`.

**1. `cluster/stage.py` — WRONG: smoke import failure**

- **WRONG — lines 18–20, 62.** C1 stages `experiments/mr-burgers2d/engines.py`, but stages `experiments/separable-decoder/sep_common.py` only for C2. `engines.py:17` imports `sep_common` unconditionally. A clean staged ladder job therefore raises `ModuleNotFoundError`.
- **CORRECT — lines 27–29, 58–74.** Files come from committed blobs, with working-copy equality checks and content-hash provenance.
- **CORRECT — lines 27–29.** The committed lane subtree includes the `quad3d` dependencies needed by `cohorts.py → common.py → b3d_common.py`. Those dependencies are present.
- **CORRECT — lines 83–111.** GPU partition, one GPU, paralab paths, absolute cluster interpreter, x64, highest precision, GPU preflight, `JCPREF_STORE`, and `JCPREF_META` are configured.
- **NEEDS-FIX — lines 43–49.** Any attempt name beginning `smk` bypasses resource certification, irrespective of its workload. Bind the exemption to an explicit smoke configuration.
- **NEEDS-FIX — lines 36, 78.** H200 is the default, but other GPU types remain permitted despite §7’s H200 requirement.

**2. `fomrun2.py` — core arithmetic CORRECT; evidence persistence NEEDS-FIX**

- **CORRECT — lines 58–80.** DST interpolation scales orthonormal coefficients by `M/L`, pads both spectral axes correctly, and restores zero walls. Restriction pads before striding, preserving lattice alignment.
- **CORRECT — lines 120–147.** FD timing dispatches solver then output map without intervening host synchronization, as A2.6 requires.
- **CORRECT — lines 150–173.** Convergence checks stage count, elapsed time units, buffer coverage, finite residuals, tolerance, and Newton bounds. A stage reaching iteration 20 with tolerance met is accepted and flagged.
- **CORRECT — lines 156–158.** PS validity requires finite fields and finite padded-grid running maximum.
- **NEEDS-FIX — lines 171–173.** Saved evidence contains summaries, not per-stage residual/count arrays or actual `units`. The offline validator cannot independently reconstruct the complete-stage predicate from persisted records.

CPU checks found coincident-node interpolation errors of `3.55e-16`, `5.65e-16`, and `4.03e-16` for `L=32,64,128 → 256`. A synthetic stage at the Newton limit with its residual met passed; missing-stage evidence failed.

**3. `psjax2d.py` — CORRECT for the audited C1 path**

- **CORRECT — lines 49–68.** ETDRK4 arithmetic matches `pscore`; all four nonlinear stage evaluations contribute to the running maximum. Five intervals plus the initial field give the correct six output times.
- **CORRECT — lines 29–47.** FFT synthesis uses the correct coefficient convention and zero walls. Full-field synthesis explicitly refuses insufficient output resolution.
- **CORRECT — lines 73–75.** IC quadrature uses E0’s 256 panels × 32 Gauss points.
- **NEEDS-FIX — lines 83–94.** Empty fixture lists pass both gate and control through `all([])`. Assert the prescribed two fixtures. The current smoke supplies them, so this is not its immediate failure.

Executed G-PS1 on both prescribed CPU fixtures: relative differences were `7.12e-16` and `3.54e-16`; running maxima agreed exactly; half-step differences were `3.59e-6` and `1.17e-5`, so both controls fired.

**4. `gates.py` — mostly correct fixtures; acceptance gaps**

- **CORRECT — lines 38–44.** G-FD1 verifies pinned hashes. I independently verified the source files and source commit, rather than relying solely on this gate.
- **CORRECT — lines 61–75.** The actual pinned dev6 cases 0 and 1 resolve to `a0_L2048`, `dt=0.00125`, `(ntol,ltol)=(1e-11,1e-9)`, central CN-R.
- **CORRECT — lines 82–93.** G-PS2 loads the named stored PS run with manifest verification and reproduces its recorded resolution/time step.
- **CORRECT — lines 96–118.** G-INT covers the registered interpolation sizes, anisotropic fields, coincident nodes, and FFT-versus-matrix PS synthesis.
- **NEEDS-FIX — lines 47–55.** G-FD2 accepts tolerance arguments but hardcodes the lane side to level `1`. Defaults are consistent; nondefault arguments silently compare unequal tolerances.
- **NEEDS-FIX — lines 121–130.** G-CT accepts an empty specification list as PASS; reproduced directly. Assert required coverage.
- **CORRECT — lines 133–138.** K-swap implements the specified shifted times and swapped-reference score sensitivity.

The small `L=32` G-FD2 check returned zero field difference and identical Newton counts. This does **not** replace the registered `L=256` gate.

**5. `ladder.py` — calibration CORRECT; acceptance and resume WRONG**

- **CORRECT — lines 234–245.** Failed `t0` makes the group unavailable. Otherwise selection chooses the loosest converged level satisfying agreement on every calibration case.
- **CORRECT — lines 265–275.** Recovery runs all production cases once at the nearest tighter **calibration-passing** level, satisfying A2.6.
- **CORRECT — line 200.** `S` scoring is restricted to UB at physical `dt=0.005`.
- **WRONG — lines 134–163, 293–316.** Failed gates and controls are recorded but never invalidate the completed result. G-PS1’s `control_fires` is also separate from `passed`. The job still writes `COMPLETE`.
- **WRONG — lines 113–117, 135–136, 221–224, 281–282.** Resume overwrites config/provenance/store metadata, then reuses old gates and configurations without checking their original identities. Changing config, code, store, or population can mix evidence under new metadata.
- **NEEDS-FIX — lines 122, 204–215.** Checkpoint JSON is overwritten nonatomically. Resume skips records without verifying their NPZ files and hashes.
- **WRONG — lines 195–197.** Screening automatically accepts non-bitwise results within `1e-12`. C1 requires hash equality; the documented fallback requires smoke evidence and a declared family policy, neither checked here.
- **NEEDS-FIX — lines 186, 194–197.** The warm-up execution is unchecked. Timing-row coverage is not enforced.
- **NEEDS-FIX — lines 299–302.** K-swap chooses the first available configuration and first configured case. The current smoke produces the intended case, but the fixed dev6 `0→1` fixture should be explicit.
- **NEEDS-FIX — lines 304–310.** Pilot measurements do not burn in before timing and do not separately persist compilation/per-stage timing.

**6. `match2.py` — tie arithmetic CORRECT; multiple false-acceptance paths**

- **CORRECT — lines 35–38.** Median and worst comparisons apply uncertainty to each case before aggregation, exactly as specified.
- **CORRECT — lines 82–113.** Offline calibration recomputes field differences and enforces the exact next tighter passing recovery level.
- **WRONG — lines 47–56, 293–301.** Completed inputs are accepted without checking smoke gates or K-loose/K-swap. Offline score disagreement is only reported, not used to block matches. Failed matcher controls likewise leave selections populated.
- **WRONG — lines 91, 126–136.** Validation trusts `converged` and `hashes_ok` booleans instead of checking saved evidence and timing rows. A CPU counterexample with `evidence.finite=False`, `umax=inf`, and a failed timing identity row was still admissible when the summary booleans were true.
- **WRONG — line 287.** Required production population is inferred from whatever records exist. A six-case smoke—or an accidentally incomplete campaign—can therefore be treated as the complete admissible ladder. Require the registered 38 cases for production and label smoke analysis separately.
- **NEEDS-FIX — lines 48–49, 145–151, 292–300.** Store versions are recorded but never compared among jobs, targets, and the active store. Equal `eta` is not proof of equal reference identity.
- **WRONG — lines 195–203.** The minimum-three check counts target eligibility, not the actual common eligible population. Eligibility disagreement can leave fewer than three common cases.
- **WRONG — lines 243–255.** K-false assumes the largest **raw-error** case is also above the median after subtracting heterogeneous uncertainties. That is false. For `e=[1,2,3]`, `eta=[0,0,2]`, the constructed perturbation changes `median(e−eta)` from `1` to `2`, while correctly passing median matching and failing worst matching. The extra `median_unchanged` assertion falsely fails the control. Current P1 targets do not trigger this defect.
- **CORRECT — lines 174–189, 256–258.** K-syn uses positive synthetic costs and checks rejection reasons plus admissibility of the inaccurate fixture.
- **WRONG — lines 301–305.** Empty/all-too-small target sets can produce `controls_all_fire=True`; a too-small target subsequently crashes the printing loop because family keys are absent.
- **WRONG — line 219.** “Projected cost” contradicts A2.4’s replacement label, “deferred.”
- **NEEDS-FIX — lines 155–171, 214–225.** Coordinate edge and screening boundary are distinguished, but step-edge reasons, extension state, and the deterministic A2.5 extension round are not represented.
- **NEEDS-FIX — lines 193–225.** These are C1 screening selections, not A2.1-confirmed selections. The output should make that distinction explicit. PS-anchor differences are not implemented here.

**7. `targets.py` — preview usable; production contract incomplete**

- **CORRECT — lines 26–56.** P1 fields are rescored through `refstore`, eligibility is honored, and the output is labelled preview.
- **NEEDS-FIX — lines 49–54.** P1 comparison values are recorded but agreement is not asserted.
- **NEEDS-FIX — lines 33–35.** An empty eligible population crashes at `max`; return the explicit insufficient-population status.
- **NEEDS-FIX — lines 49–56, 71.** FRONT ingestion, full configuration identity, and separate `|H`/`|X` populations remain unimplemented. These block production matching/C2, not the numerical smoke.

**8. `ledger.py` — sound reservation foundation; cap enforcement incomplete**

- **CORRECT — lines 31–67.** Locked read/modify/write, atomic replacement, and spent-plus-reserved reservation accounting.
- **CORRECT — lines 75–96.** Ambiguous submissions retain reservations; failed accounting queries do not automatically release them.
- **NEEDS-FIX — lines 60–66.** Reservation hours are not required to be finite and positive.
- **NEEDS-FIX — lines 59–67, 97–115.** New reservations do not first reconcile prior requeues/raised limits. Automatic requeues can consume additional hours beyond the original reservation before accounting is refreshed. Disable automatic requeue or explicitly reserve/reconcile retries.
- **NEEDS-FIX — lines 101–109.** Reconciliation ignores subprocess return status and assumes the final accounting row is the current attempt. Fail closed on query errors and establish attempt ordering explicitly.

**9. `cluster/submit.sh` — lane cap CORRECT; account cap race**

- **CORRECT — lines 13–27.** A lane lock serializes submissions; lane cap 3, account checks 8, Slurm-limit equality, and budget reservation are present.
- **CORRECT — lines 28–40.** Pre-submission failure releases the reservation; ambiguous submission retains it.
- **WRONG — lines 13, 18–26, 35.** The lock is lane-local. Two lanes can both observe seven account jobs and submit, reaching nine. Account cap 8 needs a shared submission lock/reservation mechanism.

**10. `cluster/collect.py` — checksums CORRECT; retry behavior NEEDS-FIX**

- **CORRECT — lines 25–38, 44–51.** Checksums, completion markers, `ALL-DONE`, queue absence, and terminal accounting checks protect normal cleanup.
- **NEEDS-FIX — line 41.** Any interrupted collection—or successful partial collection—leaves `archive/` present, so the next collection fails at `exist_ok=False`. Provide a resumable transfer/finalization path.
- **NEEDS-FIX — lines 28–29.** Cleanup verifies discovered output directories, not the expected staged step list. Bind completeness to the reservation/run specification.

**11. `configs/smk1.json` — correct gate fixtures; incomplete resource coverage**

- **CORRECT — lines 29–100.** Calibration cases, screening cases/repetitions, four calibration levels, gate list, K-loose fixture, and K-swap are configured.
- **CORRECT — line 144.** Includes A2.4’s informational `N=1024`, `dt=0.05/800` PS pilot.
- **NEEDS-FIX — lines 102–132.** FD resource pilots cover only `t0` and `t3`, whereas A1.6 requires every tolerance level at every family/resolution. The smoke cannot supply that promised certificate as written.
- **NEEDS-FIX — line 58.** A fixed 50 ms burn has no demonstrated adequacy for the registered timing procedure.

**12. `INHERITED.json` and inherited modules — CORRECT**

All five modules—`refcore.py`, `pscore.py`, `cohorts.py`, `verdict.py`, `refstore.py`—are byte-identical to both the specified E0 worktree and recorded commit `cc04f4b54a1b717d00eb803a276d7efb86994773`. Both copied metadata files also match their sources and pinned hashes.

- `refcore.py:151–168,205–217`: output indexing and CN-R stage schedule are consistent. The registered FD ladder needs at most 400 stages, below `NST=1300`; schedule overflow is asserted against.
- `refstore.py:93–155`: manifest verification, per-case eligibility, and scoring are preserved.
- `cohorts.py:27–32`: explains the unconditional runtime dependency chain responsible for the staging failure.

**13. `c2time.py` — draft only; obvious defects**

These are later-stage findings, not additional reasons to run C2 now:

- **WRONG — lines 188–189.** G-RR skips ROM subjects lacking `target_e`, permitting vacuous PASS.
- **WRONG — lines 263–280.** Selection uses each candidate’s own valid cases, rather than the group-wide intersection with at least five cases. Tie-breaking omits stage-0 cost. Universal `[:3]` truncation also does not implement pooled union shortlists.
- **WRONG — lines 195–200, 275–280.** Identity-vetoed candidates remain eligible for selection.
- **WRONG — lines 325–331.** T-id deliberately vetoes the real subject and leaves that veto in the final production result.
- **NEEDS-FIX — lines 291–292.** A is excluded from stage-2 subjects; using the baseline configuration itself as a target needs explicit handling.

**Ranked fixes**

1. **Fix C1 staging:** include `sep_common.py`; verify imports against the isolated staged file set.
2. **Make acceptance fail closed:** gate/control failures, offline disagreement, insufficient populations, invalid evidence, and unapproved identity fallbacks must block claims.
3. **Bind resume to immutable identities:** config, code, store, population, fields, and gates; use atomic checkpoints.
4. **Repair matcher population/store checks and K-false’s heterogeneous-uncertainty assertion.**
5. **Complete smoke resource evidence:** every tolerance level, credible burn-in, compilation and stage/call measurements.
6. **Enforce the shared account cap and retry-aware GPU-hour budget.**
7. **Before production:** finish H/X populations, FRONT ingestion, extension/anchor handling, amended labels, and the separate C2 audit.

The numerical core has encouraging CPU checks, but the **current smoke submission remains NOT READY**.