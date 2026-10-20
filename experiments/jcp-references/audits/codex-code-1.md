**Overall verdict: NOT READY for the real-size cluster smoke.**

Audited commit `d44e1837d619d2addf309e5c5fa82270a5aa3daf` against DESIGN through A4. No files modified, no GPU jobs run, and no lab-log entry written, per your read-only instruction. Checks used source inspection and small NumPy reproductions.

The principal numerical kernels look sound. The blockers are failure handling, gates that can falsely pass, missing escalation/audit paths, store eligibility, sealing, and absent memory instrumentation.

Tables below use **blocking** for defects affecting correctness, acceptance, evidence integrity, or safe execution; **non-blocking** for documentation or diagnostic limitations.

**[cohorts.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/cohorts.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 35–38, 81–92, 133–145 | CORRECT | non-blocking | Historical seeds and **exact draw counts** match `qcore.cohort`, `engines.params_draw`, and `common.table`. Historical 2D column-wise RNG is preserved. |
| 66–78 | CORRECT | non-blocking | New 2D rows use `default_rng([seed,j])`, the specified draw order, and correct family/OOD ranges. |
| 97–121 | WRONG | blocking | New 3D cohorts retain one sequential generator. A2.11, read literally under “later governs,” specifies seed sequence `[seed,j]`. This follows A1.10/vendor semantics but not A2.11. At public seed 5550002, row 1 has `B=3`; independently seeded row 1 has `B=2`. Historical 3D draws should remain unchanged. |
| 109–125, 143–178 | CORRECT | non-blocking | OOD width/viscosity substitutions preserve vendor draw consumption; all three candidate blobs are drawn. Mask, amplitude, fixed 257-grid `s_star`, and x-slowest layout match the vendor. |
| 86, 139 | WRONG | blocking | These checks reject historical-seed collisions, but no shipped generation/validation path enforces distinct seeds across all four new cohorts or records redraws. |
| 181–188 | CORRECT | non-blocking | Selection and tie-breaking are correct. Computed selections: 2D **0,1,2,3,4,5,19,26**; 3D **0,1,2,3,5,33,43,47**. |
| 184 | NEEDS-RESTATEMENT | non-blocking | Selecting 3D indices constructs the entire normalized cohort, including GPU-backed peak calculations. Selection needs only raw viscosities; this also makes collection unexpectedly invoke JAX work. |

**[refcore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refcore.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 48–86, 111–113 | CORRECT | non-blocking | Central/upwind stencils, zero walls, Laplacian scaling, and \(F=N-\nu\Delta\) have the correct signs. Upwind arithmetic follows each vendor. |
| 124–148 | CORRECT | non-blocking | Residual is \(u-u_n+\theta\tau F(u)+(1-\theta)\tau F(u_n)-\tau[\theta g_{n+1}+(1-\theta)g_n]\). CN, BE, forcing times, previous-state predictor, and Helmholtz preconditioner match A1.3. |
| 124–166, 184–199 | CORRECT | non-blocking | Four BE half steps replace two CN steps. Integer `dt/2` bookkeeping gives correct output indices. Finest/refined ST runs have 802/1202 stages in 2D and 402/602 in 3D; evidence buffer accommodates them. |
| 136–167 | CORRECT | non-blocking | Newton stopping uses the previous full-grid norm; achieved residual/count are retained per stage. Running maximum includes IC and completed stage states. Acceptance is checked separately by the caller. |
| 207–227 | CORRECT | non-blocking | 2D restriction includes walls; 3D indices are `k*(L/64)-1`, coordinate-wise, flattened x-slowest. This matches `qpanel.lattice65_index`. |
| 115, 169–180 | CORRECT | non-blocking | Production eigenvalue arrays are explicit JIT arguments, not captured large constants. |
| 160, 163 | NEEDS-RESTATEMENT | blocking | The output buffer contains **six full-grid fields**, not six lattice fields. `lax.cond(... .at[idx].set(...))` permits compiler buffer reuse but is not evidence that this execution is in-place. No compiled-memory or runtime peak measurement establishes the claim. |

**[pscore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/pscore.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 31–53 | CORRECT | non-blocking | Odd/even rFFT extensions, division by two, and analysis factor \(2/M\) are consistent for continuous sine-series coefficients. |
| 64–84 | CORRECT | non-blocking | Padding, truncation, and derivative synthesis order are correct in both dimensions. NumPy execution of these formulas agreed with independent direct tensor sums to relative errors **2.1e−15 / 2.7e−15**. |
| 3–6 | NEEDS-RESTATEMENT | non-blocking | Implementation uses unnormalized continuous sine coefficients, while DESIGN describes orthonormal coefficients/transforms. This is an internally consistent change of representation, not a demonstrated numerical error; document the conversion explicitly. |
| 95–108, 124–134 | CORRECT | non-blocking | ETDRK4 contour expressions and stages match Cox–Matthews. At zero linear eigenvalue, NumPy gives \(Q/h=1/2\), \(f_1/h=f_2/h=f_3/h=1/6\). Forcing uses the four specified stage times. |
| 160–209 | CORRECT | non-blocking | Direct lattice synthesis and composite GL coefficients are consistent. The 3D product of `4*x*(1-x)` factors gives the required mask, followed by `A/s_star`. |
| 240–287 | CORRECT | non-blocking | MMS forcing has the correct temporal, advective, and diffusion terms. Separable coefficient factors correctly use \(\varphi\varphi'\), \(\varphi^2\), and \(\varphi''\). |
| 136–154 | NEEDS-RESTATEMENT | non-blocking | `finite` summarizes lattice values and running amplitude; it is not explicit evidence of finiteness of every stage coefficient array. The final updated coefficient state is not directly checked. |
| 281–288 → 140 | WRONG | non-blocking | G4 captures three full coefficient tensors—`base`, `adv`, `lp`—inside the jitted block through the forcing closure. Thus “all large arrays are traced arguments” is false for the forcing path. Production PS does not have this capture. |

**[verdict.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/verdict.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 41–58 | WRONG | blocking | Dyadic levels bypass the `[0.25,8]` bracket. Ratios 1, 0.5, and 1024 return orders **0, −1, 10**, rather than undefined under A1.5. |
| 61–63, 89–98 | WRONG | blocking | Equal nonzero adjacent differences give `p=0`, then Richardson division by zero. **Reproduced: `judge()` raises `ZeroDivisionError`**, instead of storing an order failure. |
| 61–70, 96–105 | CORRECT | non-blocking | Within the valid order regime, Richardson factor, fixed-\(t_0\) normalization, direction residual, arithmetic floor, accumulated proxy, both floor inequalities, and running-max Péclet are correct. |
| 108–120 | WRONG | blocking | `t0_shared=False` does not prevent acceptance. **Reproduced:** all levels with IC twice the canonical IC can receive `outcome='accepted'`. Missing G5 makes this especially consequential. |
| 109–118 | WRONG | blocking | Péclet failure takes precedence over terminal floor failure. After tight solves still fail both, `solve_case` stops because of the floor, but the recorded outcome is `peclet-fail`, contrary to A2.1’s terminal `unresolved-floor` classification. |
| 151–178 | CORRECT | non-blocking | FD validity, PS validity/order/stability, resolution, and discrepancy are checked in the specified order. Compared field is the half-step field; \(\sigma=s_x+s_t\). |
| 181–185 | CORRECT | non-blocking | For the required eight-case input, FAIL suspends; otherwise six PASS validate. The collector later overrides this behavior incorrectly. |

**[gates.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/gates.py)**

| Lines / gate | Verdict | Severity | Finding |
|---|---|---|---|
| 41–60, G1 | CORRECT | non-blocking | Correct vendor solvers, meshes, ICs, dt, tolerances, and numerical bar. A discrepant field can fail. Newton equality is recorded but not required by the stated field-parity bar. |
| 65–92, G2 | NEEDS-RESTATEMENT | non-blocking | Uses A4 meshes, independent analytic forcing, positive-time errors, and correct effectivity bars. “True order at three levels” actually gates only the finest pair; the coarse level enters the self-difference estimate. State that interpretation explicitly. |
| 101–115, G3a | WRONG | blocking | **NaNs can PASS.** `max(0.0, nan)` remains zero. Running the actual gate with a stub returning all-NaN fields produced `worst_rel=0.0, pass=True`, even with `finite=False`. |
| 107–112, G3a | WRONG | blocking | Exact and computed fields share `PS.synth_lattice`; a common lattice-normalization/indexing defect is invisible. Required independent mode-by-mode transform tests are absent. |
| 118–134, G3b/c | CORRECT | non-blocking | Implement A4’s composite-rule and contour bars and can fail on finite discrepancies. |
| 118–134; job integration | WRONG | blocking | Checks cover one dev IC and one viscosity. PS jobs check the configured base finest size, not necessarily an overridden/escalated size or selected timestep. This does not establish the required coefficient checks for all runs being certified. |
| 156–172, G4 | WRONG | blocking | Spatial `N=32` output is computed but contributes neither to `q_sp` nor an error-order bar. A finite corrupted coarse field can leave G4 passing. Temporal self-difference ladder and half-step separation implement A3.4. |
| 175–186 | CORRECT | non-blocking | Projection example, exact coefficient, two observed orders, and rejection bars are correct. |
| 191–252, units | WRONG | blocking | Order unit misses dyadic unbracketed cases. RNG golden checks omit new 3D family rows and most 3D descriptor components. They do not test all new-seed collision requirements. |
| 200–220, X1b | NEEDS-RESTATEMENT | non-blocking | Proper deterministic first-order rejection/control test, but uses random synthetic fields rather than the specified stored dev6 case-0 fields. |
| Entire release, G5–G7 | WRONG | blocking | No lane `audit_ref.py` or G5 implementation; no G6 swapped/perturbed-store tests; no G7 loader-refusal test. `Ucrypto` tests encryption, not loader sealing. Consequently `all_pass=True` does not mean all DESIGN gates passed. |

**[sealio.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/sealio.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 18–51 | CORRECT | non-blocking | Required AES-256-CBC/PBKDF2/SHA256 invocation, hex-key validation, in-memory plaintext, round-trip verification, and plaintext/ciphertext hashes are implemented. |
| 54–58 | WRONG | blocking | Seed decryption never verifies the expected **ciphertext** hash. Plaintext checking is optional at the API level. A2.9 requires both checks for every decryption. |
| 46–50 | WRONG | blocking | Ciphertext writes are non-atomic and not made immutable. Combined with `mode_fd`’s existence-only resume logic, interrupted writes can be treated as completed artifacts. |
| Entire file / release | WRONG | blocking | No implemented secret-seed generation, collision/redraw ledger, or prescribed custody-file creation. Encryption helpers alone do not complete A1.11/A2.9. |

**[refjob.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 35–45 | CORRECT | non-blocking | Base/refinement meshes, dt values, schemes, production/retry/tight/tighter targets, and X1 ladders match the amendments. |
| 93–141 | CORRECT | non-blocking | For successful solver attempts: prod → all-level tight on floor failure → refine on order/Péclet failure → all-level tight if the refined triple newly fails the floor. If already tight, the new finest level is tight too. Attempt fields and scalar evidence are retained. |
| 105–122 | NEEDS-RESTATEMENT | blocking | Failed prod/refine solves get one retry, then terminate as `solver-fail`. They never reach refinement. Original §4 says “then treated as (b)”; A1.5 names a terminal solver-fail outcome without clearly withdrawing that instruction. Resolve this procedural conflict explicitly. |
| 137–166 | NEEDS-RESTATEMENT | non-blocking | Fields/evidence are in NPZ and metadata in a JSON sidecar, rather than NPZ `meta`. This is workable because the collector hashes both, but amend the schema. Job manifest currently hashes only the NPZ for non-sealed cases. |
| 101–102, 208–239 | WRONG | blocking | **Sealed per-case metadata leaks to plaintext stdout:** cohort, case index, kind, attempt, residual, Newton total, and timing. The outer sealed branch suppresses only the final summary; inner `solve_case` logging remains enabled. |
| 223–245 | WRONG | blocking | Resume skips existing JSON/ciphertext without verification and does not reconstruct prior manifest entries or QC records. A resumed completed job can publish a manifest/QC covering only its newly processed cases. |
| 185–205 | NEEDS-RESTATEMENT | non-blocking | Finite-only quantiles include denominators and failed solver attempts are counted. However, `cases` counts case-kind records, kinds are pooled, and historical failed outcomes are not tabulated separately. This is not an unambiguous frozen QC schema. |
| 251–313 | WRONG | blocking | Calibration discards `res`/`nwt`; saves only base-ladder fields, not D1/X1/X4 fields and evidence; excludes refinement-level calibration; does not install evidence under `store/<dim>/dev*/floor/`. Calibration cannot be independently reconstructed as specified. |
| 273–275, 432–440 | WRONG | blocking | Failed proxy calibration does not refuse completion or block later reference production. No implemented amendment/refusal step enforces recalibration before production. |
| 316–375 | CORRECT | non-blocking | Initial triple, finest half-step, one stability-halving rerun, and selected run IDs are implemented correctly within one job. Canonical and PS ICs remain separate. |
| 344–375 | WRONG | blocking | No ordered cross-check-driven spatial escalation. Manually overriding `Ns` starts IDs again at attempt 0 and writes the same case filenames. Collector rejects the changed files rather than preserving and selecting the escalation history. |
| 329–330, 368–371 | WRONG | blocking | A3.6 requires job gate evidence on every immutable run record. Runs carry neither that evidence nor a per-run binding to it; only a case-level Boolean is stored. |
| 398–408, 425 | WRONG | blocking | Smoke overrides are not restricted to local CPU use. PS gates use `PS_CFG[dim]['Ns'][-1]`, ignoring `cfg['Ns']`, so a larger manually supplied ladder is checked at the wrong size. |
| 409–431 | CORRECT | non-blocking | Reported gate failure refuses solver execution with exit 4. Scope is limited by the defective/missing gates above. |
| 62–79, 145–148 | WRONG | blocking | No `peak_bytes_in_use` measurement. Timings include IC generation, compilation, extraction, and transfers; there is no burn-in or separate steady-state measurement. This cannot deliver the specified smoke memory/cost calibration. |

**[refstore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refstore.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 71–98 | CORRECT | non-blocking | Sealed/OOD cohorts are refused without an opening path. Both ordinary files are SHA256-checked; `load` checks metadata identity. |
| 63–68, 119–124 | WRONG | blocking | `STATUS.json` is outside the hashed manifest, and its `store_version` is never compared with the current manifest. Stale status can grant eligibility to a different store release. |
| 121–122 | WRONG | blocking | Missing cross-check entry is treated as “not a cross-check case.” A required cross-check case omitted from STATUS becomes eligible when the dimension is validated. Required membership must come from the pinned selection, not map presence. |
| 141–156 | CORRECT | non-blocking | Pairwise identity/version checks and rereading uncertainty from the store implement A3.7; caller-provided `eps_ref` is ignored. |
| 159–181 | WRONG | blocking | **`restrict()` crashes in both dimensions.** Stored `t0` has no time axis, but `restrict_field` expects six times. Reproduced 2D `IndexError`; 3D cannot reshape 250047 elements into `(6,63,63,63)`. |
| 172–184 | WRONG | blocking | No restricted PS cross-check recomputation, still required by A2.10. A3.3 makes the result diagnostic; it does not remove that computation. |
| 110–112 | WRONG | non-blocking | Richardson fields return unchanged metadata containing the finest-field uncertainty, with no explicit diagnostic/no-uncertainty marker. This contradicts the loader contract. |

**[collect.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/collect.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 29–34, 120–141 | CORRECT | non-blocking | OLD paths, names, `f257` versus `c<j>`, and hash conventions match the historical writers. Checked an actual 2D field/hash and actual 3D shape `(6,250047)`, `n=513`, `dt=.0025`. |
| 33–34, 152–154 | NEEDS-RESTATEMENT | non-blocking | No old 3D dev4 reference is configured; absence is silently omitted. Report comparison coverage and missing cases explicitly. |
| 41–75 | WRONG | blocking | Hard links match A4, but collection does not verify pulled files against `OUTPUTS.sha256` or original result hashes before generating a fresh trusted manifest. Corruption during transfer can be blessed as the new reference. |
| 57–66, 197–205 | WRONG | blocking | Smoke fields are skipped, but smoke gate/calibration results still enter—and can overwrite—the summary. Incomplete or gate-failed ordinary jobs are not refused before linking fields. |
| 61–65 | WRONG | blocking | Sealed-file refusal only handles `.npz`/`.json`; `.enc` files are silently ignored. The collector does not enforce the stated prohibition on pulling sealed artifacts. |
| 89–117 | WRONG | blocking | Unresolved cross-checks are never escalated; there is no multi-attempt PS collection/selection mechanism. |
| 111–113 | WRONG | blocking | X2 assigns PASS/FAIL whenever `sigma` exists, including `UNRESOLVED(resolution)`. It fails to require \(\sigma_{PS}\le\epsilon_{ref}/2\), contrary to A1.14. |
| 214 | WRONG | blocking | Any MISSING cross-check forces `unvalidated`, even when another case FAILs. FAIL must take precedence and suspend the dimension. |
| 181–190, 216–230 | WRONG | blocking | **E0 ignores all sealed/OOD case-kinds.** Only historical cohorts are enumerated; QC is not incorporated. E0 can report PASS without generating either new cohort. A3.5 requires every cohort × both kinds. |
| 144–162 | CORRECT | non-blocking | Available old/new distances use positive times and the canonical fixed-\(t_0\) norm. |
| 144–162, 233–237 | WRONG | non-blocking | X3 only records individual `fires`; no cohort coverage/completeness or ≥90% finding is computed. Required PS-versus-Richardson diagnostic is also absent. |

**[cluster/stage.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/cluster/stage.py)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 38–51, 89–91 | CORRECT | non-blocking | Stages source bytes verified against the recorded commit; includes required imported vendor modules and a content manifest. Does not rely on cluster ancestor Git state. |
| 55–85 | CORRECT | non-blocking | GPU partition, absolute cluster venv, f64/highest precision, GPU preflight exit 42, paralab outputs/cache/temp, input verification, and output hashes are present. |
| 24–25, 32–34 | WRONG | blocking | No resource guard routes refinement to H200 with appropriate host memory. Defaults allow 3D L=768 refinement on an A100-80G/120G allocation, contrary to DESIGN §12. |
| 19–23 | NEEDS-RESTATEMENT | non-blocking | Provenance omits DESIGN and the submit wrapper. Executed numerical code is pinned, but governing specification/submission policy is not included in the staged evidence. |

**[cluster/submit.sh](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/cluster/submit.sh)**

| Lines | Verdict | Severity | Finding |
|---|---|---|---|
| 6–19 | CORRECT | non-blocking | Own namespace/directory, direct paralab transfer, input hash verification, duplicate-name check, and `squeue` before/after successful submission are present. |
| 11–18 | WRONG | blocking | Concurrency check and submission are not atomic. Two submitters can both observe three jobs and submit a fifth. |
| 16 | WRONG | blocking | `test ! -e ... && mkdir -p ...` is also racy. Two submitters for the same attempt can both pass and stage/submit into one directory. Exclusive `mkdir` or locking is required. |
| 18 | WRONG | non-blocking | Unrestricted `$EXTRA` can override audited partition, resources, or output paths. The documented dependency use is narrower than the actual interface. |

**Memory and time feasibility**

Exact f64 allocations, using interior states:

| Dimension / L | One state, GB | Six-output buffer, GB | Assessment |
|---|---:|---:|---|
| 2D / 8192 | 0.537 | 3.220 | Plausible on A100-80G; not measured for this solver. |
| 2D / 12288 | 1.208 | 7.247 | Potentially feasible on A100-80G by capacity, but not established; DESIGN routes refinement to H200. |
| 3D / 512 | 1.067 | 6.405 | Plausible on H100/H200; requires measurement. |
| 3D / 768 | 3.610 | 21.658 | H100-80GB is a serious OOM risk; H200 feasibility remains unverified. |

These are **not peak-memory estimates**. Additional live allocations include:

- Previous/current/Newton states, residuals, CN explicit term, eigenvalues and preconditioner denominator.
- BiCGStab vectors and its preconditioned/JVP intermediates.
- FFT extensions, complex results, transposes, and library workspace. The 2D vendor DST uses full complex FFTs.
- Retained eigenvalue arrays in `_SOLVERS`, keyed separately by Newton budget.
- Full-grid padding during 2D lattice extraction.
- Unchunked 3D IC coordinate and blob construction.

For scale, **20–30 state-equivalents at L=768 consume 72.2–108.3 GB**, before additional retained allocations. That range illustrates the risk; it is not a measured bound.

JVP is matrix-free and does not build a dense Jacobian. Production FD/PS large inputs are passed explicitly. The identified captured tensors are in G4 forcing. Buffer reuse inside `lax.cond` must be established from compiled/runtime evidence, not its source comment.

The old timings cannot establish this implementation’s runtime: tighter nonlinear solves, CN work, FFTs, retries, compilation, and refinement change the cost. No measured peak-memory field or usable revised ≤120 GPU-hour budget is produced.

**Required before approving the smoke:** repair the reproduced order/restriction/G3a failures; complete independent audit and loader fault gates; fix status/E0/X2 and escalation handling; preserve calibration evidence; eliminate sealed plaintext telemetry; add peak-memory instrumentation and resource routing; and make submission/directory acquisition atomic.