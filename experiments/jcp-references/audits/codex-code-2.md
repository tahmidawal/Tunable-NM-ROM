**Overall verdict: NOT READY for the specified real-size smoke wave.**

Audited HEAD `1c224902a07c23f297194cf23336de23284acdcb` against DESIGN through A5. No files modified, no GPU jobs run, and no lab-log entry written, following your read-only instruction. Validation used source inspection, in-memory NumPy reproductions, Python syntax parsing, and shell syntax checks.

The three configs use real sizes and bypass the production-calibration prerequisite. The remaining blockers concern trustworthy calibration, retained measurement evidence, collection, and gates that can misclassify results. **The 12-hour allocation looks plausible from the old 2D timings, but is not established.**

“Blocking” below means a correctness, required-evidence, or execution defect. Some blockers affect later collection/reporting rather than execution of these three jobs.

**Disposition of every previous WRONG / NEEDS-RESTATEMENT finding**

The numbering below follows those 52 findings in their original order. “CORRECT” means the specific previous defect is repaired; new defects are discussed afterward.

| # | Previous finding | Current verdict | Severity | HEAD disposition |
|---:|---|---|---|---|
| 1 | `cohorts`: new 3D RNG conflicts with A2.11 | CORRECT | non-blocking | A5.8b explicitly preserves the vendor’s sequential 3D generator. Specification repair. |
| 2 | `cohorts`: no cross-cohort seed collision enforcement | CORRECT | non-blocking | `make_seeds.py` rejects historical, public-check, and previously drawn new seeds; records redraw count. |
| 3 | `cohorts`: cross-check selection computes GPU-backed normalization | CORRECT | non-blocking | Selection now uses raw parameter-table viscosities only. |
| 4 | `refcore`: six full-grid outputs; unsupported memory/in-place claim | NEEDS-RESTATEMENT | blocking | Outputs remain full-grid; runtime high-water instrumentation exists, but `# in place` remains unproved and measurement limitations remain. |
| 5 | `pscore`: continuous versus orthonormal coefficients | NEEDS-RESTATEMENT | non-blocking | Implementation remains internally consistent, but DESIGN’s normalization convention still differs; conversion is not documented. |
| 6 | `pscore`: incomplete stage-finiteness evidence | NEEDS-RESTATEMENT | non-blocking | Final coefficients are now checked. Every intermediate coefficient array is still not explicitly checked; running amplitudes provide indirect detection. |
| 7 | `pscore`: G4 captured forcing tensors | CORRECT | non-blocking | A5.7 explicitly permits these three small G4 tensors. The module’s blanket “all large arrays” wording should still be narrowed. |
| 8 | `verdict`: dyadic order bypasses bracket | CORRECT | non-blocking | Original ratios 1, 0.5, 1024 now return `None`. |
| 9 | `verdict`: `p=0` Richardson division crash | CORRECT | non-blocking | Original reproduction no longer reaches Richardson with zero order. |
| 10 | `verdict`: mismatched IC accepted | CORRECT | non-blocking | `ic-mismatch` now blocks acceptance with highest precedence. |
| 11 | `verdict`: Péclet overrides terminal floor failure | CORRECT | non-blocking | A5.2 precedence implemented. |
| 12 | `gates`: G2 “three-level true order” wording | NEEDS-RESTATEMENT | non-blocking | True-error order still uses only the finest pair; all three enter self-difference effectivity. |
| 13 | `gates`: G3a NaNs pass | CORRECT | non-blocking | Explicit finiteness checks now reject them. |
| 14 | `gates`: G3a shares exact/computed lattice synthesis | CORRECT | non-blocking | Exact solution now uses independent direct NumPy sums; transform tests added. |
| 15 | `gates`: coefficient checks do not cover selected PS runs | CORRECT | non-blocking | Each PS run now checks its actual IC, size, viscosity, and timestep. |
| 16 | `gates`: G4 ignores coarse spatial result | CORRECT | non-blocking | Both spatial pairs now contribute to acceptance. |
| 17 | `gates`: incomplete order/RNG units | WRONG | non-blocking | Order edge cases and complete 3D golden descriptors added. Collision rejection/redraw behavior still lacks the specified unit test. |
| 18 | `gates`: X1b uses random fields instead of stored dev6 fields | NEEDS-RESTATEMENT | non-blocking | Unchanged; A2.8’s specified stored-field construction is not implemented. |
| 19 | Release: G5–G7 absent | WRONG | blocking | Files exist, but G5 is incomplete and G6/G7 execution is broken by the loader issues below. |
| 20 | `sealio`: seed ciphertext/plaintext verification missing | CORRECT | non-blocking | Both expected hashes are mandatory and checked in the required order. |
| 21 | `sealio`: non-atomic mutable ciphertext writes | CORRECT | non-blocking | Helper and job artifact writes use replacement plus read-only permissions. New seed-file exception discussed below. |
| 22 | Release: secret-seed/custody generator absent | CORRECT | non-blocking | Generator exists and never prints seed values. |
| 23 | `refjob`: terminal solver failure conflicts with refinement wording | CORRECT | non-blocking | A5.1 explicitly resolves the conflict. |
| 24 | `refjob`: JSON sidecar versus NPZ metadata schema | NEEDS-RESTATEMENT | non-blocking | Both files are now manifested, but metadata remains a sidecar rather than the specified NPZ `meta`. |
| 25 | `refjob`: sealed per-attempt stdout leakage | CORRECT | non-blocking | Inner logging obeys `QUIET`; ordinary sealed progress contains only item counts. |
| 26 | `refjob`: unsafe resume and incomplete resumed manifests | WRONG | blocking | Ordinary resumes restore manifest entries but bless current bytes without comparison to a previous trusted hash. Sealed refusal is per requested case, not an upfront refusal of any directory containing ciphertext. |
| 27 | `refjob`: ambiguous pooled QC schema | CORRECT | non-blocking | Per-kind counts, zero outcomes, failed-attempt counts, and quantile denominators are present. |
| 28 | `refjob`: calibration evidence discarded/incomplete | WRONG | blocking | Fields and residual/Newton arrays now survive, but D1/X1/X4 attempt records are discarded, and no floor-store installation exists. A5 legitimately removes refinement-level calibration. |
| 29 | `refjob`: failed calibration does not block production | CORRECT | non-blocking | FD now requires a matching, passing calibration-status file. The status’s own validity check remains defective, below. |
| 30 | `refjob`: no ordered PS escalation/history | WRONG | blocking | Distinct ladder attempts and filenames exist, but no implemented unresolved→one-escalation workflow or enforcement of the one-escalation limit. |
| 31 | `refjob`: PS runs lack job-gate binding | CORRECT | non-blocking | Runs carry `job_gates_ok` and their individual coefficient checks. |
| 32 | `refjob`: cluster smoke overrides/wrong-size PS gates | CORRECT | non-blocking | Staging and Slurm execution reject overrides; PS gates use the selected size. Local GPU smoke is permitted by A4.4. |
| 33 | `refjob`: absent memory/usable timing instrumentation | WRONG | blocking | AOT and device high-water fields added, but no burn-in; “solve” includes setup/transfers, and control-attempt measurements are discarded. |
| 34 | `refstore`: stale STATUS can grant eligibility | CORRECT | non-blocking | Store-version mismatch is rejected. A new list-comparison defect prevents normal status use. |
| 35 | `refstore`: missing cross-check entry grants eligibility | CORRECT | non-blocking | Membership comes from the pinned list; missing required verdict prevents eligibility. |
| 36 | `refstore`: restriction crashes | WRONG | blocking | 2D repaired; 3D still reshapes a one-frame IC as six frames. |
| 37 | `refstore`: restricted PS cross-check absent | WRONG | blocking | Implemented for 2D ST; 3D cannot reach it because restriction crashes. |
| 38 | `refstore`: Richardson uncertainty mislabeled | CORRECT | non-blocking | Diagnostic marker and cleared uncertainty fields added. |
| 39 | `collect`: missing old 3D dev4 silently omitted | CORRECT | non-blocking | `old_coverage` now exposes an empty dev4 coverage entry. |
| 40 | `collect`: transferred case files trusted without original hashes | CORRECT | non-blocking | Each linked case file is checked against `OUTPUTS.sha256`. Result/calibration files remain an exception, below. |
| 41 | `collect`: incomplete, failed, smoke results collected | CORRECT | non-blocking | Explicitly excluded before case linking and summarization. |
| 42 | `collect`: ciphertext silently ignored | CORRECT | non-blocking | `.enc` artifacts under local run outputs cause refusal. |
| 43 | `collect`: no multi-attempt selection/escalation | WRONG | blocking | Latest ladder selection exists; unresolved cases still do not trigger or require their prescribed escalation. |
| 44 | `collect`: X2 evaluated without PS resolution condition | CORRECT | non-blocking | Requires `sigma <= eps_ref/2`. |
| 45 | `collect`: MISSING overrides FAIL | CORRECT | non-blocking | FAIL now suspends before other statuses are considered. |
| 46 | `collect`: E0 ignores sealed/OOD cohorts | CORRECT | non-blocking | QC counts now enter E0; missing cohorts prevent PASS. Duplicate-coverage vulnerability remains, below. |
| 47 | `collect`: no X3 aggregate or PS/Richardson diagnostic | CORRECT | non-blocking | Per-kind fractions, threshold result, available coverage, and PS/Richardson distance added. |
| 48 | `stage`: insufficient 3D FD resource guard | CORRECT | non-blocking | A5’s H200/≥200G host requirement enforced for 3D FD. |
| 49 | `stage`: DESIGN/submit omitted from provenance | CORRECT | non-blocking | Both included in staged committed inputs. |
| 50 | `submit`: concurrent cap check/submission race | CORRECT | non-blocking | Local `flock` serializes this lane’s wrapper, as A5.6 specifies. |
| 51 | `submit`: remote directory acquisition race | CORRECT | non-blocking | Exclusive final `mkdir` implemented. |
| 52 | `submit`: unrestricted additional sbatch flags | CORRECT | non-blocking | Only the prescribed `afterok` dependency syntax is accepted. |

**New-file and changed-code findings**

| Location | Verdict | Severity | Finding |
|---|---|---|---|
| [`refstore.py:65`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refstore.py:65) | WRONG | blocking | **Every normal `_status()` call rejects the pinned list.** The left side sorts strings lexically; the right side formats numerically sorted indices. For 2D, `val32/19` sorts before `val32/2`; analogous failure occurs in 3D. This breaks `reference_status`, `score`, `resolved`, and completion of G7. |
| [`refstore.py:179`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refstore.py:179), 189 | WRONG | blocking | Adding `[None]` fixes 2D IC restriction only. 3D still forces `(6,63,63,63)`. Reproduced: `ValueError: cannot reshape array of size 250047 into shape (6,63,63,63)`. |
| [`gates.py:276`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/gates.py:276) | WRONG | blocking | **Utransforms repeats the original NaN-pass bug.** Running the actual function with all transform results stubbed to NaN returns `worst_rel=0.0, pass=True`. `max(worst,nan)` hides the discrepancy; no finite-value condition follows. |
| `gates.py:142–146`; `refjob.py:411–419` | WRONG | blocking | Contour aggregation has the same pattern: a finite first coefficient discrepancy followed by a NaN discrepancy can leave the finite maximum and pass. Each coefficient comparison needs an explicit finite result. |
| [`refjob.py:337`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:337) | WRONG | blocking | **Calibration can pass failed solves.** `proxy_ok` checks only field differences against proxies; it ignores each record’s `steps_ok`, stage completeness, and proxy finiteness. Reproduced logically with the actual proxy functions: residual `inf` gives `steps_ok=False`, `tau=inf`, yet a zero field difference satisfies the calibration inequality. Missing the tighter target is allowed by A3.2; accepting non-finite/failed evidence is not valid calibration. |
| `refjob.py:320–324, 348–388` | WRONG | blocking | `keep()` saves arrays but returns the attempt record without retaining it. D1/X1/X4 callers discard that return. Their compile/solve times, peaks, achieved acceptance, running maxima, and other attempt metadata disappear. X4 substitutes `umax=0`. |
| [`collect.py:50`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/collect.py:50), 85–89 | WRONG | blocking | **Collecting only this smoke wave fails on a fresh checkout.** `store/` is absent at HEAD. Gates/calibration outputs create no case links, and `write_manifest()` writes into the nonexistent directory without creating it. |
| `collect.py:60–81, 242–243` | WRONG | blocking | The collector verifies case files, but not `result.json` or calibration NPZs. It trusts calibration/gate/QC status from an unchecked result, does not verify `calib_fields_sha256`, and neither links nor manifests the floor evidence. `pull.sh` helps only if it was used successfully and the files stayed unchanged afterward. |
| `collect.py:244–252`; `make_configs.py` | WRONG | blocking | Sealed coverage is summed by counts without checking disjoint configured case sets. Two jobs covering the same 32 cases can count as 64 distinct cases and conceal missing coverage. Generated configs are disjoint, but the collector does not enforce that invariant for retries/replacements. |
| `verdict.py:46–50` | WRONG | non-blocking | Finite positive inputs can still crash if their ratio underflows: `solve_order(1e-300,1e300,.5,.25,.125)` raises `ValueError: math domain error`. This should return undefined. Not an expected physical-scale smoke failure. |

**G5: independence, identity, fault detection, and semantics**

[`audit_ref.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:1) is a substantial improvement, but does not yet implement the specified independent audit.

| Check | Verdict | Severity | Result |
|---|---|---|---|
| Independent of `verdict.py`, `refcore.py`, and `refstore.py` | CORRECT | non-blocking | None is imported. Order, norms, Richardson factors, direction, floor, and outcomes are independently implemented. |
| Strict “NumPy, no JAX” requirement | WRONG | non-blocking | `draws_2d()` imports `engines`; `table_3d()` imports `b3d_common`. Both import JAX, and the latter also Optax. The called generators are NumPy, but the audit is not dependency-independent of JAX. |
| Pinned case identity | CORRECT | non-blocking | Expected descriptors come from fixed dimension/cohort/index mappings, not stored metadata. Independent 2D IC exactly matched the vendor formula in the CPU check. The 3D masked-blob formula and fixed 257-grid maximum match the definition by inspection. |
| Swap fields and metadata together | CORRECT | non-blocking | In-memory synthetic val32 records were swapped; the independent IC check detected the wrong identity even with the metadata case index rewritten. |
| Perturb middle level at \(t_3\) by 1% | NEEDS-RESTATEMENT | blocking | The synthetic fault was detected. However, the test accepts any outcome change or mismatch at its \(10^{-12}\) comparison tolerance; A1.13 specifically requires a \(p\) or uncertainty change exceeding \(10^{-6}\). The code never measures that bar. |
| Finite-input order/uncertainty/outcome formulas | CORRECT | non-blocking | They implement the intended formulas and A5 precedence for ordinary finite valid inputs. |
| Exceptional-value outcome equivalence | WRONG | blocking | `if t0dev > threshold` differs from production’s `not(t0dev <= threshold)` for NaN. Reproduced NaN IC: production returns `ic-mismatch`; audit returns `solver-fail`. Missing/non-finite metadata can also cause exceptions rather than audited failure records. |
| Agreement tolerance near zero | WRONG | blocking | Pure relative comparison of \(c\) falsely fails pristine data when independent arithmetic differs by roundoff. Reproduced \(c\approx1.7164\times10^{-12}\), differing by only \(1.22\times10^{-19}\), nevertheless reported as a mismatch. Another small test failed with absolute difference \(3.64\times10^{-17}\). |
| Required FD numbers and history audited | WRONG | blocking | Only final attempts and selected summary quantities are checked. Stored `n0`, `d12`, `d23`, `eps_t`, Péclet, descriptors, and check flags are not compared. Superseded attempts are not independently audited. |
| Cross-check numbers audited | WRONG | blocking | Only the verdict label is compared. No comparison of stored \(q,s_x,s_t,\sigma,x_{PS},\eta\); no independent recomputation of gate/stability flags from run evidence. |
| Cross-check exceptional-value semantics | WRONG | blocking | With a NaN half-step field but stale `finite=True`, production returns `UNRESOLVED`; G5 returns `FAIL`. Reproduced in memory. |
| Old-versus-new distances | WRONG | blocking | Promised in the docstring and A1.13, but no implementation exists. |
| Completeness and command failure | WRONG | blocking | It audits existing files without enforcing required coverage. Both audit scripts also exit successfully after writing `all_pass=False`, so exit status alone cannot gate reporting. |

The independence is therefore **real but incomplete**: it successfully catches the intended swapped identity and a representative perturbation, yet neither verifies all required evidence nor matches production failure semantics.

**G6/G7, seeds, configs, resources, and transport**

| File/path | Verdict | Severity | Assessment |
|---|---|---|---|
| `test_refstore.py:25–66` — G6 | CORRECT | non-blocking | Pristine-load control and swapped/perturbed file checks use copies and the original pinned manifest. They should exercise hash refusal correctly once a store exists. |
| `test_refstore.py:69–81` — stale-status test | WRONG | blocking | Temporarily overwrites the real canonical `STATUS.json`, rather than an isolated copy. Concurrent readers see corrupt status; process termination can leave it behind. I did not run this mutating test. |
| `test_refstore.py:82–114` — G7 | WRONG | blocking | Sealed refusals are correctly exercised under A3.8. The later normal `score()` call hits the deterministic sorting defect, so the suite cannot finish. Only 2D restriction is tested, missing the surviving 3D crash. |
| [`make_seeds.py:25`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/make_seeds.py:25) | CORRECT | non-blocking | Correct secret draw source, collision set, collision-only redraw rule, custody key format/permissions, exclusive creation, memory-only plaintext, and hash-only stdout. No seed values are printed. |
| `make_seeds.py:57–59` | WRONG | blocking | Seed ciphertext is written directly to its final filename, unlike other ciphertexts. An interrupted/short write leaves a final-named partial file; the retained key then makes rerunning refuse. This violates A5.5’s atomic-ciphertext requirement. |
| [`make_configs.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/make_configs.py:1) | CORRECT | non-blocking | Historical chunks are complete/disjoint; PS selections match independent NumPy recomputation. Sealed configs contain paths/hashes, not seeds. The three smoke configs contain no override. |
| `cohorts.py:143–167` | CORRECT | non-blocking | Chunked coordinates preserve x-slowest ordering, active blobs, mask, and fixed normalization. It bounds coordinate temporaries while retaining the full host output. The unit compares against the vendor at a small size. |
| `cohorts.py:203–215`; `gates.py:247–273` | CORRECT | non-blocking | Cross-check selections reproduced exactly; extracted NumPy RNG unit passed against committed golden values. |
| `refcore.py:180–199` | NEEDS-RESTATEMENT | blocking | AOT compilation is implemented correctly for the fixed argument shapes/types. `peak_bytes` is a process-lifetime device high-water mark, not the isolated peak of that attempt; missing stats silently become `None`. |
| `refjob.py:72–98` | WRONG | blocking | No explicit burn-in before timed blocks. `solve_seconds` includes solver construction, eigenvalue setup and input transfer, less compilation. It is not a clean steady-state solve measurement. |
| `cluster/stage.py` | CORRECT | non-blocking | Requested smoke resources are accepted; code/config/specification are pinned; f64, highest precision, GPU preflight, paralab paths, and 12-hour wall time are present. Calibration has no production-status prerequisite. |
| `cluster/submit.sh` | CORRECT | non-blocking | Implements A5.6’s local lock, exclusive remote directory, cap, restricted dependency, and queue checks. Actual queue availability was not checked. |
| [`cluster/pull.sh`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/cluster/pull.sh:1) | CORRECT | non-blocking | For successfully completed jobs, excludes `.enc`, pulls logs/results/nonsealed evidence, and verifies nonencrypted output hashes. Shell syntax passes. |
| `cluster/pull.sh` completion/cleanup | NEEDS-RESTATEMENT | non-blocking | Assumes completion rather than checking it; does not remove verified nonsealed remote job directories. Cleanup remains a separate operational step. |

**The proposed three-job smoke wave**

| Requirement | Verdict | Severity | Assessment |
|---|---|---|---|
| Gates job uses real sizes | CORRECT | non-blocking | Runs both dimensions, including G3b/G3c at escalation sizes \(N=3072\) and \(384\), on the requested A100-80G. |
| Calibration uses every base level, both kinds, three tolerances, two cases | CORRECT | non-blocking | Exactly 36 base calibration solves per dimension. No smoke override. |
| Jobs bypass missing calibration-status refusal | CORRECT | non-blocking | Refusal applies to `mode='fd'`; these jobs are `gates`/`calib`. |
| Jobs will necessarily pass pre-run gates | NEEDS-RESTATEMENT | non-blocking | No config-based refusal found. Numerical gate success on the allocated GPUs is unmeasured; a gate failure legitimately stops the job. |
| A2.1/A3.2/A5.4 trustworthy calibration | WRONG | blocking | Raw base evidence is saved, but failed-solve acceptance can produce a false passing calibration status. |
| D1 evidence | WRONG | blocking | Required ladders, fields, residuals, counts, orders and magnitudes are produced; their attempt records/timing/memory are discarded. |
| X1 evidence | WRONG | blocking | All dev cases are run and orders/outcomes retained; attempt records are discarded. |
| X4 evidence | WRONG | blocking | Correct 2D case/middle-timestep perturbation; raw arrays and order retained, but attempt record discarded and Péclet amplitude fabricated as zero. |
| Floor evidence reaches specified store location | WRONG | blocking | Collector does not install it; fresh smoke-only collection also fails because `store/` is absent. |
| Production prerequisite handoff | NEEDS-RESTATEMENT | non-blocking | Checker expects the bare `calib_status` fields in `checks/calib_<d>d.json`; copying the complete `result.json` there would be refused. No exporter is provided. |
| Timing/memory re-estimate | WRONG | blocking | Partial instrumentation is useful, but no burn-in, no control-attempt records, no refinement measurements, and no production-size PS trajectory timing. This wave cannot establish all costs in §12. |
| 12-hour feasibility | NEEDS-RESTATEMENT | non-blocking | Plausible for 2D under moderate slowdown; neither 3D runtime nor actual peak memory is established without GPU measurements. |

A5.9 correctly defers G5–G7 until the collected reference store exists. These three jobs alone cannot run those store gates, and their absence during this wave is **not itself** a violation.

**2D calibration cost estimate**

Using the requested old reference cost and an \(L^2\)-per-stage model,

\[
T(L,s)\approx355\text{ s}\left(\frac{L}{8192}\right)^2\frac{s}{800}.
\]

CN-R has 202, 402, and 802 stages at the three ST base levels. This estimate assumes equal per-stage cost across schemes and tolerances; that assumption must be measured.

| Work in `calib2d` | New solves | Estimated seconds |
|---|---:|---:|
| ST: 2 cases × 3 levels × 3 tolerances | 18 | 2,436.5 |
| S: 2 cases × 3 levels × 3 tolerances | 18 | 174.7 |
| D1: four additional solves per calibration case; finest ST reused | 8 | 758.5 |
| X1: 6 dev cases × 3 levels | 18 | 75.9 |
| X4: doubled middle timestep; outer levels reused | 1 | 22.4 |
| **Total, excluding gates/compilation/IC/I/O** | **63** | **3,468.1 s = 0.963 h** |

| Average slowdown against that model | Solve-work estimate |
|---:|---:|
| 2× | 1.93 h |
| 4× | 3.85 h |
| 8× | 7.71 h |
| 12× | 11.56 h |

The allowance is about **12.46× before overhead**. Tight/tighter targets may repeatedly hit Newton budgets 40/60, so the old BE timing does not guarantee that allowance is sufficient.

For memory scale, the largest base FD states are approximately:

| Calibration | One f64 interior state | Six full-grid output states |
|---|---:|---:|
| 2D, \(L=8192\) | 0.537 GB | 3.220 GB |
| 3D, \(L=512\) | 1.067 GB | 6.405 GB |

These exclude Krylov/FFT workspace, extraction temporaries, cached eigenvalues and executables. The requested devices are plausible choices; **240G is host memory and does not establish H200 device fit**.

Before approving this wave, the essential repairs are finite/valid calibration acceptance, retention of every control-attempt record, usable burn-in/timing evidence, and smoke-only collection/floor-evidence handling. The reproduced loader and audit defects must also be repaired before any resulting references are reported.