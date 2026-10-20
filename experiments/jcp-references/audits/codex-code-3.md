**Overall verdict for the first real-size wave: NOT READY.** One blocking-now defect remains: `Utransforms` still passes NaNs in its nonlinear and lattice-synthesis checks. The calibration repairs work in CPU reproductions; staging and resource selection have no demonstrated execution blocker.

Audited HEAD `1458bbeff` against DESIGN through A6. No files modified, no GPU jobs launched, and no lab-log entry written. Checks used source inspection, actual functions extracted into in-memory NumPy harnesses, and filesystem doubles. Full G5–G7 execution awaits a populated store.

“CORRECT” below means the cited defect is repaired, not that every surrounding path is certified.

**Disposition of all prior blocking findings**

The numbered references follow `codex-code-2.md`. Repeated findings in that audit are explicitly grouped.

| Prior finding | Verdict at HEAD | Severity now | Evidence / remaining issue |
|---|---|---|---|
| #4: full-grid outputs, unsupported in-place/memory claim | CORRECT | non-blocking | Unsupported “in place” comment removed. A6.5 accurately limits memory evidence to process-lifetime device high-water measurements. Actual device fit remains unmeasured. |
| #19: incomplete/broken G5–G7 | NEEDS-RESTATEMENT | blocking-later | Loader defects repaired; G5 remains incomplete and mishandles failed records, detailed below. |
| #26: unsafe resume and non-upfront sealed refusal | WRONG | blocking-later | Existing file pairs are now compared against `PREV_MANIFEST`, but JSON without NPZ raises instead of recomputing. Sealed refusal occurs after overwriting `result.json`. |
| #28 and repeated changed-code finding: discarded calibration/control evidence | CORRECT | non-blocking | `attempt_records` retains base, D1, X1 and X4 records. Synthetic execution retained **63/63 2D** and **56/56 3D** field/record pairs. X4 uses recorded running maxima. |
| #30 and #43: absent PS escalation workflow | CORRECT | non-blocking | Under A6.3, first unresolved ladders receive `UNRESOLVED-PENDING-ESCALATION`; attempts beyond 1 are refused. Escalation remains a separately submitted job. |
| #33: unusable timing/memory evidence | NEEDS-RESTATEMENT | non-blocking | Records now survive, and A6.5 makes these budgeting measurements. `solve_seconds` still includes solver construction/setup on first use, beyond compiled execution plus IC transfer. No benchmark claim is justified. |
| #36 and repeated changed-code finding: 3D restriction crash | CORRECT | non-blocking | One-frame IC and six-frame rollout restrict successfully to `(1,29791)` and `(6,29791)` at factor 2. |
| #37: restricted 3D PS cross-check unreachable | CORRECT | non-blocking | The reshape obstruction is repaired; the restricted PS path is reachable. |
| Changed-code: STATUS list sorting | CORRECT | non-blocking | Both sides now use lexical sorting. Both pinned lists pass the reproduction. |
| Changed-code: Utransforms NaN acceptance | WRONG | **blocking-now** | The first transform loop is repaired; later nonlinear/lattice comparisons still hide NaNs through `max(worst, nan)`. |
| Changed-code: contour NaN acceptance | CORRECT | non-blocking | Both gate and per-run comparisons require every discrepancy to be finite. Mixed finite/NaN reproduction rejects. |
| Changed-code: calibration accepts failed solves | CORRECT | non-blocking | Requires all three `steps_ok` flags and finite proxies. Injected infinite residuals produce `proxy_ok_all=False`. |
| Changed-code: fresh smoke-only collection lacks `store/` | CORRECT | non-blocking | `write_manifest()` now creates the directory. |
| Changed-code: unchecked result/calibration evidence; absent floor installation | NEEDS-RESTATEMENT | blocking-later | Present files are checked and floor NPZs linked/manifested. Missing calibration NPZs are silently skipped rather than rejected. |
| Changed-code: overlapping sealed jobs inflate coverage | CORRECT | non-blocking | Overlapping case sets are refused; full cohort coverage requires the exact index union. |
| G6: stale-status test overwrites canonical STATUS | CORRECT | non-blocking | Uses isolated metadata through `JCPREF_META`; canonical STATUS is untouched. |
| G7: normal scoring fails; 3D restriction untested | CORRECT | non-blocking | Sorting repaired and a 3D restriction call added. Full store-dependent suite was not executed. |
| Seeds: final-named partial ciphertext | CORRECT | non-blocking | Temporary ciphertext is decrypted/verified before exclusive atomic linking. Final filename no longer exposes an incomplete write. |

**Disposition of the prior blocking G5 findings**

| Prior G5 finding | Verdict at HEAD | Severity now | Evidence / remaining issue |
|---|---|---|---|
| Perturbation must exceed the specified \(10^{-6}\) bar | WRONG | blocking-later | `dp`/`de` are now measured, but an outcome change independently passes. Also, either uncertainty being zero/undefined forces `de=inf`, which can claim detection without establishing the required numerical change. |
| Exceptional-value FD outcome semantics | NEEDS-RESTATEMENT | blocking-later | NaN IC reproduction now agrees: both return `ic-mismatch`. Serialized failed evidence remains broken: infinite proxy becomes JSON `null`, causing a false mismatch; `umax_running=null` causes `TypeError`. |
| Near-zero agreement tolerance | CORRECT | non-blocking | Implements A6’s relative-plus-absolute tolerance. Both prior roundoff-scale examples now agree. |
| Required FD numbers and attempt history audited | NEEDS-RESTATEMENT | blocking-later | Adds `n0`, differences, per-time uncertainty, Péclet, floor flag, viscosity and acceptance of superseded attempts. Array comparisons use unchecked `zip`, so truncated vectors can escape; other stored check flags remain unchecked. |
| Cross-check numbers/validity audited | WRONG | blocking-later | Adds \(s_x,s_t,\sigma,x_{PS}\), but still does not compare stored \(q,\eta\), or independently recompute stability from timestep, resolution and running maxima. Corrupting both `q` and `eta` to 123 still returned `match=True`. |
| NaN half-step cross-check semantics | CORRECT | non-blocking | Reproduction now returns unresolved in production and the corresponding pending-escalation verdict in G5, with `match=True`. |
| Old-versus-new distances absent | CORRECT | non-blocking | Recomputes `delta_old` from old and new fields for each reported comparison. |
| Coverage enforcement and successful exit on failure | NEEDS-RESTATEMENT | blocking-later | Historical FD coverage and nonzero failure exits are implemented in both scripts. Missing PS evidence can still be treated as a matching `MISSING`, without independently failing G5 completeness. |

**Remaining defects and patch-specific findings**

| Location | Verdict | Severity | Finding |
|---|---|---|---|
| [gates.py:308](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/gates.py:308), 313 | WRONG | **blocking-now** | Nonlinear and lattice discrepancies enter `max` before any finite check. The final `isfinite(worst)` cannot detect an already-discarded NaN. Check each discrepancy before aggregation. |
| [refjob.py:275](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:275) | WRONG | blocking-later | The new resume condition reads both files without testing existence. Actual `mode_fd` with a JSON-only leftover raised `FileNotFoundError`; deletion/recomputation was never reached. |
| [refjob.py:537](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:537), refusal at 263 | WRONG | blocking-later | A sealed rerun executes gates and saves a fresh incomplete `result.json` before refusing existing ciphertext. This destroys the previous QC/manifest result and invalidates its output checksum. Refusal must precede the first save. |
| [collect.py:75](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/collect.py:75) | WRONG | blocking-later | Evidence discovery uses a glob instead of requiring the calibration artifact named by the result/manifest. In-memory reproduction accepted a complete calibration job whose required NPZ was absent, linking zero floor files. |
| [audit_ref.py:194](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:194), 205 | WRONG | blocking-later | Audit comparison does not account for production’s non-finite-to-`null` serialization. Legitimate failed attempts can produce false audit failures or abort the audit. |
| [audit_ref.py:212](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:212), 217 | WRONG | blocking-later | Added per-time comparisons do not validate vector lengths. An empty/truncated stored vector skips required comparisons. |
| [audit_ref.py:263](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:263), 277 | WRONG | blocking-later | Stability still trusts `ps['stability_ok']`; stored order/effectivity are not compared. The advertised independent cross-check validation remains incomplete. |
| `refstore.py`: `JCPREF_META`; `test_refstore.py` | CORRECT | non-blocking | Metadata override consistently controls manifest, version and STATUS reads, as A6.4 specifies. The stale-status test uses that isolated location. |
| `make_seeds.py`: publication permissions | NEEDS-RESTATEMENT | non-blocking | Verified atomic publication is repaired, but chmod occurs after linking. An interruption in that small interval leaves complete ciphertext writable. Set read-only permissions before linking to satisfy the invariant immediately. |

The decisive gate reproductions were:

| Injection into actual `u_transforms()` | HEAD result |
|---|---|
| Finite direct-sum control | `pass=True`, worst discrepancy \(1.21\times10^{-15}\) |
| NaN in initial transform checks | `pass=False` |
| NaN confined to nonlinear output | **`pass=True`, `worst_rel=0.0`** |
| NaN confined to lattice synthesis | **`pass=True`, worst discrepancy \(1.21\times10^{-15}\)** |

**First-wave assessment**

| Component | Verdict | Severity | Assessment |
|---|---|---|---|
| `configs/gates.json`, A100-80G | WRONG | **blocking-now** | Exercises the incompletely repaired Utransforms gate. G3b/G3c use escalation sizes as intended. |
| `configs/calib2d.json`, A100-80G, 12 h | CORRECT | non-blocking | Real base levels, no smoke override; retains 63 attempts including D1/X1/X4; failed calibration evidence cannot pass the proxy check. |
| `configs/calib3d.json`, H200, 240G host, 12 h | CORRECT | non-blocking | Real base levels; retains 56 attempts including D1/X1. X4 is the prescribed 2D probe. |
| `cluster/stage.py` | CORRECT | non-blocking | All **18 required input files** matched committed HEAD. Generated all three scripts in memory; each passed `bash -n` with requested resources, GPU preflight and precision settings. |
| `cluster/submit.sh` | CORRECT | non-blocking | Shell syntax passes; inspected locking, exclusive remote directory creation, lane cap, queue checks and dependency-only extra flag handling. No submission performed. |
| Runtime/device-memory fit | NEEDS-RESTATEMENT | non-blocking | Twelve hours and the selected devices remain plausible, unmeasured budgets. Host allocation does not prove device fit. This wave is intended to obtain that evidence. |

**Fix the two remaining NaN aggregation paths before this wave.** Resume, collection, sealed-job and G5 defects can remain explicitly blocking-later because these three jobs produce calibration and gate evidence, not released references.