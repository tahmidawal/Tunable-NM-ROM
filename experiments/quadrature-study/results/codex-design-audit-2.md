**Do not accept production results yet.** The numerical rewrite is substantially improved, but acceptance, reference validation, and timing verification remain incomplete.

Paths below are relative to `experiments/quadrature-study/`. Audit was read-only and offline; no JAX execution or file changes.

**A. Previous findings**

| # | Verdict | Evidence |
|---|---|---|
| 1 | **PARTLY FIXED** | Rejection exits 3 (`refjob.py:78–82`); loader requires completeness, acceptance, cohort hash and field hash (`qstudy.py:159–174`). It still does **not validate reference mesh, timestep, tolerances, shape or actual residual against the required contract**. |
| 2 | **PARTLY FIXED** | Independent audit exists; restricted/full evidence saved (`qstudy.py:214–216,343–348`). However, missing audit files silently reduce coverage, injected checks are weak, and rho coverage is narrower than claimed (`audit_qs.py:215–299`). |
| 3 | **FIXED** | Extra meshes are built and timed interleaved within the same allocation (`qstudy.py:181,428–465`); B4 uses those times (`audit_qs.py:335–347`). |
| 4 | **FIXED** | Cached burn kernel (`qstudy.py:44–53`), initial invocation before measurement (`:133`), and every timing subject/case warmed (`:452–454`). The initial burn can include compilation, but the measured blocks follow it. |
| 5 | **PARTLY FIXED** | Main-mesh ROM hashes and timed FOM accuracy/convergence are checked (`qstudy.py:469–475`). Extra-mesh ROM outputs remain unchecked. |
| 6 | **FIXED** | Gauss-192 now appears in the rho ladder (`make_configs.py:34–35`), including generated production configs. |
| 7 | **FIXED** | Historical `IP.make_fom(L, dt, 'fft')` is actually invoked (`qstudy.py:188,196`). Metadata still incorrectly calls this `lean_tight`. |
| 8 | **PARTLY FIXED** | Driver enforces disjointness and test hash (`qstudy.py:135–148`). Freeze gate exists (`cluster/stage.py:47–51`), but its amendment check is ineffective; cross-job development hashes and once-only opening remain unenforced. Reference-test exemption is explicitly documented. |
| 9 | **FIXED** | G6 evaluates both populations (`qstudy.py:387–398`). Gauss-768 rollout is added only to dv1024/dev6 (`make_configs.py:124–126`), with the correct `0.1 × B3` comparison (`audit_qs.py:349–356`). Missing required coverage still needs enforcement. |
| 10 | **PARTLY FIXED** | Setting-major cache and chunked device construction materially reduce residency (`qstudy.py:226–281,486`; `qcore.py:209–227`). Final rollout tables remain fully resident; peak memory and production runtime are unestablished. |
| 11 | **FIXED** | Switching-surface qualification is explicit (`DESIGN.md:220–221`). This is a documentation fix; no new nonidentity-tangent directional-check evidence was supplied. |
| 12 | **FIXED**, with stale text | Correct exception and no renormalization documented (`qcore.py:209–212`). Unconditional unit-sum statements remain in `qcore.py:22,84`. |

**B. Remaining/new defects**

1. **BLOCKER — Failed audits do not prevent recommendation or freezing.**  
   `select_rule.py:23–25` merely copies `failed_gates`; candidate selection ignores them (`:49–52`). Missing B1 is explicitly accepted through `x is not False`. `audit_qs.py:386–388` also exits normally after printing failed gates.  
   **Fix:** require complete, successful applicable gates before selection; require B1/B3/B5 explicitly true at every required mesh; exit nonzero on rejection.

2. **MAJOR — Gate applicability is wrong.**  
   Every production test mesh has parity targets, but no dev6 rows: `audit_qs.py:157–165` therefore reports G3 false for every `t*` job. Conversely, incomplete development parity coverage can pass when only the surviving entries pass. G7 runs on every mesh/cohort (`:200–210`), although DESIGN requires the dv1024 panel. G6 rollout checking disappears silently if its rows are missing (`:349–356`).  
   **Fix:** define required gate coverage by job role; require all expected development parity/G6 entries, and attach the accepted development evidence to test jobs.

3. **MAJOR — Reference hashes authenticate files, not the required reference calculation.**  
   `qstudy.py:159–171` trusts `accepted` without checking `mesh=8192`, ST/S timesteps, solver tolerances, acceptance residual, array shape, or manifest uniqueness. An accepted reference from another resolution can satisfy the existing checks. `audit_qs.py:183–194` does not repair this.  
   **Fix:** validate the complete registered reference contract before loading any rollout reference.

4. **MAJOR — G8 can pass incomplete evidence and excludes numerical failures from aggregates.**  
   Missing truth/arm files are skipped (`audit_qs.py:218–219,232–234`); missing comparison/full-field evidence is optional (`:240–249`). A nonempty surviving subset suffices (`:261–263`). `qstudy.clean` converts nonfinite values to null (`qstudy.py:69–70`); `stats`, B1 and B3 drop null metrics (`audit_qs.py:115–119,144–153`), while arm `finite` is not an acceptance condition.  
   **Fix:** enumerate required case/arm/file/metric coverage and reject missing or nonfinite required results. Require exact dense-matched case equality for B1.

5. **MAJOR — Extra-mesh timing has no output verification.**  
   Verification is conditional on `sj[1] == L` (`qstudy.py:469–470`); G8 then considers only invocations containing that verification field (`audit_qs.py:317–322`).  
   Skipping comparison against the **main mesh’s** hash is correct: different meshes need not produce identical fields. Skipping verification entirely is not.  
   **Fix:** retain an untimed baseline per extra mesh/arm/case, hash using that mesh’s stride, and verify every timed invocation against it.

6. **MAJOR — Test-freeze amendment check already succeeds without an amendment.**  
   `cluster/stage.py:51` searches for `FROZEN SELECTION`, but that phrase already occurs in the explanatory text (`DESIGN.md:115`). It also checks only cohorts exactly equal to `['test64']` (`stage.py:47`).  
   **Fix:** parse a dedicated frozen manifest, validate its successful source audits and identities, gate any cohort list containing test64, and enforce the registered once-only opening.

7. **MAJOR — Cross-job cohort equality is not checked.**  
   G2 compares only the recorded test hash (`audit_qs.py:106–107`), without recomputing it from physical descriptors. `select_rule.py:22–33` compares “all” errors without checking matching cohorts, case sets, or development hashes.  
   **Fix:** recompute descriptor hashes and require identical evaluated case sets across meshes. Current supplied development configs do match; the acceptance machinery does not establish that.

8. **MAJOR — G8 injection checks do not exercise the audit’s rejection path.**  
   Field perturbation merely changes a hash; “swapped case” merely compares two recorded hashes; timing perturbation merely differs from the original median (`audit_qs.py:251–260,310–316`). With one case, the fallback substitutes another arm, which is not a swapped case.  
   **Fix:** feed altered evidence through the same validation functions used for acceptance. Use two actual cases for the swap test.  
   Supplied smoke explicitly has **`G8_injected_controls=false`**, `swapped_case_detected=false` (`checks/smoke/smk128-summary.json:217–222`).

9. **MAJOR — Independent rho coverage is narrower than G8’s wording.**  
   Only Gauss-64 and one Fibonacci rule are recomputed (`audit_qs.py:277–291`); only their argmax states are guaranteed inclusion. The “mesh” check recomputes the dense-to-continuum gap (`:292–297`), not rho for the mesh quadrature rules.  
   **Fix:** explicitly register this limited audit or include required rule families and their argmax states, particularly any recommended rule and mesh-rule claims.

10. **MAJOR — Recommended rule may lack a cross-mesh timing panel.**  
    Candidates include Gauss-32/48/96/192 and Fibonacci-1597/4181 (`make_configs.py:18–19`), but those are absent from `XMESH_ARMS` (`:92`). They can be recommended while B4 is null.  
    **Fix:** cover every selectable candidate or schedule the recommended rule’s same-GPU panel before answering question (iii). B2/B4 correctly are **not** recommendation filters under DESIGN §8.

11. **MINOR — B4 accepts invalid solve-time denominators.**  
    `audit_qs.py:343–346` checks truthiness, not positivity/finiteness. Two negative query-minus-decode estimates can yield a passing positive ratio.  
    **Fix:** require finite, strictly positive endpoint solve times and the intended 256/4096 endpoints.

12. **MINOR — Labels and documentation drift.**  
    Historical FFT truth is labelled `lean_tight` (`make_configs.py:52`; `qstudy.py:204`); DESIGN’s secondary metric repeats that label (`DESIGN.md:172`). Unit-sum docstrings still contradict the Smolyak exception.  
    **Fix:** correct provenance labels and unconditional documentation claims.

**Specific checks that are correct**

| Check | Verdict |
|---|---|
| Per-setting closures | **No active late-binding defect found.** `fq/data/tab/dec` are bound in defaults. `cold` is late-bound, but calls finish before the next setting changes it (`qstudy.py:259–279,421–487`). Explicit binding would make this less fragile. |
| `values()` mesh selection | **Correct.** It is exclusively the main-mesh rho evaluator; extra meshes enter only phase 3 (`qstudy.py:365–381,428–435`). |
| Cohort restrictions / dense coverage | **Correct for supplied configs.** `:301–304` honors restrictions; dense coverage is all development cases through 1024, dev6 at 4096, and the first six test cases at 4096. |
| Population alignment / rho labels | **Correct.** States and labels append together; `[1:]` maps to steps 1–50 (`:339–342,359–363,400–412`). |
| `stats()` argmax | **Correct for finite nonnegative errors.** Index is computed over the original row list, avoiding filtered-index misalignment (`audit_qs.py:119`). |
| Timing keys / decode | **Correct.** Producer and consumer agree on `setting\|kind\|mesh\|name`, including decode’s `None`. Decode depends on setting/mesh, not the chosen quadrature arm (`qstudy.py:426–449`; `audit_qs.py:302–329`). |
| B1 / B3 arithmetic | **Correct on complete finite coverage.** B1 uses the dense-matched set and `max(2e-4, .02*wd)`; B3 fractions `5e-4/2e-3/5e-3` equal 0.05/0.2/0.5%. |
| B2 / B5 / recommendation formula | **Correct subject to acceptance/coverage defects above.** B2 uses max/min ≤1.02; B5 uses continuum rho ≤0.116; recommendation minimizes point count among eligible Gauss/Fibonacci arms. |
| NumPy rho mathematics | **Correct inspected formulas.** Fibonacci indices 19/20 give 4181/6765; mode selection matches `hops.modes_lean`; production target is Gauss-640. Smoke intentionally uses Gauss-256. |
| Reference filenames / strides | **Correct.** Both sides use `ref_{tag}_{cohort}_{case:03d}.npz`, `cohort_sha256`; 8192→257 stride is 32 (`refjob.py:54,63–71`; `qstudy.py:163–170`). |
| Reference path inference | **Correct for `runs/<attempt>/archive`.** Resolves to sibling `runs/refdv/archive/output` (`audit_qs.py:178–181`). Other layouts require `--refs`. |
| Config lists | **Correct inspected generated configs.** Parity arms append correctly; every extra-mesh arm exists; `gref_check` occurs only in dv1024 with `cohorts=['dev6']`. Gauss-192’s addition is documented in A0. |
| G1 / G4 | Driver enforces x64/highest and truth residual acceptance. G1 audit itself checks GPU/completion only; it should also verify precision evidence. G4’s numerical threshold is correct. |

The smoke supports the limited implementation checks: error recomputation discrepancy **6.94e-18**, rho relative discrepancy **2.35e-10**, and **18** main-mesh timed ROM outputs matched. It does not establish production acceptance or feasibility.

**C. dv4096 feasibility re-estimate**

Static estimates below use decimal GB; they are array-storage estimates, not measured peaks.

| Resident storage | GB |
|---|---:|
| 4096 mesh bank, 512 columns | 68.686 |
| 256 + 1024 mesh banks, 512 columns each | 4.553 |
| **All three banks** | **73.239** |
| Accurate-setting main rollout off-mesh blocks, excluding Smolyak | 11.702 |
| Accurate-setting extra-mesh off-mesh blocks | 5.782 |
| **Subtotal before mesh-rule blocks, operators and workspaces** | **90.723** |
| Host truth cache: 38 cases × six 4097² fields | 30.617 |
| One dense 16-tangent interior-field array | 2.146 |

Evidence: `qstudy.py:178–203,241–281,428–435`; `qcore.py:166–189,209–227,293–298`.

- **GPU: plausible, not established.** Nominal 90% allocation is **126.9 GB**, leaving about **36.2 GB** beyond the subtotal. Mesh-rule caches, rho blocks, fields, dense linearization intermediates, compiler workspaces and fragmentation consume that margin. Chunked construction still concatenates complete tables; Gauss-640 accurate blocks alone total about **7.55 GB**, with temporary duplication during construction.
- **Host: 200G appears sufficient for known arrays.** The dominant persistent truth cache is approximately 30.6 GB. Compiler/transfer peaks remain unmeasured.
- **Setting-major residency is improved, not perfectly bounded.** Loop locals such as `fn`, `sj`, `built_x` and `arm` can retain individual previous-setting objects after `del arms, cache, subjects, pops` (`qstudy.py:486`). A function-scoped setting phase would make lifetime guarantees clearer.
- **Runtime: 12 hours remains unsubstantiated.** dv4096 entails **2,222 phase-1 ROM queries**, **1,998 timed invocations**, **666 warm-up invocations**, 38 truth solves, and the full rho ladders. The single-case smoke cannot establish that budget.

**Feasibility verdict: reasonable for an instrumented feasibility run after the blockers are fixed; insufficient evidence to promise memory fit or completion within 12 hours.**