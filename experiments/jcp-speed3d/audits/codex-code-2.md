**NOT READY FOR CLUSTER SMOKE.** The predictor, replay, timestep, and grid-hash repairs are substantial. Remaining blockers concern smoke coverage, FOM qualification, and evidence collection. Deferring analysis is reasonable only where the jobs already save sufficient evidence; that condition is not yet satisfied everywhere.

Reviewed through `74b72c6c9`. I modified no files or git state, submitted no jobs, and performed no GPU work. Checks used sequential CPU-only processes with bytecode and persistent compilation caching disabled. Per your read-only instruction, I did not append to the lab log.

**Round-one required fixes**

The numbering below follows the nine-item “Required fixes” list at the end of round one.

| # | Disposition | Evidence |
|---|---|---|
| 1 | **RESOLVED** | `canon_dt()` reconstructs all four registered timesteps correctly; CPU round-trip checks pass. FSP now carries `Jp3`, constructs cubic `J3`, evaluates P3 through `rj`, and contracts only when P4 wins. Replay iv supplies `force_pred=pidx` and records fresh terminal certification. |
| 2 | **NOT RESOLVED** | G2 rejects nonfinite values; standalone tests now exit nonzero on their enumerated failures. T1 expected hashing and missing-hash rejection passed independent in-memory checks. The rerun grid has zero hash failures and T1 median **2.01135**. However, the supplied smoke commands do not invoke those standalone tests, W256 G5 coverage remains absent, and `s3run` can mark failed-gate jobs complete. |
| 3 | **NOT RESOLVED** | Reference eligibility, ADT gates, T1, and controls now affect selection. But cohort/row/timing completeness and score provenance remain unenforced. I reproduced `selectable=True` with a nominal 64-case cohort, **one solver row**, one candidate timing invocation, and no score provenance. |
| 4 | **DEFERRED-OK** | BE FS-K2 is restored on deployed and reduced rules. Final H1.4/H2 verdicts can wait until direct confirmation panels exist. Grid screening records suffice for that later workflow, subject to fixing acceptance before selection. |
| 5 | **NOT RESOLVED** | Warm FOM repetition arrays and hash checks are implemented. Matcher/K10 can wait. However, convergence qualification still accepts incomplete evidence; matched timing remains restricted to ROM mesh L=64. Incorrect tolerance qualification can require rerunning production FOM cases. |
| 6 | **NOT RESOLVED** | Profile mesh-free condition gating is implemented; G8’s stored result passes at **1.80e−16** relative discrepancy. H3.2/K5 verdicts and G3 arithmetic can wait for analysis. The profiling kernels still use synthetic initial-state histories, and required compile/preparation measurements are missing. |
| 7 | **NOT RESOLVED — before replication** | Runtime and staging now hash the freeze file; runtime checks ROM triples and FOM membership. But the freeze check does not validate paths or all frozen decisions, such as mesh-free qualification and comparator/match associations. Replication ADT comparisons are now collected. |
| 8 | **NOT RESOLVED** | Cluster configs exist and staging enforces H200. Production resource certificates can legitimately follow smoke. The current smoke coverage and inventory are insufficient for the promised complete certificate without further work. |
| 9 | **NOT RESOLVED, mostly non-blocking** | QN damping is saved; `n_exec` is annotated as per-batch; K6 uses maximum distance; timed batches are hash-checked; K9 resolves case 0 and shares `ec_pass`. K4 still uses positional cases, ROM preparation times remain absent, and in-job reference hashes are checked but not saved explicitly. |

**Adversarial findings**

1. **FOM convergence still accepts incomplete records.**  
   In [fomrun.py:80](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fomrun.py:80), `complete` checks only `0 < stages <= len(res)`. It does not require the expected schedule length, sufficient Newton-count records, or completed time units.

   CPU probes accepted:

   - `stages=2`, two residuals, but only **one Newton count**;
   - `stages=1` for UB with `dt=.01`, whose complete schedule requires **25 stages**;
   - `nwt=20` with a small residual.

   The last case now has an explicit last-iteration-convergence interpretation, unlike round one. That interpretation should be reconciled with A1.9/A2.5’s “no max-Newton exit” wording. Counts above the solver limit are also not rejected.

   PS finite-output checking works when callers supply `F`, as the production accuracy path does. Calling `converged(..., F=None)` still succeeds without finite-output evidence.

   **Why this affects deferral:** calibration chooses a tolerance before full-cohort execution. Rejecting that tolerance later may leave the newly selected tighter level without full-cohort accuracy or warm timings.

2. **Smoke completion is not smoke acceptance.**  
   [s3run.py:145](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3run.py:145) runs the selected mode, then writes `complete=True` and `COMPLETE`. It does not enforce a combined mandatory-gate report.

   In particular:

   - The default staged commands do not run `test_s3.py` or `test_fom.py`.
   - Failed G5/K7 suppresses higher-mesh ROM work but does not fail the job.
   - Missing expected hashes skip a panel but do not fail the job.
   - T1’s band is checked by grid analysis, not by smoke completion.
   - G6/G7 and their controls are not exercised merely by running FOM accuracy.

   This is a coverage/acceptance gap, even though the individual test scripts now have failure exits.

3. **Analysis remains bypassable and conflates unavailable evidence with failed hypotheses.**  
   [analyze.py:64](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze.py:64) checks cohort length, not exact unique case membership, and does not check one row per required case or the timing population. Score files are not bound to the current result/W artifacts.

   The independent synthetic test described above returned `selectable=True`. Separately, marking every reference ineligible raises a zero-size reduction exception instead of producing an unavailable result.

   Replacing H1 eligibility with `selectable` prevents some unsupported positive claims, but causes missing/invalid timing or blocked controls to appear as `False` for H1.2/H1.3. Those should be unavailable/blocked, not scientific failures. H1.4 also uses raw FS eligibility when deciding rescue/comparison, without separately requiring an acceptable comparator timing record. K1/K2 currently suppress all selection in a setting, broader than their specified H1 protection.

   These repairs can wait until **before selection**, because grid rows/W/distances are retained.

4. **Cross-resolution matched timing is still incomplete.**  
   [s3modes.py:440](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:440) replaces the old FOM-resolution restriction with:

   ```python
   if L != 64:
       continue
   ```

   This permits any matched FOM resolution against the L=64 ROM, but still omits matches against the other ROM meshes. The comment limiting H4.3 to n=65 is not an amendment in DESIGN. Round one explicitly required each relevant ROM mesh.

5. **Profile algebra improved, but actual-stage profiling remains unfixed.**  
   The configured scheme, timestep, viscosity scaling, and diagonal damping now appear in the kernels. However, [s3modes.py:249](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:249) profiles `w0` with synthetic history `c_n=c_{n-1}=c`, not extracted solver-stage inputs. The QN iteration measures one trial residual, without the production acceptance/backtracking path.

   G3’s component-sum check can be computed later from timing invocations. That does not recover actual-stage measurements or separately recorded JIT compilation time. ROM input preparation is also unmeasured.

6. **Freeze verification is stronger but incomplete.**  
   [s3ctx.py:45](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3ctx.py:45) verifies `(setting, rule, spec)` membership. Changing an allowed configuration’s `paths` to include `mf` is not rejected by that check. Frozen qualification, paths, required comparators, and match associations need validation before replication. This need not delay grid/FOM collection.

7. **The supplied local evidence is partly stale.**  
   The rerun grid validates the new hash/T1 path. The other local artifacts do not validate their corresponding repairs:

   - `runs/local/fom`: no `warm` arrays.
   - `runs/local/prof`: old `n_full_per_query=None` profile records; no batch `hashes_ok`.
   - `runs/local/mesh`: only `M2|512` G5 evidence; no new combined `gate` field.
   - `test_s3-M2.json`: mandatory failures empty, but cases are **0,5**, not the full prescribed smoke set **0,5,7**.

**Configuration and staging review**

| Config | Assessment |
|---|---|
| `g1` | Settings, deployed/reduced rules, adaptive starts, four timesteps, full cohort, and 4×2 screening match DESIGN. |
| `g2` | W512/W768 settings and rules match; same timing/cohort agreement. |
| `g3` | W1024 settings and rules match; same timing/cohort agreement. |
| `f1` | A1.9’s main matching ladder is correct: **17 FD configurations and 15 PS configurations**, calibration cases 0–3, full-cohort accuracy, warm 4×2 timings. L=256/384 diagnostics belong to mesh mode. |
| `smkgrid` | Correct cases 0,5,7; exercises all solver families, reduced-rule code, replay and T1. Omits the 5/3 timestep and W256/W512/W768 settings; does not invoke G0–G2 tests. |
| `smkfom` | Exercises UB, CC, PS, calibration and warm timing; omits UC and standalone G6/G7/K8 checks. |
| `smkprof` | Exercises mesh-free projection, K5, profiling, batching, confirmation and converged-rule sensitivity. Its “reduced ADT” subject actually uses the deployed rule. It does not cover W1024 batching or BDF2 profiling. |
| `smkmesh` | Includes L=384 UB/CC and n=385 W1024 tables, satisfying the requested pilot targets in principle. **W256 is absent**, so mandatory G5 coverage is incomplete. Only PS is supplied as a cross-resolution match, and matched panels run only at L=64. |

All four smoke configs use cases 0,5,7 and reduced 2×1 timing. That is useful diagnostic coverage, not production timing acceptance.

The **active import graph is covered by `cluster/stage.py FILES`**. CPU imports succeeded, and every active lane module loaded by the staged drivers/tests was in the manifest list. The earlier caveat remains: cluster `test_fom.py` needs `--skip-e0` unless its locally absolute E0 comparison dependency is made available. Merely staging that test does not execute it.

**Deferral judgment**

- **Selection/freeze script:** DEFERRED-OK until before selection/replication; grid evidence is largely sufficient.
- **Matcher/K10:** DEFERRED-OK only after fixing convergence qualification before FOM production.
- **Final H1.4/H2:** DEFERRED-OK until confirmation panels; screening alone cannot supply final verdicts.
- **H3.2/K5 verdicts:** DEFERRED-OK until after properly configured profile collection; saved W, validity/certification, conditions and ADT distances support analysis.
- **G3 calculation:** DEFERRED-OK; actual-stage profiling and missing measurement collection are not analysis-only deferrals.
- **Resource certificates:** DEFERRED-OK after smoke, but the inventory must include warm-ups, replay, certification, baseline work and mode-specific repetitions. Current certificate checks do not establish that completeness or smoke provenance.

**Blocking fixes**

1. **Before smoke:** provide an executable mandatory-gate sequence covering G0–G2, G5/K7 including W256, G6/G7/K8, T1 and expected hashes; preserve diagnostics and fail acceptance explicitly. Distinguish diagnostic completion from acceptance.
2. **Before smoke/FOM production:** fix FOM evidence completeness and settle/enforce Newton-limit semantics before tolerance selection.
3. **Before accepting profile smoke as evidence for later profiling:** collect faithful stage inputs/timings and required compile/preparation measurements; current synthetic kernels cannot be repaired by post-processing.
4. **Before mesh production:** remove the L=64-only matched-panel restriction and exercise both FD and PS cross-resolution matching.

**Non-blocking for initial smoke, required later**

1. Repair analysis completeness/provenance checks, empty reference populations, and blocked/unavailable verdict semantics before selection.
2. Implement selector/freeze, matcher/K10, confirmation verdicts and H3 qualification before their respective decisions.
3. Validate all frozen decisions, including paths and qualification, before replication.
4. Generate complete smoke-derived resource certificates before production staging.
5. Resolve K4 by explicit case IDs; persist reference provenance and clear `n_exec` aggregation metadata.
6. Treat the old local FOM/profile/mesh artifacts as historical evidence, not validation of the new code.

**NOT READY FOR CLUSTER SMOKE.**