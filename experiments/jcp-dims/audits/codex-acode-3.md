**Overall: NEEDS-FIX.** Four round-2 findings are resolved; five remain partly unresolved. Synthetic checks still reproduce false qualification.

Read-only audit: no files changed, jobs submitted, production results opened, or scratch files created. Python checks used CPU-only execution with bytecode writes disabled.

Paths below are relative to `experiments/jcp-dims/`.

| Round-2 finding | Disposition | Assessment |
|---|---|---|
| 1. Confirmation-job certification | **NEEDS-FIX** | `cert_of_job` is consumed, but accepts incomplete anchor evidence. |
| 2. q++ identity and distance | **NEEDS-FIX** | Independent distance added; mode identity and actual successor validity remain unchecked. |
| 3. Population completeness | **NEEDS-FIX** | Ladder validity lengths improved; adaptive Q1 and missing-anchor classification remain incomplete. |
| 4. B6 control blocking | **CORRECT** | Primary transitions and applicable per-form/both-form controls now block verdicts. |
| 5. Exact timing inventories | **CORRECT** | Both fused and split `(case, repetition)` inventories are checked. |
| 6. Independent 3D checks | **NEEDS-FIX** | Two-case and round recomputations added, but some required configurations are skipped and cached checks lack evidence binding. |
| 7. Qualified-comparator ratios | **CORRECT** | `Cq` pairs enter confirmation; analysis reports screening and confirmation ratios. |
| 8. Analysis-cache validity | **NEEDS-FIX** | LMF score suppression works; dependency coverage remains incomplete. |
| 9. Elasticity resampling | **CORRECT** | Uses 10,000 joint dev6 resamples. |

Ranked remaining defects:

1. **Timing-job Q1 can still accept incomplete or invalid anchors.**  
   `analyze3d.py:503–510` checks neither anchor-validity array lengths nor tighter-case identities for ladder certificates. **Reproduced:** eight production cases and eight zero distances, with empty `adt_valid` and `adt_tighter_valid`, return `True`.

   Adaptive certification consumes summary fields that contain no anchor-validity evidence. `d3run.py:209–212` defines `valid` from production runs only. Thus finite, close but invalid tight/tighter runs are not rejected by `cert_of_job`. Read the archived adaptive evidence and require complete, valid anchors before accepting screening or confirmation.

2. **3D Q2 still accepts inconsistent successor evidence.**  
   `analyze3d.py:233–250` checks setting, spec, M and case IDs, but omits `entry['mode_hash']`. It also uses the predecessor’s stored `next_valid` instead of validating the actual successor entry.

   **Reproduced:** changing q++’s mode hash still returns `pass`; setting every q++ production-valid flag to false while leaving stored `next_valid` unchanged also returns `pass`. Validate both successors’ numerical identities and their own validity inventories.

3. **Missing populations can still pass—or be treated as terminal failures.**  
   `analyze3d.py:141–145` omits lengths/shapes for adaptive `valid_tight`, `gcert`, and certificate evidence. **Reproduced:** empty tight-validity and G-cert arrays produce an entirely true Q1 result.

   In 2D, `analyze2d.py:67` defines `population_ok` solely from production `case_ids`. Missing registered tighter runs therefore become `not qualified` at `:138–140`, rather than **missing**. This lets `mstar` advance beyond incomplete lower candidates. The synthetic classification check reproduced that outcome.

   Round analysis has the same empty-array risk at `analyze3d.py:285–301`: it does not enforce production/successor validity lengths or the complete rho inventory.

4. **Caches can bypass current evidence and checks.**  
   `analyze2d.py:32–33,55–62` hashes selected code, the NPZ and expected cases, but excludes the consumed `result.json` rows, `dcert.py`, and scoring/reference dependencies. Changing an in-job distance or removing its JSON row leaves the cached offline-agreement verdict usable.

   `analyze3d.py:155–159` is weaker: the rho cache key contains names only. Changed trajectories, stored rho, rule/bank data or checker code can reuse an old `agree=True`. Bind caches to all consumed evidence and numerical dependencies. The LMF cache-hit score restriction itself is correct.

5. **Independent rho coverage still omits rho-carrying ladder members.**  
   `analyze3d.py:226–227` returns immediately for the top two rungs. The second-to-top rung has a successor and stored rho, but never reaches `rho_offline` at `:255`, even when it fits the stated CPU threshold. Other early returns also skip the check.

   Keep the ladder-top qualification label, but run the required independent audit separately from Q2 verdict selection. The adaptive distance and round-distance recomputations are present.

6. **New defect: the 2D loader does not resolve retry attempts.**  
   `analyze2d.py:27,95–96` only searches `runs/f2b/archive/output/dacc_f2b`. DESIGN A7 records the replacement attempt as `f2br1`; collection stores archives under the attempt name (`cluster/collect.py:25–26`). There is no retry mapping in the loader.

   Without an explicit collection alias, the replacement R512 evidence is silently omitted. Resolve logical jobs to their accepted attempt and output stem, preserving provenance. This finding follows from code/configuration conventions; no production archive was opened.

| File | Verdict |
|---|---|
| `dscore.py` | **READY** |
| `analyze2d.py` | **NEEDS-FIX** |
| `analyze3d.py` | **NEEDS-FIX** |
| `score3.py` | **READY** for the mapping edit |
| `make_configs_late.py` | **READY** |
| `make_round.py` | **READY** for predecessor-member coverage |
| `waitjobs.sh` | **READY** |
| `d3run.py` | **READY** for the requested adaptive `Wstore` change |

Passing checks include zero missing 2D manifest/config keys, the restricted `score3.py` diff, adaptive scoring-key consistency, explicit-M late settings, certification blocks before timing panels, `Cq` confirmation generation, and `bash -n` for `waitjobs.sh`.