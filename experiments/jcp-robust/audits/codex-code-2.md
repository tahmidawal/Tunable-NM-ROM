**Overall verdict: NEEDS-FIX. Not ready for smoke.**

Audited checkout `e0b78a2e4f604108771fab8ad58a13f71ae9c232`. No files changed, scratch created, jobs submitted, or SSH/cluster commands run. Small checks used CPU-only Python with bytecode writing disabled. E5 implementation was excluded.

Paths below are relative to `experiments/jcp-robust/`. “BLOCKING” means fix before the smoke.

**Round-one dispositions — original numbering preserved**

1. **NOT RESOLVED — WRONG — BLOCKING: integrity precedence remains incomplete.**  
   `e1verdicts.py:39–68,74–98` now applies control/gate precedence and suppresses GAL inference. However, target/check distance disagreement only disables target certification in `e1analyze.py:83–90` and `e1analyze3d.py:74–78`; it does not enter integrity mismatches. The analysis can consequently substitute own-family inference after an integrity failure. Moreover, 2D `gcert_fail` includes diagnostic bad-rule runs (`e1analyze.py:93–117,172–173`), so a deliberately invalid bad rollout can invalidate the entire arm instead of remaining a recorded control failure.  
   **Fix:** Separate integrity disagreement, numerical failure, and target unavailability. Apply integrity checks to the relevant executions without allowing expected diagnostic failures to invalidate production draws.

2. **NOT RESOLVED — NEEDS-FIX — BLOCKING: population validation is still incomplete.**  
   `e1stats.py:23–29,125–136` correctly rejects three draws and empty validity lists; CPU checks confirmed both. But 2D expected cases come from the supplied configuration (`e1analyze.py:61–62`), certificate step counts are unchecked (`:145`), and duplicate JSON rows collapse into dictionaries (`:76,100`). In 3D, the target check does not require precisely cases 0–7 (`e1analyze3d.py:73–76`), and integrity checks only population length against the supplied count (`:205`).  
   **Fix:** Validate registered production populations, unique identities, exact anchor subsets, and certificate lengths independently of supplied output metadata. Give smokes an explicit separate population contract.

3. **NOT RESOLVED — WRONG — BLOCKING: failure-state handling still has holes.**  
   Failed production now receives infinite error/distance ranks, and the own-family fallback exists. However, `e1analyze3d.py:143` assigns `econv` only when eligible cases exist, then unconditionally uses it at `:209`. Zero eligible cases cause an unbound-variable exception—or reuse the previous configuration’s value. Both dimensions also discard every case’s available target contribution when just one target case fails (`e1analyze.py:202`; `e1analyze3d.py:143`), contrary to A4.2’s per-case fallback.  
   **Fix:** Initialize per-configuration state before branching and apply the target-free margin only to affected cases. Emit explicit per-case failure states.

4. **NOT RESOLVED — NEEDS-FIX — BLOCKING: non-finite handling is not consistently safe.**  
   `E.num()` repairs many paths. Nevertheless, `e1analyze3d.py:32–35` still calls `np.isfinite(job)` directly; the CPU probe `agree(0., None)` raises `TypeError`. Diagnostic summaries still reduce raw JSON indicator lists at `:199`, bypassing normalization. Missing files/keys also raise before unavailable states are emitted.  
   **Fix:** Normalize all serialized numeric evidence before arithmetic, including target checks and diagnostics; handle absent evidence explicitly.

5. **RESOLVED — CORRECT — MINOR: 3D Q1 distances are independently recomputed.**  
   `e1analyze3d.py:97–113` loads tight/tighter coefficients, recomputes distances, compares job values, and rebuilds pooled stationarity from `gam_cert`. Population/completeness deficiencies remain under item 2.  
   **Fix:** None to the basic recomputation path.

6. **RESOLVED — CORRECT — MINOR: Sobol’s empty-check mismatch is repaired.**  
   `r2acc.py:207–209` now writes empty check IDs when no check executes. `e1analyze.py:85–90` leaves that target uncertified.  
   **Fix:** None.

7. **RESOLVED — CORRECT — MINOR: 3D C-E1.2 now exercises rejection.**  
   `e1run3d.py:43–49,160–168` checks the table’s rule hash, deliberately makes the wrong claim, requires rejection, and verifies the true claim. `e1analyze3d.py:170–171` requires both outcomes.  
   **Fix:** None to this control.

8. **RESOLVED — CORRECT — MINOR: the E2 3D scoring adapter installs the job’s bank.**  
   `e2score3.py:20–24` rebuilds the setting map and clears the decode cache; `select3d.py:43` invokes it. `c3common.py:95` loads the hash-verified bank.  
   **Fix:** Ensure all E2 scoring uses this adapter. The smokes still fail to exercise a newly trained bank; see item 26.

9. **RESOLVED — CORRECT — MINOR: 3D front timing now has preceding certified executions.**  
   `make_configs_e2_3d.py:80–90` prepends ladders for the anchor and subjects. `d3run.py:247` records internal states, satisfying the evidence linker at `:480–495`.  
   **Fix:** None to execution ordering; downstream acceptance remains broken under item 23.

10. **RESOLVED — CORRECT — MINOR: 2D selection now distinguishes unavailable evidence.**  
    `select2d.py:43–56` blocks on integrity failure, stops on undetermined candidates, and labels target-unverified picks diagnostic.  
    **Fix:** None to the original false-“none” behavior.

11. **NOT RESOLVED — NEEDS-FIX — BLOCKING: reproduction gates are only partially integrated.**  
    Owning-lane hash checks and enumerated E2 replays now exist (`e1analyze.py:220–278`; `e2analyze.py:127–142`; `e2analyze3d.py:143–165`). But the smoke replay coverage promised by A3.6 has no matching acceptance entry point: the E2 analysis requires production selection/verdict outputs and full populations. The 3D replay also lacks explicit finite/shape checks before `max(rel)` (`e2analyze3d.py:162–164`), which can ignore a later NaN.  
    **Fix:** Add population-aware smoke replay acceptance, explicit finite/shape validation, and staged archive/hash inputs. Preserve the full production replay requirements.

12. **NOT RESOLVED — WRONG — BLOCKING: resource certification is incomplete and sometimes inaccurate.**  
    Certificates/inventories are now staged for production, but:
    - `certify.py:105,131` has no `dtime` inventory handler, although `make_configs_e2.py:99` generates that driver.
    - `certify.py:85–96` ignores block-specific case sets. CPU inspection of `smk3q` counted eight W1024 production cases although the block executes four.
    - Certificate fields at `certify.py:158–162` do not bind code identity or rounded time limit; staging does not validate such bindings.
    - Smoke jobs carry neither certificate nor inventory because `cluster/stage.py:63,100` conditions both on `production`.

    **Fix:** Inventory actual driver executions and reuse, support `dtime`, bind code/config/spec/time/GPU identity, and define an explicit smoke reservation/inventory contract.

13. **RESOLVED — CORRECT — MINOR: training completion contracts now match collection.**  
    `jobs/smk2.json:24` and `jobs/smk3.json:17` name `SEED-COMPLETE`; `cluster/collect.py:30–38` checks the expected staged steps rather than discovered directories.  
    **Fix:** None.

14. **RESOLVED — CORRECT — MINOR: empty large-file cleanup is handled.**  
    `cluster/collect.py:42–45` skips the large-archive checksum invocation when `LARGE.txt` is empty.  
    **Fix:** None.

15. **RESOLVED — CORRECT — MINOR: principal per-draw execution structure is retained.**  
    `r2acc.py:275–290` and `e1run3d.py:63–95` retain tight/tighter, successor, and validity evidence.  
    **Fix:** None to this structure.

16. **RESOLVED — CORRECT — MINOR: draw generation remains consistent with round one.**  
    `rules2d.py:50–55` and `rules3d_names.py:58–63` retain the inspected generators and deployed seed-zero reconstruction.  
    **Fix:** None.

17. **NOT RESOLVED — WRONG — MINOR: fixture admission still passes an empty reduction.**  
    The independent fixture construction remains present, and `fixture_fired()` now requires six states. But `establish_e1.py:56–63` still returns `admitted=True` for empty inputs. CPU reproduction: `admit([], [], []) → {'states': [], 'admitted': True}`.  
    **Fix:** Require equal, nonzero input lengths—and the registered six states—inside admission itself.

18. **RESOLVED — CORRECT — MINOR: inspected 2D bank substitution remains complete.**  
    `p1common.py:153–168`, `e2bank.py:20–28`, and now `dtime.py:89–91` use the supplied bank/rotation and reset offline caches.  
    **Fix:** None to these paths.

19. **RESOLVED — CORRECT — MINOR: frozen training sources retain their provenance.**  
    Read-only hashing found zero mismatches across 17 train2d and 10 train3d provenance entries. `train2d/run_seed.sh:54–63` separates smoke settings from production settings.  
    **Fix:** None to production recipe preservation.

20. **RESOLVED — CORRECT — MINOR, within the original scope: principal submission safeguards remain.**  
    `cluster/submit.sh:13–46` retains lane serialization, lane/account checks, campaign lock, reservation, and post-submit queue inspection. `ledger.py:48–68,103–118` charges reservations and GPU-count-adjusted elapsed time.  
    **Fix:** Preserve these safeguards; resource-certificate deficiencies are item 12.

**Additional findings**

21. **WRONG — BLOCKING: every generated target-free E2 2D verdict run loses own-family qualification.**  
    `make_configs_e2.py:75–78` deliberately sets `target=None`. `e1analyze.py:149–151` therefore passes `target=None, rho_tgt=None` into `draw_qualified()`, which rejects these at `e1stats.py:126–134` before computing own-family qualification.

    CPU reproduction with otherwise passing evidence returned:
    ```
    (None, None, 'missing')
    ```
    **Fix:** Validate and compute Q1/own-family Q2 independently; require target evidence only for cross-family qualification.

22. **WRONG — BLOCKING: E2 H-acc-r can treat nonstationary or failed-anchor cases as qualified.**  
    The 2D `case_q` at `e1analyze.py:167–170` and 3D `case_q()` at `e2analyze3d.py:44–48` omit stationarity and tighter-anchor validity/agreement. These values feed H-acc-r directly (`e2analyze.py:83–84`; `e2analyze3d.py:84–85`).

    CPU reproduction returned `[True]` for a 3D case with every step nonstationary, a failed tighter solve, and tight/tighter distance 1.  
    **Fix:** Derive H-acc-r qualification from the complete registered Q1/Q2 predicate, including pooled stationarity and the required anchor gate.

23. **WRONG — BLOCKING: rejected or incomplete timing evidence can produce front membership.**  
    `e2analyze3d.py:116–142` uses individually hash/drift-passing invocations while merely reporting `tm['valid']`; it does not enforce panel acceptance or certified-run links. The 2D front similarly accepts any surviving valid ratio (`e2analyze.py:95–119`) without validating expected population/repetitions or incorporating the timing job into integrity.  
    **Fix:** Recompute and enforce the complete timing acceptance contract before assigning costs or front states. Include timing-job identity, completion, bank identity, and population in acceptance.

24. **WRONG — BLOCKING: E2 3D selection bypasses important evidence checks.**  
    `select3d.py:86` trusts serialized tight/tighter distances; `:95` derives the expected tighter population from that same supplied list. Recomputed distances are not compared with job distances. Job completeness/backend and `memory_rejected` are recorded (`:44–45`) but do not govern selection. `e2analyze3d.py:68` defines integrity essentially as completion alone.  
    **Fix:** Validate exact case/key mappings, independently recompute tighter distances, enforce distance agreement and memory rejection, and propagate these checks through cross-bank verdicts.

25. **WRONG — BLOCKING: identical selected/deployed configurations disappear from fixed-configuration verdicts.**  
    `make_configs_e2.py:69–72` correctly avoids duplicate execution when the selected rule equals the deployed rule. But `e2analyze.py:154–160` supplies no deployed alias in that case. Consequently the `dep` H-acc-N/H-acc-r series requested at `:180–186` becomes missing despite available identical evidence.  
    **Fix:** Reuse the execution under both analysis identities, with explicit provenance.

26. **WRONG — BLOCKING: the training smokes do not exercise the trained-bank pipeline.**  
    `smk2` executes incumbent replay before training (`jobs/smk2.json:11–27`) and has no subsequent selection or verdict step. `smk3` trains a pilot, then evaluates settings explicitly bound to incumbent `W1024` (`configs/smk3q.json`, `settings`); it supplies no pilot-bank registry.  
    **Fix:** Register each smoke’s emitted checkpoint/rotation and run the promised selection/evaluation stages against it. Keep incumbent reproduction as a separate explicit step.

27. **NEEDS-FIX — BLOCKING: no complete smoke acceptance path exists.**  
    DESIGN §6 names `smoke_accept.py`, but that file is absent. Production analysis cannot substitute directly: 3D `e1analyze3d.py:84–85,205` intentionally rejects the smoke’s three draws/eight cases, while E2 replay/selection entry points require production layouts.  
    **Fix:** Implement read-only acceptance of smoke outputs with explicit smoke populations, all required control checks, memory limits, and resource measurements.

**Smoke staging and coverage**

The driver imports succeeded on CPU, and their imported repository modules are included in the staging file set. Generated step commands have consistent job-relative paths. The revised completion markers match collection.

**As the checkout currently stands, none of the three jobs stages successfully:** `cluster/stage.py:77–79` rejects the untracked E5 files visible in `git status`. This is a staging-state observation, not an E5 code finding. Also, staging creates `runs/<attempt>` before this rejection (`:73–79`), so a failed attempt leaves a directory that prevents an immediate retry.

After resolving that staging state, I found no additional immediate driver-import or completion-marker mismatch. **That does not establish successful cluster completion or control acceptance**; runtime, memory, and the following missing coverage remain unresolved.

| Smoke | Present execution coverage | Controls/gates it cannot currently demonstrate end to end |
|---|---|---|
| `smk1` | 2D/3D E1 draws, bad-vector fixtures, identity rejection, permuted replay | Accepted **G-E1.2** reproduction lacks a complete smoke acceptance path. **C-E1.1, C-E1.2, G-E1.1, G-E1.3** produce evidence but have no unified smoke acceptance. **C-E5.1/G-E5.1** are absent from this job spec; E5 implementation was excluded. |
| `smk2` | Incumbent-bank replay through explicit bank substitution; truncated training and rotation | **G-E2.1 run path**: no candidate selection. **G-E2.3**: replay output exists but no smoke-compatible checker. No newly trained-bank identity/evaluation check; no promised 256²/1024² trained-bank verdict cells. |
| `smk3` | Training memory instrumentation; incumbent W1024/W256 ladders | **Memory gate** is recorded but lacks smoke acceptance. No selection/evaluation using the trained pilot bank. **G-E2.3** lacks a smoke-compatible checker. |

The revised code repairs several concrete round-one defects, but the remaining qualification, acceptance, and smoke-coverage failures require another fix-and-audit round.