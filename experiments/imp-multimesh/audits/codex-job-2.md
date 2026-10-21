**FIX-FIRST for S1 and J2.** The restaging fixes most earlier blockers. Submission reconciliation can still release a reservation without proving that no job exists, and early batch exits still bypass `OUTPUTS.sha256`.

Read-only audit completed: no files changed, jobs submitted, GPU computations run, or writes to main or `LAB-LOG.md`. No scratch files were created.

All paths below are relative to `experiments/imp-multimesh/`. **FIXED** means the implementation is present; numerical success still requires execution.

Verified staging and prerequisites:

- Both manifests pass: **S1 146 files; J2 63 files**.
- Both resource-certificate hashes match their designated staged config/script.
- S1’s staged numerical and acceptance modules match the current sources.
- Submission, collection, and both batch scripts pass `bash -n`; inspected Python modules parse.
- Tufts advertises H200 GPUs. The absolute venv executable and reference-store directory exist.
- Both remote attempt directories are absent. The account queue was empty; paralab had **3.4 TB available**.

**Disposition of every job-audit blocking fix**

| Previous blocker | Round-3 disposition and evidence |
|---|---|
| S1 sentinel absent | **FIXED.** Before/after sentinel measurements, drift predicate, and combined control-7 result are present: `mmsmoke.py:199–208`. Acceptance consumes that result: `mmaccept.py:39`. Detailed measurements remain in `smoke.json`; `ACCEPT.json` records the control outcome. |
| Ambiguous submission releases reservation | **NOT FIXED.** Failed SSH lookup now preserves an `AMBIGUOUS` reservation, but successful-empty lookup still releases it: `cluster/submit.sh:54–67`. Additionally, `squeue; sacct` returns the last command’s status, so a failed `squeue` followed by successful-empty `sacct` reaches release. |
| Validate collection arguments and exact attempt/JID | **FIXED.** Arguments are validated before remote interpolation; the ledger checks the exact pair; accounting must return the expected job name: `cluster/collect.sh:9–15`; `cluster/ledger.py:68–69`. |
| Reject duplicate JIDs | **FIXED.** Numeric validation and cross-attempt duplicate rejection: `cluster/ledger.py:57–61`. |
| Atomic ledger replacement | **FIXED.** Temporary file followed by `os.replace`, under the exclusive lock: `cluster/ledger.py:33–43`. This provides atomic replacement, not an explicit fsync durability guarantee. |
| Propagate checksum failure | **FIXED on the normal finalization path:** S1 `runs/s1/run.sbatch:61`; J2 `runs/j2/run.sbatch:54`. **NOT FIXED overall:** manifest/preflight exits still bypass finalization: S1 `:24–28`; J2 `:17–18`. |
| J2 extraction/head TERM→KILL escalation | **FIXED.** Both commands have 60-second kill-after limits: `runs/j2/run.sbatch:43`, `:49`. |
| Restage and refresh affected hashes | **FIXED.** Both manifests and certificate bindings verify. |

**Disposition of the relevant code-audit findings**

| Finding | Round-3 disposition and evidence |
|---|---|
| Full-grid basis-prefix copies | **NOT FIXED**, explicitly retained: `mmcore.py:313–320`. Nonblocking for this empirical feasibility smoke under `DESIGN.md:619–623`. |
| Obsolete full-factor rank criterion | **FIXED.** Orthogonality and per-prefix conditioning are separate: `mmcore.py:135`, `:149–151`; `mmbuild.py:243–249`. |
| Missing val32 tighter anchors | **FIXED.** First two cases selected independently per cohort: `mmcore.py:383–398`. |
| Basis failures abort all ranks / wrong terminal states | **FIXED.** Initial unresolved inventory, per-rank certificate handling, and missing-rule status: `mmbuild.py:201–212`, `:235–262`, `:287–305`. |
| Missing source-certificate evidence accepted | **FIXED.** Missing evidence leaves configurations unresolved and returns: `mmbuild.py:205–212`. |
| κ=4/κ=8 agreement attached only to one member | **FIXED.** Shared result attached to both configurations: `mmbuild.py:313–326`. |
| PM trajectory timers lack burn-in | **FIXED.** Burn-in precedes each trajectory: `mmbuild.py:339–345`. |
| Failed controls do not fail acceptance | **FIXED.** Aggregated control failure produces exit 1: `mmaccept.py:84–89`. |
| Missing control-4 fallback / positive fixture | **FIXED.** Half-candidate fallback and positive-fixture requirement: `mmcore.py:346–354`; `mmsmoke.py:167–178`; `mmaccept.py:40–41`. |
| Missing ours-table/query pilot | **FIXED.** Bank construction and query are exercised: `mmsmoke.py:211–223`. |
| Missing host peaks and production matrix assertion | **FIXED.** Host high-water readings and matrix shape are recorded; shape gates acceptance: `mmsmoke.py:104–118`, `:222–223`; `mmaccept.py:42`. |
| “12 real states” discrepancy | **NOT FIXED.** Design still says 12: `DESIGN.md:538–539`. Implementation records actual states and tiles to the asserted production shape: `mmsmoke.py:107–111`. Documentation discrepancy, not a submission blocker. |
| Missing kept-state-mask comparison | **FIXED.** Explicit comparison: `mmsmoke.py:152–158`. |
| Missing unmapped-coordinate assertion | **FIXED.** Coordinate mutation is measured and included in the predicate: `mmsmoke.py:92–97`. |
| Missing sentinel | **FIXED**, as above: `mmsmoke.py:199–208`. |
| Missing historical replication reference / controls 8–9 unreachable | **FIXED.** Reference is staged and manifest-verified; acceptance sections are independently guarded: `configs/smoke.json:48–49`; `mmaccept.py:51–83`. |
| Decision fixture might lack an admissible witness | **FIXED.** Two synthetic ours configurations derive from measured competitor errors: `mmaccept.py:64–77`. |
| Control-8 preconditions, expectations, vacuity | **FIXED for S1’s reduced fixtures.** Required outcomes and second-X precondition are enforced: `cost.py:308–350`. |
| Missing prerequisite yields no structured acceptance | **FIXED for ordinary acceptance-input failures.** Exceptions become failed records, followed by the final write: `mmaccept.py:25–35`, `:80–89`. Early batch exits still bypass acceptance. |
| Q2 ignores finer-rule convergence | **FIXED.** All six finer-rule distances enter acceptance: `oursq.py:88–92`. |
| Invalid/nonfinite evidence handling | **NOT FIXED overall.** Q2 preserves `UNVERIFIED`; Q1 invalid tight/tighter evidence can still become `FAIL`: `oursq.py:78–92`. **Production, deferred**; S1 replication consumes per-case errors rather than this qualification verdict. |
| Row/NPZ, mesh, coefficient and descriptor identity | **NOT FIXED overall.** Mesh, scheme, rank, coverage and coefficient hashes are checked, but complete rule/policy/configuration and descriptor identity remain unchecked: `oursq.py:48–66`. Remaining qualification semantics are **production, deferred**. |
| Shell operators evade timeout / command failures ignored | **FIXED for these staged commands.** Script execution under timeout and failure aggregation: `cluster/stage.py:80–89`; `runs/s1/run.sbatch:33–58`. |
| Timeout totals exceed allocation | **FIXED for the staged nominal budgets**, including kill graces; details below. |
| Deferred imports and runtime dependency staging | **FIXED for the identified dependencies.** Explicit imports and historical analysis file: `configs/smoke.json:35–49`. |
| CPU preflight exit assertion | **FIXED.** Exact exit 42 required: `runs/s1/cmds/00_preflight_cpu.sh:2–5`. |
| J2 bank not reaped | **FIXED.** TERM, KILL, `wait`, and status logging: `runs/j2/run.sbatch:33`. |
| J2 validates only final marker | **FIXED.** Generation and required phase markers are checked: `runs/j2/run.sbatch:36–39`. |
| Head runs after extraction failure / failed evidence deleted | **FIXED.** Head depends on extraction success; NPZ removal requires overall success: `runs/j2/run.sbatch:45–53`. |
| Complete extraction/head invocation wall times | **FIXED.** Timers surround both complete process invocations: `runs/j2/run.sbatch:42–50`. |
| Arbitrary submission kinds / certificate binding | **NOT FIXED overall.** Unknown kinds are rejected and hashes are checked, but requested kind is never compared with certificate kind: `cluster/submit.sh:9`, `:19–34`. The specified `s1 smoke` and `j2 probe` invocations themselves are consistent. |
| Stale/unbound J2 resource certificate | **FIXED.** Current script hash matches; certificate describes current budgets: `checks/resources-j2.json:5–7`; `cluster/submit.sh:27–28`. |
| Submission atomicity under ambiguous response | **NOT FIXED**, as detailed above: `cluster/submit.sh:54–67`. |
| Collection terminal check, archive, hashes and cleanup | **FIXED on the normal completed-output path:** `cluster/collect.sh:12–36`. Missing early-exit checksum files still prevent that path. |
| Ledger double counting, invalid projections, concurrent writes | **FIXED.** Terminal elapsed plus active reservations, finite positive bounds, exclusive locking: `cluster/ledger.py:27–43`, `:46–56`. |
| Duplicate bindings / atomic persistence | **FIXED:** `cluster/ledger.py:33–37`, `:57–62`. |

The prior timed-output, A2/J5 identity, outlier and retry fixes remain present in `mmtime.py:36–66`, `:100–131`, `:218–244`. Their production adequacy does not gate these jobs.

**Production, deferred:** complete production decision universes—including entirely missing X configurations—`cost.py` production adapter, aliases/input gates, accounting variants, and A7 compile handling in `mmtime`/`cost`. These do not block S1 or J2.

S1’s execution path is suitable for an H200 smoke: GPU partition, absolute venv, f64/highest precision, reference-store environment, and lane-local output/cache paths are configured at `runs/s1/run.sbatch:3–28`. Its command limits total **5,220 seconds**; five 30-second kill graces bring that to **5,370**, leaving **330 seconds** within the 5,700-second allocation for other work.

All nine controls now have executable paths and acceptance outcomes, including mapping mutations, Q1 fallback, timing injection/sentinel, decision fixtures, and replication rejection. Numerical success and 4096² memory feasibility remain empirical. An early manifest or GPU-preflight failure still produces neither `ACCEPT.json` nor `OUTPUTS.sha256`.

J2’s successful path supplies the required phase-wise bank-fit evidence: generation, step 1, 5,000→15,000 subsampled rate, 20,000→25,000 full-grid rate, transition window, and reconstruction tail. The historical trainer emits elapsed timestamps at `runs/j2/code_r3a/sep_solvers.py:398–405`, `:439–442`. Extraction and head wall timers include loading, fitting, diagnostics, emission, and process exit.

J2’s nominal budgets total **6,600 seconds**. Adding two 60-second kill graces and the five-second bank termination delay leaves approximately **775 seconds** within its 7,500-second allocation, before polling and other overhead. Its measurement processes are bounded; runtime success is not guaranteed by inspection.

Normal job paths stay within this lane, and collection now rejects malformed arguments and mismatched attempt/JID pairs. The remaining GPU-hour accounting risk is submission reconciliation: a live job can lose its reservation after an inconclusive empty lookup.

| Job | Verdict | Blocking fixes |
|---|---|---|
| **S1 smoke** | **FIX-FIRST** | Preserve reservations under inconclusive submission reconciliation; finalize structured failure/checksum evidence on early controlled exits. |
| **J2 probe** | **FIX-FIRST** | Same submission reconciliation and early-exit checksum fixes. Measurement-path blockers are otherwise fixed. |

Blocking fixes:

1. **Keep the reservation whenever submission remains uncertain.** Check both lookup commands independently; successful-empty accounting is not proof of non-submission. Do not select the largest JID silently if reconciliation returns multiple candidates.
2. **Add failure finalization for explicit early exits.** Preserve the original failure status, write structured failure evidence and `OUTPUTS.sha256`, and make checksum failure nonzero. S1 should record failed acceptance when it cannot reach its controls.
3. **Restage and refresh manifests and affected certificate hashes after those changes.**