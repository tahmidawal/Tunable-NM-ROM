**FIX-FIRST for both S1 and J2.** Most measurement-path fixes are present, but submission ambiguity, collection validation, and ledger persistence remain unresolved. S1 also lacks the previously requested sentinel control.

Read-only audit completed: no files changed, jobs submitted, GPU computations run, or writes to main or `LAB-LOG.md`. CPU checks used mocked, in-memory inputs; no scratch files were needed.

Paths below are relative to `experiments/imp-multimesh/`.

**Verified staging and cluster prerequisites**

- Manifest verification passes: **S1 145 files; J2 63 files**.
- Both resource-certificate hashes match their designated staged config/script.
- S1’s staged `mmsmoke`, `mmaccept`, `oursq`, `mmbuild`, `mmcore`, and `cost` match the current sources.
- Both batch scripts pass `bash -n`.
- Tufts reports H200 resources; the absolute venv is executable and the reference-store directory exists.
- Both remote attempt directories are absent. The account queue was empty; paralab reported **3.4 TB available**.
- An in-memory acceptance test passes with complete synthetic evidence and returns exit **1**, with structured failed controls, when smoke evidence is missing. This verifies aggregation, not numerical controls.

**Disposition of previous findings**

“FIXED” here means the implementation is present; numerical success still requires the smoke.

| Previous finding | Disposition and evidence |
|---|---|
| Full-grid basis-prefix copies | **NOT FIXED**, explicitly retained in `mmcore.py:313`. A5.3 permits empirical feasibility testing, so this alone does not block S1. |
| Obsolete full-factor rank criterion | **FIXED**: orthonormality and prefix conditioning remain separate, `mmcore.py:135`; `mmbuild.py:243`. |
| Missing val32 tighter anchors | **FIXED**: anchors selected per cohort, `mmcore.py:384`. |
| Basis failures abort all ranks / wrong terminal states | **FIXED**: initial unresolved inventory and per-rank handling, `mmbuild.py:201`, `236`, `259`. |
| Missing source-certificate evidence accepted | **FIXED**: missing evidence returns with configurations unresolved, `mmbuild.py:205–212`. |
| κ=4/κ=8 agreement attached only to one member | **FIXED**: shared result attached to both, `mmbuild.py:323–326`. |
| PM timing lacks burn-in | **FIXED**: burn precedes each trajectory, `mmbuild.py:339–345`. |
| Failed controls do not fail acceptance | **FIXED**: aggregate failure exits nonzero, `mmaccept.py:84–89`. |
| Missing control-4 fallback / unqualified positive fixture | **FIXED**: fallback and positive-fixture requirement, `mmsmoke.py:167–178`; `mmaccept.py:40–41`. |
| Missing ours-table/query pilot | **FIXED**: bank construction and query are exercised, `mmsmoke.py:205–217`. |
| Missing host-memory peaks and matrix-shape assertion | **FIXED**: host peaks recorded at state, contribution, NNLS and ours-pilot stages, `mmsmoke.py:104–118`, `216`; shape gates acceptance, `mmaccept.py:42`. `pipeline.host_rss()` uses `ru_maxrss`, so these are process high-water readings. |
| “12 real states” versus actual pilot recipe | **NOT FIXED** in the design: `DESIGN.md:538` still says 12. The implementation records actual counts and tiles 10 states to 120, `mmsmoke.py:107–111`. The intended matrix shape is checked; this is a remaining documentation discrepancy. |
| Missing kept-state-mask comparison | **FIXED**: explicit mask comparison, `mmsmoke.py:152–158`. |
| Missing unmapped-coordinate assertion | **FIXED**: coordinate mutation enters the predicate and recorded evidence, `mmsmoke.py:92–97`. |
| Missing sentinel in timing control | **NOT FIXED**: `mmsmoke.py:193–202` measures three competitor blocks, with no sentinel. `mmaccept.py:17–18`, `39` accepts that control without sentinel evidence. |
| Missing staged historical replication reference | **FIXED**: explicit extra file, `configs/smoke.json:48–49`; present and manifest-verified in S1. |
| Controls 8/9 unreachable; witness may not exist | **FIXED**: separately guarded acceptance sections; measured competitor errors generate two accuracy-admissible synthetic ours configurations, `mmaccept.py:51–83`. |
| Control-8 missing preconditions / changing expectations / cumulative vacuity | **FIXED** for these fixtures: required outcomes and second-X precondition enforced, `cost.py:308–350`. |
| Missing prerequisite produces no `ACCEPT.json` | **FIXED** for ordinary prerequisite exceptions: independent guarded sections and final write, `mmaccept.py:25–35`, `80–89`. |
| Q2 ignores finer-rule convergence | **FIXED**: six check distances now enter Q2 acceptance, `oursq.py:88–92`. |
| Invalid successor/check evidence becomes numerical failure | **FIXED** for Q2: invalid or incomplete evidence yields `UNVERIFIED`, `oursq.py:89–92`. **NOT FIXED overall**: invalid Q1 tight/tighter evidence can still produce `FAIL`, `oursq.py:78–81`. |
| NPZ mesh/configuration identity and `W_sha` unchecked | **FIXED in part**: mesh, scheme, rank and coefficient hashes checked, `oursq.py:55–59`. **NOT FIXED overall**: full rule/policy/configuration identity and descriptor identity are not validated, `oursq.py:48–66`. |
| Shell command failures/operators escape timeout | **FIXED**: commands run as scripts under timeout and contribute to failure, `cluster/stage.py:80–89`. |
| S1 timeout totals exceed allocation | **FIXED** nominally: 5,220 seconds versus 5,700 allocated; five 30-second kill graces still leave 330 seconds for overhead, `runs/s1/run.sbatch:33–56`. |
| Deferred-import checks / CPU exit assertion | **FIXED**: explicit imports, `configs/smoke.json:35–45`; CPU exit 42 asserted in `runs/s1/cmds/00_preflight_cpu.sh:5`. |
| J2 bank killed without reap/status | **FIXED**: `wait` and status logging, `runs/j2/run.sbatch:33`. |
| J2 validates only final marker | **FIXED**: generation and all required training markers checked, `runs/j2/run.sbatch:36–39`. |
| J2 head runs after extraction failure / failure evidence deleted | **FIXED**: head gated on extraction exit; NPZ deleted only if overall success, `runs/j2/run.sbatch:45–53`. |
| J2 complete extraction/head invocation times missing | **FIXED**: shell timers surround both complete commands, `runs/j2/run.sbatch:42–50`. |
| Arbitrary submission kinds bypass gate | **FIXED** for unknown kinds, `cluster/submit.sh:9`. **NOT FIXED overall**: the requested kind is never compared with the certificate’s kind, `cluster/submit.sh:19–34`. The supplied smoke/probe choices themselves are correct. |
| Unbound/stale J2 resource certificate | **FIXED**: script hash binding and updated budgets, `cluster/submit.sh:27–28`; `checks/resources-j2.json:5–7`. |
| Ambiguous submission releases reservation | **NOT FIXED**: failed or empty reconciliation still releases it, `cluster/submit.sh:54–59`. |
| Collection terminal check, archive, hashes and cleanup | **FIXED** on the normal path, `cluster/collect.sh:13–35`. |
| Collection attempt/JID validation | **NOT FIXED fully**: job-name check added, but ledger membership is only a global grep and neither argument is validated before remote-shell interpolation, `cluster/collect.sh:8–12`. |
| Ledger double counting / invalid projections / concurrency | **FIXED**: terminal elapsed plus active reservations, positive finite validation, exclusive lock, `cluster/ledger.py:26–45`. |
| Duplicate JID bindings / atomic persistence | **NOT FIXED**: no duplicate-JID rejection; direct overwrite remains, `cluster/ledger.py:32–34`, `54–58`. |

The prior timed-output retention, A2/J5 identity, outlier and retry findings remain **FIXED** in the inspected implementation (`mmtime.py:36–66`, `200`, `238–242`). Their production adequacy is outside this job gate.

**S1 execution and control coverage**

Its H200 request, absolute venv, f64/highest precision, GPU preflight, reference-store environment, and output paths are correctly configured (`runs/s1/run.sbatch:3–28`). Command failures reach the final exit status; acceptance runs even after earlier command failures.

Controls **1–6, 8 and 9** have executable paths and acceptance records. Control **7 remains incomplete**: injection and clean-block comparison exist, but the sentinel requirement identified in the previous audit remains absent. Therefore S1 can report `all_passed=true` without exercising all previously required timing evidence.

The memory pilot now records the required peaks and checks the production-shaped matrix. Its feasibility and 3,600-second smoke budget remain empirical, not proven by this read-only audit.

**J2 measurement coverage**

The successful path retains enough evidence for the registered phase decomposition:

- generation timer and step 1;
- steps 5,000 and 15,000 for the subsampled rate;
- steps 20,000 and 25,000 for the full-grid rate;
- transition overhead and final reconstruction tail.

The historical trainer emits elapsed timestamps at those markers (`runs/j2/code_r3a/sep_solvers.py:398–405`, `435–442`). Extraction and head-fit wall timers include startup, fitting, diagnostics, and output writing.

Nominal budgets total **6,600 seconds**, versus **7,500 seconds** allocated. However, extraction and head fitting use plain `timeout`, with **no kill-after escalation** (`runs/j2/run.sbatch:43`, `49`). A process that does not terminate on TERM can consume the remainder of the allocation and prevent final timing/checksum records.

**Shared safety and output defects**

- **Submission remains unsafe under uncertainty.** An SSH or accounting failure is converted into an empty lookup, then treated as proof that submission failed. A successfully submitted job can consequently lose its reservation.
- **Collection is not namespace-safe for unchecked arguments.** Normal S1/J2 paths stay within the lane, but arguments enter remote shell commands before validation. The ledger check also does not establish the exact attempt/JID pair.
- **Ledger writes are not crash-atomic**, and duplicate JIDs remain possible.
- **Checksum failure does not fail either job.** S1 explicitly ignores it (`runs/s1/run.sbatch:61`); J2 leaves `FAIL` unchanged after it (`runs/j2/run.sbatch:54–57`). Early preflight/manifest exits also bypass final checksum generation.

**Production, deferred:** production adapter and aliases/input gates; complete production decision universes, including entirely missing X configurations; accounting variants; A7 compile handling in `mmtime`/`cost`; and remaining full qualification identity/state semantics. These do **not** block these two jobs.

| Job | Verdict | Blocking fixes |
|---|---|---|
| **S1 smoke** | **FIX-FIRST** | Add and acceptance-gate sentinel evidence; fix shared submission, collection, ledger and checksum failure handling. |
| **J2 probe** | **FIX-FIRST** | Add bounded TERM→KILL escalation for extraction/head commands; fix shared submission, collection, ledger and checksum failure handling. |

Before submission: retain reservations whenever submission status is uncertain; validate and bind collection arguments to the exact ledger entry; reject duplicate JIDs and atomically replace the ledger; propagate checksum failures; add S1’s sentinel control and J2’s kill-after limits. Restage and refresh affected manifests/certificate hashes afterward.