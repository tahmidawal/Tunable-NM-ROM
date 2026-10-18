**FIX-FIRST** at `7444c4dfc`. Paths below are relative to `experiments/jcp-qualify/`.

| Code-3 must-fix | Round-4 disposition | Evidence |
|---|---|---|
| 1. Reservation lifecycle | **PARTIALLY / WRONG** | Positive finite hours and pre-submit release restoration fixed. Failed/timeout jobs still lose pending coverage; `reserve` never revalidates mandatory inventory. `ledger.py:71,132–149`. |
| 2. Pilot footprint | **PARTIALLY / NEEDS-FIX** | Original and fallback mains now remain resident (`qpilot.py:57–76`). Full timing footprint still unrepresented; details below. |
| 3. Fallback timing pricing | **RESOLVED / CORRECT** | Rule-size scaling now applies to certification and triplets. `resources.py:93–101`. |
| 4. Attempt discovery/source | **RESOLVED / CORRECT** | Overlaps have recorded lexicographic supersession; source lookup checks the requested arm. `analyze.py:189–208`; `front2d.py:34–40`. |
| 5. P1 reproduction manifest | **RESOLVED / CORRECT** | Smoke now requires numerical-manifest equality alongside trajectory agreement. `smoke_accept.py:51–56`. |

Ranked must-fix before submission:

1. **WRONG — reservation enforcement remains bypassable.** `pending` accepts `fballow256/1024/4096: 0`, despite A3.1 requiring two hours per unknown mesh. Alternatively, bare `q2known*` markers remove allowances without validating fallback reservations or enlarged timing projections. Failed jobs are not restored by reconciliation. Fix these transitions and validate inventory at every reservation. `ledger.py:71–94,132–149`.

2. **NEEDS-FIX — pilot cannot establish the full A2.4 footprint.** It holds only one initial field and runs fused rollouts, without timing’s split-query certification path. Actual timing retains initial fields for every case/mesh and holds decoded outputs while invoking certifiers and subsequent replays. The pilot’s peak therefore is not an established upper bound. `qpilot.py:46–48,65–73`; `qtime.py:71–78,137–185`. Reproduce that live allocation pattern or provide a justified conservative bound.

3. **NEEDS-FIX — initialize the submission inventory.** No `runs/GPU-LEDGER.json` exists; the submission wrapper’s reservation will refuse until a valid pending inventory is installed. `cluster/submit.sh:23`; `ledger.py:71–72`.

| Staged-job check | Judgment |
|---|---|
| Scheduler/environment | **CORRECT:** all use `gpu`, GPU preflight exit 42, x64/highest, isolated paralab outputs/cache/tmp. smkA: A100-80G, 200G, 2.5h; smkB: H200, 240G, 2h; smk3: H200, 200G, 1.5h. Each `runs/<attempt>/run.sbatch:3–27`. Runtime sufficiency remains unmeasured. |
| HEAD/provenance/checksums | **CORRECT:** all 136 staged source files per job equal committed HEAD; every manifest checksum passes. `runs/{smkA,smkB,smk3}/PROVENANCE.json:1`. |
| Dependencies/assets | **CORRECT:** staged CPU imports succeed for both driver families; required vendors, engines, checkpoint, replay, banks and smoke rules are present. Both bank pins and loaded rule pins pass. `cluster/stage.py:19–22`; `c3common.py:35–44`; `s3common.py:34–54`. |
| smkA controls | **CORRECT:** TH06 selects six dev6 cases; combined/P1 reproduction, positive GAL, under-resolution, both budget-zero, zero-update and wrong-mesh controls are wired. Successor roles and schedules check out. `qacc.py:130–132,264–366`; `smoke_accept.py:43–89`. |
| smkB controls | **CORRECT** for identity, injected fault and doubled work; **NEEDS-FIX** for pilot coverage above. `qtime.py:144–145,249–310`. |
| smk3 controls | **CORRECT:** 64-case populations, eight check-rule cases, under-resolved control, unique coefficient keys, reproduction and offline-agreement gates are wired. `q3run.py:39–68`; `smoke_accept.py:165–173`. Offline recomputation must run after collection. |

No deterministic import, missing-role, schedule or rule-shape failure found. No files changed, scratch created, cluster jobs submitted, or GPU runs performed.