**Overall: NEEDS-FIX.** Two round-3 findings remain partially unresolved, and two additional defects affect missing-evidence handling and cost statistics.

Read-only audit completed. No files changed, jobs submitted, production results opened, or scratch files created. CPU checks used bytecode writes disabled.

Paths below are relative to `experiments/jcp-dims/`.

| Round-3 finding | Disposition | Assessment |
|---|---|---|
| 1. Timing-job certificates and adaptive anchors | **NEEDS-FIX** | Ladder anchor lengths/identities and adaptive anchor validity are fixed. Adaptive certificate and distance inventories remain unchecked. |
| 2. Successor mode identity and own validity | **CORRECT** | Mode mismatch blocks Q2; invalid q++ runs produce “finer rule unavailable.” Both reproduced. |
| 3. Population completeness | **CORRECT** | The reported adaptive/2D/round validity-inventory defects are guarded. Missing 2D tighter populations now remain missing. Separate downstream round handling is defective below. |
| 4. Evidence-bound caches | **NEEDS-FIX** | JSON rows and additional code dependencies are included, but numerical dependencies remain outside cache keys. |
| 5. Second-to-top independent rho | **CORRECT** | Runs the independent check and retains its audit flag without changing ladder-top precedence. |
| 6. Retry resolution | **CORRECT** | `dscore.out_dir` resolves completed numbered retries using the logical output stem. |

Ranked remaining defects:

1. **Adaptive timing certification still accepts missing evidence.**  
   [analyze3d.py:532](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:532) checks validity-array lengths, but not `noncert`, certificate row count, or either distance inventory.

   **Reproduced:** eight registered cases, valid anchors, `noncert=[]`, `cert.shape=(0,100)`, and only one tight and one tighter distance return `cert=True`. Screening/confirmation can therefore accept incomplete numerical certification. Require the complete case/step shapes and distance counts before evaluating thresholds.

2. **Caches still bypass changes to the numerical checker and its inputs.**  
   [analyze3d.py:161](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:161) binds rho results to trajectories, stored rho and two source files, but excludes `vendor/quad3d/npcore.py`, bank/rotation data, pinned rule data and explicit mode identity. Cache hits return before loading or validating these dependencies.

   The fixgate cache at `analyze3d.py:221` checks only the evaluated rule’s hash—not the fixture, bank, target rule or checker code. The 2D cache at `analyze2d.py:32–61` still omits the vendored `npbank.py` and bank/rotation data. Consequently, a changed checker or numerical input can retain an old agreement verdict. Bind these dependencies or verify immutable content pins before accepting cache hits.

3. **New: incomplete round evidence is consumed as a completed test.**  
   [analyze3d.py:320](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:320) assigns incomplete populations `blocked (population)` and exports that label. [make_round.py:34](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/make_round.py:34) accepts predecessor evidence solely because its key exists.

   `front3d.round_next` then removes every tested non-pass key; `analyze3d.py:623` likewise omits it from open candidates. Missing evidence can therefore permanently remove a candidate from further testing and the outstanding-work inventory. Preserve missing/blocked-evidence states separately and require sufficient predecessor evidence before advancing.

4. **New: cost Shapley and elasticity bootstrap change the registered statistic.**  
   [analyze2d.py:287](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:287) computes cost Shapley from the median of per-case medians. At `:519`, elasticity bootstrap does the same, while its point estimate uses the pooled triplet median.

   DESIGN §4 defines cost as the median over valid fused triplets. These statistics differ even with complete dev6 × two-repetition evidence: synthetic pairs `(1,100), (2,3), (4,5), (6,7), (8,9), (10,11)` give registered cost **6.5**, versus **7.5** in the Shapley path. Preserve repetition arrays and resample cases with their valid triplets, applying the registered statistic consistently.

| File | Verdict |
|---|---|
| `dscore.py` | **READY** |
| `analyze2d.py` | **NEEDS-FIX** — cache dependencies; cost statistic |
| `analyze3d.py` | **NEEDS-FIX** — certificate inventories; caches; incomplete rounds |
| `score3.py` | **READY** for the mapping edit |
| `make_configs_late.py` | **READY** |
| `make_round.py` | **NEEDS-FIX** — predecessor evidence |
| `waitjobs.sh` | **READY** |
| `d3run.py` | **READY** for adaptive `Wstore`/`Cstore` storage |

Passing checks included zero missing 2D manifest/config keys, the restricted speed3d scoring diff, adaptive scoring-name mapping, and shell syntax. On the permitted smoke archive, original Q1 passed; removing the tighter population failed Q1 and triggered the missing-population predicate.