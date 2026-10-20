**f2kA: FIX-FIRST. b2f: FIX-FIRST.** Certificates and staging pass; A9.3 retry enforcement remains incomplete.

| Check | f2kA | b2f |
|---|---|---|
| Recomputed certificate | 11.266700 h → **17 h** | 1.954786 h → **3 h** |
| Cost fixes | R384 square allowance raised to 17 s | Both fixgate blocks included: 240 s |
| Staging | **176/176 hashes pass** | **176/176 hashes pass** |
| Provenance | Current sources match; 172 git-verifiable files match commit `93d652370` | Same |
| Inventory | 96 settings, 228 arms | 246 arms, two blocks |

Two remaining blockers affect both jobs:

- **Input identity remains incomplete.** [`NUMERIC_FILES`]( /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:126) omits the decoder checkpoint actually loaded by `Q.Model`. An in-memory mutation of that checkpoint’s provenance hash left the retry signature unchanged. Include its hash in the comparison against the original attempt.
- **Configuration conflicts are recorded but still consumed.** [`load_all()`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:140) retains the newest arm and merely appends `arm_conflicts`; downstream qualification proceeds. Reject incompatible arms or abort analysis. Validate each retry arm against the original configuration, including arms without duplicate archived results.

GPU-type comparison and exported attempt identity are implemented.

**Submission must still wait:** live queue is **3/3 lane and 8/8 account**; neither remote job directory exists. Under the current ledger:

| Reservation scenario | Total against 55 GPU-h |
|---|---:|
| f2kA alone, retaining other pending projections | **57.3879 — blocked** |
| b2f alone, retaining other pending projections | **52.6979 — fits** |
| Both | **58.4279 — blocked** |

After fixing A9.3, reconcile completed jobs and require **spent + live reservations + requested limit + remaining pending projections ≤ 55 GPU-h**, lane count **<3**, and account count **<8**, with the shared submission lock.

No files changed, submissions made, or LAB-LOG written.