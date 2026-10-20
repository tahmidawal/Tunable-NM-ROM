Two blockers remain. No files changed, no SSH, no submissions, and no arrays copied.

| Item | Assessment | Evidence |
|---|---|---|
| 1. Coverage | **CORRECT** | FD covers dev6 0–5 and val32 0–31 exactly once, split 13/13/12. PS splits the registered cases into {0,1} and {5,19}. Outputs remain job-local; no sealed/OOD or E0 writes. |
| 2. Scripts/staging | **NEEDS-FIX** | H200, `gpu`, 240G, preflight, x64/highest and isolated directories are correct. All five manifests verify. However, staged commit `0f7fb2d1` no longer equals HEAD `c676046f`, which advanced during this audit. |
| 3. Certificates | **FD NEEDS-FIX; PS CORRECT** | FD reserve omits permitted retry work, detailed below. PS now projects 2.29 h per split; 1.5× = 3.435 h, covered by each 3.5 h limit. |
| 4. Collection/analysis | **CORRECT** | A7 specifies serial promotion, full ref2a collection, fine-file verification before cleanup, and joint analysis of all five archives. Writer/promoter/analyzer layouts agree. |
| 5. Release gate | **CORRECT** | A7 gates 2D on smk1 alone; its acceptance file reports `all_passed: true`. |

Blocking fixes only:

1. **Correct the FD “full retry path” reserve.** [resources.py:152](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/resources.py:152) charges a tight **base** triple plus one refinement level. But [refrun.py:201](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ref/refrun.py:201) can refine first, then tighten the **refined** triple—including another L=12288 solve. Failed production/refinement solves can additionally trigger `retry`. These costs remain unrepresented even for A7’s assumed two difficult cases. Enumerate those paths, regenerate the three FD certificates, and set limits ≥1.5× the corrected projections. Current limits—4.25/3/2.75 h—pass the existing arithmetic but do not establish the claimed coverage.

2. **Reconcile the staged snapshot with the required HEAD audit.** All jobs retain 149 intact provenance files from `0f7fb2d1`; current HEAD differs in `DESIGN.md`, `resources.py`, `make_configs.py`, and the removed `configs/b1024.json`. The intervening changes concern b-jobs, so this is not evidence of changed reference behavior. Nevertheless, the requested “staged trees equal HEAD” condition fails. Restage against the settled commit after the FD correction, or explicitly pin the submission audit to the older snapshot.

| Job | Verdict |
|---|---|
| ref2a | **FIX-FIRST** |
| ref2b | **FIX-FIRST** |
| ref2c | **FIX-FIRST** |
| ps2a | **FIX-FIRST** — snapshot reconciliation only |
| ps2b | **FIX-FIRST** — snapshot reconciliation only |