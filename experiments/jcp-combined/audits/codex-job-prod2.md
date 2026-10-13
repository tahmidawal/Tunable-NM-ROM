No production-blocking defect found for the three named jobs under A7. Read-only review completed: no files modified, no project modules imported, no jobs submitted.

1. **Prior finding 1 — ACCEPTED-AS-AMENDED. Former BLOCKER: runtime projection.**  
   [DESIGN.md:812](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md:812) explicitly makes this an engineering estimate, requires 48-hour requests, and requires reporting/rerunning overruns. [inventory.py:95](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:95) includes the enlarged ladder and revised timing costs; calibration uses recorded timings. All four projections reproduce exactly from the current estimator, inventories and smoke JSONs. The estimates remain approximations, as A7 permits. **Required fix: none for submission.**

2. **Prior finding 2 — ACCEPTED-AS-AMENDED. Former BLOCKER: production residency.**  
   [inventory.py:146](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:146) now requires each smoke peak, adds missing-bank tables and operator intermediates; the 2D calculation adds resident initial fields. [c3run.py:303](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:303) chunks operator evaluation into 16-state batches, preserving the per-state calculation. These are estimates rather than validated production peaks, consistent with A7. **Required fix: none for this gate.**

3. **Prior finding 3 — NOT RESOLVED completely; residual MINOR for these jobs.**  
   [inventory.py:303](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:303) now checks the complete `staged_sources` manifest, source/configuration hashes, smoke hashes/completion/gates, allocation-derived capacity, and finite positive resource inequalities. All certificates pass against HEAD and checkout bytes. In-memory mutations for stale smoke hashes, missing sources, 999-GB capacity, excessive runtime and NaN memory are rejected.

   Remaining generic weaknesses: line 319 strips the A100 capacity qualifier; [stage.py:60](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/cluster/stage.py:60) still permits certification bypass through an unrestricted `smoke` flag. **Fix:** distinguish A100 capacities and validate bounded smoke configurations. Neither affects these configurations: none sets `smoke`, and the authenticated 2D smoke explicitly records **NVIDIA A100 80GB PCIe**.

4. **Prior finding 4 — RESOLVED for production evidence collection. Former MAJOR: ladder qualification.**  
   [c3run.py:341](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:341) runs `SEL + DIAG` for every ladder rule and saves A6.2 eligibility. [c3run.py:246](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:246) requires valid fixed-sweep runs, verified same-rule adaptive counterparts, and finite worst Gram-field distance ≤2.5e−4. All three 3D configurations use the full 64-case validation cohort. **Remaining reporting requirement:** the later selector must consume these eligibility fields; this review does not certify that deferred analysis layer.

5. **Prior finding 5 — RESOLVED. Former MAJOR: test-cohort descriptors.**  
   [c2common.py:55](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2common.py:55) permits only `dev6`/`val32` and generates only configured evaluation cohorts, plus training descriptors for overlap checking. The unconditional `test64` generation is gone. **Required fix: none.**

6. **Prior finding 6 — RESOLVED by explicit correction. Former MINOR: evidence wording.**  
   [DESIGN.md:821](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md:821) corrects the BE distance, pairwise maxima and allocation comparison. [c3run.py:4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:4) correctly identifies CN K=2. **Required fix: none.**

Verified certificate values—all specify **48 hours and 200G host memory**:

| Attempt | GPU | Projected hours, including 1.5× | Projected device GB |
|---|---|---:|---:|
| c2grid | A100-80G | 13.882 | 17.569 |
| c3descW1024 | H200 | 31.144 | 74.158 |
| c3descW768 | H200 | 17.635 | 74.158 |
| c3descB, later | H200 | 17.628 | 75.193 |

No deterministic crash or missing qualification evidence was established for these jobs. Resource overruns remain possible under the explicitly accepted estimate policy.

**SUBMIT**