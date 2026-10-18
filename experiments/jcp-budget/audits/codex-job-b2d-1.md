**FIX-FIRST for b1024a/b1024b; SUBMIT for b256/b4096**, subject to the stated reference-promotion prerequisite. The blocking issue is split-result aggregation, not the GPU driver.

Read-only audit of `281fc66d0dff82bf692001037ad9b3376c6363b2`. No writes, SSH, submissions, or scratch copies.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Configuration inventory | **CORRECT** | Exact equality with the SHA-pinned FRONT-2D v1: b1024a has 8 R128/R256 members; b1024b has 3 R384 members; b256 has 10; b4096 has 12. Populations are all38/all38/all38/sub16. |
| 1. Designs and checks | **CORRECT** | 1024² uses 38 corners: the 32-corner factorial plus M and its five pairs. Other meshes use 22 pairwise corners. B2 check population is dev6. |
| 1. B7/B8 and controls | **CORRECT** | All four use B7 T=0.5, q2=sub16, tighter=dev6; B8 uses the namespace’s `refs`, `REFS-MANIFEST.sha256`, and dev6 checks. Controls occur only in b1024b/b256/b4096, each with R384 present. |
| 2. Resource certificates | **CORRECT as estimates** | Recomputed projections exactly match all four certificates. Time limits exceed 1.5× projection; details below. |
| 3. Scripts and provenance | **CORRECT** | All 150 provenance-listed files per job match HEAD byte-for-byte; all staged manifest entries verify. Unique `jb_*` names, task directories, output/cache/tmp paths. H200, gpu partition, absolute cluster Python, f64/highest, GPU preflight and output checksums are present. |
| 4. Individual outputs and archive gates | **CORRECT structurally** | Required archive fields exist and are finite for every configured member/case. B2/B7/B8 inventories align with the per-job analysis gates. Numerical gate outcomes remain runtime-dependent. |
| 4. Combined 1024² analysis | **NEEDS-FIX** | The A8 member-wise merge is not implemented in the inspected analysis path. Duplicate meshes overwrite each other; B7/B8 aggregate decisions remain partition-local. |
| 5. Other submission blockers | **None found** | Remote reference readiness and queue state were not independently inspected, per the no-SSH constraint. |

The certificate calculations are:

| Job | Projection, hours | 1.5× projection | Sbatch limit |
|---|---:|---:|---:|
| b1024a | 4.5693 | 6.8540 | 7.00 |
| b1024b | 3.1133 | 4.6699 | 4.75 |
| b256 | 4.1996 | 6.2993 | 6.50 |
| b4096 | 5.0889 | 7.6334 | 7.75 |

[resources.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/resources.py:72) derives R128/R384 phase rates from the smoke. Unmeasured R256 receives R384’s slower rates. Checks scale to six cases, pairwise corner/check factors are 22/38 and 6/17, and 4096² receives ×2 plus the 67.3-second pilot. Check-phase timing includes setup/shared checks and some B7 work, providing conservatism; these remain plausible projections rather than guaranteed bounds.

Memory is also plausible on H200: approximately 48 GiB for the 4096² bank; approximately 15.6 GiB for the two R384 Gauss640 point-table sets at M=1536/2048; approximately 3 GiB for the fine bank; and approximately 6 GiB total for cached fine-floor QRs at R128/R256/R384. Production releases check tables before B8. The pilot measured 87.46 GiB, versus approximately 118.33 GiB at the configured 0.92 allocation fraction. It does not measure the entire production lifetime, but the additional cached QRs and case arrays do not indicate an unavoidable OOM.

The 256² B8 route correctly constructs a separate `Bank(1024)` and decodes on the 1025² lattice. Archive coverage includes the P1-cited members. In particular, 4096² R128 LSPG-BE factor 1 correctly pools 22 finite qualify cases with 16 finite P1 cases.

The aggregation defect is explicit: [ana2d.py:626](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana2d.py:626) assigns `D[j['mesh']] = j`, silently retaining only the last 1024² partition. [ana7.py:634](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana7.py:634) computes front order and subsequent ranking summaries only over that job’s members.

**Final decisions:** b1024a **FIX-FIRST**; b1024b **FIX-FIRST**; b256 **SUBMIT**; b4096 **SUBMIT**.

Blocking fixes only:

1. Implement and verify the A8 aggregation of both 1024² partitions: preserve all 11 members and both partitions’ gates/controls; reject duplicate-member conflicts; recompute B7 front order and B8 ranking/whole-front summaries over the combined population. Verify that input order cannot discard a partition. This requires an offline analysis fix, not additional GPU work.