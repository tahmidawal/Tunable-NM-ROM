**No remaining blockers found at HEAD `538cee4cf`.**

| Audit item | Verdict | Evidence |
|---|---|---|
| Populations and members | **CORRECT** | ref3: val64 0–15 plus 43; ps3: {0,43}; b3d: four registered members × cases 0–15. Controls/check subsets match; B8 3D remains deferred. |
| Staging and sbatch | **CORRECT** | All 155 manifest entries per job verify; tracked staged contents match HEAD. Separate directories, H200, 240G host RAM, f64/highest precision and GPU preflight present. |
| Memory certificates | **CORRECT as estimates** | All three reproduce from source evidence. FD: 30.96 GiB measured → 104.49 GiB at L768, including cubic-scaled output/workspace, below 128.62 GiB allocator limit. Retained observation arrays are host-side. PS: 7.25 GiB E0 peak; added horizon stores lattice outputs. b3d: 35.79 GiB estimate; smoke includes identical W512 Gauss 56³/64³ tables. |
| Time certificates | **CORRECT as estimates** | Requested/projected hours: ref3 **6.75/4.346**, ps3 **3.50/2.290**, b3d **2.50/1.613**. Each exceeds 1.5×. Retry reserves and smoke scaling match the stated models. |
| Downstream | **CORRECT** | `anaref` implements 3D {0,43} per-window tiers. `ana3d --refs --manifest --anaref` verifies promoted-reference identity/hashes. Inventory: **114 arrays / 1,248 case rows**. Qualify archives cover all **64** required member/case pairs for G-B3. |
| Other blockers | **None found** | Numerical production acceptance remains a post-run gate. |

**ref3: SUBMIT · ps3: SUBMIT · b3d: SUBMIT**

Blocking fixes: none.

Read-only audit completed; no files changed, jobs submitted, or SSH used.