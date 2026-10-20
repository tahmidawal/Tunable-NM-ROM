**FIX-FIRST for all three jobs: the required memory certificates are missing.** No files changed, jobs submitted, or SSH used.

| Audit item | Verdict | Evidence |
|---|---|---|
| Populations and members | **CORRECT** | `ref3`: val64 0–15 plus 43. `ps3`: {0, 43}. `b3d`: exactly the four registered members × cases 0–15; case 43 excluded. |
| Controls and deferrals | **CORRECT** | B2 checks and controls use 0–3; B7 successor uses 0–7, second successor 0–3, tighter anchor 0–5. B8 3D remains deferred under A4. |
| Staging and sbatch | **CORRECT** | All 155 manifest entries per job verify; staged provenance matches HEAD `08ed2932e`. Separate paralab directories, H200, 240G host RAM, f64/highest precision, GPU preflight and output checksums are present. |
| Time certificates | **CORRECT as estimates** | See below. These are plausible projections, not worst-case guarantees. |
| Memory certificates | **NEEDS-FIX** | All three contain `"memory": {}`. `resources.py:memory_note` implements neither 3D FD, PS nor b3run estimates, despite DESIGN A2.7 requiring them. |
| Downstream analysis | **CORRECT with the specified inputs** | `anaref` uses 3D {0,43} and per-window tiers. `ana3d --refs --manifest --anaref` checks promoted-reference hashes and consumes dimension-3 eligibility. |
| Inventory and reproduction coverage | **CORRECT** | `expected_inventory` generates 114 required coefficient arrays / 1,248 case rows. Qualify archives contain all 64 required member/case pairs: CN 0.01 in `q3d2`; the other three members in `q3d3`. Numerical production reproduction remains a post-run gate. |

| Job | Projection | 1.5× projection | Requested |
|---|---:|---:|---:|
| ref3 | 4.346 h | 6.519 h | 6.75 h |
| ps3 | 2.290 h | 3.435 h | 3.50 h |
| b3d | 1.613 h | 2.420 h | 2.50 h |

`ref3` includes the enumerated retry reserve for two cases, using the 5.1× refinement factor. `ps3` uses E0 ×2 horizon ×3 stability-retry pricing. `b3d` scales smoke rollout seconds by 4/3 members ×8 cases; this conservatively over-scales check populations, which increase only from two to four.

**Memory evidence needs a correction:** 28.63 GiB is the **ROM** smoke peak. The FD L512 smoke recorded **30.96 GiB**. At L768, the output buffer alone is **36.98 GiB**; cubic scaling of the measured FD peak suggests roughly **105 GiB**, before establishing refinement-specific headroom. This looks potentially feasible, but the submitted certificate supplies no accounting.

**Job decisions:** ref3 **FIX-FIRST**; ps3 **FIX-FIRST**; b3d **FIX-FIRST**.

1. Populate and regenerate all three memory certificates: ref3 must account for L768 outputs, solver workspace and retained arrays; ps3 for its largest FFT/workspace allocation and retry lifetime; b3d for W512 Gauss 56³/64³ construction and retained tables, plus the additional member’s compiled kernels. Compare estimated peaks with the actual device allocator limit and host allocation. Update staging if the committed source changes.