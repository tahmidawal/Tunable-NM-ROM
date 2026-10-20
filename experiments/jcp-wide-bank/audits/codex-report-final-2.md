Read-only audit completed; no files modified. **All 15 numerical spot-checks pass. All 12 PNGs inspected.** Remaining blockers concern reporting and one misleading plot connection.

References below are to [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md).

**`codex-report-final.md` dispositions**

| Prior item | Disposition |
|---|---|
| §1 Number provenance | **NOT-CLOSED** — measured results and extrapolations are now distinguished, but failed-job metadata remains hard-coded in `make_report.py`, rather than sourced from the requested machine-readable ledger. |
| §2 Provisional table/tensor labels | **NOT-CLOSED** — rollout captions and tensor paragraphs now carry PROVISIONAL, but the bank-floor table at lines 113–126 still lacks its own status qualification. |
| §2 Native-grid versus refined-reference accuracy | **NOT-CLOSED** — line 3 retains “every accuracy number is scored against first-order references” before contradicting it with the correct native-grid bank-floor exception. |
| §2 Missing glossary terms | **CLOSED** — the requested definitions for \(A\), conditioning, timing, checks, bank identities, POD, Smolyak and decimal GB are present. |
| §2 Nodes versus cells | **CLOSED** — the glossary explicitly maps 65/129 nodes to 64³/128³ cells. |
| §2 Nominal \(\kappa\) | **NOT-CLOSED** — line 254 explains nominal \(\kappa\), but line 238 still defines it unconditionally as the exact ratio \(M/R'\). |
| §3 “Widening does not lower error” | **CLOSED** — both summary and detailed discussion acknowledge small improvements in every case while keeping H4 unavailable. |
| §3 “No better than M2” | **CLOSED** — the report now specifies worse worst-case error, slightly lower median error, and no overall dominance. |
| §3 G2/G4 scope | **CLOSED** — line 233 identifies the 32-column checks and the restricted checks above rank 512. |
| §3 Linear-span scope | **CLOSED** — the opening and detailed conclusion explicitly exclude a nonlinear-manifold head. |
| §3 Registered diagnostics | **NOT-CLOSED** — 3D \(m_d/m_\rho\), memory and converged-rollout iteration medians are added, but exit/rejected-step summaries and 2D iteration/memory summaries remain absent without an explicit omission statement. |
| §3 Storage conclusion | **CLOSED** — the claim is limited to advection storage at tested settings and accompanied by sampled total device memory. |
| §3 Cost-growth attribution | **CLOSED** — line 103 states that \(R'\), \(M\) and \(m^\star\) grow together without assigning an isolated cause. |
| §4 `2d_error_vs_Rp.png` | **CLOSED** — the regenerated plot uses a readable linear axis with useful ticks. |
| §4 All four J2 PNGs: post-hoc disclosure | **CLOSED** — every J2 plot now prominently identifies A7 post-hoc selection. |
| §4 J4 error legend/diagnostics | **CLOSED** — the legend is outside the curves and hollow markers explicitly denote gate-failed diagnostics. |
| §4 J4 diagnostic cost series | **CLOSED** — mesh series now have distinct colours and markers. |
| §4 J4 \(m^\star\) bank identities/unavailable settings | **CLOSED** — bank/family identities and unavailable ranks 768/1024 are identified. |
| §4 J4 memory identities/capacity | **NOT-CLOSED** — deployed points identify bank and mesh and the line says total capacity, but the hollow diagnostic series still omits its mesh identity. |
| §4 J2 hypothetical H200 comparison | **CLOSED** — the H200 line is now described as total device memory “for scale.” |
| §5 Coordinator readiness | **CLOSED** — all five specifically identified misleading interpretations are now explicitly corrected. |

The previously passing `2d_cost_vs_Rp.png` and `2d_memory_vs_Rp.png` remain consistent and readable; `2d_mstar_vs_M.png` retains its earlier nonblocking axis-label improvement opportunity.

**`codex-results-j4.md` dispositions**

| Prior item | Disposition |
|---|---|
| §2 “Floor halves” | **CLOSED** — the summary says “nearly halves” and the detailed discussion reports the reduction as 47%. |
| §2 Mesh dependence grows with width | **CLOSED** — the report explicitly says growth is nonmonotone and notes that \(M\) also changes. |
| §2 Prefix “no better than M2” | **CLOSED** — worst and median errors are distinguished. |
| §2 Storage limitation | **CLOSED** — advection-block storage is distinguished from the 75.76 GB sampled device footprint and sampling limitations. |
| §3 Wrong accuracy headline | **CLOSED** — the revised paired-improvement claims agree with recomputation at both meshes. |
| §3 Joint rank/test-count comparison | **CLOSED** — line 228 explicitly describes H4 as a joint \((R',M)\) comparison. |
| §4 Missing G2/G4 disclosure | **CLOSED** — the limited gate scope is stated explicitly. |
| §4 Historical solver comparability | **NOT-CLOSED** — the warning names J2 but omits the historical three-adaptive-step lane, despite the historical comparison at line 178 and DESIGN A8’s explicit requirement. |
| §5 \(m^\star\) plot identification | **CLOSED** — bank identities and unavailable wide settings are now shown. |
| §5 Error-plot legend overlap | **CLOSED** — the legend no longer overlaps the floor curve. |

**Numerical spot-checks: 15/15 PASS**

Each semicolon-separated value below is one checked number.

| JSON source | Independently checked report values |
|---|---|
| J1 `result.json`, rank 128 rows and final timing invocations | ST worst **4.64%**; S worst **3.45%**; query **28.4 ms** |
| J1 `result.json`, rank 512 rows and final timing invocations | ST worst **2.84%**; S worst **1.60%**; query **688.8 ms** |
| J3b `training.json` | Total **1.85 h**; 129-node rank-1024 floor **1.29%** |
| J2 `result_A7.json`, rank 512, κ=4 | Median iterations **27**; sampled peak **21.7 GB** |
| J4 `result.json`, 129-node W1024 rank 1024 | Converged worst **3.44%**; sampled peak **75.8 GB**; median iterations **28**; Gauss \(m_d\) **64000**; tensor **34.37 GB** |

Additionally, `checks/j1-audit.json`, `j2-audit.json` and `j4-audit.json` all report `accepted: true`; their full field-reconstruction audits were not rerun.

**New blocking error**

[3d_j2_mstar_vs_M.png](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/plots/3d_j2_mstar_vs_M.png) connects different ranks into one curve per quadrature family, creating an artificial drop near \(M=1027\) from the rank-512 value to the rank-256 value; split curves by rank or remove connecting lines so the plot does not suggest a within-rank reduction in required quadrature.

Resolve the **NOT-CLOSED** items above and this plot error; no sampled numerical correction is needed.

REPORT-OK: NO