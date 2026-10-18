**Overall: NEEDS-RESTATEMENT.** The four batch-2 configurations are admissible and match the qualification export. Two issues remain: A28.5 revives an obsolete defer policy, and A29 overstates automatic compliance with batch accounting and provenance requirements. No numerical WRONG finding.

References: `D` = `experiments/jcp-hr-baselines/DESIGN.md`; `Q` = the supplied qualify worktree’s `experiments/jcp-qualify/runs/EXPORT.json`. Audited baseline-worktree HEAD: `23945d31f`.

1. **CORRECT — A28.1 resolves round-18 item 1.**  
   It explicitly corrects the gradient-flop figure and applies the partial-workload exclusions to every projection. Evidence: `D:1533–1535`, `D:1459–1465`.  
   **Fix:** None required; retain those exclusions in generated resource descriptions.

2. **CORRECT — A28.2 resolves round-18 item 2.**  
   Memory acceptance is explicitly retrospective, restricted to the executed path, and no longer attributes the cumulative peak solely to ECSW. Evidence: `D:1537–1540`.  
   **Fix:** None required; use the stated rejection wording.

3. **CORRECT — A28.3 resolves round-18 item 3.**  
   Classified Stage-A timers replace elapsed wall time; the allocation is explicitly a start threshold; subsequent Stage-B expenditure and overruns remain reportable. Requests are frozen, deduplicated and nonrecursive. Evidence: `D:1542–1548`.  
   **Fix:** None required; preserve these distinctions in the ledger and extension implementation.

4. **CORRECT — A28.4 resolves round-18 item 4.**  
   The corrected allocation figures agree with the stated arithmetic and remain upper bounds, distinct from measured expenditure and the V-budget band. Evidence: `D:1550–1552`, `D:788–791`, `D:1262–1263`.  
   **Fix:** None required.

5. **NEEDS-RESTATEMENT — A28.5 only partially resolves round-18 item 5.**  
   Truncation and retry accounting are now explicit and conservative (`D:1554–1558`). However, “the defer order … remains §12’s” (`D:1558–1559`) revives a superseded policy: §12 permits dropping common-solver non-front runs (`D:303–305`), whereas A2.15 makes that ablation complete and removes those runs from deferral (`D:633–635`). A1.14 also superseded §12’s treatment of high-rank native `none` runs (`D:505–509`). This changes which evidence may be omitted.  
   **Fix:** Replace the §12 cross-reference with one explicit, consolidated defer list. Identify the governing treatment of native `none`, common-solver runs, extensions, probes and exploratory timing, consistent with A26.5’s required timing coverage. Keep the incomplete-cell and retry rules.

6. **CORRECT — A28.6 resolves round-18 item 7’s provenance clarification.**  
   Generation and freeze commits are explicitly distinguished. Evidence: `D:1561–1562`, `D:1519–1527`.  
   **Fix:** None required.

7. **CORRECT — A28.7 resolves round-18 item 8’s acceptance-scope clarification.**  
   Design acceptance satisfies only A19.4(a); code/gates, the resource artifact and amended smoke acceptance remain separate, revision-pinned requirements. Evidence: `D:1564–1566`, `D:1264–1269`. Round-18 item 6 required no correction; A26.5 still freezes membership before Stage B and selects picks afterwards (`D:1502–1506`).  
   **Fix:** None required to this wording.

8. **CORRECT — every A29 batch-2 configuration uses a registered baseline option.**  
   A1.4 registers BE/CN/BDF2 × Δt₀{1, 1.25, 5/3, 2.5}, with Δt₀ = 0.01 (`D:367–370`).

   | Configuration | Registered option | Export evidence |
   |---|---|---|
   | `W256|gl24|fs|CN|0.01|K1` | CN × 1 | `Q:1895` |
   | `W256|gl24|fs|CN|0.0125|K1` | CN × 1.25 | `Q:1909` |
   | `W256|lat8192|qn|BDF2|0.025|k4` | BDF2 × 2.5 | `Q:1905` |
   | `W512|lat16384|qn|BDF2|0.025|k4` | BDF2 × 2.5 | `Q:1911` |

   These are exactly the four `threeD.tested` passes. Different ROM solvers do not violate the registered time-option requirement.  
   **Fix:** None required; retain A2.4’s exclusion if a corresponding baseline option is actually deferred (`D:556–558`).

9. **NEEDS-RESTATEMENT — registration before production permits shared execution, but does not automatically satisfy A21.3.**  
   Running batch 2 in the original tuning job is sound: its X-specific fronts and Stage-B membership can be formed together with batch 1. But A29’s assertion that **all** A21.3(i–iv) are “satisfied by construction” merely because registration precedes production is too strong (`D:1579–1581`). A21.3 independently requires cost certification and charging, including qualification work in ours’ V ledger (`D:1326–1331`). Those obligations affect budget-comparability wording.

   A29 also names the qualification commit without pinning its export hash (`D:1572–1573`), despite A21.3’s explicit file-hash requirement. The supplied export’s SHA-256 is:
   `6cb394d149be8852260b75a585db8102791ef52e9963ab6934f67d536c374431`.

   **Fix:** State that shared initial execution replaces a separate timing pass, while A21.3’s remaining obligations still apply, with pick freezing superseded by A26.5. Pin the qualification export hash and identify the imported qualification work counted once in ours’ V ledger. Include batch-2 workload in resource projections.

10. **CORRECT — diagnostic versus qualified-subset labels are accurate.**  
    A22 withdraws batch 1’s misattributed quadrature evidence and prohibits H-base labels (`D:1337–1342`). A29 preserves that restriction and expressly disclaims completeness (`D:1576–1577`). The qualification export uses the same subset label and records untested candidates (`Q:1864`, `Q:1934–1952`).  
    **Fix:** None required; never promote batch-1 diagnostic outcomes into qualified H-base conclusions.

11. **CORRECT — A29.2’s query contract is consistent.**  
    The mesh query projects the interior full field, evolves using the specified FS or QN solver, and decodes six fields onto the same 63³ interior lattice. Evidence: `D:1583–1586`; `experiments/jcp-hr-baselines/s3common.py:135–159,162–211`. The three copied modules’ hashes match `INHERITED.json:172–190`. This preserves A21.2’s full-field contract rather than importing mesh-free timing.  
    **Fix:** None required; keep G-ours and both-job replay requirements (`D:1423`).

12. **NEEDS-RESTATEMENT — A19.4(a) is not yet discharged by this round.**  
    A26.7/A28.7 provide a valid successor-audit mechanism (`D:1512–1513`, `D:1564–1566`), but the restatements in items 5 and 9 remain unapplied. They affect allowed coverage and budget/provenance claims, so this is not CLEAN under the requested definition. I found no additional production-number inconsistency among A19–A29 beyond those items.  
    **Fix:** Apply items 5 and 9 and record acceptance against the exact amended design revision. Production still requires the independent code/gate, resource and smoke evidence.

No files changed, jobs submitted, GPU work or tests run, or scratch created.