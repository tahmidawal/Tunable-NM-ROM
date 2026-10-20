**HOLD: c2budA, c2budB and c2budC.** The rollout inventory and pricing repairs cover the required work, and all three certificates reproduce. Smoke validation still does not fully fail closed.

| Item | Verdict | Finding |
|---|---|---|
| `c2bud --dry-run` call coverage | **CORRECT** | Includes baseline work, check-rule twins, B1 replays, control calls, selected-arm checks, successor and restoration. |
| Inventory described as an exact count | **NEEDS-RESTATEMENT** | It conservatively double-counts selected-arm check calls already performed during convergence checking. |
| `project_bud2` rollout pricing | **CORRECT** | All inventoried calls receive prices. Independent branch accounting confirms the estimate covers ordinary, successful-successor and failed-successor/restoration paths for these configs. |
| Unmeasured-setting pricing | **CORRECT** | R128 uses fast’s same-R′ rates, scaling main-role rates by **4181/1597**. R256, R384 and acc use R512, the available larger-R′ smoke setting. |
| 2D smoke gate | **WRONG** | Row completeness is repaired, but B1 evidence counts and offline B0/G-score evidence remain unchecked. |
| 3D budget smoke gate | **WRONG** | Checks run signatures without checking their required cases or multiplicities; missing control rows also pass. |
| Split configs and frozen binding | **CORRECT** | Exact requested partition; every frozen setting appears once, with unchanged arms and matching frozen SHA256. |
| Certificate hashes and arithmetic | **CORRECT** | Source/config/smoke bindings and `verify_certificate()` pass for all three allocations. This does not repair the smoke-gate omissions. |
| Scientific independence of split | **CORRECT** | Each setting’s budget uses its own trajectories and parameters. No budget definition or case population changes. |
| Offline analysis workflow | **CORRECT — run per job** | `budget2d.py` accepts one `--run`, not several. Analyze A, B and C separately, each with the production `--grid` argument. |

The ordinary execution performs **1,370 calls per setting**, including **1,338 saved rows** and **32 unsaved B1 calls**. The inventory prices **1,402**: the extra **32** are two selected check trajectories × 16 check cases. Successor plus restoration adds **704** inventoried calls. This overcount is conservative.

The certificate results are:

| Job | Settings | Projected hours, including ×1.5 | Projected device peak | Allocation | Decision |
|---|---|---:|---:|---|---|
| c2budA | R512, R256 | 46.181540 | 40.820703 GB | A100-80G, 48 h, 128G | **HOLD** |
| c2budB | R384, acc | 46.181540 | 40.820703 GB | A100-80G, 48 h, 128G | **HOLD** |
| c2budC | fast, R128 | 4.587112 | 40.820703 GB | A100-80G, 48 h, 128G | **HOLD** |

In-memory mutations against the actual smoke results produced:

| Mutation | 2D gate | 3D gate |
|---|---|---|
| B1 false; B2 incomplete; setting evidence removed; ordinary row failed | Reject | Reject |
| Delete one required row | Reject | **Accept** |
| Delete an entire case | Reject | **Accept** |
| Duplicate a row | Reject | **Accept** |
| Remove control rows | Reject | **Accept** |
| Set B1 evidence count to zero, retaining `all_equal=true` | **Accept** | **Accept** |

The defects are in [smoke_gates()](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:334). The 3D branch constructs only `(rule, spec)` sets; its `cases` variable merely receives a nonempty check. Neither budget branch binds or validates the offline B0 fixtures, real-data mismatch check, or G-score reproduction and negative controls. The previously audited successful smoke analysis does not make those certificate checks fail closed.

**Release conditions:** validate exact required case coverage and multiplicity in 3D, require complete B1/control evidence, bind the offline machinery-gate results to the smoke and analyzer sources, and regenerate certificates. H3’s companion comparison and H5’s scope clarification remain report items, not additional job blockers.

No files were written, created or appended by this audit. No jobs were submitted.