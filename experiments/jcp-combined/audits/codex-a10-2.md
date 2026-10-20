**Overall: WRONG — the W768 post-hoc selection is supported, but not every round-1 finding is fixed.** Label propagation still stops before downstream analyses, and diagnostic rule coverage remains incomplete.

1. **`freeze.py` — CORRECT.**  
   `3d_rules` preserves `posthoc`, `label`, selection arms, and both convergence outcomes. For post-hoc input, effective `converged` comes from `posthoc_converged`; `pre_registered_converged` retains the original `None`.

2. **Config generation — CORRECT.**  
   `make_budget_configs.py` uses effective convergence and carries the label and post-hoc flag. The new `make_c3grid_config.py` derives deployed rules from frozen selected settings, records excluded settings, carries the frozen-file hash and labels, and requires an available common baseline.

3. **`select3d.py` — CORRECT for the supplied archives.**  
   The header now accurately describes raw median fixed-sweep `tB` times and the diagnostic timing baseline. `ids_of` correctly handles the last trajectory block and checks consistent case ordering across repeated blocks. I checked every trajectory join in both production archives. Adaptive eligibility includes all 128 `gl48` rows, versus 64 for each other rule; all W768 adaptive-CN rows are valid.

4. **`cnfail3d.py` — partially fixed; coverage remains WRONG.**  
   Comparator validity is now reported, invalid comparators withhold the cross-rule verdict, case populations are checked, and `lat256` is excluded from the failing-set comparison. The current W768 output has complete coverage and valid comparators.

   However, [line 45]( /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/cnfail3d.py:45) discovers rules only from existing fixed-CN arrays. If an entire rule’s arrays are absent, that rule silently disappears; two surviving rules can still produce `true`. There is no expected-rule inventory check or `complete` guard. This leaves part of round 1’s coverage finding unresolved.

5. **A10.1 — NEEDS-RESTATEMENT.**  
   W768 confirms fixed CN K=2 is ineligible at **all 13 rules**, all three escalation selection axes are `unresolved`, original convergence is `None`, and pre-registered selection is unavailable.

   The identical failing set is **seven cases: 5, 7, 11, 30, 40, 43, 62**, across the **12 non-control rules**. `lat256` fails the distance threshold on **all 64 cases**. Thus “the same cases at every rule” needs “every non-control rule.” Smoke cases 0 and 1 exclude the seven production failures.

6. **A10.2 — CORRECT in scientific scope; downstream-label promise remains WRONG.**  
   The revised text correctly limits agreement to fixed BE and adaptive CN, distinguishes stopping verification from an error bound, and preserves later arm-specific gates.

   But “every downstream file” remains false: [analyze3d.py:79](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/analyze3d.py:79), [budget3d.py:69](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget3d.py:69), and [score.py:348](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/score.py:348) construct outputs without copying the rule-selection label or flag. Configs and raw results preserve them; these derived outputs do not. A10.3’s universal inheritance claim therefore also remains **WRONG**.

7. **A10.2a — NEEDS-RESTATEMENT, minor.**  
   The repeated-run explanation and last-block join are correct. In both archives, corresponding `gl48` rows match in every field **except `seconds_first`**. Replace “bitwise-identical rows” with “identical saved numerical diagnostics apart from elapsed time.” Both trajectory copies are not retained, so their bitwise equality cannot be independently established from the archive.

8. **W768 post-hoc convergence and selection — CORRECT.**  
   Independent read-only NumPy checks reproduced eligibility and pair statuses from archived rows and escalation records:

   | Pair | Fixed BE | Adaptive CN | Combined |
   |---|---|---|---|
   | `gl48|gl56` | fail | fail | fail |
   | `gl56|gl64` | pass | fail | fail |
   | `gl64|gl80` | pass | pass | pass |

   Hence effective convergence is **`gl64`**. Applying eligibility and the threshold to saved ladder distances gives:

   | Family | Qualifying rungs | Minimum rung | Fixed-arm median-time sum |
   |---|---|---|---|
   | Lattice | `lat32768`, `lat65536` | `lat32768` | 0.279608 s |
   | Gauss | `gl40`, `gl48` | `gl40` | 0.513338 s |

   The control does not qualify. Deployment is correctly **`lat32768`**, with fallbacks `lat65536`, `lat131072`, then unavailable. These costs rank the original fixed-sweep timed arms, not the adaptive selection pair.

No files were created, modified, or appended. The independent checks used archived statuses, rows, saved distances and timing arrays; I did not rerun full bank decoding.