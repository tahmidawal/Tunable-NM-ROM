**All four B-3 WRONG items are fixed. No qualifying regression found.**

| Item | Verdict | Evidence |
|---|---|---|
| Cost certificate populations | **CORRECT** | Requires exactly `seed0` and `seed1000`, each measured on 576 trajectories, scaled to 1,152 and 4,032 respectively. Arbitrary populations fail certification. |
| Dry-run evidence | **CORRECT** | Mandatory executed solves require timing rows; state-machine replay checks decisions, terminals and scored runs. Skipped tight solves and invented terminals are rejected. |
| Admission NaN | **CORRECT** | Non-finite and non-positive proposed ceilings are rejected before budget arithmetic. The NaN/20-hours-spent bypass is closed. |
| Verdict ID normalization | **CORRECT** | Eligible IDs, floors, policy records and reference-error maps normalize consistently; duplicate aliases are rejected. Numeric and prefixed inputs produce identical verdicts. |
| Timing summary authority | **NEEDS-RESTATEMENT** | Safe for admission under the stated contract: `resources.train_item` recomputes from population timings and ignores the summary. Probe: summary **4.3922 h / fundable**, authoritative **4.6161 h**, exceeding **4.5667 h** cutoff. The informational summary remains misleading. |

Validation: **149 tests passed**, resource self-test passed, training static checks **0 failures**. No numerical GPU validation.

No repository/main/LAB-LOG writes, SSH, jobs, or sealed/OOD access. Scratch **29 MB**, deleted.