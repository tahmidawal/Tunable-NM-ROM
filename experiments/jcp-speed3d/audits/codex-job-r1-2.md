**SUBMIT.** Round 1’s duplicate-ADT blocker is resolved.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Change scope | **NEEDS-RESTATEMENT** | Literally, `37860512d → c79a3e0b8` also adds the initial `runs/r1` snapshot. Outside staging and DESIGN/audit/heartbeat/checksum records, the only runtime change is the ADT cache in `mode_rep`. No unrelated runtime changes found. |
| 2. Cache isolation and distances | **CORRECT** | Key includes setting, rule, path and serialized ADT spec, including scheme, timestep and tolerance. Each configuration retains its own `W` and uses the matching cached ADT `W`. All **56** distance pairings passed CPU assertions. |
| 3. Analyzer populations | **CORRECT** | CPU execution of the staged ROM loop with simulated rollouts and actual writer tag/row expressions produced **56 configuration runs + 34 unique ADT runs = 2,880 rows**. Actual `rows_elig` passed all **112** configuration/ADT population checks: exactly cases **0–31**, once each. |
| 4. Restaging and batch requirements | **CORRECT** | All **35** provenance files byte-match source commit `c79a3e0b8`; all **39** manifest checks pass. All **40** staged files match lane commit `94658d092`. `COMMIT.txt` and frozen-selection digest agree. Sbatch is byte-unchanged: `gpu`, H200, preflight exit **42**, highest precision, x64 and paralab paths. |

Item 1 is a wording clarification, not a submission blocker. Verification covers the local staging directory.

No files changed, jobs submitted, GPU execution, LAB-LOG/main writes or scratch copies.