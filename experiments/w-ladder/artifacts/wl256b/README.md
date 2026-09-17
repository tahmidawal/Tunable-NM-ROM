# wl256b (job 3780448) — gate-failed attempt, retained for the record

This job completed every measurement at 256² (432 timed invocations, 8 decompositions) and then
**raised on its own retained-value gate** and exited 1, per `DESIGN.md` §10 A4: the gate compared a
quantity that flips with a tie among the eight cold-fit starts. Its `result.json` is kept here with
its logs; the numbers are superseded by `wl256c`, which reran the identical configuration with the
corrected gate. Nothing here is cited in the report.

`result.json` has `"complete": false` because the driver raised before setting the flag; the
invocation records themselves are complete. The full field arrays were not collected: they are
superseded and the share is at 92 %.
