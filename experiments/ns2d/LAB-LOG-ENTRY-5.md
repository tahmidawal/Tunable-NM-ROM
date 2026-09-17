## 2026-09-17

### ns2d — correction to the two entries above: 3–5 % of the held-out oracle fits are LM budget exits, not "none"; verdicts unchanged

DESIGN §A8, commit on `exp/2026-09-17-ns2d`. The ns203 entry said "no budget exits"; that was
read off the `oracle_reasons` dict without the code map. `ns2d_decoder.make_lm_fit`: reason
0 = budget (300 iterations), 4 = stationary. Stored: ns203 {0: 12, 4: 372}, ns204 {0: 18, 4: 366}
of 384 held-out states at 256² (3.1 % / 4.7 %). Those states' oracle errors are upper bounds on
the best fit; the medians over 384 states and both H-ORACLE verdicts (1.19 / 1.15 vs bar 2.0)
stand. The report now carries the count per mesh (`oracle.budget_exits_of_states`), and the
self-audit's claim 16 is marked WRONG with the correction in addendum 3.
