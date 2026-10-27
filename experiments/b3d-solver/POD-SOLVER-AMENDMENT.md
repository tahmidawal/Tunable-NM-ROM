# One bounded POD stopping-policy repair

The original five-schedule, four-case controls are independently audited and retained.
Three original policies miss their common full-trajectory field-distance check on case43,
despite valid native anchors and no fresh production-stationarity misses. Root approved
one prospective stronger production stopping tolerance,1e-4 versus the original1e-3.
Basis, trust radius, all five integrators/steps, Q1 thresholds and native anchor tolerances
remain unchanged. There is no tolerance ladder or new support/basis construction.

Run the five stronger schedules and the two original qualified15-/40-step policies in
one allocation on cases0,5,30,43. Reuse every independently valid native anchor by exact
source/file, basis, trust, input, case, cohort, schedule and initial-state bindings. Recompute
each new production-to-tight field distance with the current Phi Gram matrix, and freshly
certify every actual production output. Stronger production states are not required to
match the old states; original-policy controls must match their complete old state hash.
Native anchors are independent of the production stopping tolerance.

The copied `sch.gtol=1e-4` is passed to the unchanged typed compiled query and evolution
functions in both the tuning driver and final adapter. No global tolerance dictionary is
mutated. The original1e-3 Q1 stationarity bar remains the qualification threshold.

Use three A-B-A repetitions with original15-step POD as anchor. Both descriptor and native
drift masks are retained. Cost selection requires all four admitted case populations and
uses the median of per-case medians. Error selection is independent but requires all four
qualified, reference-eligible cases. Freeze the minimum-cost and minimum-worst-error picks
from these seven subjects; no wall-clock comparison across the two tuning jobs. Retain
all prior failures as their original observations. Budget15 minutes on one H200/200G.
