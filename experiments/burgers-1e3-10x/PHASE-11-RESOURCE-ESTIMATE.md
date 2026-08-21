# Phase-11 resource estimate

The sole cell requests one fixed H200, 8 CPUs, 96 GiB host memory, and 16
hours. The 35,904 immutable coefficient targets occupy about 0.96 GiB in f64;
the largest full-train field cohort is about 0.85 GiB. G2 has 165,954 generator
parameters, 164,384 encoder parameters, and 2,533 predictor parameters. Its
model, optimizer, q32 train states, trust telemetry, and atomic checkpoints fit
comfortably within the request. Resolution-homogeneous batches 8/2/1 prevent
simultaneous device retention of all three full grids.

Feasibility is decided on the allocated H200 before update 1, not inferred from
another job. After data generation, compilation, and the mandatory live
structural panel, the cell measures materialized and host-transferred encoder,
joint, predictor, Cox full-cohort, and q32 trust work at every N using three
warmups and ten retained repetitions. The maximum projection names and sums:

- 705,024 encoder/joint updates and conditional 176,256 predictor updates;
- 72 pre-gate epoch Cox cohorts, terminal and globalized train Cox cohorts,
  and two train K3-equivalent cohorts, all evaluated in batches of at most 8;
- conditional 18 predictor epoch cohorts, its terminal Cox cohort, and one K3
  equivalent;
- 35,904 by 40 train trust attempts in resolution-homogeneous 8/2/1 batches;
- conditional two selection initial cohorts, two terminal Cox and two K3
  equivalent cohorts, plus 2 by 5,712 by 40 trust attempts;
- an independently recomputed data/audit reserve equal to elapsed preflight
  work and a fixed 1,800-second compression/checkpoint/audit reserve.

The driver subtracts actual monotonic elapsed time from 57,600 seconds and
requires `1.15 * projected_remaining <= actual_remaining`. Otherwise it stops
before update 1 with bitwise-unchanged weights. The structural panel must also
pass before this decision. Thus the 16-hour request is feasible by a measured
same-job kill gate; no cross-job wall time is a scientific comparison.
