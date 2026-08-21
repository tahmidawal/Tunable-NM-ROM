# Phase-9 resource estimate

Phase 9 uses one fixed H200, 8 CPUs, 96 GiB host memory, and a 16-hour cap per
arm. T1 is the only arm staged initially. Its 35,904 coefficient targets occupy
about 0.96 GiB in f64, the largest full train FOM cohort is about 0.85 GiB, and
parameters, optimizer states, autolatents, persisted epoch scalars, and trust
work remain well below the 96 GiB host request. Device work is
resolution-homogeneous with batches 8/2/1, avoiding simultaneous retention of
all full-grid fields.

The 528,768-update schedule is not assumed to fit merely from these byte
counts. Before update 1 the same-job no-update preflight compiles and measures
all three update phases at each resolution, persists 3 warmups and 10 measured
repetitions, and projects all locked updates, full-cohort evaluations, and
terminal trust/capacity work, the independent audit, and a conservative second
data-regeneration reserve equal to elapsed pre-preflight work. The decision
subtracts actual monotonic data-generation, compile, and preflight elapsed time
from 16 hours. If the projected remainder plus 15% exceeds the actual remaining
allocation, execution stops with bitwise-unchanged weights;
the cell is an infrastructure-only result. This makes the 16-hour request
feasible by construction without relying on a cross-job timing extrapolation.

T2 has the same request only if independently licensed. Its 165,954-generator,
164,384-encoder, and 2,533-predictor trees remain small relative to data, but
its mandatory same-job structural preflight and its own measured no-update
projection must both pass before update 1. The total preregistered cap is two
cells and 32 allocated GPU-hours.
