# Phase-9 terminal-retraction audit resource estimate

The one audit-only cell requests one H200, 8 CPUs, 96 GiB host memory, and 16
hours.  It performs zero optimizer/science updates and never runs the recovery
driver.  It regenerates the locked train cohort, independently evaluates the
terminal Cox/K3 fields, and repeats the final-only capacity work solely to
record the already-retracted portability disagreement.

Job 2735251 completed the recovery driver plus this class of independent audit
work in 14m29s on an H200.  Removing the driver cannot add work.  The 16-hour
limit is therefore conservative by more than an order of magnitude; no timing
or speedup claim is made from this cross-job estimate.
