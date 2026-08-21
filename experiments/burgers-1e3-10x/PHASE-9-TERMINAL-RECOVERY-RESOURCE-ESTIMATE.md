# Phase-9 terminal-recovery resource estimate

The single recovery cell requests one H200, 8 CPUs, 96 GiB host memory, and 16
hours.  It performs no training.  Its immutable work checkpoint is 19.8 MB;
the train coefficient targets are about 0.96 GiB in f64, and the largest train
full-field cohort is about 0.85 GiB.  Field evaluation remains
resolution-homogeneous with the locked Phase-9 batches 8/2/1.

The failed T1 invocation completed all training, terminal Cox/K3 evaluation,
and one final-capacity computation before failing in 47m20s.  Recovery performs
only data regeneration, checkpoint checks, terminal Cox/K3, final capacity,
output compression, and one independent repetition of field/capacity work.
Even a conservative 10x multiplier on the complete failed-job elapsed time is
under eight hours.  The 16-hour request therefore provides more than a 2x
margin without a cross-job performance claim.  No metric in this recovery is a
timing or speedup result.
