A100 80 GB repeat. cuDSS files come from job 7178602 (a100-all, pax105), which was cancelled after its 28 cuDSS tasks
because its AmgX and CG parts were re-submitted as parallel jobs when the lane job cap was raised (a100-amgx 7181638,
a100-cg 7181639). The AmgX files that 7178602 had already written are not used; the AmgX set is taken from 7181638 only.
Job 7180190 (h200-hybrid) was cancelled after its Poisson 192^3 and 256^3 tasks; heat 256^3 ran in 7181622.
