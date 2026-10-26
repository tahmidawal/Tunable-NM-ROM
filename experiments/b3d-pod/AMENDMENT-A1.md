# A1: one predeclared training-only support extension and complete no-HR fallback

Status: proposed; independent design approval required before implementation.
This is the only support extension. It gives the POD comparator one bounded tuning
opportunity after the frozen supports failed. It is not an open support search.

Freeze the existing central rank-512 POD basis. Regenerate only the original ECSW
training states, with training seed 923701 and count 1536, trajectory-selection seed
20261017, original 24 training trajectories and unchanged option/state schedule.
The contribution matrix and full candidate list were not archived by the original
implementation. Rebuild them with the original routine and candidate seed 20261018,
32768 candidate grid nodes. Persist the actual candidate list, training descriptors,
state hashes, block scalings and contribution hashes before starting the fit.
No validation reference, descriptor, trajectory or error enters fitting.

Run exactly one inherited greedy-NNLS path with support target 16384 (k32), cap 16384,
the inherited drop/refactor/KKT rules, maximum 40000 iterations and a 5400-second path
deadline. No retry, rank change, alternative support target or validation-informed
weight selection. Accept an earlier KKT optimum only if its inherited certificate
passes; otherwise an unreached target is a construction failure. Reject nonpositive
weights, nonfinite values, drift-quality failure or device peak above 120 GB.

The hard fitting allocation is at most 2.25 GPU-hours, with an internal 2.15-hour
deadline including state regeneration, contribution assembly and fitting. This is
a cap, not a reliable cost bound. The frozen k32 support hash is recorded before
any validation. Validate it against the exact full-residual no-HR model using the
same BDF2 step, common production solver, tight/tighter anchors, own-state full
gradient and stationarity rules, field-distance bar and timing identity rules.
The original diagnostic cases are checked first. If any admission predicate fails,
STOP the support search and retain the failure. Full validation of k32 is released
only after the subset result passes and the support/data/configuration are frozen.

Reserve at most 0.75 GPU-hours for support validation. Separately run the no-HR
rank-512 model on the complete ordered validation population, retaining its own Q1,
E0 scores, cost outputs and iteration evidence, with at most 0.4 GPU-hours. It is a
complete fallback comparator; it is not hyper-reduced. Reuse completed pilot cases
where identity and cost contracts permit; do not rerun full k8/k16 populations.
All lane jobs, including failures, remain within the original four-GPU-hour cap.

Historical observed POD domination remains evidence with its original limitations.
Failure of the frozen supports or this one extension is not a claim that POD loses
to the coordinate bank. A qualified HR comparator remains open if k32 fails.

Implementation, small controls, exact source/config/batch and results retain their
independent audit gates. Fix the final-print NumPy-boolean serialization for future
jobs only; preserve the failed pilot's original checkpoint and log unchanged.
