# Independent code and small-control review

Reviewer: `/root/independent_review`, Codex agent, read-only review.
Verdict: **READY for the initial diagnostic pilot only**.

Reviewed commit `39a707bb3`; replay.py SHA256
`f42bc1508f01f4b0c42b4676f10988b18ee81e08f3db9905c069fcea077bd319`.
The reviewer checked whole-residual metrics against the inherited full and ECSW
residuals/Jacobians, BDF2 histories/startup, provenance/reference/artifact pins,
explicit case order, all three field chains, Q1 anchors, output identity and
paired timings. The saved gradient vectors support offline recomputation.

The all-node equivalence, finite-difference gradient, weight-scale, exact-zero
and output-schedule controls substantiate the equations. The split evolution
is warmed before timing, every phase is synchronized, and its coefficients are
checked against the scored fused output; split cost stays diagnostic.
Timing coverage explicitly retains unmatched or drift-rejected panels.

No blocking code finding. This verdict permits staging only. Exact source tar,
configuration and batch must pass an independent job review before submission.
The pilot is subset evidence; full validation and comparative claims remain gated.
