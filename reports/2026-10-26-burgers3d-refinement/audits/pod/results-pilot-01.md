# Independent pilot results review

Reviewer: `/root/independent_review`, read-only Codex agent.
Verdict: **READY for diagnostic publication**, with the post-completion failure disclosed.

All three sampled-objective Q1 checks pass on the diagnostic population. Both frozen
HR supports fail the new whole-residual admission contract. The reviewer independently
recomputed field chains, checked saved gradient/norm coherence, and separately
implemented a NumPy full-PDE discrete-adjoint gradient and Jacobian norm for one
saved step; this corroborated the saved full stationarity diagnostic.

The selected difficult-case subset cannot qualify the complete validation population.
Report sampled Q1 separately from HR failure, retain the frozen bars and case scope,
and do not present HR failure as POD losing to the coordinate bank.

The job completed and checkpointed its numerical evidence before a final-print NumPy
boolean serialization failure. Preserve original outputs and logs; no numerical rerun
is required. The raw metadata finite flag is a string, quarantined from admission;
finite values are independently checked from numerical arrays. Future serialization
must retain proper JSON booleans.
