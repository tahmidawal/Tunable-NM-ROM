# Independent Codex numerical and result review

Verdict: accept the scoped cost-screen result and stop both tested arms. The implementing coordinator did not perform this independent review.

| Arm | Median paired ratio | Fine corrections (total) | Fine Jacobians (total) | Coarse Jacobians (total) | Accepted coarse proposals |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 1.000000000 | 324 | 624 | 0 | 0 |
| affine2 | 1.391964393 | 283 | 583 | 606 | 300 |
| pure2 | 1.439692301 | 308 | 608 | 608 | 18 |

The independent reviewer recomputed every paired ratio and found every paired candidate timing slower than direct. All timed hashes resolve to saved numerical witnesses; recorded and timed fields and coefficients are exactly equal for this run. Fresh per-step certificates, original baseline states/decisions, real-bank field/derivative transfer and source/output checksums pass. Logs confirm GPU/f64/highest; stderr is empty, with no captured-constant, OOM, truncation or disk errors.

The affine arm accepts every proposed coarse initialization, so rejection alone does not explain its cost loss. Pure-prefix rejection is common; neither arm has numerical coarse failures. The original temporal-warm P1 fused query remains the timed comparator.

Scope: failed development cost screen only. Tight/tighter trajectory agreement, physical E0 error, and finer quadrature were not evaluated; no matched-accuracy or general multilevel conclusion is permitted. No validation, 3D or repeated cycles should follow this gate. Timing begins with the supplied device-resident initial field, includes initial projection/evolution/six output fields, and excludes descriptor generation and output transfer equally.

Reviewer: `/root/independent_design_review`, read-only. Raw source/result linkage and numerical arrays were independently checked. Tables above are generated from saved run JSON, not transcribed.
