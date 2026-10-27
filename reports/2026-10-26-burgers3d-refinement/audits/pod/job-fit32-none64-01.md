# Independent exact k32-fit and no-HR-validation job review

Reviewer: `/root/independent_review`, read-only Codex agent.
Verdict: **READY for one submission each** of the two exact packages.

The reviewer checked every file in both source archives, the approved numerical
source hashes, frozen configurations, remote source/batch hashes and resource caps.

| job | source archive SHA256 | batch SHA256 |
|---|---|---|
| fit32 | 8c37e1224f224c9fb7f68acec860573cf396c6d926c537dd62285e2ff6858725 | 0dad6f99d5427bb960d49291d157ba994e6e6a43bc69e2f721b872c83c741f55 |
| none64 | 1cbb7d13f30191a67c897044441ccf1cdd06e14440306d985758daa6cd40fa4a | f7de508cf98de46b3dbe3936ac8fceaf6207035a6c982f5d46acedb852567223 |

Fitting has a training-only runtime and no E0 environment. Validation has the
complete ordered population and prescribed tighter anchors. Both use independent
paralab directories, H200 allocations, precision/backend/source checks and bounded
runtime. Combined hard allocations and the reserved support validation remain
below the lane cap. A fitting failure stops support search; k32 validation remains
gated on the frozen artifact and results. Queue checks are required around each
submission and duplicate submission is forbidden.
