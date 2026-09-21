# SPEED-LOG — burgers-heldout

The query kernel, solver variants and timing contract are inherited unchanged from hires-burgers
(`experiments/hires-burgers/SPEED-LOG.md`: Cholesky, trust clipping, damping carry-over, `pred2` predictor —
all kept there with same-job parity/before-after timings). This lane changes the MODEL (bank, head,
directions), not the kernel. Rows below are added only for measured effects in this lane's jobs.

| hypothesis | change | measured effect | kept? |
|---|---|---|---|
| a rank-512 bank keeps the incumbent's per-query cost | `cpod512` replaces `inc512` at identical R | pending (bh2/bh3) | — |
