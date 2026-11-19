# One bounded spatial refinement within B2

The completed L8192-to16384 check keeps the representations, CN step0.00125, Q512,22016 tests and solver tolerances fixed. Both methods remain physically accurate and every returned step converges, but dev000 changes by0.101566% for the direct tree and0.101589% for global960. The unchanged sensitivity limit is0.1%. The common near-equal change and roughly halved spatial error are consistent with the shared first-order upwind discretization, not a changed decoder. **L8192 is not admitted.** Keep that failed comparison and its raw states.

Reuse the immutable b2space run as the new **L16384 baseline**, on the strength of its physical and full-convergence checks only. This reuse does not turn its failed comparison with L8192 into a pass, and does not consume another baseline allocation. Register exactly one further spatial check, **L32768 versus L16384**. If it fails0.1%, stop spatial refinement in this batch; do not continue an automatic ladder.

Re-evaluate the other independent numerical controls at the new baseline; they are not automatically transferred from L8192:

| Owned run | Setting changed from b2space | Maximum reservation |
|---|---|---:|
| b2rspace | L32768 |25min|
| b2rquad | Q1024, with fine spacing1/16384 |30min|
| b2rtests |44032 common weak tests |25min|
| b2rtime | dt0.000625 |30min|
| b2rtight | The same100x tighter initial/inner/outer tolerances registered in B2 |25min|

The already running b2quad at L8192 finishes and is retained as diagnostic evidence. The not-yet-submitted original b2tests, b2time and b2tight are superseded by the finer-setting runs above; no allocated job is cancelled or hidden. Each new run compares directly with b2space and preserves all frozen representation hashes, initial-callable input, truth-free stepping, actual4096 final output, physical reference allowance and complete returned stopping rule. The packed implementation remains disabled.

This amendment does **not** raise the5400-second additional B2 allocation or28144-second cumulative ceiling. Completed time plus full outstanding reservations must fit that guard; reserve shorter jobs only when completed pilot times justify them. Two owned jobs maximum remains in force. If the required checks cannot fit, record an incomplete batch and review a specific finite continuation before further submission. All-case development qualification, adaptation, native NM-ROM confirmation, complete-runtime claims and final-test fields remain beyond this pilot.
