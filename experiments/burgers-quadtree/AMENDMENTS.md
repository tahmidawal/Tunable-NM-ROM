# Prospective implementation amendments

2026-11-17, before representation01 submission: the 200-update throughput recipe uses uniform depth-1 trees and k=8, with the prescribed three 128-wide tanh layers and 32 outputs. It evaluates all registered feasible uniform depth-1/2 prefixes after training. This is only a throughput/gradient fixture, not the full trained candidate. The full recipe still requires the equal mixture of uniform, random-connected and training-guided configurations and refitted prefixes. Classical controls run first in the same job. Data remain provisional until reference qualification; no projection result is a PDE solve or final-test claim.

The first local POD control uses bilinear evaluation of a 17x17 shared residual basis and retains its interpolation error. It is an approximate local POD control, not an asserted optimum. The smooth polynomial and neural supports remain C2; interpolated POD features are not called C2 and require separate derivative qualification before PDE use.

## Explicit user architectural freedom — 2026-11-17

The user authorizes any necessary architectural changes: coarse/local decoders, shared/specialized networks, linear/nonlinear online coordinates, hierarchy/windows, refinement, objectives, coupling and hyper-reduction. Original candidate lists are starting points, not restrictions. Preserve physics-based evolution, per-case 1% fine-reference target, actual 4096x4096 output, complete cost accounting and other NM-ROM benchmarks. Register changes before runs and retain failures; no test-driven retuning without fresh confirmation. Finite compute expansions remain authorized. No unrelated PDE/3D expansion.

## Batch A1, registered before submission

Reference dense extension: at most 30 minutes on one A100. Use the already computed FD 256/512 sine interpolants and independent spectral solution; evaluate 512,1024,2048,4096 nodes, save every development final 4096 field, compare scoring changes with 0.0002 absolute gate. This qualifies the extension of evolved fields, not initial discontinuous wall traces.

Two representation repair hypotheses, at most 120 minutes: (1) coarse ranks 32/64 with existing local features repair missing global structure; compare matched-data global POD ranks 16/32/64/128. (2) 20,000 mixed-configuration updates with shared network, rank32 coarse space, eight trajectory-time pairs, sampled prefixes 4/8/16/32 and equal configuration-class probabilities. A finite pool contains uniform depth1/2, random connected depth1-plus-child trees, and TRAIN-residual-guided trees. Select checkpoint by fixed development projection every 1000 steps; no test outcomes. Evaluate all cases, retain all checkpoints. Gate remains <=0.5% projection on at least 90% of development trajectories; initial errors are separate. The previous 200-update pilot failed, retained. These reservations use reconciled terminal allocation seconds plus outstanding limits under the original eight-hour batch.
