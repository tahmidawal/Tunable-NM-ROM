# Coarse P4 component factor equivalence

One coarse retained P4/g4 h1 candidate mesh; no truth/training admission. Verify original
FE layout, actual component coefficient maps/essential masks and both native H1/L2 block
identities. Factor the original full velocity free H1 matrix once, then release it. Factor
the matching scalar free H1 once and solve each component through the actual maps. Compare
both results for four deterministic random RHS and four original weak pressure-divergence
RHS. Keep the unchanged 1e-10 full linear-residual/solution comparison threshold. Persist
all four original sparse metric matrices, maps, masks, RHS, pressure coefficients, both
solutions, factor estimates, native versions and peak RSS. Existing adapter capacities
remain active; the coarse input is additionally limited to 2,000 cells/40,000 mixed DOFs.

One fresh native-factor05 allocation, gpu partition, one GPU, sixteen CPUs, 64 GiB and
20 minutes; eight GiB working/one GiB result planning envelope. No automatic retry. No
fine factorization, eigenproblem, bank training or production cap change is authorized.
The exact package must pass independent source review and coordinator release first.
