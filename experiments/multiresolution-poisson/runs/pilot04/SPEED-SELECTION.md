# Proposed next Poisson speed experiment: checkpoint selection

This generated selection uses every development case and both completed query meshes. It prepares a concrete experiment for coordinator review; no new job is submitted.

| Scheduled endpoint | Eligible | Worst adjusted physical error | Median adjusted physical error | Checkpoint SHA256 |
|---|---|---:|---:|---|
| original_global | True | 0.0776427372 | 0.0184951152 | `1b170670c7948f098d7c57f3030c19437df7a212dac4113376a6ed1509e679e1` |
| expanded_global | True | 0.0850077911 | 0.016423953 | `8a5af8674ad60c33d99792d6c8ede870379d3b915418b75b4a14c386417752ca` |
| original_relative | True | 0.0680196727 | 0.0184979689 | `81f945571da60bbfe9adfba5969727ade137c940b212bc6a6e25c525273d417a` |
| expanded_relative | True | 0.0747532372 | 0.0164215045 | `89578430f32a183d0319753a3f723d34fc58eceec73a469217ed0aa1c934d7e8` |

Proposed frozen controls are `original_frozen` and `original_relative`. The continuation minimizes worst adjusted error over 30 cases on meshes [256, 512], among complete endpoints with solver-valid tighter generic outputs throughout. Ties use median error and then model name. No inspected-case-only selection is used.

The proposed job retains both development cohorts and uses 3 repetitions, 1800 total measured invocations, and a `02:00:00` limit on generic A100. Requested test count is 256; primary residual-reduction targets are [0.01]. Separate stationary accuracy controls add 480 fully recorded single calls excluded from speed selection. The projection-by-initialization factorial, direct parity checks and DST controls are specified in the linked proposal.

See [algorithm and interpretation](../../AFTER-PILOT04-SPEED-PROPOSAL.md) and [complete budget](../../speed_factorial_proposal.json). All errors remain development measurements with empirical reference adjustment, without a rigorous continuum bound.

## Plain-language glossary

- **Endpoint / eligible:** fixed final training checkpoint / complete training and solver-valid tighter outputs on every declared case and mesh.
- **Adjusted physical error:** common-observation relative error enlarged using the recorded empirical reference-refinement difference.
- **Worst / median:** largest value over both meshes and all sources / middle value of that same collection.
- **SHA256 / frozen control / factorial:** content identity / unchanged comparison checkpoint / independently crossed implementation choices.
