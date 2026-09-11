# Poisson accuracy campaign integration inventory

All runs below are numerically audited and retained. Accuracy eligibility is shown separately. The experiment branch remains separate; no merge has been performed.

| Run | Source commit | Archive commit | Model | Finest error % | GPU ms | Host ms | Pass 5% | Invalid calls |
|---|---|---|---|---:|---:|---:|---|---:|
| staged_accuracy08 | `db66d194efa56a9fecbee9f0c116f42d7f356e7e` | `3d4b607313108ad01db94070288c1504400ccedc` | joint_matched | 7.571199 | 2.545561 | 6.116468 | False | 0 |
| staged_accuracy08 | `db66d194efa56a9fecbee9f0c116f42d7f356e7e` | `3d4b607313108ad01db94070288c1504400ccedc` | original_relative | 7.280248 | 2.613091 | 6.211456 | False | 0 |
| staged_accuracy08 | `db66d194efa56a9fecbee9f0c116f42d7f356e7e` | `3d4b607313108ad01db94070288c1504400ccedc` | staged_head | 7.757472 | 2.597750 | 6.231384 | False | 0 |
| staged_accuracy08 | `db66d194efa56a9fecbee9f0c116f42d7f356e7e` | `3d4b607313108ad01db94070288c1504400ccedc` | staged_joint | 7.660328 | 2.565345 | 6.163464 | False | 0 |
| capacity_accuracy09 | `f3e3c21a440a31eb97b06fdb9ab9951662e17935` | `787f1beac4fd496ab0b3e1d6366e44747632431d` | original_relative | 7.280248 | 2.610272 | 6.245764 | False | 0 |
| capacity_accuracy09 | `f3e3c21a440a31eb97b06fdb9ab9951662e17935` | `787f1beac4fd496ab0b3e1d6366e44747632431d` | r128_head | 7.386210 | 2.870484 | 6.464140 | False | 0 |
| capacity_accuracy09 | `f3e3c21a440a31eb97b06fdb9ab9951662e17935` | `787f1beac4fd496ab0b3e1d6366e44747632431d` | r128_joint | 7.553765 | 2.849253 | 6.453854 | False | 0 |
| capacity_accuracy09 | `f3e3c21a440a31eb97b06fdb9ab9951662e17935` | `787f1beac4fd496ab0b3e1d6366e44747632431d` | r64_head | 7.785384 | 2.623935 | 6.214082 | False | 0 |
| capacity_accuracy09 | `f3e3c21a440a31eb97b06fdb9ab9951662e17935` | `787f1beac4fd496ab0b3e1d6366e44747632431d` | r64_joint | 7.681653 | 2.584367 | 6.217932 | False | 0 |
| correction_accuracy10 | `d5146f31268f6c4302c745280420d857c98ebcef` | `0f5683124099eeab3f6e736d3f71c6c7e6a1a22b` | original_relative | 7.280248 | 3.927122 | 11.218030 | False | 0 |
| correction_accuracy10 | `d5146f31268f6c4302c745280420d857c98ebcef` | `0f5683124099eeab3f6e736d3f71c6c7e6a1a22b` | r128_q0 | 7.553765 | 4.412962 | 11.692039 | False | 0 |
| correction_accuracy10 | `d5146f31268f6c4302c745280420d857c98ebcef` | `0f5683124099eeab3f6e736d3f71c6c7e6a1a22b` | r128_q16 | 7.032119 | 5.232792 | 12.490237 | False | 0 |
| correction_accuracy10 | `d5146f31268f6c4302c745280420d857c98ebcef` | `0f5683124099eeab3f6e736d3f71c6c7e6a1a22b` | r128_q32 | 6.110576 | 6.043163 | 13.092479 | False | 0 |
| correction_accuracy10 | `d5146f31268f6c4302c745280420d857c98ebcef` | `0f5683124099eeab3f6e736d3f71c6c7e6a1a22b` | r128_q8 | 7.207505 | 5.031713 | 12.251328 | False | 0 |

Runtime columns are comparable within each run only. Exact paths, hashes, base-bank diagnostic target outcomes and retained implementation files are in `integration-inventory.json`.

Do not merge automatically; root presents the requested merge decision
Staged08 and capacity09 are accepted numerical evidence of target failures, not successful physical-accuracy methods
Correction10 remains one frozen family with 32 primary; its target and cost outcomes below are authoritative, regardless of acceptance of numerical audits
No training or source-family descriptors are needed online; exact supplied-source contraction and source-based training-code initialization are retained
Correction intervention adds linear capacity and analytic elimination; q0 parity and old-manifold inclusion are audited
All correction prefixes share the fixed128 bank and retain the separate 3% bank diagnostic target
Original 5% adjusted physical/numerical target is never relaxed; no universal FOM speed advantage is claimed
All full fields, repetitions, references and raw checkpoints restore from ordered archive parts

## Glossary

Run: one isolated GPU job and archive. Source commit: frozen scientific implementation used by that job. Archive commit: commit retaining the checked result parts. Finest error: worst current-relative full-field error over all opened development cases at the finest mesh. GPU ms: pooled median device query time. Host ms: pooled median time including transfers. Pass 5%: all cases pass adjusted physical error, reference refinement and numerical validity. Invalid calls: retained invocations failing numerical gates. Bank target: separate 3% full-span projection diagnostic. Prefix: 8,16 or32 columns of the same fixed training-only correction basis.
