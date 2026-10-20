**GO for submission.** No blocking defect found. Read-only audit; no files modified.

| Item | Verdict | Finding |
|---|---|---|
| Arms/spec | **CORRECT** | `sob001`: λ=0.01; `sob1`: λ=1; `coarse`: 129 nodes; `base_s1`: seed 1. All retain σ=4 and the prescribed training recipe. Matches §3.3 as amended by A1.5/A1.7. |
| Provenance | **CORRECT** | Both jobs’ manifests, file sizes and provenance hashes match staged files and recorded git blobs. Trainer, dependencies and staging code are byte-identical to audited tr1b. |
| Seed behavior | **CORRECT** | `--seed 1` changes parameter/latent-code initialization and value-point sampling. State selection explicitly stays seed 0; the CLI seed does not change trajectory generation. |
| 129-node path | **CORRECT** | Generator, coordinates, neighbours and rotation use the supplied node count. Interior has **16,129 points**, exceeding `p_sub=4096`; FD spacing is **1/128**, with zero boundary neighbours. Independent indexing/FD check passed to <8×10⁻¹⁴. Rotation uses that same interior. Full coarse GPU training remains untested. |
| Gates | **CORRECT** | Coarse skips R0 and R2a appropriately; its FOM residual check remains active. `base_s1` retains R0 but skips seed-0 R2a. Sobolev arms likewise retain R0 and skip R2a. |
| Batch/resources | **CORRECT** | Same allocation as completed tr1b: four A100s, 320G host memory, eight CPUs, seven hours. Separate GPU/output per arm, f64/highest precision, GPU preflights, checksums and failure propagation retained. Bash syntax passes. |
| Memory/time claims | **NEEDS-RESTATEMENT** | tr1b **actually completed on A100-PCIE-40GB**; `sob01` took **4.675 h training, 4.732 h total**. λ=1 changes a scalar weight, not workload dimensions or iteration count, so comparable cost and seven-hour feasibility are well supported—but not measured for `sob1`. A5’s “under 2GB extra” remains an unmeasured bound. |

No **WRONG** items in the staged job.