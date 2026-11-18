# Burgers tree embeddings: first representation evidence

Provisional development screening for the new study. These results concern reconstructed fields, not PDE trajectories or complete solve speed. All numbers below are generated from saved job evidence.

**Decision: do not advance these checkpoints to PDE performance claims.** No tested tree passes the all-snapshot 0.5% representation gate. R1 is registered to repair the training-code optimization gap before interpreting the early stop as a learning plateau. Final-test fields remain unopened.

![Representation capacity](representation-capacity.png)

| Method | Active variables | Worst development/alignment error |
|---|---:|---:|
| Global nonlinear k16 | 16 | 40.86953% |
| Global nonlinear k32 | 32 | 31.59653% |
| Global nonlinear k64 | 64 | 18.64784% |
| Global direct128 | 128 | 12.88335% |
| Direct local r4, uniform | 96 | 26.65210% |
| Direct local r4, prescribed | 128 | 18.64280% |
| Direct local r8, uniform | 48 | 33.51156% |
| Direct local r8, prescribed | 112 | 21.74389% |
| Direct local r16, uniform | 80 | 28.89365% |
| Direct local r16, prescribed | 80 | 28.89365% |
| Nonlinear child4, uniform | 96 | 26.13875% |
| Nonlinear child4, prescribed | 128 | 17.09895% |

Errors use the current snapshot's reference L2 norm. Panels contain16 random development trajectories and12 alignment trajectories, with11 snapshots each. These first comparisons use64-by64 Gauss integration; independent finer scoring and actual4096 output are being evaluated separately. Uniform and prescribed allocations share the same local checkpoint and parent. Prescribed trees use only the known initial field. A best-found nonlinear fit is not a certified global optimum.

![Parent and corrections](parent-children-witness.png)

The parent is refitted jointly with children, so the summed child correction adds to the refitted parent shown separately. Colors show physical field/error values; they are not a pointwise1% guarantee.

Even optimistic span diagnostics fail0.5% on the initial checkpoint. These controls release the parent to128 linear coefficients and every child to16 coefficients. They exceed the eligible128-variable budget and are used only to diagnose feature limitations.

| Optimistic local features | Free linear coefficients | Worst error |
|---|---:|---:|
| neural16 | 576 | 2.04852% |
| polynomial16 | 576 | 2.10900% |

The initial bank selected update11000; the direct local features selected update2000; the nonlinear child head selected update1000. Best and final checkpoints, raw fitting statuses and unsuccessful recipes are retained. The initial training optimized codes with Adam; a subsequent frozen-checkpoint audit found materially better training fits under explicit code refitting. This limits claims about training convergence, while preserving the measured finite-checkpoint failures.

The primary fine-upwind tested weak residual has an independent dense fixture. No fixed-tree trajectory, deployable adaptation, faithful CROM/SNF-ROM/masked-autoencoder confirmation, final-test result, or complete runtime claim has been produced at this stage. Those stages remain gated by representation evidence.

## Evidence and glossary

Sources: jobs7351850 (global),7351946 (direct local),7352069 (nonlinear children),7352154 (diagnostics). Raw arrays and file hashes are in the lane's results and manifests directories. Seed2026111851; float64/highest precision on verified A100 allocations. No timing is compared across allocations.

- Active variables: independently adjustable parent and retained-child numbers.
- Projection error: best reconstruction found with the reference field available, distinct from a PDE solve.
- Worst error: maximum over every listed trajectory and snapshot, including the alignment panel.
- Uniform: a complete tree at the deepest level fitting the variable cap.
- Prescribed: a connected tree chosen from the initial-field callable; no future solution access.
- Span diagnostic: a deliberately more permissive linear fit used to test whether a checkpoint contains useful field directions.
- R1: the prospectively registered repair that explicitly refits codes during weight training.
