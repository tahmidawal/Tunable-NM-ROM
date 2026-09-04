Overall verdict: r3 is improved but still not implementation-ready. In the current phase-0 artifact, the one-Newton F3 control now fires and F11 passes, but the replacement F8 control still does not fire; `complete=false`.

Line references below are to [B3D-DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/B3D-DESIGN.md:1).

## Round-2 items 34–50

34. **RESOLVED** — undamped Picard is replaced by explicitly heuristic, residual-monotone Helmholtz defect correction (L98–111).

35. **RESOLVED** — cubic bootstrap and self-generated zero-correction history are defined, with predictor/output work charged (L104).

36. **RESOLVED** — the nested 33/65/129 grids and fixed 257-node normalization remain coherent (L62, L76).

37. **RESOLVED** — the architecture-matched IC is disclosed and D4 is reported with \(k=0\) excluded (L63–65, L191).

38. **RESOLVED-WRONGLY** — F8 now mutates a nonzero \(x\)-advection term (L178), but the current control is \(1.904\times10^{-5}\) against a \(10^{-4}\) firing threshold and still fails.

39. **PARTIAL** — downwind replaces the non-firing central control (L175), but the observed \(-5685\) blow-up is unchecked for residual convergence and is therefore a trivial rather than credible solver control.

40. **PARTIAL** — F7 is fully specified and F11 supplies an independent MMS (L177, L181), but fixed-\(\Delta t\) contaminates F11’s spatial order and its observed result sets F7’s moving acceptance band.

41. **RESOLVED** — promotion uses validation only, requires all gates, and stops if neither candidate passes before opening test seed 1 (L124–129).

42. **RESOLVED-WRONGLY** — a clustered lower bound was added (L216), but bootstrapping eight already-required-to-exceed-one point estimates makes that lower bound automatically exceed one and ignores timing-repetition uncertainty.

43. **PARTIAL** — C1 now uses each checkpoint’s own latents, is kernel-only, and has a numerical rule (L154–157), but a 1.5 ratio does not establish flatness.

44. **PARTIAL** — starts, budgets, doubling, worst error, POD comparison, and usefulness are added (L191, L217), but optimality is merely recorded and the N=129 POD construction is undefined under pool-only training.

45. **RESOLVED** — T0-truth is demoted to scope evidence and direct candidate/path fidelity carries tensor validity (L203–206, L212).

46. **PARTIAL** — M1 uses actual N=129 shapes and explicit host/device caps (L182), but omits the largest trainer, oracle, and EQ workloads.

47. **RESOLVED** — separate immutable seed-0 and seed-1 tables, exact prefixes, and fixed validation indices are specified (L48–52).

48. **RESOLVED-WRONGLY** — controls were added (L161–182, L190, L201), but D3’s duplicated row need not reduce column rank and the supposedly prevalidated F8 control demonstrably does not fire.

49. **RESOLVED** — a dedicated explicit-argument 3D trainer replaces the 2D-hard-wired trainer (L117–122).

50. **RESOLVED** — physical time and stored step index are now unambiguous (L10, L164–175).

## New findings

51. **NEEDS-RESTATEMENT — safeguarded defect correction.** The finite backtracking is a legitimate monotone safeguard, though not Armijo sufficient decrease; the cubic bootstrap and zero-correction rung are correct. However, [b3d_common.py:641](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_common.py:641) increments `its` when every alpha fails, so “one/two corrections” can mean failed attempts rather than accepted corrections.

52. **WRONG — F11 does not isolate spatial order.** The manufactured field satisfies the walls and is strictly positive on interior nodes, so its exact-state upwind branch is fixed. The continuous autodiff forcing has the correct residual sign, but the exact field does not satisfy the discrete BE time difference; with fixed \(\Delta t\), error against the exact field contains a nonvanishing temporal floor.

53. **NEEDS-RESTATEMENT — F7’s band is self-calibrated.** F11 is an independent datum, so this is not literal same-data circularity, but using its observed \(p=0.852\) creates the post-observation F7 band \([0.552,1.152]\); freeze F7’s band theoretically and report F11 separately.

54. **WRONG — F3’s specification and implementation disagree.** R3 still says two Newton iterations (L173), while [b3d_fom_gates.py:190](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_fom_gates.py:190) now runs one. One iteration does fire at \(4.875\times10^{-5}\); the exact Helmholtz preconditioner accelerates the linear solve but does not make nonlinear Newton one-step exact.

55. **WRONG — F5 can treat numerical failure as a successful control.** The driver ignores the downwind rollout residual and maps any nonfinite trajectory to \(-\infty\), directly violating “NaN anywhere is FAIL” ([b3d_fom_gates.py:196](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_fom_gates.py:196)).

56. **WRONG — D4 is not yet a converged, well-defined POD comparison.** A 0.5× train-POD-\(K\) bar is defensible as a deliberately strong nonlinear-benefit criterion, but that projection is a baseline, not a held-out “floor.” At N=129, a POD fitted only on the \(63^3\) pool cannot project full-grid held-out states, and D4 has no required optimality threshold.

57. **NEEDS-RESTATEMENT — STEP/ROLL uses a scale-dependent forward tolerance.** Eager-versus-jitted \(\max|\Delta z|\le10^{-10}\) may reject harmless reassociation or miss field-significant differences in a poorly scaled latent chart; gate normalized latent/field/residual differences and LM decision traces.

58. **WRONG — C1’s 1.5 rule does not justify “\(N\)-independent.”** A 50% timing spread can hide iteration-count or compilation differences; certify identical reduced jaxpr/FLOPs, report LM attempts, and use fixed-work timing or a materially tighter interval.

59. **WRONG — C2’s bootstrap is tautological under condition (i).** If all eight trajectory point speedups exceed one, every resample of those eight values also exceeds one; the bootstrap adds no timing uncertainty information.

60. **WRONG — M1 does not exercise the peak-memory workloads.** Bank/tensor construction plus one online step does not bound an 8192-snapshot trainer step, D4 full-grid multistart Jacobians, NNLS work arrays, or their compilation/allocator peaks.

61. **NEEDS-RESTATEMENT — FOM control parameters are not currently leaking, but the API permits it.** All supplied production call sites use defaults, yet `adv`, `zscale`, `zadv`, and `xadv` remain public production arguments and even an invalid `adv` string silently enters the scaled-upwind branch ([b3d_common.py:335](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_common.py:335)).

62. **WRONG — the supposedly fixed Fourier matrix is trained.** `B` is inside `params`, and the trainer zeros only the `out_scale` gradient before optimizing the whole tree ([b3d_common.py:895](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_common.py:895)).

63. **NEEDS-RESTATEMENT — only the reconstruction estimator is unbiased.** Uniform point subsampling gives an unbiased global reconstruction-MSE numerator and there is indeed no full-grid finishing, but squaring the sampled Gram deviation adds sampling variance and is not an unbiased estimator of the full-pool orthonormality penalty.

64. **CORRECT — test-mode normalization.** [test_modes_3d](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_common.py:723) produces unit-Euclidean-norm tensor-product DST-I columns and eigenvalues consistent with the assembled discrete Laplacian.

65. **NEEDS-RESTATEMENT — the indexing documentation is wrong, not the indexing.** `i*n²+j*n+k`, reshaping, interior extraction, and assembled operators are mutually consistent, but \(k/z\), not \(i/x\), is fastest-varying; [grid_coords_3d’s docstring](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_common.py:62) says the opposite.

66. **CORRECT — tensor derivative axes.** [backward_diff_bank_3d](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-03-burgers3d-tensor/experiments/separable-decoder/b3d_tensor_common.py:26) uses axes 0/1/2 and ghost zeros exactly consistently with `backward_adv_field_3d` and the assembled \(D_x^-+D_y^-+D_z^-\).

67. **WRONG — the r3 F8 replacement still fails empirically.** The current artifact records correct residual/JVP values of zero but an \(x\)-mutation control of only \(1.904\times10^{-5}\), below the implementation’s \(10^{-4}\) firing threshold; phase 0 remains incomplete.

68. **WRONG — D3’s negative control is not rank-forcing.** Replacing one row by another in a tall \(M\times R\) matrix with \(M=2R\) generally leaves full column rank; duplicate or zero a column instead.

69. **WRONG — the implemented F3/F5 cohort scope is smaller than r3 claims.** The phase-0 driver rolls out only the eight test trajectories, while r3 requires F3 on all trajectories and F5 on train, validation, and test truth.

70. **NEEDS-RESTATEMENT — the mode selector can cut a degenerate eigenshell.** Stable sorting does not guarantee that fixed \(M=128/256\) retains complete permutation-degenerate 3D sine shells, so the weak test space can acquire an artificial axis preference.

| item | verdict | one-line fix |
|---|---|---|
| 51. Defect safeguard/rungs | NEEDS-RESTATEMENT | Call it monotone finite backtracking and store attempted and accepted corrections separately. |
| 52. F11 spatial order | WRONG | Use the exact BE time quotient in the forcing or refine \(\Delta t\) with \(h\), and assert MMS positivity. |
| 53. F7 band | NEEDS-RESTATEMENT | Freeze an independent order band before phase 0; do not derive it from observed F11 output. |
| 54. F3 control | WRONG | Change r3 to one Newton iteration, retain the measured witness, and prevalidate it at every \(N\). |
| 55. F5 control | WRONG | Require a finite, residual-accepted control rollout; nonfinite output must fail, never map to \(-\infty\). |
| 56. D4/POD-\(K\) | WRONG | Define a streaming full-grid train-POD baseline, its aggregation, and a mandatory oracle-optimality threshold. |
| 57. STEP/ROLL | NEEDS-RESTATEMENT | Gate normalized latent, decoded-field, residual, and accept/reject discrepancies with conditioning recorded. |
| 58. C1 flatness | WRONG | Add identical-graph/fixed-work evidence and replace 1.5 with a tight uncertainty-qualified criterion. |
| 59. C2 bootstrap | WRONG | Bootstrap paired raw timing differences hierarchically over repetitions and trajectories before applying cohort rules. |
| 60. M1 coverage | WRONG | Compile and measure a real trainer step, D4 oracle batch, EQ fit, and full online path at each candidate snapshot count. |
| 61. Control-parameter leakage | NEEDS-RESTATEMENT | Expose a production residual with no mutation knobs and keep mutated operators private to the gate driver. |
| 62. Fixed Fourier matrix | WRONG | Remove `B` from the optimizer tree or explicitly zero its gradient and assert bitwise invariance. |
| 63. Subsampled Gram loss | NEEDS-RESTATEMENT | Define it as a stochastic regularizer or use an unbiased two-sample/full-pool Gram estimator. |
| 64. Mode normalization | CORRECT | No change; retain an explicit orthonormality/eigenvector gate. |
| 65. Flat-index documentation | NEEDS-RESTATEMENT | State that \(i/x\) is slowest and \(k/z\) fastest; the implementation order itself is consistent. |
| 66. Bank derivative axes | CORRECT | No change; retain a direct bank-versus-field derivative identity gate. |
| 67. F8 control | WRONG | Increase the deterministic mutation or use a separation-based threshold that demonstrably fires at every \(N\). |
| 68. D3 control | WRONG | Duplicate or zero a column so rank loss is guaranteed. |
| 69. F3/F5 cohort scope | WRONG | Run or enforce the acceptance/non-negativity checks whenever every train, validation, and test cohort is generated. |
| 70. Degenerate mode shells | NEEDS-RESTATEMENT | Promote complete discrete eigenshells and let \(M\) be the resulting shell size. |

**NO — the single blocking defect is C2: its trajectory-only bootstrap lower bound is logically guaranteed once all eight point speedups exceed one, so it cannot support the claimed uncertainty-qualified FOM win.**