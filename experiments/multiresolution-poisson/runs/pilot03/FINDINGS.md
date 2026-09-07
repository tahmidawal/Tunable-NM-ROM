# Guarded Poisson small-system kernel findings

These generated results are provisional development evidence on the unchanged separable checkpoint and source cohort. They compare complete host-input/host-output queries while changing only the damped-normal-system solve, with explicit numerical-agreement and fallback checks.

Job `3352868`, source `20f96592b14c2d9727eb40341d2c6cbcb568f464`, GPU `NVIDIA A100 80GB PCIe`. The archive audit covers 672 invocations and 1200 replayed linear systems. No training or sealed-final evaluation ran.

## Full-query accuracy and latency

| Intervals | Arm | Requested/retained modes | Tau | Latency ms | Worst physical error | Invalid | Nonstationary | Time outliers | Fallbacks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | dst | — | — | 1.92482 | 0.000154599776 | 0 | 0 | 0 | 0 |
| 256 | dst_coarse128 | — | — | 2.19808 | 0.000552815664 | 0 | 0 | 0 | 0 |
| 256 | rom_modular | 64/64 | 0.01 | 4.16751 | 0.0736259009 | 0 | 5 | 0 | 0 |
| 256 | rom_fused | 64/64 | 0.01 | 4.16844 | 0.0736259009 | 0 | 5 | 0 | 0 |
| 256 | rom_gj | 64/64 | 0.01 | 4.03638 | 0.0736259009 | 0 | 5 | 0 | 0 |
| 256 | rom_modular | 64/64 | 0.0 | 5.48181 | 0.0731865097 | 0 | 0 | 0 | 0 |
| 256 | rom_fused | 64/64 | 0.0 | 5.2462 | 0.0731865097 | 0 | 0 | 0 | 0 |
| 256 | rom_gj | 64/64 | 0.0 | 5.10264 | 0.0731865097 | 0 | 0 | 0 | 0 |
| 256 | rom_modular | 256/257 | 0.01 | 4.64251 | 0.0725423651 | 0 | 3 | 0 | 0 |
| 256 | rom_fused | 256/257 | 0.01 | 4.55431 | 0.0725423651 | 0 | 3 | 0 | 0 |
| 256 | rom_gj | 256/257 | 0.01 | 4.09843 | 0.0725423651 | 0 | 3 | 0 | 0 |
| 256 | rom_modular | 256/257 | 0.0 | 5.00866 | 0.0725423651 | 0 | 0 | 0 | 0 |
| 256 | rom_fused | 256/257 | 0.0 | 4.75293 | 0.0725423651 | 0 | 0 | 0 | 0 |
| 256 | rom_gj | 256/257 | 0.0 | 4.38236 | 0.0725423651 | 0 | 0 | 0 | 0 |
| 512 | dst | — | — | 2.43503 | 3.6766576e-05 | 0 | 0 | 0 | 0 |
| 512 | dst_coarse128 | — | — | 2.98738 | 0.000552815664 | 0 | 0 | 0 | 0 |
| 512 | rom_modular | 64/64 | 0.01 | 5.3643 | 0.0736295517 | 0 | 5 | 0 | 0 |
| 512 | rom_fused | 64/64 | 0.01 | 5.02619 | 0.0736295517 | 0 | 5 | 0 | 0 |
| 512 | rom_gj | 64/64 | 0.01 | 4.71972 | 0.0736295517 | 0 | 5 | 0 | 0 |
| 512 | rom_modular | 64/64 | 0.0 | 5.95339 | 0.0731890805 | 0 | 0 | 0 | 0 |
| 512 | rom_fused | 64/64 | 0.0 | 5.9294 | 0.0731890805 | 0 | 0 | 0 | 0 |
| 512 | rom_gj | 64/64 | 0.0 | 5.42196 | 0.0731890805 | 0 | 0 | 0 | 0 |
| 512 | rom_modular | 256/257 | 0.01 | 5.27062 | 0.0725423124 | 0 | 3 | 0 | 0 |
| 512 | rom_fused | 256/257 | 0.01 | 5.25773 | 0.0725423124 | 0 | 3 | 0 | 0 |
| 512 | rom_gj | 256/257 | 0.01 | 4.85762 | 0.0725423124 | 0 | 3 | 0 | 0 |
| 512 | rom_modular | 256/257 | 0.0 | 5.60024 | 0.0725423124 | 0 | 0 | 0 | 0 |
| 512 | rom_fused | 256/257 | 0.0 | 5.79684 | 0.0725423124 | 0 | 0 | 0 | 0 |
| 512 | rom_gj | 256/257 | 0.0 | 5.35629 | 0.0725423124 | 0 | 0 | 0 | 0 |

Latency is the median over cases of each case's median repetition time. Nonstationary tau-controlled outputs remain eligible only when their explicit residual-reduction stopping condition was reached and independently checked physical accuracy qualifies. Tighter nonstationary outputs remain invalid. Every outlier and unsuccessful output stays in the data.

## Paired complete-query kernel effect

| Intervals | Modes | Tau | Original modular / specialized | Original fused / specialized | Same-grid DST / specialized |
|---:|---:|---:|---:|---:|---:|
| 256 | 64 | 0.01 | 1.03168 | 1.04168 | 0.471639 |
| 256 | 64 | 0.0 | 1.08381 | 1.0459 | 0.376237 |
| 256 | 256 | 0.01 | 1.18537 | 1.10838 | 0.461835 |
| 256 | 256 | 0.0 | 1.17051 | 1.11903 | 0.433825 |
| 512 | 64 | 0.01 | 1.11022 | 1.05805 | 0.498426 |
| 512 | 64 | 0.0 | 1.11543 | 1.10631 | 0.465981 |
| 512 | 256 | 0.01 | 1.08014 | 1.05515 | 0.493805 |
| 512 | 256 | 0.0 | 1.07935 | 1.06255 | 0.448728 |

Every ratio first divides the two case-median times, then takes the median over cases. A ratio above unity favors the specialized query. These same-mesh ratios do not imply target qualification and do not replace the cheaper qualifying FOM envelope below. No times from earlier jobs are used.

| Intervals | Target | Qualifying ROM / modes / tau | Qualifying FOM | ROM ms | FOM ms | FOM / ROM case ratio |
|---:|---:|---|---|---:|---:|---:|
| 256 | 0.1 | rom_gj / 64 / 0.01 | dst | 4.03638 | 1.92482 | 0.471639 |
| 256 | 0.05 | unattained | dst | unattained | 1.92482 | unattained |
| 256 | 0.01 | unattained | dst | unattained | 1.92482 | unattained |
| 256 | 0.001 | unattained | dst | unattained | 1.92482 | unattained |
| 512 | 0.1 | rom_gj / 64 / 0.01 | dst | 4.71972 | 2.43503 | 0.498426 |
| 512 | 0.05 | unattained | dst | unattained | 2.43503 | unattained |
| 512 | 0.01 | unattained | dst | unattained | 2.43503 | unattained |
| 512 | 0.001 | unattained | dst | unattained | 2.43503 | unattained |

Qualification requires every case to satisfy solver and numerical-agreement gates, $(e+\delta)/(1-\delta)\leq\epsilon$, and the configured reference-difference allowance. The reference adjustment is empirical, not a rigorous continuum bound. This envelope is selected on development data and is not independent final confirmation.

## Numerical agreement and linear systems

Maximum specialized-versus-original field discrepancy is 2.39879533e-11, latent discrepancy 5.01199588e-10, and initial-residual-scaled terminal residual-norm change 8.34418452e-16. Failed agreement gates: 0; changed counters: 3; changed stopping reasons: 0; replay counter mismatches: 3.

Independent CPU replay checks cover 1200 SPD normal systems. Maximum condition number is 18940.7472; maximum linear backward error is 1.53414879e-16; maximum relative step discrepancy against CPU generic solve is 3.83810106e-13. Timed specialized fallback count is 0. Online guards and any fallback execute within the measured fused query.

| Requested condition | Maximum known-solution relative error | Maximum backward error | Forward-error limit |
|---:|---:|---:|---:|
| 1 | 1.86282204e-16 | 4.71439518e-17 | 1e-12 |
| 10000 | 8.67719925e-14 | 9.56368074e-17 | 1e-10 |
| 1e+08 | 7.0824058e-10 | 1.53138027e-16 | 2e-07 |
| 1e+12 | 1.51922134e-05 | 5.4482195e-16 | 0.002 |

The separate non-SPD smoke fixture asserts that fallback is taken. Small backward error alone does not guarantee a small forward error for ill-conditioned systems; known-solution synthetic checks and actual final-field agreement provide complementary evidence. The original modular/fused control uses untouched `ctol_tol` source. Replays archive the actual matrices and steps but supply no timing data.

## Cost components and reference scope

| Intervals | Arm | Modes | Tau | Input ms | Projection/init ms | Solver ms | Fused device ms | Output ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | dst | — | — | 0.557592 | 0 | 0.164596 | — | 1.17525 |
| 256 | dst_coarse128 | — | — | 0.598632 | 0 | 1.13043 | — | 0.379083 |
| 256 | rom_modular | 64 | 0.01 | 0.576108 | 0.800776 | 2.1804 | — | 0.581438 |
| 256 | rom_fused | 64 | 0.01 | 0.577665 | — | — | 3.06628 | 0.538892 |
| 256 | rom_gj | 64 | 0.01 | 0.573733 | — | — | 2.96843 | 0.517633 |
| 256 | rom_modular | 64 | 0.0 | 0.564312 | 0.891189 | 3.37434 | — | 0.591447 |
| 256 | rom_fused | 64 | 0.0 | 0.536637 | — | — | 4.20613 | 0.463547 |
| 256 | rom_gj | 64 | 0.0 | 0.544074 | — | — | 4.04716 | 0.515436 |
| 256 | rom_modular | 256 | 0.01 | 0.577284 | 0.883144 | 2.74867 | — | 0.582326 |
| 256 | rom_fused | 256 | 0.01 | 0.552264 | — | — | 3.56244 | 0.461891 |
| 256 | rom_gj | 256 | 0.01 | 0.617573 | — | — | 2.83562 | 0.497489 |
| 256 | rom_modular | 256 | 0.0 | 0.594719 | 0.61097 | 3.26013 | — | 0.620868 |
| 256 | rom_fused | 256 | 0.0 | 0.604267 | — | — | 3.66165 | 0.40706 |
| 256 | rom_gj | 256 | 0.0 | 0.554639 | — | — | 3.2447 | 0.536224 |
| 512 | dst | — | — | 0.838042 | 0 | 0.195659 | — | 1.42303 |
| 512 | dst_coarse128 | — | — | 0.86858 | 0 | 1.54882 | — | 0.615245 |
| 512 | rom_modular | 64 | 0.01 | 0.836559 | 0.818282 | 2.48418 | — | 0.799419 |
| 512 | rom_fused | 64 | 0.01 | 0.791019 | — | — | 3.56031 | 0.668794 |
| 512 | rom_gj | 64 | 0.01 | 0.844075 | — | — | 3.16845 | 0.692033 |
| 512 | rom_modular | 64 | 0.0 | 0.811841 | 0.868698 | 3.48929 | — | 0.807812 |
| 512 | rom_fused | 64 | 0.0 | 0.821542 | — | — | 4.3884 | 0.687151 |
| 512 | rom_gj | 64 | 0.0 | 0.831001 | — | — | 3.80575 | 0.654112 |
| 512 | rom_modular | 256 | 0.01 | 0.822743 | 0.902425 | 2.84906 | — | 0.795059 |
| 512 | rom_fused | 256 | 0.01 | 0.83347 | — | — | 3.74594 | 0.671 |
| 512 | rom_gj | 256 | 0.01 | 0.786409 | — | — | 3.34133 | 0.695585 |
| 512 | rom_modular | 256 | 0.0 | 0.806859 | 0.803627 | 3.10658 | — | 0.77655 |
| 512 | rom_fused | 256 | 0.0 | 0.810344 | — | — | 4.34137 | 0.68114 |
| 512 | rom_gj | 256 | 0.0 | 0.819521 | — | — | 3.9008 | 0.654677 |

Fused projection, solve, residual checks and field decoding share one device interval. The original modular query provides separately synchronized components; these values are not substituted into fused invocations. Post-query physical and final-gradient validation is excluded from all query timings. Setup, operator construction and compilation/warmup remain separately recorded.

References use [512, 1024, 2048] intervals with common observation at 256 intervals. Maximum final empirical refinement difference is 7.35117705e-06. The artifact audit independently recomputes every recorded field metric, with maximum discrepancy 5.30084409e-16. Physical error is relative discrepancy on common nested observation nodes; same-grid discrepancy is retained separately in JSON. No new representation oracle is run here and no global approximation-floor claim follows from the kernel study.

## Plain-language glossary

- **Intervals / requested / retained modes:** cells per grid axis / desired number of smooth PDE tests / actual complete sine shells retained.
- **Arm / tau / latency:** implementation / initial-residual reduction target / median of case-median complete-query times.
- **ROM / FOM / DST / GJ:** reduced model / full discrete model / direct sine-transform solver / diagonally scaled Gauss–Jordan elimination.
- **Physical / same-grid error:** common-observation relative discrepancy from refined reference / full-mesh relative discrepancy from the same discrete full solver.
- **Stationary / invalid / outlier / fallback:** normalized objective gradient below threshold / failed numerical or solver gate / repetition above three aggregate latencies / charged use of generic linear solve.
- **Modular / fused / specialized:** separately dispatched query stages / one compiled device pipeline / fused query using guarded small-system elimination.
- **Condition / backward / forward error:** sensitivity to perturbations / normalized linear equation residual / relative solution discrepancy.
- **SPD / normal system / damping:** symmetric positive definite / equations formed from the residual Jacobian / regularization of the proposed step.
- **Replay / counter / numerical agreement:** untimed diagnostic rerun / recorded solver work / scale-aware output comparison under frozen limits.
- **Input / projection / solver / output:** host source upload / weak-source construction and initialization / latent or direct solve / full host field and metadata return.
- **Target / reference allowance / envelope:** requested error ceiling / permitted empirical refinement difference / cheapest qualifying tested development configuration.
- **Delta / e / epsilon / ms:** last refinement discrepancy relative to fine-reference norm / observed field error / requested ceiling / milliseconds.
