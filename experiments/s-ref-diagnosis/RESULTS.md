# Why the frozen 2D Burgers quadratures fail against the space-only (S) reference

**Post-hoc diagnosis of an opened and closed sealed cohort.** Every number below is recomputed from the saved per-case scores of the eight sealed-closeout panel jobs (lane `exp/2026-11-15-jcp-sealed-closeout`, freeze F1). No model was re-run, no reference was regenerated, and no frozen decision changes. The script reproduces every frozen median and worst-case band verdict (`analyze.py` asserts equality with each `*-decision.json`). Numbers are final for this diagnosis. GPU use: none.

## Answer

**Measured.** All 2D S failures are failures of the *median* band. The worst-case band passes on every mesh. Against S, the dense 2D anchor (Gauss 256²) has a median error of only 0.0395–0.0420%, against 0.9511–0.9560% for ST, a per-case S/ST ratio of 0.051–0.052. The registered 5% margin is relative to that anchor, so on S it is 0.0020–0.0021 percentage points (pp) wide, compared with 0.0476–0.0478 pp on ST. Fibonacci 987's median exceeds the anchor by 0.0039–0.0073 pp on S, which is 2–4 times the S margin but far inside the ST margin. The rule term is not larger on S. Fibonacci 987's median shift on ST is 0.0124–0.0149 pp, and the per-case median |rule − dense| is 0.0030–0.0034 pp on S against 0.0031–0.0032 pp on ST. What changes between S and ST is the margin. Counterfactual (a) judges each S difference against the ST absolute margin, and it flips every S failure to a pass. Setting the reference uncertainty to zero (b) flips none, because the S uncertainty (0.000006–0.000012 pp at the anchor median) is under 1% of the band. Rescoring on the cases eligible for both references (c), or on all 128 cases (d), rescues none of the fib987 failures. The same rescoring turns Gauss 45²'s two S passes (2048², 4096²) into failures. The 21 S-ineligible cases are all `unresolved-floor`, and they have *smaller* S errors (anchor median 0.0120% against 0.0395%). Eligibility filtering therefore does not cause the failures. **Derived.** The S reference uses backward Euler at Δt = 0.005, the same scheme and step as the 2D reduced model. The S error therefore leaves out the reduced model's temporal truncation, which dominates the ST error. The anchor's per-case ST − S gap is 0.9156–0.9176 pp (median). The same-mesh central/Crank–Nicolson full-order solve at Δt = 0.005 differs from R_S by 0.9250–0.9274% (median), which is the size of the backward-Euler time error. The 2D S test thus measures the quadrature perturbation against a tolerance about 20 times tighter than the ST test. The 3D model steps at Δt = 0.01 while R_S uses 0.005, so its S anchor error stays large (S/ST 0.488–0.495) and the S margin stays wide. The 3D rules also sit closer to their dense anchor (median |Δ| ≤ 0.0002 pp; per-case median |rule − dense| ≤ 0.0007 pp). **Hypothesis (not supported as the cause).** H2 proposed that the S reference's time error moves the anchor. The data show the opposite: S *removes* the reduced model's time error. H5 proposed that mixing continuum advection with mesh-dependent discrete diffusion makes S measure a different target. That mix may explain the nonzero S floor, but it does not explain the failure: the 2D S anchor median does not fall as the scoring mesh approaches the reference mesh (0.0395% at 256², 0.0420% at 4096²). No controlled run was needed, because the existing same-mesh full-order arms already measure the reference time-step difference.

## Suggested sentence for the paper

> Against the secondary space-only reference, which shares the reduced model's backward-Euler step (Δt = 0.005) in 2D, the dense-rule median error falls to about 0.04% (versus 0.95% against the space–time reference), so the registered 5% relative margin shrinks to about 0.002 percentage points. The frozen 2D rules exceed this tighter tolerance in median error (Fibonacci 987 on 5/5 meshes, Gauss 45² on 3/5), although their worst-case errors pass. The 3D model steps at Δt = 0.01, so its space-only reference retains a time-step mismatch, and both 3D rules pass. We therefore report sufficiency only against the space–time reference, and we read the 2D space-only failure as unresolved quadrature error at the 0.005-percentage-point scale, which the space–time error otherwise masks.

## Configuration facts behind the diagnosis

| Item | 2D | 3D | Source |
|---|---|---|---|
| Reduced-model time stepping | backward Euler, Δt = 0.005 | backward Euler, Δt = 0.01 | saved `setup.dt`; plateau `DESIGN.md` / `offmesh.py` |
| R_S (secondary) | central + backward Euler, Δt = 0.005, space-refined | same, Δt = 0.005 | E0 `report.md` |
| R_ST (primary) | central + Crank–Nicolson, Δt ∝ h, space and time refined | same | E0 `report.md` |
| Dense anchor | gauss256 | lat32768 | config `conv_arm` |
| Pass rule | median and worst inside ±5% of anchor after eps bands, whole suffix | same | `frozen_decisions.band_interval` |

## Table 1: Per-mesh decomposition of the median (T) band, target rules (2D)

A = anchor statistic and X = subject statistic, both in %. Δ = X − A and the margin = 0.05·A, both in percentage points (pp). eps = half-width of the anchor's uncertainty band (pp). Columns (a)–(d) are counterfactual verdicts.

| Mesh | Rule | Ref | A (%) | X (%) | Δ (pp) | Δ/A | Margin (pp) | eps (pp) | Frozen T | (a) within ST margin | (b) eps=0 | (c) common set | (d) all cases |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 256² | fib987 | S | 0.0395 | 0.0449 | +0.00532 | +13.46% | 0.00198 | 1.2e-05 | FAIL | True | FAIL | FAIL | FAIL |
| 256² | fib987 | ST | 0.9560 | 0.9683 | +0.01236 | +1.29% | 0.04780 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 256² | gauss45 | S | 0.0395 | 0.0433 | +0.00379 | +9.58% | 0.00198 | 1.2e-05 | FAIL | True | FAIL | FAIL | FAIL |
| 256² | gauss45 | ST | 0.9560 | 0.9550 | -0.00098 | -0.10% | 0.04780 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 512² | fib987 | S | 0.0402 | 0.0441 | +0.00391 | +9.73% | 0.00201 | 1.2e-05 | FAIL | True | FAIL | FAIL | FAIL |
| 512² | fib987 | ST | 0.9522 | 0.9666 | +0.01441 | +1.51% | 0.04761 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 512² | gauss45 | S | 0.0402 | 0.0428 | +0.00260 | +6.47% | 0.00201 | 1.2e-05 | FAIL | True | FAIL | FAIL | FAIL |
| 512² | gauss45 | ST | 0.9522 | 0.9517 | -0.00049 | -0.05% | 0.04761 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 1024² | fib987 | S | 0.0400 | 0.0473 | +0.00729 | +18.23% | 0.00200 | 1.2e-05 | FAIL | True | FAIL | FAIL | FAIL |
| 1024² | fib987 | ST | 0.9513 | 0.9663 | +0.01493 | +1.57% | 0.04757 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 1024² | gauss45 | S | 0.0400 | 0.0427 | +0.00269 | +6.73% | 0.00200 | 1.2e-05 | FAIL | True | FAIL | FAIL | FAIL |
| 1024² | gauss45 | ST | 0.9513 | 0.9509 | -0.00042 | -0.04% | 0.04757 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 2048² | fib987 | S | 0.0420 | 0.0474 | +0.00547 | +13.02% | 0.00210 | 5.8e-06 | FAIL | True | FAIL | FAIL | FAIL |
| 2048² | fib987 | ST | 0.9511 | 0.9660 | +0.01492 | +1.57% | 0.04756 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 2048² | gauss45 | S | 0.0420 | 0.0431 | +0.00114 | +2.71% | 0.00210 | 5.8e-06 | PASS | True | PASS | FAIL | FAIL |
| 2048² | gauss45 | ST | 0.9511 | 0.9508 | -0.00037 | -0.04% | 0.04756 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 4096² | fib987 | S | 0.0420 | 0.0474 | +0.00546 | +13.01% | 0.00210 | 5.8e-06 | FAIL | True | FAIL | FAIL | FAIL |
| 4096² | fib987 | ST | 0.9511 | 0.9660 | +0.01492 | +1.57% | 0.04756 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |
| 4096² | gauss45 | S | 0.0420 | 0.0431 | +0.00114 | +2.72% | 0.00210 | 5.8e-06 | PASS | True | PASS | FAIL | FAIL |
| 4096² | gauss45 | ST | 0.9511 | 0.9507 | -0.00037 | -0.04% | 0.04756 | 2.8e-04 | PASS | n/a | PASS | PASS | PASS |

Columns (b)–(d) give the whole-rung decision, combining T and W. Column (a) applies to the S median band only.

## Table 2: Every non-passing band in the 2D and 3D suffixes

| Mesh | Target | Suffix rung | Ref | Band | Δ (pp) | Margin (pp) | Verdict | Within ST margin |
|---|---|---|---|---|---:|---:|---|---|
| 256² | fib987 | fib987 | S | T | +0.00532 | 0.00198 | FAIL | True |
| 256² | gauss45 | gauss45 | S | T | +0.00379 | 0.00198 | FAIL | True |
| 256² | gauss45 | gauss50 | S | T | +0.00280 | 0.00198 | FAIL | True |
| 512² | fib987 | fib987 | S | T | +0.00391 | 0.00201 | FAIL | True |
| 512² | gauss45 | gauss45 | S | T | +0.00260 | 0.00201 | FAIL | True |
| 1024² | fib987 | fib987 | S | T | +0.00729 | 0.00200 | FAIL | True |
| 1024² | gauss45 | gauss45 | S | T | +0.00269 | 0.00200 | FAIL | True |
| 2048² | fib987 | fib987 | S | T | +0.00547 | 0.00210 | FAIL | True |
| 4096² | fib987 | fib987 | S | T | +0.00546 | 0.00210 | FAIL | True |

No ST band and no 3D band fails. Every failing S band is a median band, and each lies inside the corresponding ST absolute margin.

## Table 3: Reference term: S versus ST on the dense anchor and on same-mesh full-order solves

Medians over the cases eligible for both references. CC is central + Crank–Nicolson. UB is upwind + backward Euler. hist is the historical 3D full-order solve.

| Mesh | Common cases | Anchor e_S (%) | Anchor e_ST (%) | Anchor per-case ST − S (pp, median [min, max]) | Per-case S/ST | Same-mesh full-order medians, S / ST (%) |
|---|---:|---:|---:|---|---:|---|
| 256² | 103 | 0.0395 | 0.9560 | 0.916 [0.208, 2.392] | 0.052 | UB_L256 1.101 / 1.645; CC_L256 0.927 / 0.077 |
| 512² | 103 | 0.0402 | 0.9522 | 0.918 [0.196, 2.394] | 0.051 | UB_L512 0.559 / 1.298; CC_L512 0.926 / 0.077 |
| 1024² | 103 | 0.0400 | 0.9513 | 0.918 [0.191, 2.393] | 0.051 | UB_L1024 0.283 / 1.092; CC_L1024 0.925 / 0.076 |
| 2048² | 103 | 0.0420 | 0.9511 | 0.918 [0.182, 2.393] | 0.051 | UB_L2048 0.142 / 1.044; CC_L2048 0.925 / 0.076 |
| 4096² | 103 | 0.0420 | 0.9511 | 0.918 [0.182, 2.393] | 0.051 | UB_L4096 0.071 / 1.017; CC_L4096 0.925 / 0.076 |
| 65³ | 64 | 0.7955 | 1.6067 | 0.818 [0.218, 1.907] | 0.495 | hist_dt0.005 2.239 / 2.257; hist_dt0.01 2.401 / 2.653; UB_L64 2.489 / 2.649; CC_L64 0.600 / 0.351 |
| 129³ | 64 | 0.7935 | 1.6041 | 0.818 [0.221, 1.923] | 0.489 | hist_dt0.005 1.101 / 1.177; hist_dt0.01 1.347 / 1.869; UB_L128 1.357 / 1.888; CC_L128 0.645 / 0.296 |
| 257³ | 64 | 0.7932 | 1.6050 | 0.818 [0.221, 1.927] | 0.488 | hist_dt0.005 0.566 / 0.896; hist_dt0.01 0.924 / 1.687; UB_L256 0.940 / 1.795; CC_L256 0.658 / 0.285 |

The anchor error is a median over the cases eligible for that reference. The ST − S gap and the ratio use common cases. In 2D, the same-mesh CN solve's S error is about the size of the anchor's ST − S gap. Both measure the backward-Euler time error at Δt = 0.005, which ST sees and S does not.

## Table 4: Rule term against both references (size of the quadrature perturbation)

| Mesh | Rule | Median per-case abs(rule − dense), S (pp) | same, ST (pp) | Max per-case, S (pp) | Max per-case, ST (pp) |
|---|---|---:|---:|---:|---:|
| 256² | fib987 | 0.00338 | 0.00324 | 0.8949 | 0.2344 |
| 256² | gauss45 | 0.00204 | 0.00132 | 0.2116 | 0.1358 |
| 512² | fib987 | 0.00325 | 0.00317 | 0.8825 | 0.2353 |
| 512² | gauss45 | 0.00182 | 0.00130 | 0.1097 | 0.1061 |
| 1024² | fib987 | 0.00319 | 0.00310 | 0.8827 | 0.2345 |
| 1024² | gauss45 | 0.00168 | 0.00130 | 0.1085 | 0.1055 |
| 2048² | fib987 | 0.00304 | 0.00307 | 0.8827 | 0.2342 |
| 2048² | gauss45 | 0.00171 | 0.00131 | 0.1083 | 0.1054 |
| 4096² | fib987 | 0.00304 | 0.00307 | 0.8827 | 0.2342 |
| 4096² | gauss45 | 0.00171 | 0.00131 | 0.1082 | 0.1054 |
| 65³ | lat4096 | 0.00018 | 0.00018 | 0.0466 | 0.0683 |
| 65³ | gl20 | 0.00068 | 0.00053 | 0.1290 | 0.1825 |
| 129³ | lat4096 | 0.00018 | 0.00018 | 0.0406 | 0.0687 |
| 129³ | gl20 | 0.00073 | 0.00053 | 0.1266 | 0.1762 |
| 257³ | lat4096 | 0.00018 | 0.00017 | 0.0413 | 0.0686 |
| 257³ | gl20 | 0.00071 | 0.00051 | 0.1268 | 0.1752 |

## Table 5: Uncertainty term and the margin each rule would need

The needed margin is approximately (|Δ| + subject eps half-width + anchor eps half-width)/A, maximised over the suffix and over both bands. The registered margin is 5%.

| Mesh | Ref | eps median (pp) | eps max (pp) | Needed margin, rule 1 | Needed margin, rule 2 |
|---|---|---:|---:|---|---|
| 256² | S | 1.2e-05 | 1.2e-04 | fib987 13.51% | gauss45 9.62% |
| 256² | ST | 2.9e-04 | 1.3e-03 | fib987 1.52% | gauss45 0.91% |
| 512² | S | 1.2e-05 | 1.2e-04 | fib987 9.79% | gauss45 6.54% |
| 512² | ST | 2.9e-04 | 1.3e-03 | fib987 1.66% | gauss45 1.19% |
| 1024² | S | 1.2e-05 | 1.2e-04 | fib987 18.27% | gauss45 6.77% |
| 1024² | ST | 2.9e-04 | 1.3e-03 | fib987 1.68% | gauss45 1.20% |
| 2048² | S | 1.2e-05 | 1.2e-04 | fib987 13.07% | gauss45 2.74% |
| 2048² | ST | 2.9e-04 | 1.3e-03 | fib987 1.68% | gauss45 1.20% |
| 4096² | S | 1.2e-05 | 1.2e-04 | fib987 13.05% | gauss45 2.74% |
| 4096² | ST | 2.9e-04 | 1.3e-03 | fib987 1.68% | gauss45 1.20% |
| 65³ | S | 1.3e-03 | 9.9e-03 | lat4096 1.64% | gl20 1.33% |
| 65³ | ST | 2.4e-03 | 1.4e-02 | lat4096 1.45% | gl20 3.90% |
| 129³ | S | 1.3e-03 | 9.9e-03 | lat4096 1.44% | gl20 1.26% |
| 129³ | ST | 2.4e-03 | 1.4e-02 | lat4096 1.43% | gl20 3.77% |
| 257³ | S | 1.3e-03 | 9.9e-03 | lat4096 1.48% | gl20 1.29% |
| 257³ | ST | 2.4e-03 | 1.4e-02 | lat4096 1.42% | gl20 3.74% |

## Table 6: Eligibility (why 107 versus 123 in 2D)

| Mesh | S status counts | ST status counts | Ineligible for both | S-eligible anchor e_S median (%) | S-ineligible anchor e_S median (%) | ST-ineligible anchor e_ST median (%) |
|---|---|---|---:|---:|---:|---:|
| 256² | accepted 107, unresolved-floor 21 | accepted 123, order-fail 5 | 1 | 0.0395 | 0.0120 | 1.1024 |
| 512² | accepted 107, unresolved-floor 21 | accepted 123, order-fail 5 | 1 | 0.0402 | 0.0109 | 1.1071 |
| 1024² | accepted 107, unresolved-floor 21 | accepted 123, order-fail 5 | 1 | 0.0400 | 0.0104 | 1.1072 |
| 2048² | accepted 107, unresolved-floor 21 | accepted 123, order-fail 5 | 1 | 0.0420 | 0.0104 | 1.1082 |
| 4096² | accepted 107, unresolved-floor 21 | accepted 123, order-fail 5 | 1 | 0.0420 | 0.0104 | 1.1082 |
| 65³ | accepted 64 | accepted 64 | 0 | 0.7955 | — | — |
| 129³ | accepted 64 | accepted 64 | 0 | 0.7935 | — | — |
| 257³ | accepted 64 | accepted 64 | 0 | 0.7932 | — | — |

The 2D S exclusions are `unresolved-floor`: the E0 rule requires both level differences to exceed ten times the solver-residual floor proxy. The E0 report attributes these exclusions to high-viscosity cases, where backward Euler at Δt₀ on L = 8192 cannot push the step residual below about 1e-11. The ST exclusions are `order-fail`. Eligibility comes from the original frozen E0 status. It was not chosen here, and Table 1 columns (c) and (d) show that it does not cause the fib987 failures.

## Attribution summary

| Dimension | Rule | Frozen S by mesh | Term that flips each S failure |
|---|---|---|---|
| 2D | fib987 | 256² FAIL, 512² FAIL, 1024² FAIL, 2048² FAIL, 4096² FAIL | margin (5% of a small, time-error-free anchor); not uncertainty, not eligibility |
| 2D | gauss45 | 256² FAIL, 512² FAIL, 1024² FAIL, 2048² PASS, 4096² PASS | margin (5% of a small, time-error-free anchor); not uncertainty, not eligibility |
| 3D | lat4096 | 65³ PASS, 129³ PASS, 257³ PASS | none (passes) |
| 3D | gl20 | 65³ PASS, 129³ PASS, 257³ PASS | none (passes) |

## Files

- `PROTOCOL.md`: written and committed before the per-case pull.
- `reduce_cluster.py`: reduces saved `result.json` files on the cluster to per-case scalars. The output is not committed; its SHA256 is in `INPUTS.sha256`.
- `analyze.py`: reproduces the frozen verdicts, then decomposes them and runs the counterfactuals; writes `decomposition.json`.
- `make_results.py`: generates this file from `decomposition.json`.

## Glossary

- **S / ST reference.** S is the space-only reference: central differences refined in space, with backward Euler at the reduced model's original step Δt₀ = 0.005. ST is the space-and-time reference: central differences with Crank–Nicolson, refining space and time together (primary).
- **Anchor / dense rule.** The quadrature rule with the most points in the panel (Gauss 256² in 2D, CBC lattice 32768 in 3D). Every rule is judged relative to its error.
- **T / W band.** T is the median and W the worst case error over eligible cases. A rung passes when both lie within ±5% of the anchor's after widening by reference uncertainty.
- **Suffix.** The selected rule and every larger rung of its family. All must pass.
- **eps (eps_ref).** Per-case Richardson estimate of the reference's numerical error, from E0.
- **Eligible.** Cases whose reference passed the E0 checks for that reference kind (`accepted`). `unresolved-floor`: level differences too small relative to the solver residual floor. `order-fail`: observed convergence order outside [1.8, 2.2].
- **pp.** Percentage points, the absolute difference between two errors given in %.
- **Counterfactual (a)–(d).** Recomputations that change one term: (a) judge the S difference against the ST absolute margin; (b) set eps to 0; (c) restrict to cases eligible for both S and ST; (d) use all 128 cases with eps 0.
- **CC, UB, hist, PS.** Full-order solvers: central + Crank–Nicolson, upwind + backward Euler, the historical 3D solver, and pseudo-spectral.
- **Measured / Derived / Hypothesis.** Measured: read from saved scores. Derived: follows from measured numbers together with documented configuration. Hypothesis: proposed and not established by this analysis.
