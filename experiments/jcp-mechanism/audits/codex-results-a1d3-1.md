**No numerical errors found in report sections 1, 3, or 5.** Independently recomputed with NumPy; did not execute `make_report.py`, modify files, or use a GPU. All 160 checked numerical table cells match at printed precision.

**(a) CORRECT — per-case recovery statistics and amendment-2 labels.**

For all 64 cases in each configuration, computed \(f_j=1-d_j/s_j\) from the JSON distance arrays. Bootstrap: 2,000 case resamples, NumPy `default_rng(0)`, 2.5th percentile of resampled medians.

Distances below are fractions, not percentages.

| Mesh | R′ | Median \(s_j\) | Median \(d_j\) | Maximum \(d_j\) | Median \(f_j\) | Bootstrap bound | Minimum \(f_j\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64³ | 256 | 0.03527344256 | 1.156434e-7 | 1.156534e-5 | 0.999996055 | 0.999994836 | 0.999879519 |
| 64³ | 512 | 0.03527038010 | 1.362498e-7 | 5.490417e-5 | 0.999995265 | 0.999993829 | 0.999440618 |
| 128³ | 256 | 0.01847940127 | 1.658694e-8 | 4.989645e-6 | 0.999999191 | 0.999998877 | 0.999906700 |
| 128³ | 512 | 0.01848304918 | 2.627635e-8 | 5.066869e-5 | 0.999998593 | 0.999997871 | 0.999084799 |

Every configuration has:

- Complete cases 0–63; no exclusions below \(s_j=10^{-3}\).
- Fraction \(f_j\ge0.75\): **64/64 = 1.000**.
- No invalidity trigger; median separation exceeds 0.01.
- All three recovery criteria pass, giving **R**.

I also reconstructed all these per-case distances from saved coefficients using the cached evaluation-bank Gram matrix and reference initial norms. Maximum disagreement with JSON: **\(2.9143\times10^{-16}\)**. Saved coefficients exactly match every fifth internal state.

**(b) CORRECT — continuum rho tables.**

Recomputed maximum and median from `rho_R*.npz`, selecting only `k>=1`: **600 states per configuration**. Entries are **worst / median**.

| Rule | 64³, R′256 | 64³, R′512 | 128³, R′256 | 128³, R′512 |
|---|---|---|---|---|
| dense upwind | 0.193616 / 0.128691 | 0.194517 / 0.128646 | 0.0972043 / 0.0658482 | 0.0977076 / 0.0658539 |
| tensor | 0.193616 / 0.128691 | 0.194517 / 0.128646 | 0.0972043 / 0.0658482 | 0.0977076 / 0.0658539 |
| gl24 | 2.11232e-3 / 3.33820e-4 | 1.00035e-2 / 1.19276e-3 | 2.45535e-3 / 3.53330e-4 | 1.26609e-2 / 1.34012e-3 |
| lat4096 | 2.37286e-2 / 1.27467e-3 | 8.30368e-2 / 7.82865e-3 | 2.87051e-2 / 1.34047e-3 | 1.04977e-1 / 8.88435e-3 |
| lat32768 | 2.90968e-5 / 7.07541e-7 | 8.61779e-5 / 1.98649e-6 | 3.64088e-5 / 7.10203e-7 | 1.40779e-4 / 2.02638e-6 |
| nodes | 8.82290e-5 / 1.98181e-6 | 1.78570e-4 / 2.69946e-6 | 6.97509e-6 / 1.21109e-7 | 1.55263e-5 / 1.63554e-7 |

All match the report. Mesh-target maxima also match, including the nonzero tensor/sign-upwind discrepancies.

**(c) CORRECT — refined errors, explicitly PROVISIONAL.**

Recomputed each case’s maximum from `err_refined[1:]`, then maximum/median across 64 cases. Entries are percentages: **worst / median**.

| Arm | 64³, R′256 | 64³, R′512 | 128³, R′256 | 128³, R′512 |
|---|---|---|---|---|
| tensor | 10.438115 / 3.401792 | 10.218572 / 3.384215 | 6.688809 / 1.989331 | 6.105117 / 1.977283 |
| gl24 | 4.702516 / 1.251184 | 3.164073 / 1.117437 | 4.691529 / 1.250249 | 3.119781 / 1.124292 |
| lat4096 | 4.712119 / 1.251349 | 3.023436 / 1.118475 | 4.701028 / 1.250034 | 3.008526 / 1.125038 |
| lat32768 | 4.702569 / 1.251303 | 3.165024 / 1.117490 | 4.691557 / 1.250172 | 3.120824 / 1.124214 |
| nodes | 4.702677 / 1.251301 | 3.165646 / 1.117490 | 4.691621 / 1.250172 | 3.121470 / 1.124215 |

All match. The mesh-ratio table also matches: tensor **1.560534 / 1.673772** for R′256/512; nodes **1.002356 / 1.014152**.

**(d) CORRECT — recorded gates and execution log.**

Each recorded gate was checked against its bar. Width-dependent entries below are **R′256 / R′512**.

| Gate | Bar | 64³ | 128³ |
|---|---:|---:|---:|
| Gram condition | ≤1e8 | 1.000000 | 1.002278 |
| tensor/direct | <1e-10 | 2.688e-16 | 2.399e-16 |
| G1 derivative | <1e-6 | 2.104e-8 | 2.279e-8 |
| G2a bank | ≤1e-12 | 0 / 0 | 0 / 0 |
| G2b tests | ≤1e-12 | 6.359e-15 / 6.987e-15 | 6.828e-15 / 7.085e-15 |
| G2c value | ≤1e-11 | 1.306e-15 / 1.949e-15 | 1.034e-15 / 1.797e-15 |
| G2c Jacobian | ≤1e-11 | 3.667e-15 / 1.280e-14 | 1.588e-15 / 1.936e-15 |
| G3 Jacobian | ≤1e-12 | 0 / 0 | 0 / 0 |
| G3 half-Jc | ≤1e-12 | 2.284e-16 / 2.628e-16 | 2.488e-16 / 2.169e-16 |
| G4 rho reproduction¹ | 1e-6 | 2.052e-12 / 5.130e-13 | 1.954e-13 / 2.314e-13 |

¹ Report-only under amendment 1; these values also satisfy the original tolerance.

Initial-reference agreement: **2.317e-16 / 0**, below **1e-12**. The executed script matches the current script and enforces finite gate inputs and reference amplitude ≥1e-8; those amplitudes are not separately persisted for independent recalculation.

The log contains `jax_backend=gpu`, `x64=True`, `precision=highest` for both meshes, two completion markers, and `ALL-DONE`. No NaN, OOM, red-zone, captured-large-constant, warning, or traceback messages found.

**(e) CORRECT — checks, validity and solver qualifications, with one evidence limitation.**

Continuum checks recomputed directly from saved target/check vectors:

| Mesh, R′ | Certification check | Nodes-reached check² | Bar |
|---|---:|---:|---:|
| 64³, 256 | 1.539292e-8 | 2.053217e-8 | ≤1e-6 |
| 64³, 512 | 3.622056e-8 | 1.716386e-7 | ≤1e-6 |
| 128³, 256 | 2.039856e-8 | 2.047332e-8 | ≤1e-6 |
| 128³, 512 | 8.012554e-8 | 1.684119e-7 | ≤1e-6 |

² Nodes-reached values match JSON and log; their per-state target arrays were not saved, so these checks cannot receive the same independent raw-array verification.

All four `continuum_target_valid` flags are correctly true. `nonfinite_targets` is absent; every saved target, check, rho, coefficient and internal-state array is finite. All validation finite flags are true; reason-3 counts are zero.

At 64³, all three required sensitivity arms have eight complete, finite cases and zero reason-3 exits:

| R′ | tensor maximum distance | lat32768 | nodes | Invalidation bar for converged/nodes |
|---|---:|---:|---:|---:|
| 256 | 8.946e-16 | 9.683e-16 | 1.462e-15 | 0.003527344 |
| 512 | 1.533e-15 | 2.104e-15 | 1.337e-15 | 0.003527038 |

Non-stationary counts at both meshes: R′256 **0/1600** throughout; R′512 tensor **2/1600 = 0.125%**, converged/nodes **1/1600 = 0.0625%**. None exceeds 1%.

Thus **64³ correctly lacks a provisional-solver mark**, and **both 128³ labels correctly carry it because sensitivity was not rerun there**. No failure-triggered invalidation should fire on these data. These passing outputs do not demonstrate fault-injection coverage.

**(f) CORRECT — scoped numerical claims; minor NEEDS-RESTATEMENT outside the requested sections.**

Section 5’s historical field distances independently reproduce from old/new coefficients, the cached bank and initial norms:

| Mesh, R′ | tensor | gl24 | lat4096 | lat32768 |
|---|---:|---:|---:|---:|
| 64³, 256 | 1.722e-15 | 1.425e-15 | 1.690e-15 | 2.167e-15 |
| 64³, 512 | 1.260e-15 | 1.260e-15 | 1.113e-15 | 1.482e-15 |
| 128³, 256 | 1.605e-15 | 1.284e-15 | 1.244e-15 | 1.571e-15 |
| 128³, 512 | 1.786e-15 | 1.220e-15 | 1.284e-15 | 1.237e-15 |

Each comparison includes 64 cases; all printed values match.

Sections 1/3/5 contain no unsupported causal conclusion. The supported statement is that **nodes recovers the designated converged rollout within the declared margins**. This does not isolate point placement as a causal variable. Refined-error tables and their mesh-ratio comparison are correctly labelled PROVISIONAL.

Minor restatement: the introduction cites “amendments 1–3”; the applicable specification now includes **1–6**. Also, printed recovery values of `1.000` are rounding, not exact equality.

**Remaining WRONG: none**