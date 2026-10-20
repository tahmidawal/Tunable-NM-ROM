**Overall: NEEDS-RESTATEMENT for interpretation; the saved numerical results and gate decisions are CORRECT.** I found no incorrect finalist decision or component classification. The qualification concerns “what limits next”: these are measured sensitivities and discrepancies, not causal contributions or guaranteed improvement opportunities.

No files were created, modified, or appended.

**1. Gates and independent CPU recomputation — CORRECT.**

I checked the design amendments, analyzer and archived driver logic, inventories, validity evidence, output checksums, frozen-selection binding, and certification hashes.

| Check | Audit result |
|---|---|
| B0 identity | Pass. Worst relative defects: first-order `1.3570e-11` / `1.3615e-11`; E0 `3.9507e-11` / `4.9657e-11`, for A / W1024r respectively. All below `1e-10`. |
| Analytic fixtures | Independently recover defects `−0.002` and `0.5`. |
| Relative normalizer-mismatch control | Correctly uses the A12 relative defect. All four pinned defects exceed `1e-6`; smallest is `0.678144`. |
| B1 sham | Archived evidence records bitwise equality on all 32 cases per setting; driver implements the required stopping gate. I did not rerun the GPU sham. |
| Inventory | Independently reconstructed: 5,952 rows in A and 1,184 in W1024r, with matching archive cases and no duplicates or omissions. Initial coefficients agree across ablations. |
| Original-reference reproduction | Correct coverage: 35,712 / 7,104 per-time comparisons, zero failures; worst differences `2.22e-16` / `1.67e-16`. |
| E0 certification | Correctly binds to the original-reference analyses, identical result hashes, and frozen version `8d106e19…`. Direct original-reference reproduction is appropriately “not applicable.” |
| Grid reproduction | Correct coverage: 768 / 128 arm-case comparisons. Archived field discrepancies are below `2.28e-15`. Independent coefficient comparisons also agree to roundoff. |
| B2 / B3 | Both discriminating in every setting: **32/32** cases exceed their respective thresholds. B2 correctly compares the control rule against the converged rule, rather than using the informational in-job comparison against the deployed rule. |
| Converged-rule check | Correctly passes W256/W512/W768/W1024 and fails M2_512 on validity, as discussed below. |

Independent CPU calculations used the saved coefficients, decoded bank fields, physical initial-field normalization, and reference projection. Representative **case 0, CN, step 0.01, K=1**, with evolved-time maxima and E0-dependent projection quantities:

| Setting | `delta_quad` | `delta_time` | `D_a` | `e_P` | `e_dyn` |
|---|---:|---:|---:|---:|---:|
| M2_512 | 6.438640e-6 | 8.597441e-4 | 8.832729e-4 | 5.773741e-3 | 3.408169e-3 |
| W1024 | 9.651367e-7 | 8.258847e-4 | 8.303141e-4 | 1.656778e-3 | 1.952342e-3 |

These agree with the saved arrays within `1.4e-14`; their independently evaluated B0 defects are below `4.2e-14`. Additional checks included case 11 and other settings. A W512 projection cross-check using QR agrees within `4.3e-16`.

The W1024 rerun is consistent with A16: the archived driver change releases the retained table reference. Both successful archives have GPU/f64/highest-precision logs and completion markers; no results from failed job 5231969 were substituted.

**2. M2_512 fails closed — CORRECT.**

Exactly one archived row is invalid:

`gl64 | ad|GAL|BDF2|0.00125|tighter | case 11`

It has **one failed verification among 200 steps**. The failing step’s residual/tolerance ratio is approximately **1.03687**, with exit reason 2. GAL requires residual verification; a finite trajectory or small inter-rule distance cannot override that failure.

The maximum converged-versus-finer distance is only `8.408988e-8`, comfortably below `2.5e-5`, but `all_valid=False` correctly makes the check fail. Both analyses explicitly withhold quadrature attribution.

This establishes **unverified convergence**, not demonstrated quadrature inadequacy. M2’s deployed sensitivities remain reportable as distances to `gl56`, with that qualification. Its `gl56` ideal-model uncertainty calculations are not themselves missing: the invalid replay is at the finer rule.

**3. Selected-arm finalist qualification — CORRECT.**

All **nine selected arm entries** pass their own 64-case checks, with valid deployed, converged, and finer trajectories. Below are the largest values across selected arms in each setting:

| Setting | Worst `delta_quad` | Worst converged–finer distance |
|---|---:|---:|
| M2_512 | 3.329994e-5 | 7.464557e-8 |
| W256 | 1.508524e-4 | 2.891014e-6 |
| W512 | 3.478413e-5 | 2.008511e-6 |
| W768 | 1.685685e-4 | 8.020493e-6 |
| W1024 | 1.466456e-4 | 1.836433e-5 |
| Required maximum | **2.5e-4** | **2.5e-5** |

M2’s finalist passes do **not** contradict its broader budget failure: A3.6 checks the selected arms’ own trajectories, whereas A2.4 additionally requires every converged-rule-dependent budget trajectory type.

The selections match the pinned frozen record. Under A15, **no fast-equal-accuracy selection exists**; these qualification passes do not restore that retracted claim or establish a new speedup.

**4. Largest-component classifications — CORRECT; “what limits next” — NEEDS-RESTATEMENT.**

I independently reconstructed every classification and summary from the per-case arrays. They correctly use only the first **32 budget cases**, excluding the extra finalist-only cases.

For the frozen accurate-equal-cost arms, the E0 classifications are:

| Setting and selected arm | Largest sensitivity/discrepancy counts |
|---|---|
| M2_512, BDF2 / 0.0125 / K1 | Time package 28; projection 3; ideal-ROM discrepancy 1 |
| W256, CN / 0.0125 / K1 | Projection 30; time package 2 |
| W512, BDF2 / 0.0125 / K1 | Time package 28; projection 2; ideal-ROM discrepancy 2 |
| W768, CN / 0.016666667 / K2 | Time package 20; ideal-ROM discrepancy 7; projection 5 |
| W1024, CN / 0.01 / K1 | Ideal-ROM discrepancy 16; time package 14; projection 2 |

The defensible interpretation is:

- Time-package sensitivity remains prominent for the selected M2/W512/W768 arms.
- Representation is the clearest measured limitation at W256.
- W1024 has a mixed picture, not a universal dynamics bottleneck.
- Quadrature is never the largest component; every reported `delta_quad` is below `3U_star`, hence unresolved under the registered indicator. M2 additionally lacks quadrature attribution.
- Larger-step CN diagnostics retain substantial solver-sensitive outliers.

Do not describe `delta_time` as pure time-truncation error, `e_dyn` as an isolated dynamics error, or the largest component as the intervention guaranteed to improve accuracy most. E0 also materially changes the classifications; the first-order “dynamics dominates” narrative should not be carried forward.

**5. Reference-free agreement between first-order and E0 — CORRECT.**

Exact equality holds across **64,512 per-time entries, including unavailable markers**, covering all reference-free sensitivities, mixed differences, ideal-end contrasts, `D_a`, `U_star`, and finalist-finer distances. Finalist outcomes are identical. Reference-dependent errors, B0 diagnostics, and largest-component classifications appropriately change.