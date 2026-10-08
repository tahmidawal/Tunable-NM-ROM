No files modified; no GPU used.

| Item | Verdict | Finding |
|---|---|---|
| Bare `M =` | **FIXED** | All four setting descriptions now use LaTeX. |
| Bare `h` | **FIXED** | Now in math delimiters. |
| Unicode rho outside tables/code | **FIXED** | No remaining occurrences found. |
| Per-case mesh spreads | **FIXED** | All 20 added values match independent recomputation using matched case IDs. |
| Registered p90 summaries | **NOT FIXED** | All 24 added **3D continuum** p90 entries match raw-array quantiles over the 600 evolved states. **2D continuum and 2D/3D mesh-target p90 remain absent without disclosure.** |

Recomputed largest per-case spreads, in percentage points:

| Setting | Arms, in order | Recomputed spreads |
|---|---|---|
| 3D 512 | tensor, gl24, lat4096, lat32768, nodes | 4.113455, 0.046311, 0.046306, 0.046311, 0.046310 |
| 3D 256 | same | 3.749306, 0.042947, 0.042946, 0.042948, 0.042948 |
| 2D acc | dense, lat64, gauss96, gref, nodes | 2.822221, 2.821337, 0.008222, 0.008467, 0.008939 |
| 2D fast | dense, lat64, fib1597, gref, nodes | 1.891804, 1.891882, 0.017504, 0.017464, 0.017744 |

The missing 2D continuum p90 values are recoverable from `rho_per_state_*.npz`, using the same arm orders above:

| Mesh | acc | fast |
|---|---|---|
| 256² | 0.0562706, 0.0570703, 0.000249065, 0, 0.000710833 | 0.0559665, 0.0555028, 0.00240349, 0, 0.000556821 |
| 1024² | 0.014645, 0.0149325, 0.000244278, 0, 0.00000282921 | 0.0143531, 0.0141641, 0.00258249, 0, 0.00000106033 |

**New WRONG:** the added `p90` and `pp` terms lack the required plain-language glossary definitions: 90th percentile and percentage points. No numerical mismatch found in the added values.

Remaining WRONG: undisclosed missing 2D continuum and 2D/3D mesh-target p90 summaries; missing glossary definitions for p90 and pp.