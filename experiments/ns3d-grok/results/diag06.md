# diag06 coefficient-space tracker, development

Generated from `runs/diag06/output/summary.json`. Job 4147975, commit `b447d836137a39de55a746c4e547ed79035e7894`. Development seed 202609202 only. The sealed seed was not opened. A rewrite is kept when its relative field gap versus the grid tracker is at most 1e-6 and every case stays within 5%. Times and errors for each method come from the same timed calls.

| method | kept | parity worst | evolved worst | cases over 5% | median ms |
|---|---:|---:|---:|---:|---:|
| grid tracker | reference | 0.000e+00 | 4.563% | 0/16 | 15.238 |
| coeff_full | True | 5.824e-15 | 4.563% | 0/16 | 10.923 |
| f32 | False | 2.549e-06 | 4.563% | 0/16 | 9.210 |
| trunc_1em12 (9260 freq) | True | 5.996e-15 | 4.563% | 0/16 | 6.838 |
| trunc_1em08 (9259 freq) | True | 9.242e-09 | 4.563% | 0/16 | 6.615 |
| trunc_1em06 (9222 freq) selected | True | 6.919e-07 | 4.563% | 0/16 | 6.584 |
| trunc_1em04 (8942 freq) | False | 4.077e-05 | 4.564% | 0/16 | 6.495 |
| trunc_1em03 (8354 freq) | False | 2.866e-04 | 4.565% | 0/16 | 6.577 |

| FOM dt | evolved worst | median ms |
|---:|---:|---:|
| 0.004 | 0.094% | 6.766 |
| 0.005 | 0.152% | 5.507 |
| 0.01 chosen | 0.788% | 3.079 |
| 0.02 | 76.694% | 1.856 |

Development paired speedup (chosen CNAB2 ms / selected ms): 0.467606.

Piece medians for one trajectory length, separate from the accuracy calls:

| piece | median ms |
|---|---:|
| coeff_advection_twice_per_step | 0.579 |
| coeff_shift_reproject | 6.032 |
| grid_advection_twice_per_step | 8.262 |
| grid_centroid_shift_reproject | 3.349 |
| six_output_shifts | 0.622 |

Centered-snapshot floor of a low-wavenumber Fourier basis:

| rank | evolved worst | cases over 5% |
|---:|---:|---:|
| 64 | 69.881% | 16/16 |
| 128 | 50.441% | 16/16 |
| 256 | 30.386% | 16/16 |

