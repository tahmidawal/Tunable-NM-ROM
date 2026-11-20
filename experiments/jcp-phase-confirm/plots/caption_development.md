# Development: physical position and centroid-aligned shape

Measured diagnostics from saved accepted trajectories. Qualification and sources are retained in the adjacent JSON; the three-panel plot shows median histories and full case ranges.

Both field errors use the original physical initial-field norm. Phase is the Euclidean norm of wrapped energy-weighted physical-centroid differences, shown as percent domain length. Alignment uses reference information and is diagnostic only; it is not an online correction or a globally optimal translation. The head and span start from the same reconstructed field and have the same three frame unknowns.

| Diagnostic, maximum over evolved times | Head median | Span median | Head worst | Span worst | Median paired span/head ratio | Head lower cases |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary field error (%) | 0.096675023 | 0.66970766 | 0.18563258 | 1.2706914 | 7.05904 | 16/16 |
| Centroid-aligned field error (%) | 0.095293259 | 0.66973737 | 0.18596911 | 1.2706923 | 7.16006 | 16/16 |
| Physical displacement (% domain length) | 0.0022847756 | 0.0031078368 | 0.0042955196 | 0.0078156303 | 1.1817 | 12/16 |

Circular-moment concentration is pending; phase interpretation remains provisional until the GPU replay is qualified.

The aligned-error comparison retains its own measured advantage or disadvantage. It cannot be obtained by subtracting phase error from ordinary error, and cannot prove a unique deformation mechanism. No speed or uniform-stability conclusion follows from this plot.

## Glossary

A head is a nonlinear restriction on spatial coefficients; a span permits arbitrary coefficients. Common IC means identical initial reconstructed fields. Median is the middle case statistic; worst is the largest. A paired ratio divides span by head on the same case before summarizing. Circular concentration measures how well a periodic centroid is defined. The full case range is variability across physical inputs, not a statistical confidence interval.
