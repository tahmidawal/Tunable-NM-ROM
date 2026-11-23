# Fresh cohort (initial-fit qualification incomplete): physical position and centroid-aligned shape

Measured diagnostics from saved trajectories. Qualification and sources are retained in the adjacent JSON; the three-panel plot shows median histories and full case ranges.

Both field errors use the original physical initial-field norm. Phase is the Euclidean norm of wrapped energy-weighted physical-centroid differences, shown as percent domain length. Alignment uses reference information and is diagnostic only; it is not an online correction or a globally optimal translation. The head and span start from the same reconstructed field and have the same three frame unknowns.

| Diagnostic, maximum over evolved times | Head median | Span median | Head worst | Span worst | Median paired span/head ratio | Head lower cases |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary field error (%) | 0.083754933 | 0.69005114 | 0.24730869 | 1.703199 | 7.24456 | 16/16 |
| Centroid-aligned field error (%) | 0.083399323 | 0.68996635 | 0.24768048 | 1.7033593 | 7.33031 | 16/16 |
| Physical displacement (% domain length) | 0.0021485191 | 0.0035753534 | 0.0058405825 | 0.0084907663 | 1.5394 | 12/16 |

**Provisional confirmation:** the frozen initial-fit stationarity check fails for cases [8, 10]. All trajectories remain shown. This cohort is not accepted as fully numerically qualified; no gate was relaxed or case removed. See IC_QUALIFICATION_DIAGNOSIS.json for exact gradients and candidate costs.

The minimum circular-moment concentration is 0.66278416; 0 coordinate/time entries fall below the prospectively declared descriptive flag. This assesses centroid definition, not uniqueness of the underlying physical structure.

The aligned-error comparison retains its own measured advantage or disadvantage. It cannot be obtained by subtracting phase error from ordinary error, and cannot prove a unique deformation mechanism. No speed or uniform-stability conclusion follows from this plot.

## Glossary

A head is a nonlinear restriction on spatial coefficients; a span permits arbitrary coefficients. Common IC means identical initial reconstructed fields. Median is the middle case statistic; worst is the largest. A paired ratio divides span by head on the same case before summarizing. Circular concentration measures how well a periodic centroid is defined. The full case range is variability across physical inputs, not a statistical confidence interval.
