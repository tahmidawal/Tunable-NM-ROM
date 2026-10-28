# Retained boundary diagnostics

Diagnostics only; no added thresholds or reference admission.

| Case | Max radial deviation | Max normal angle (radians) | Outlet backflow / UH | Raw trace estimate / U | Trace bound / U |
|---|---:|---:|---:|---:|---:|
| P3g3 | 8.97701802e-10 | 4.12004953e-06 | 0 | 0 | 1e-12 |
| P3g4 | 5.85990978e-12 | 4.47522763e-08 | 0 | 0 | 1e-12 |
| P4g4 | 5.85990978e-12 | 4.47522763e-08 | 0 | 0 | 1e-12 |

## Glossary

Radial deviation: distance of a retained obstacle quadrature point from the exact circle. Normal angle: difference between the recorded fluid-outward normal and the inward radial direction. These are sampled diagnostics, not continuous-boundary maxima. Backflow: inward outlet flux normalized by inlet mean speed times channel height. Raw trace estimate: polynomial estimate of error against prescribed FE boundary velocity. Trace bound: conservative coefficient bound including the fixed roundoff allowance.
