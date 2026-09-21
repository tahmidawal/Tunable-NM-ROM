# Kim et al. masked-autoencoder NM-LSPG versus the project NM-ROM on the shared Burgers 2D family

Reproduction gate: **PASSED**. **Adapted** reproduction: passed on attempt 3 of the pre-registered three (activation sigmoid), not on the paper-default attempt 1; median 1.45 % against the 1.5 % bar (published < 1 %), i.e. above the published figure. Hyper-reduction was NOT reproduced (HR gate failed on every attempt), so every HR arm is exploratory. Kim-baseline rows are admissible only where marked (same code hashes and the gate activation).
All numbers are generated from run JSONs by `reports/gen_report.py`; state: gate only.

## 1. Reproduction gate (Kim et al. 2022, Section 6.2; published NM-LSPG < 1 %, NM-LSPG-HR 0.93–0.98 %, LS-LSPG-HR 34–38 %)

Pass rule (pre-registered): median of three seeds ≤ 1.5 % and a valid LS-LSPG control ≥ 10 %; HR: median at 55/58 ≤ 2 %.

| job | attempt | activation / scaling | NM-LSPG per seed | median | autoencode-only per seed | LS-LSPG control | HR 55/58 per seed (basis) | gate | HR gate |
|---|---|---|---|---|---|---|---|---|---|
| 4059370 | 1 | swish / feature | 1.67 %, 1.45 %, 1.73 % | 1.67 % | 1.04 %, 0.85 %, 1.21 % | 31.60 % | 131.69 %, 318.89 %, 154.03 % (residual) | FAIL | FAIL |
| 4072224 | 2 | swish / global | 709.38 %, 11.22 %, 10.89 % | 11.22 % | 5.60 %, 3.81 %, 3.86 % | 31.60 % | 9006.73 %, 13143.74 %, 2050.67 % (sns) | FAIL | FAIL |
| 4072224 | 1 (weights reloaded) | swish / feature | 1.67 %, 1.45 %, 1.73 % | 1.67 % | 1.04 %, 0.85 %, 1.21 % | 31.60 % | 386.56 %, 13005.47 %, 18777.20 % (sns) | FAIL | FAIL |
| 4077574 | 3 | sigmoid / feature | 1.45 %, 1.83 %, 1.44 % | 1.45 % | 1.58 %, 1.79 %, 1.49 % | 31.60 % | diverged, diverged, diverged (sns) | pass | FAIL |

## Glossary

- **NM-LSPG**: nonlinear-manifold least-squares Petrov–Galerkin; each backward-Euler step minimises the full discrete residual over the latent vector. **HR**: hyper-reduction (residual evaluated at sampled rows only).
- **worst evolved**: largest, over cases and the five output times after t = 0, of the field error against the same-grid full-order solve, divided by the norm of the initial field.
- **autoencode-only**: encode then decode the true states; not a lower bound on the manifold error.
- **solved unknowns**: number of unknowns in the online nonlinear solve (K, or K+q for the project ROM with q corrections).
- **tune**: 16 cases carved from the training split, used for every choice; **validation**: 32 held-out cases, never used to choose.
- **GN cap hits**: time steps whose Gauss–Newton solve hit the 20-iteration cap. **compiled-query memory**: XLA memory analysis (arguments + outputs + temporaries) of the jitted query.
- **POD-LSPG zero / ic**: linear basis with zero reference or with the initial field as reference. **ours_q0 / ours_q256**: frozen project checkpoint without / with 256 corrections.
- **FOM**: full-order model on the same grid; `fom_fft_tight` is the reference itself (error 0 by construction).
