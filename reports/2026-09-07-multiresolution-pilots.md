# Initial multiresolution pilots: accuracy and complete-query cost

This report covers the first frozen-network mesh-transfer pilots of the current separable NM-ROM against efficient FOM solvers. The numbers are provisional development evidence; independent confirmation and the full resolution study remain open.

The tested frozen decoders produce solutions on new meshes, but these primary configurations have not established a complete-query advantage over efficient FOMs. Increasing resolution mostly preserves the ROM error in these pilots. Further work must address representation, initialization or reduced-solver cost, according to the PDE.

The older ViT + CP architecture is excluded. Waves use only the fresh verified lineage. The final cohorts remain unopened. Different rows use different physical error definitions, stated below; they must not be ranked as a common cross-PDE accuracy score.

## Primary configurations from the first pilots

These are explicit representative settings, not a claim that every row is the cheapest configuration at a qualified accuracy target. Burgers uses a same-mesh FOM here; its native report also includes a coarse-mesh output envelope. Poisson uses the development-selected stopping setting shown. Raw cost ratios are separate from physical-accuracy qualification.

| PDE | Intervals | ROM setting | Error metric | Cases | ROM median / worst error (%) | FOM worst error (%) | ROM / FOM query (ms) | Raw paired FOM/ROM |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Heat | 64 | CN dt=0.025 | Current L2 | 4 | 5.858 / 9.736 | 0.07092 | 27.775 / 1.221 | 0.04396 |
| Heat | 128 | CN dt=0.025 | Current L2 | 4 | 5.855 / 9.736 | 0.01772 | 27.738 / 1.3691 | 0.04925 |
| Burgers | 256 | dt=0.005; stall=0.001 | Initial L2 | 4 | 1.954 / 3.961 | 2.226 | 39.284 / 18.416 | 0.4917 |
| Burgers | 512 | dt=0.005; stall=0.001 | Initial L2 | 4 | 2.88 / 5.274 | 1.582 | 42.379 / 27.875 | 0.6954 |
| Poisson | 256 | tau=0.01 | Steady L2 | 6 | 0.8927 / 7.363 | 0.01546 | 4.8257 / 2.0607 | 0.395 |
| Poisson | 512 | tau=0.01 | Steady L2 | 6 | 0.8928 / 7.363 | 0.003677 | 4.8859 / 2.4406 | 0.4995 |
| Reflective wave | 256 | dt=0.0025 | Initial wave state | 2 | 46.17 / 55.37 | 0.5047 | 3300.4 / 14.024 | 0.004249 |
| Reflective wave | 512 | dt=0.0025 | Initial wave state | 2 | 46.01 / 55.2 | 0.1445 | 3354.3 / 64.534 | 0.01924 |
| Absorbing wave | 256 | dt=0.0025 | Initial wave state | 2 | 7.576 / 7.655 | 0.08748 | 3221.3 / 94.984 | 0.02948 |
| Absorbing wave | 512 | dt=0.0025 | Initial wave state | 2 | 7.573 / 7.654 | 0.01951 | 3274.3 / 297.2 | 0.09076 |

All timings include the supplied host field, initialization/source projection, solve or evolution, and requested host field outputs. Each cost is the cohort median of per-case repetition medians. The paired ratio is the median of per-case FOM/ROM cost ratios. Each pair was measured in one job on one GPU; raw wall times must not be compared across PDE jobs as a hardware-normalized ranking.

**Heat:** errors are maxima over output times relative to the current reference norm, evaluated on the common observation grid. This is a newly verified, restricted single-bump development family. It does not yet cover the broader heat use case.

**Burgers:** errors use the initial reference norm. The first reference refinement estimate leaves target qualification unresolved. Cold-fit budget exits are retained in the native records and independent audit; an observed small error is not proof of a stationary initial fit. The follow-up separately tests more starting guesses and a finer reference.

**Poisson:** errors are relative solution norms for each steady source. Tighter stationary solves still leave a worst-case error floor. The reference has empirical refinement evidence; development qualification is not an independent final-cohort result.

**Waves:** the reported metric is the maximum of initial-normalized displacement, velocity and energy-state errors. Absorbing errors relative to the small remaining field are substantially larger and must also be reported. The time-step pair is resolved for these cases, so a smaller step does not remedy the observed error.

## Evidence and limitations

| PDE | Numerical source commit | GPU job | Native findings |
|---|---|---|---|
| Heat | `ad882df3ecb8614d4cafa6a25761e920f3074e3d` | 3349961 | [Findings / source records](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/HEAT-PILOT-NOTES.md) |
| Burgers | `96befb8815a4c20c8d9f2862502e454ab8402032` | 3350012 | [Findings / source records](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/pilot01/out/pilot.json) |
| Poisson | `82d2c3261126cb150bb83220ec3edbcf0dd5f489` | 3350079 | [Findings / source records](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/pilot01/FINDINGS.md) |
| Waves | `a02aacd9578f46a9bee0dcb093acd35e62d10739` | 3349951 | [Findings / source records](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/pilot01/analysis/FINDINGS.md) |

All first-pilot outputs were checksum-collected and their exact remote job directories removed. Numerical checks cover GPU execution, precision, relevant operators, reference refinement and state advancement. Root independently recomputed the preserved heat, Burgers and Poisson field errors. The wave root review checks source design, paired accounting and artifact hashes; it is not an independent full-grid regeneration of every wave metric.

The rigorous reference-bound field remains unspecified where only empirical refinement is available. No paper-wide speedup, optimal capacity, optimized training cost, broad-family robustness, or independent confirmation is established.

## Next experiments justified by the diagnostics

- **Burgers:** resolve reference space/time error; compare the original cold start with several training-code guesses; retain the efficient same-grid and coarse-grid FOMs.
- **Heat:** keep the checkpoint fixed and compare solver tolerances and compiled full-query execution. Treat the separate nonlinear-head reconstruction gap as an accuracy task.
- **Poisson:** increase smooth test-mode coverage and measure full-bank projection versus best recorded nonlinear-head fits; compare compiled and segmented queries.
- **Waves:** use matched-dimensional linear and nonlinear controls with a frozen spatial bank to separate compression from autonomous dynamics. Finer rendering alone cannot address the current error.

These are development decisions. The complete study still needs the full mesh ladder, separately labeled per-resolution training, independent data/training repeats, validation-selected settings and sealed final evaluation.

## Reproduction

Run `/home/tahmid/Dev/.venv/bin/python reports/generate_multiresolution_pilots.py` from a checkout with the recorded experiment worktrees. The adjacent JSON manifest identifies every source artifact by content hash. All numerical table values are generated; none are hand-entered.

## Plain-language glossary

- **PDE / FOM / NM-ROM / ROM:** partial differential equation / full spatial solver / nonlinear-manifold reduced solver / reduced solver.
- **Frozen / intervals / cases:** unchanged network weights / grid cells along one axis / distinct physical inputs. Timing repetitions are not additional cases.
- **ROM setting / dt / CN / stall / tau:** chosen solver configuration / time step / Crank–Nicolson time formula / relative improvement stopping rule / requested weak-residual reduction.
- **Current L2 / initial L2 / steady L2:** field error divided by the current reference norm / initial reference norm / steady reference solution norm.
- **Initial wave state / energy-state error:** the separately normalized displacement, velocity and energy measures / error combining displacement gradients with velocity.
- **Median / worst / query ms / paired ratio:** middle case result / largest case error / complete input-to-output milliseconds / per-case FOM time divided by ROM time, then a cohort median.
- **Bank / head / latent / weak test mode:** learned spatial features / their nonlinear coefficient map / compressed state coordinates / smooth function averaging the PDE equation.
- **Cold start / stationary / budget exit:** initial reduced-state fitting / meeting a local derivative convergence check / exhausting allowed iterations.
- **DST / coarse-grid envelope:** direct sine-transform solution / least-cost qualifying full solve allowing fewer cells and charging interpolation to the requested output.
- **Reference refinement / uncertainty / qualification:** comparing finer trusted solves / remaining reference error / meeting accuracy and numerical-validity requirements with that uncertainty included.
- **Development / validation / sealed final / provisional:** preliminary experiment data / data used to select settings / untouched independent confirmation data / evidence with the stated limitations.
- **Compiled / segmented / projection / compression:** one prepared executable / separately launched stages / representing a field in a spatial span / constraining that representation through fewer latent coordinates.
- **Commit / manifest / checksum:** saved source revision / inventory of source artifacts / content fingerprint checking exact file bytes.
