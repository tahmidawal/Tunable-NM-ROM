# Accuracy and cost knobs for Burgers in two and three dimensions

This atlas regenerates plots from preserved experiments and newly rescored saved timing witnesses. Numbers are final for those saved invocations; historical qualification labels, conditional reference estimates and missing controlled sweeps limit interpretation. No new training or PDE solve was run.

Open `index.html` for the filterable plot gallery. Each figure is available as PNG and SVG. `POINTS.csv` contains every plotted configuration; `POINTS.json` retains per-case scores, timings and repetition arrays. `PLOTS.json` maps figures to configurations. `SOURCES.json` and the retained compact data pin provenance.

## Reading the plots

The primary error is the maximum over evolved saved outputs at simulation times 0.05 through 0.25 of the field L2 difference divided by the initial reference-field L2 norm. It is neither training loss nor a pointwise error. Reference uncertainties are conditional Richardson estimates, not rigorous bounds. The main linked 2D panels use the same six development cases. The newly rescored 3D screening panel uses four exposed validation cases; the later frontier uses the full exposed validation population; historical work uses that full population; mesh diagnostics use their fixed diagnostic subset. Populations are never mixed inside a curve.

Each runtime curve stays within one allocation. Cost and error refer to the same timed returned field, or a scoring trajectory that the timed invocation reproduced by its recorded identity check. Initialization, evolution and output decoding are included in fused query time. Compilation, training and postprocessing are excluded. Separate phase measurements are diagnostic and are never summed to manufacture a fused time. Historical 3D controls without admitted warmed timing use nonlinear iteration counts; iterations do not measure equal work across methods or dimensions.

The recovered 3D screening panel is rescored from its own saved coefficient trajectories, not from a separate accuracy run. Its original in-job output links and A–B–A timing controls passed. New f64 physical-lattice Gram rescoring is checked against direct field reconstruction, with an independent NumPy implementation included for replay. The small screening population supports descriptive knob curves, not a new full-population or sealed validation claim. Minimal models, references, coefficients and products are preserved; a large replay archive is divided into checksum-verified parts so every committed file can be backed up to GitHub.

Markers retain the original numerical qualification outcome: circles pass the reported solver/quadrature checks; crosses fail or remain unverified. A passing development check is not full validation or a claim of physical accuracy. No failed configuration is used to assert an accepted speedup. Every point retains its case count, worst error and count of errors above the explicitly descriptive one-percent threshold. Timing outliers use within-case Tukey fences when at least four repeats exist; small repetition counts cannot establish tail behavior.

## Scope and gaps

- Bank width, test-space size, quadrature count/family, time step/method and mesh resolution are covered wherever saved records supply multiple settings. Exhaustive gallery pages hold the other listed settings fixed.
- The nonlinear head K panels vary K under the historical selected-rule policy. Test-space size and quadrature can change with K; those curves do not isolate a pure K effect. Separate seeds remain separate. The recorded 2D qualified K trend is inconclusive.
- The saved 3D bank is a linear-coefficient model. Its `K1` and `k4` labels are solver work policies, not nonlinear latent dimensions. A matched nonlinear-head K sweep in 3D is absent and cannot be inferred from width or work-policy curves.
- Solver tolerance and fixed/refreshed nonlinear work policies are available as diagnostic controls rather than a complete warmed runtime ladder. Their evidence is retained in the source data; this atlas does not use first-call/compilation seconds as production cost.
- Raw weak-residual norms and stationarity certificates are plotted separately from physical error. Changing test functions, residual weights, integration scheme or time step changes the objective, so residual values are not compared as a common accuracy scale. A certificate is not a loss value, and residual norm is not silently squared into training loss.
- A systematic 3D head sweep, a controlled fixed-M/fixed-quadrature K study, and any missing matched tolerance timings require new experiments. This atlas completes the saved-evidence plotting, not those unrun experiments. Full-decoder comparisons, adaptive/local refinement and other PDE extensions remain deferred at the user's request.

## Reproduce

Run `/home/tahmid/Dev/.venv/bin/python build.py` from this directory, with `OPENBLAS_NUM_THREADS=1`. It needs only the compact `data/` files and plotting libraries. `collect.py` documents the source extraction and can refresh it read-only from the original local lanes and checksum-verified cluster archives. Do not delete shared archives to reproduce these figures.

## Plot inventory


| Dimension | Knob | Matched groups | Groups with complete warmed timing | Groups with iteration counts |
|---|---|---:|---:|---:|
| 2D | r | 78 | 55 | 78 |
| 2D | K | 10 | 10 | 10 |
| 2D | M | 51 | 28 | 51 |
| 2D | m | 60 | 57 | 60 |
| 2D | dt | 36 | 36 | 36 |
| 2D | mesh | 10 | 8 | 8 |
| 3D | r | 34 | 4 | 34 |
| 3D | K | 0 | 0 | 0 |
| 3D | M | 47 | 7 | 45 |
| 3D | m | 132 | 30 | 132 |
| 3D | dt | 4 | 4 | 4 |
| 3D | mesh | 2 | 2 | 2 |

Counts describe coverage, not independent replications. Multiple sweeps can reuse configurations.

### 2D

- [Overview](figs/2d-overview.png)
- Bank width r (coefficients): [page 1](figs/2d-r-01.png), [page 2](figs/2d-r-02.png), [page 3](figs/2d-r-03.png), [page 4](figs/2d-r-04.png), [page 5](figs/2d-r-05.png), [page 6](figs/2d-r-06.png), [page 7](figs/2d-r-07.png), [page 8](figs/2d-r-08.png), [page 9](figs/2d-r-09.png), [page 10](figs/2d-r-10.png), [page 11](figs/2d-r-11.png), [page 12](figs/2d-r-12.png), [page 13](figs/2d-r-13.png), [page 14](figs/2d-r-14.png), [page 15](figs/2d-r-15.png), [page 16](figs/2d-r-16.png), [page 17](figs/2d-r-17.png), [page 18](figs/2d-r-18.png), [page 19](figs/2d-r-19.png), [page 20](figs/2d-r-20.png)
- Nonlinear latent dimension K: [page 1](figs/2d-K-01.png), [page 2](figs/2d-K-02.png), [page 3](figs/2d-K-03.png)
- Smooth test functions M (count): [page 1](figs/2d-M-01.png), [page 2](figs/2d-M-02.png), [page 3](figs/2d-M-03.png), [page 4](figs/2d-M-04.png), [page 5](figs/2d-M-05.png), [page 6](figs/2d-M-06.png), [page 7](figs/2d-M-07.png), [page 8](figs/2d-M-08.png), [page 9](figs/2d-M-09.png), [page 10](figs/2d-M-10.png), [page 11](figs/2d-M-11.png), [page 12](figs/2d-M-12.png), [page 13](figs/2d-M-13.png)
- Quadrature points m (count): [page 1](figs/2d-m-01.png), [page 2](figs/2d-m-02.png), [page 3](figs/2d-m-03.png), [page 4](figs/2d-m-04.png), [page 5](figs/2d-m-05.png), [page 6](figs/2d-m-06.png), [page 7](figs/2d-m-07.png), [page 8](figs/2d-m-08.png), [page 9](figs/2d-m-09.png), [page 10](figs/2d-m-10.png), [page 11](figs/2d-m-11.png), [page 12](figs/2d-m-12.png), [page 13](figs/2d-m-13.png), [page 14](figs/2d-m-14.png), [page 15](figs/2d-m-15.png)
- Time step Δt (simulation units): [page 1](figs/2d-dt-01.png), [page 2](figs/2d-dt-02.png), [page 3](figs/2d-dt-03.png), [page 4](figs/2d-dt-04.png), [page 5](figs/2d-dt-05.png), [page 6](figs/2d-dt-06.png), [page 7](figs/2d-dt-07.png), [page 8](figs/2d-dt-08.png), [page 9](figs/2d-dt-09.png)
- Mesh intervals / nodes per axis (source convention): [page 1](figs/2d-mesh-01.png), [page 2](figs/2d-mesh-02.png), [page 3](figs/2d-mesh-03.png)
- loss: [page 1](figs/2d-residual-01.png), [page 2](figs/2d-residual-02.png), [page 3](figs/2d-residual-03.png), [page 4](figs/2d-residual-04.png)
- training: [page 1](figs/2d-K-training.png)
### 3D

- [Overview](figs/3d-overview.png)
- Bank width r (coefficients): [page 1](figs/3d-r-01.png), [page 2](figs/3d-r-02.png), [page 3](figs/3d-r-03.png), [page 4](figs/3d-r-04.png), [page 5](figs/3d-r-05.png), [page 6](figs/3d-r-06.png), [page 7](figs/3d-r-07.png), [page 8](figs/3d-r-08.png), [page 9](figs/3d-screen-r-01.png)
- Smooth test functions M (count): [page 1](figs/3d-M-01.png), [page 2](figs/3d-M-02.png), [page 3](figs/3d-M-03.png), [page 4](figs/3d-M-04.png), [page 5](figs/3d-M-05.png), [page 6](figs/3d-M-06.png), [page 7](figs/3d-M-07.png), [page 8](figs/3d-M-08.png), [page 9](figs/3d-M-09.png), [page 10](figs/3d-M-10.png), [page 11](figs/3d-screen-M-01.png), [page 12](figs/3d-screen-M-02.png)
- Quadrature points m (count): [page 1](figs/3d-m-01.png), [page 2](figs/3d-m-02.png), [page 3](figs/3d-m-03.png), [page 4](figs/3d-m-04.png), [page 5](figs/3d-m-05.png), [page 6](figs/3d-m-06.png), [page 7](figs/3d-m-07.png), [page 8](figs/3d-m-08.png), [page 9](figs/3d-m-09.png), [page 10](figs/3d-m-10.png), [page 11](figs/3d-m-11.png), [page 12](figs/3d-m-12.png), [page 13](figs/3d-m-13.png), [page 14](figs/3d-m-14.png), [page 15](figs/3d-m-15.png), [page 16](figs/3d-m-16.png), [page 17](figs/3d-m-17.png), [page 18](figs/3d-m-18.png), [page 19](figs/3d-m-19.png), [page 20](figs/3d-m-20.png), [page 21](figs/3d-m-21.png), [page 22](figs/3d-m-22.png), [page 23](figs/3d-m-23.png), [page 24](figs/3d-m-24.png), [page 25](figs/3d-m-25.png), [page 26](figs/3d-m-26.png), [page 27](figs/3d-screen-m-01.png), [page 28](figs/3d-screen-m-02.png), [page 29](figs/3d-screen-m-03.png), [page 30](figs/3d-screen-m-04.png), [page 31](figs/3d-screen-m-05.png), [page 32](figs/3d-screen-m-06.png), [page 33](figs/3d-screen-m-07.png), [page 34](figs/3d-screen-m-08.png)
- Time step Δt (simulation units): [page 1](figs/3d-dt-01.png), [page 2](figs/3d-screen-dt-01.png)
- Mesh intervals / nodes per axis (source convention): [page 1](figs/3d-mesh-01.png)
- loss: [page 1](figs/3d-residual-01.png), [page 2](figs/3d-residual-02.png), [page 3](figs/3d-residual-03.png), [page 4](figs/3d-residual-04.png), [page 5](figs/3d-residual-05.png), [page 6](figs/3d-residual-06.png), [page 7](figs/3d-residual-07.png), [page 8](figs/3d-residual-08.png), [page 9](figs/3d-residual-09.png), [page 10](figs/3d-screen-residual-01.png), [page 11](figs/3d-screen-residual-02.png), [page 12](figs/3d-screen-residual-03.png), [page 13](figs/3d-screen-residual-04.png), [page 14](figs/3d-screen-residual-05.png)

## Plain-language glossary

- **r / bank width:** number of coefficients in the ordered linear dictionary used to represent the field.
- **K / latent dimension:** number of nonlinear-head input coordinates. Solver labels `K1` and `k4` in historical 3D records mean something different: work or refresh policies.
- **M / test-space size:** number of smooth functions against which the PDE residual is projected.
- **m / quadrature count:** number of integration points used to approximate that projection. A tensor rule's per-axis order is not its total point count.
- **Mesh:** spatial discretization size. Source conventions differ: historical 2D uses intervals per axis; the newer 3D mesh panel uses nodes per axis, including boundary nodes.
- **Time step / dt:** time advanced in one numerical step. Changing it can also change residual weights and the directions used by a least-squares solver.
- **BE, CN, CNR, BDF2:** backward Euler, Crank–Nicolson, Crank–Nicolson with the recorded startup treatment, and second-order backward differentiation: alternative time integrators.
- **LSPG / GAL:** a least-squares solve of the projected time-discrete residual / a Galerkin reduced equation. Their objectives differ.
- **fs / qn / ad:** fixed-sweep, refreshed quasi-Newton, and adaptive nonlinear-solver policies in the original records.
- **Quadrature family / rule:** integration construction, such as Gauss, lattice or Fibonacci; the rule name specifies the saved point set.
- **Seed:** recorded random initialization or sampling identifier; separate seeds are separate repetitions of the model-building recipe.
- **Conditional L2 error:** field-distance estimate relative to the reference, normalized by the initial reference field. **Median** is the middle case value; **worst** is the largest case value.
- **eps / reference uncertainty:** conditional estimated error of the reference, used to judge whether differences are resolved; it is not an absolute guarantee.
- **Residual / loss:** mismatch with the discrete equations being minimized. The saved raw norm is reported in its source units; it is not interchangeable with field error or a squared loss.
- **Stationarity / certificate:** a check that a returned solver state satisfies a recorded stopping condition. It does not prove that the reconstructed field is accurate.
- **Qualified / Q1 / Q2:** the source solver-convergence checks / source quadrature-consistency checks passed for their recorded population. A failed or missing check remains visible.
- **Fused query cost:** measured input preparation, initial fit, evolution and output synthesis under the stated source contract, in milliseconds. A **timing panel** is one allocation on one GPU.
- **Iteration / correction:** one recorded nonlinear update. Different solvers and dimensions can perform different work per update.
- **Invocation / witness / hash:** one solver execution / its saved output / a content fingerprint linking the output to a repeated timed execution.
- **Development / exposed validation:** cases already available for research and selection; neither is sealed confirmation. No sealed or out-of-distribution cases are read here.
- **POINTS table:** dimension, panel and id identify the run; r, K, M, m, mesh, dt, scheme, solver and policy specify its controls; qualified and timing_usable retain admission; n_cases records population; error columns record median, worst and count above one percent; milliseconds, iterations, residual and stationarity give the named metrics; timing_repetitions/outliers describe retained timings; source and SHA256 columns identify evidence.
