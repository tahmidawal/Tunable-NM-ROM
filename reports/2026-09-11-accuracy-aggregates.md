# Aggregate errors of the retained accuracy-campaign models

Generated from the finalized September campaign's previously audited error arrays. These are development-cohort aggregates, not final publication validation or a new independent field audit.

## Meaning of the aggregates

For case $i$ and requested output time $t_j$, let $e_{ij}$ be the recorded normalized physical error. The mean of case maxima is $n^{-1}\sum_i\max_j e_{ij}$; the mean final error is $n^{-1}\sum_i e_{iJ}$; the mean over cases and observations is $(nJ)^{-1}\sum_{i,j}e_{ij}$, where $J$ counts all saved observations including the initial state. The last statistic gives observations equal weight; it is not a continuous-time integral. Poisson has one static field per case.

Each physical case receives equal weight. All timing repetitions have identical error arrays; only one copy enters these statistics. Source hashes, case counts and primary worst errors agree with the accepted campaign report. No cases or outliers are removed.

The primary normalizations are static reference-field L2 for Poisson, current reference-field L2 for heat, initial reference-field L2 for Burgers, and initial energy-state scale for reflective waves. Their percentages are not interchangeable. Heat final time is recorded in its case arrays, as are the distinct Burgers and wave horizons.

## Primary trajectory maxima, aggregated across cases

| PDE | Intervals | Cases | Mean % | Median % | Worst % | Cases above 5% | Upper outliers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson | 64 | 42 | 1.987198 | 1.271045 | 6.111946 | 2 | 2 |
| Poisson | 256 | 42 | 1.982206 | 1.266181 | 6.110598 | 2 | 2 |
| Poisson | 1024 | 42 | 1.982130 | 1.266120 | 6.110576 | 2 | 2 |
| Heat | 64 | 16 | 2.330163 | 2.355837 | 4.762500 | 0 | 0 |
| Heat | 256 | 16 | 2.329971 | 2.353668 | 4.762515 | 0 | 0 |
| Heat | 1024 | 16 | 2.329988 | 2.353578 | 4.762515 | 0 | 0 |
| Burgers | 64 | 6 | 5.212322 | 4.175852 | 10.856979 | 2 | 0 |
| Burgers | 256 | 6 | 2.404574 | 2.195423 | 4.554611 | 0 | 0 |
| Burgers | 1024 | 6 | 2.404137 | 2.053097 | 3.884680 | 0 | 0 |
| Reflective waves | 64 | 4 | 4.059211 | 3.807446 | 5.041071 | 1 | 1 |
| Reflective waves | 256 | 4 | 4.203460 | 3.940915 | 5.136114 | 1 | 1 |
| Reflective waves | 1024 | 4 | 4.214115 | 3.953492 | 5.145194 | 1 | 1 |

Cases above the displayed error threshold are a descriptive count, not a replacement for the campaign's full physical, reference and numerical acceptance gates. Upper outliers exceed the third quartile plus one-and-a-half interquartile ranges, using NumPy's linear quantiles; they remain included. Passing on average does not establish a worst-case pass.

## Primary final-state and observation-averaged errors

| PDE | Intervals | Mean final % | Median final % | Worst final % | Mean over cases and observations % |
| --- | --- | --- | --- | --- | --- |
| Poisson | 64 | 1.987198 | 1.271045 | 6.111946 | 1.987198 |
| Poisson | 256 | 1.982206 | 1.266181 | 6.110598 | 1.982206 |
| Poisson | 1024 | 1.982130 | 1.266120 | 6.110576 | 1.982130 |
| Heat | 64 | 1.375493 | 1.256666 | 2.574316 | 1.579354 |
| Heat | 256 | 1.373646 | 1.246532 | 2.592446 | 1.576926 |
| Heat | 1024 | 1.373674 | 1.246010 | 2.593641 | 1.576870 |
| Burgers | 64 | 4.628664 | 3.313496 | 10.856979 | 4.060162 |
| Burgers | 256 | 1.677832 | 0.997986 | 4.165043 | 1.819648 |
| Burgers | 1024 | 1.136027 | 0.658487 | 2.437078 | 1.681500 |
| Reflective waves | 64 | 4.019044 | 3.727111 | 5.041071 | 2.902933 |
| Reflective waves | 256 | 4.128611 | 3.841314 | 5.051856 | 2.965985 |
| Reflective waves | 1024 | 4.134007 | 3.850980 | 5.050157 | 2.970614 |

## Other normalizations and wave components

Burgers current-field error uses the instantaneous reference norm, which differs from its primary initial-field normalization. Wave displacement is reported both relative to initial displacement and to current displacement; velocity uses the initial energy scale. Energy-state error includes displacement gradients as well as velocity and is not energy drift.

| PDE | Intervals | Metric | Mean case maximum % | Median case maximum % | Worst % | Mean final % |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers | 64 | current_field | 7.758313 | 6.376830 | 15.347717 | 7.676170 |
| Burgers | 256 | current_field | 3.095743 | 2.584814 | 5.887817 | 2.713292 |
| Burgers | 1024 | current_field | 2.810364 | 2.416546 | 4.388298 | 1.901264 |
| Reflective waves | 64 | initial_displacement | 1.196569 | 1.121129 | 1.477907 | 1.109905 |
| Reflective waves | 64 | current_displacement | 2.363423 | 2.416004 | 2.589672 | 1.792942 |
| Reflective waves | 64 | initial_energy_scaled_velocity | 2.709446 | 2.581865 | 3.260997 | 2.635732 |
| Reflective waves | 256 | initial_displacement | 1.235075 | 1.168670 | 1.542841 | 1.133824 |
| Reflective waves | 256 | current_displacement | 2.422003 | 2.448640 | 2.696466 | 1.828147 |
| Reflective waves | 256 | initial_energy_scaled_velocity | 2.826170 | 2.668458 | 3.419016 | 2.776242 |
| Reflective waves | 1024 | initial_displacement | 1.237471 | 1.171903 | 1.546468 | 1.135764 |
| Reflective waves | 1024 | current_displacement | 2.426210 | 2.451008 | 2.703948 | 1.830858 |
| Reflective waves | 1024 | initial_energy_scaled_velocity | 2.831474 | 2.677171 | 3.421022 | 2.782358 |

## Interpretation and provenance

The older CP paper primarily used case aggregates and final heat states. Current heat final-state current-relative means are the closer statistical comparison, but benchmark families, horizons, meshes, training capacity and reference protocols differ. These tables alone cannot attribute a difference to the decoder architecture. The sealed final cohorts remain unopened, and no acceptance decision changes.

[Accepted campaign](2026-09-11-accuracy-improvements-and-wave-speed.md) · [Per-case arrays, aggregates and source hashes](2026-09-11-accuracy-aggregates.json) · [Generator](generate_accuracy_aggregates.py)

Regenerate with:

```bash
OPENBLAS_NUM_THREADS=1 /home/tahmid/Dev/.venv/bin/python reports/generate_accuracy_aggregates.py
```

## Plain-language glossary

- **PDE / intervals / cases:** equation family / mesh subdivisions per axis / distinct physical inputs; timing repeats are not additional cases.
- **Retained model / development cohort / sealed final cohort:** selected method / cases used during diagnosis and selection / untouched cases reserved for later final validation.
- **Relative L2 / normalization:** field-error magnitude divided by the specified reference magnitude / choice of that divisor.
- **Case maximum / mean / median / worst:** largest error over one case's saved times / arithmetic average across cases / middle case value / largest value across cases and saved times.
- **Final / observation:** last requested output time / one saved output time, including the initial state.
- **Mean over cases and observations:** equal-weight average of saved error scalars over both axes; different from an aggregate space-time norm.
- **Upper outlier / quartile / interquartile range:** case beyond the stated upper statistical threshold / quarter-position of the sorted values / difference between upper and lower quartiles. Outliers are retained.
- **Cases above 5% / acceptance gate:** count with a primary case maximum exceeding that physical error threshold / the fuller accuracy, reference and solver-validity requirements.
- **Initial field / current field / static field:** reference at the initial time / reference at the evaluated time / stationary reference solution.
- **Displacement / velocity / energy-state / energy drift:** wave field / its time derivative / combined displacement-gradient and velocity measure / change in conserved energy, which is a different diagnostic.
- **CP / decoder / checkpoint:** canonical-polyadic tensor representation / map from reduced coordinates to a field / saved trained parameters.
- **Repetition / SHA256 / source hash:** repeated timing invocation of the same case / content fingerprint / fingerprint linking the analysis to its accepted input file.
