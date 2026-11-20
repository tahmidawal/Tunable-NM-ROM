# Conditional B8 spatial-sampling and training-strata repair

This is prospective preparation, conditional on the current expanded multidepth child's retained results. The multidepth driver now exposes optional flags; defaults preserve the current uniform-index/legacy-family recipe and no active job is modified. Register a new run and its exact settings before enabling the repair. Preserve all prior attempts. This changes training sampling and declared family weights, not decoder architecture, physical error thresholds, PDE formulation or final-test policy.

## Diagnosis to test

The current stochastic stage draws 512 uniformly selected Gauss-grid indices per field. A narrow pulse can consequently receive few samples in its most localized region; the exact count depends on the chosen region and must be measured rather than assumed. Its primary-family sampling probability is one third shared by 128 trajectories, while each 16-trajectory ridge/two-pulse family receives one third. This creates substantially lower per-trajectory exposure for original pulses and smooth controls. Separate code banks correct cross-depth interference but do not address spatial or family exposure.

Inspect the completed selected checkpoint's time-matched training and development errors, fit statuses and residual locations first. If the assembled decoder remains inaccurate, the proposed architecture-neutral repair keeps shared child feature/head weights, frozen parent weights and independent layout/depth codes. It does not open final tests or train on development residuals.

## Unbiased physical spatial loss

For the original physical quadrature weights $w_i$, let $r_i$ be a fixed parent reconstruction residual on a training field. Prepare the proposal

$$
p_i=(1-\alpha)\frac{w_i}{\sum_jw_j}
       +\alpha\frac{w_i r_i^2}{\sum_jw_jr_j^2},\qquad \alpha=0.5.
$$

A zero weighted residual falls back to normalized quadrature weights. Draw points independently with replacement for each training field and use

$$
\widehat{\mathcal L}=\frac{1}{m\,\|u_{\mathrm{ref}}(t)\|_{L^2}^2}
\sum_{s=1}^{m}\frac{w_{i_s}}{p_{i_s}}
\left(u_\theta(x_{i_s},z)-u_{\mathrm{ref}}(x_{i_s},t)\right)^2.
$$

Its expectation is exactly the original quadrature reconstruction loss. The physical-weight component retains coverage and bounds importance weights by $\sum_i w_i/(1-\alpha)$. Do not retain the old uniform-sampling grid-size multiplier, self-normalize weights inside the minibatch, or backpropagate through the NumPy proposal. Independent 256-grid scoring and actual4096 checks remain unchanged. Full-quadrature finishing/refitting needs no stochastic proposal.

The residual proposal uses training reference fields only, offline, with the frozen parent's initial fitted training codes. Store its checkpoint/data hashes and construction settings. A fixed proposal remains unbiased as training proceeds, even if its variance reduction becomes less effective. Refreshing it would require an explicitly recorded training-only operation. This is supervised reconstruction sampling, not random/importance strong-form PDE collocation or truth-assisted online refinement.

## Five explicitly weighted training strata

Use the original parameter generator to distinguish smooth controls from pulses and classify pulse width at 0.05. For the current 128 original plus 32 expanded training trajectories, the helper fixture computes:

| Training stratum | Trajectories | Proposed probability |
|---|---:|---:|
| Smooth | 32 | 0.2 |
| Narrow pulse, width at most 0.05 | 41 | 0.2 |
| Broader pulse, width above 0.05 | 55 | 0.2 |
| Ridge | 16 | 0.2 |
| Two pulses | 16 | 0.2 |

Select a stratum, distinct trajectories within it, and independent uniform snapshot times. This raises aggregate original-family probability from one third to 0.6. It is explicitly a five-stratum weighted objective, not uniform sampling over all trajectories. Keep tree-variant probabilities 0.2/0.4/0.4 for coarse4/uniform20/prescribed20 and their independent codes unless separately changed. Report family-specific errors so improvement on one family cannot conceal regression elsewhere.

## Preparation and performance

`neural/spatial_sampling.py` is pure NumPy. It prepares immutable per-field probabilities/CDFs, then draws selected fields with compiled binary search. Prepare once in chunks or reuse per-field CPU CDFs; avoid GPU-to-CPU residual transfers and rebuilding CDFs every training update. Probabilities and CDFs together require 16 bytes per field-point pair, about 461 MB for 1760 fields on 128-square quadrature, plus negligible quadrature storage. This is offline training memory and still counts toward resource accounting. The sampler explicitly rejects held-out cohort identifiers.

`test_spatial_sampling.py` checks exact finite-sum expectation, deterministic Monte Carlo agreement within computed sampling uncertainty, bounded importance weights, zero-residual fallback, finite very-large residuals, complete/nonoverlapping actual training strata and held-out guards. These establish arithmetic correctness, not improved training or PDE performance.

Before launch, record whether spatial sampling alone or both sampling and strata are changed. Applying both together is a practical training repair but cannot identify which mechanism caused any improvement. No new architecture search or automatic PDE admission follows from these helpers.

Implementation switches: `--spatial-sampling residual-mixture --sampling-alpha .5` and independently `--family-sampling split-primary`. `--reuse-codes` retains its verified-parent/code reuse behavior. Proposals are prepared in four-field GPU chunks, then cached CPU CDFs; preparation seconds, allocation bytes, probability SHA256, parent/root-initialization/data provenance, and stratum counts/probabilities are recorded in result metadata. Full-quadrature updates still use N*w with every point. No launch is authorized by this preparation alone.
