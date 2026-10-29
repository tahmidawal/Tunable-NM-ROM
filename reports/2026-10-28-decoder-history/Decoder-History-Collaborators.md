# How our decoders evolved

## A guide for collaborators

We have been studying how to solve a PDE using a small number of unknowns instead of solving for the field at every mesh point. The **decoder** turns those few unknowns into the full solution field. During a reduced solve, we adjust the unknowns so that the decoded field satisfies the governing equations as closely as possible.

The main question throughout this work has been: **which decoder gives an accurate solution that is also inexpensive to compute?** A network that reconstructs saved solutions well is a promising starting point, but it still has to work inside the physics solver.

### The story in five steps

**We started with grid-based tensor decoders.** These assembled fields from learned spatial factors. They were inexpensive to evaluate, but the factors were tied to a particular grid, and accuracy did not reliably improve when the grid was refined.

**We tried networks that take spatial coordinates as inputs.** These could evaluate a solution at new locations without learning a new output grid. They improved several reconstruction tests, but evaluating and differentiating a large network during every solver iteration was costly. Some also produced unreliable solves despite good reconstruction.

**We separated the spatial functions from the small nonlinear network.** The expensive spatial functions could then be evaluated once and reused. A small network, called the head, produced their coefficients. This became the main learned-bank approach.

**We tested whether the nonlinear head was always needed.** We tried different heads, added correction directions, and also solved directly for coefficients in the learned spatial bank. Direct coefficient solving became an important option. Classical data-derived bases remained strong competitors.

**We improved the numerical solver and tested stronger alternatives.** Better integration and time stepping improved accuracy. However, careful comparisons showed that a tuned conventional solver could often reach the same accuracy faster. The latest architecture proposal adds local detail where needed; it remains a hypothesis to test.

### What we can say confidently

The learned coordinate bank can support reduced computations whose cost does not grow with the output mesh in the qualified settings. It also allows evaluation at selected spatial points. Producing a large full-field output still has a cost.

We have **not established a general speed and accuracy advantage over the best conventional methods** for the tested Burgers problems. There are useful results for particular settings, including moving-frame models for translating flows, but they should keep their stated scope.

The most consistent lesson is simple: **a better reconstruction is not necessarily a better PDE solve.** Representation, solver convergence, physical accuracy and total cost must be checked separately.

### How to read the rest

The following sections describe what each architecture changed, why we tried it, what happened, and its status. Blue links open code or supporting evidence directly on GitHub. The code links point to preserved source files at a fixed revision; they do not depend on access to a local workstation. Running an experiment still requires its original configuration, dependencies and checkpoints.

This edition exports the historical review completed on October 28, 2026. It is not a fresh numerical audit. A short update below distinguishes subsequent campaign developments from that frozen review. Earlier results that were corrected or withdrawn remain clearly marked.

**Repository:** [Tunable NM-ROM on GitHub](https://github.com/tahmidawal/Tunable-NM-ROM). **Full code map:** [decoder code and evidence index](https://github.com/tahmidawal/Tunable-NM-ROM/blob/main/reports/2026-10-28-decoder-history/CODE-INDEX.github.md).


### Update for this edition

Since the historical review was frozen, a shared-decoder latent hierarchy was also tested. Both bounded initialization/replay policies failed their cost checks against the direct solver on the same trained model. This adds another scoped negative result; it does not rule out every multilevel method.

The obstacle-flow campaign has also progressed: references for the chosen fixed channel were admitted, and a neural bank was trained and tested. Its representation and actual reduced-solve accuracy did not meet the required targets. Further classical-basis comparisons remain unfinished. There is still no accepted end-to-end learned-ROM advantage for that campaign.

These updates follow the [dated lab-log excerpt preserved for this edition](https://github.com/tahmidawal/Tunable-NM-ROM/blob/main/reports/2026-10-28-decoder-history/collaborator-status-2026-10-29.txt). The detailed historical sections below retain their earlier snapshot: statements there about work awaiting training describe the state at the original review, not today's status. No new experiment was run to prepare this PDF.


<!-- PAGEBREAK -->

## The architectural distinctions that explain the history

A **decoder** maps reduced coordinates to a physical field. An **encoder** maps an observed field to reduced coordinates. A ROM need not use an encoder online: it can solve directly for reduced coordinates using the PDE.

For the old grid CP decoder, the spatial factors are fixed after training:

$$u(i,j;z)=b+\sum_{a=1}^{R}h_a(z)X_{a,i}Y_{a,j}.$$

For a joint coordinate decoder, spatial and latent inputs interact inside the network:

$$u(x;z)=b_{\mathrm{bc}}(x)f_\theta(x,z).$$

For the separable coordinate bank, the spatial network and latent head are separate:

$$u(x;z)=b_{\mathrm{bc}}(x)\sum_{a=1}^{R}g_a(x)h_a(z).$$

For the ordered-bank path, the coefficients are solved directly:

$$u(x;c)=B_r(x)c.$$

The last representation is **linear in its online coefficients**, even when a neural network learned its spatial functions. Its PDE solve can still be nonlinear because Burgers or Navier–Stokes is nonlinear. Conversely, a nonlinear head inside a fixed bank describes a curved subset of that bank's linear span; it cannot escape the span merely by increasing head complexity.

Here $K$ is latent-head dimension, $R$ is total bank width, $r$ is retained bank width, $M$ is the number of weak tests and $m$ is quadrature size. These are different cost and accuracy controls. Network parameter count is another quantity again.

The development branches are described chronologically below. They are not all merged into a single implementation.



## The beginning: CP, attention and SIREN

### Plain CP decoder — original Heat model

**Idea.** A small MLP converts the latent vector into channel weights. Products of learned axis-factor tables turn those weights into a field. The factor tables are tied to a grid; the coefficient map is nonlinear, but every output lies in the span of the fixed spatial products.

**Why it was used.** Tensor factors make selected-node evaluation and contraction inexpensive. The heat pipeline used temporal warm starts.

**What happened.** This became the baseline that later studies challenged for resolution transfer, trainability and end-to-end accuracy. The frozen public heat rollout has a known bug and must not be used as today's behavioral reference. Subsequent heat work repaired execution and studied training and algebra separately.

**Status.** Historical baseline; retain for reproducibility, not as the current recommended implementation. **Code:** [cp](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/cp/decoder.py). The original source docstring's claim that warm-starting removes the need for cold-start regularity is a design rationale, not a general convergence theorem.

### LinearCP — CP with a direct latent skip

**Idea.** Add a linear latent-to-coefficient branch to the nonlinear head before the same spatial CP contraction. The name does **not** mean the whole decoder is linear.

**Why.** Poisson begins from a cold latent guess, making the local behavior of the head important.

**What happened.** This was the original Poisson architecture. Later controlled decoder tests did not find that adding the skip solved the spatial resolution wall. The fixed grid-factor representation remained the important limitation.

**Status.** Historical baseline. **Code:** [linearcp](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/linearcp/decoder.py). A caution for anyone reusing the comments: an arbitrary nonlinear branch can cancel a linear branch's derivative. A skip alone does not mathematically guarantee full rank everywhere; the later protected-anchor construction addresses that issue explicitly.

### Mirrored ViT decoder — May attention ablation

**Idea.** Replace the CP output map with a symmetric transformer decoder mirroring the encoder.

**What happened.** Reconstruction could be competitive, while Poisson reduced solves failed and heat rollouts became physically unusable. Attention over output tokens also defeated the cheap independently queried output structure needed by the tested hyper-reduction path.

**Status.** Tested negative for this ROM implementation, not a universal rejection of transformer decoders. **Code:** [vit](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/vit/decoder.py). **Original evidence:** [vit_status](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/vit_status/STATUS.md); canonical log's May prologue. Reconstruction, solver behavior and cost must remain separate in citing this result.

### Cross-attention INR — distinct from the mirrored ViT

**Idea.** Each spatial query attends to encoder tokens, followed by an output MLP. This supports coordinate queries but introduces dependence on per-snapshot encoder information.

**What happened.** The May journal records good reconstruction/off-grid query behavior, followed by failed reduced solves. At solve time there was no known solution snapshot from which to obtain the sample-specific tokens. The tested frozen-reference-token replacement did not recover a useful solver.

**Status.** Tested reconstruction model; failed tested ROM coupling. A learned latent-to-token replacement was suggested, not established as a successful repair. It is misleading to lump this failure together with output-token self-attention cost: the token-information mismatch is a separate issue. **Code:** [xattn](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/xattn/cross_attn_inr.py). **Evidence:** [may_journal](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/may_journal/JOURNAL.tex). These May comparisons are historical, not freshly audited performance claims.

### Modulated SIREN — coordinate decoder with sine activations

**Idea.** Spatial coordinates enter a sine network; the latent vector modulates its layers. Unlike CP, spatial features can change with the latent state.

**What happened.** An accurate autoencoder could still give a bad cold-start reduced solve. Diagnostic probes found a spurious attraction basin near the starting latent state; simple rank-deficiency was not the explanation. Training and solver changes recovered useful accuracy, but evaluating and differentiating the joint coordinate network during every iteration remained expensive.

**Status.** Important demonstrated failure mechanism and historical recovery; not the main deployed architecture. **Code:** [siren](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/siren/modulated_siren.py). **Detailed iteration history:** [may_journey](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/may_journey/journey-to-good-manifold.md).

### Default-field, latent-centering and anchored SIREN variants

**Variants tried.** Add an unmodulated default field; regularize the latent distribution; anchor the decoded zero latent to a representative field; change width; add operator-aware training; and compose an initialization-friendly checkpoint with a more accurate checkpoint during solving. Continuation and iteration-budget changes accompanied these trials but are solver changes, not new decoder families.

**What happened.** The anchor-based recipe improved cold-start behavior, but scaling capacity or improving warm-start reconstruction could destroy that behavior again. Composite solving eventually recovered accuracy. Thus “larger network” and “lower reconstruction loss” were unreliable proxies for the geometry needed by the solver.

**Status.** Historical variants with mixed outcomes. The collection includes a linear-skip SIREN implementation, but its presence alone is not a separate accepted benchmark result. **Code:** [default_siren](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/default_siren/default_field_siren.py), [skip_siren](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/skip_siren/linear_skip_siren.py). **Evidence:** [may_journey](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/may_journey/journey-to-good-manifold.md) and [may_speed](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/may_speed/SUMMARY.md).

### Nonlinear affine-in-latent constructions versus LinearAffineZ

**Idea.** Move expensive spatial feature computation into a SIREN trunk. The nonlinear affine variants still let latent dependence alter the effective map; the successful simplification made the output affine in the online latent coordinates:

$$u(x;z)=\Phi(x)Az+b(x).$$

**What happened.** The nonlinear variants retained bad cold-start behavior in the historical study. Making the Jacobian constant in the online variables gave a much easier solve, at a different accuracy/cost tradeoff.

**Status.** A useful early precursor of the cached-bank idea. The old speedups used a particular CG comparator and predate later fairness audits; they are not reinstated here. **Code:** [affine](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/affine/affine_z.py), [linear_affine](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/linear_affine/linear_affine_z.py). **Evidence:** [may_speed](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/may_speed/SUMMARY.md).

### Shared-latent multiresolution and FAS — a solver branch, not a new decoder victory

The May record also contains coarse-to-fine latent initialization, a learned residual encoder, and a full-approximation-scheme correction cycle. The initial transfer attempts did not establish an improvement. The promising FAS headline used best-across-cycle selection against truth, so it was not a demonstrated deployable stopping rule. See the May multiresolution subsection in the preserved lab log and the session files on `origin/decoder-explorer`.

This belongs in the lineage because later continuation experiments revisit related questions, but the May result does not certify today's prefix or separate-head methods. The historical journal is the evidence locator; this report does not claim to have reconstructed an independently runnable May FAS package.

## Recovered precursor and July work: fixed POD, residual MLPs and CP optimization

### Frozen POD as the decoder

**Idea.** Use a mean field and a fixed data-derived basis, with coefficients as the reduced unknowns. This is a linear trial space even when the reduced residual is nonlinear.

**What happened.** The recovered Burgers and wave studies used frozen POD when nonlinear-chart rollouts were unreliable. Later campaigns made POD a much stronger comparator through matched training snapshots, better solvers, hyper-reduction and appropriate tuning.

**Status.** Essential baseline and, in several current comparisons, the more competitive method. Historical “POD fails” claims cannot be carried forward across changed formulations and stopping rules. **Evidence:** [old_ledger](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_ledger/MASTER-RESULTS.md), [old_readme](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_readme/README.md). **Historical implementation reference:** [pod_hr](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/pod_hr/hrcore.py).

### Frozen POD lift plus residual MLP

**Idea.** Start near the POD reconstruction and train an additive nonlinear correction:

$$u(z)=\bar u+Vz+f_\theta(z).$$

The recovered general implementation uses a frozen POD encoder/lift and a residual network. The wave implementation uses a normalized stacked displacement/velocity state and a smaller correction network.

**What happened.** This is a real historical architecture, distinct from both pure FiLM and a learned spatial bank. The recovered records describe mixed gains and training-data sensitivity; favorable reconstruction did not establish a universally better rollout.

**Status.** Historical comparator, not the pure learned-bank method. **Code:** [old_podmlp](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_podmlp/podmlp.py), [old_podmlp_rom](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_podmlp_rom/podmlp_rom.py), [old_wave](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_wave/pod_mlp.py). **Evidence:** [old_ledger](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_ledger/MASTER-RESULTS.md). The recovered scripts include legacy reduced-precision arithmetic; do not silently reuse them under today's precision claims. Their comment that a POD skip guarantees a full-rank Jacobian for any nonlinear correction is too strong without additional constraints.

### CP heat repairs and reduced-algebra optimization

The recovered heat work compiled previously eager stages, changed small-system algebra, and studied reduced Gram contractions, Cholesky and execution loops while retaining a nonlinear CP solve. These are implementation and solver improvements, not evidence for an entirely new decoder.

**Status.** Useful historical engineering evidence; contemporary parity and paired timing are needed before transferring a claimed benefit. **Code:** [old_heat](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_heat/h3red.py). **Evidence:** [old_heat_log](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_heat_log/HEAT_ROUNDS.md) and [the recovered optimization report](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-07-30-cp-heat-optimization-transfer.md). That report distinguishes the earlier May repair/training work from the July timing work.

## August: coordinate decoders and the move to a true reduced solve

### Concatenated-coordinate MLP and grid-tied control

**Idea.** Feed coordinates and family descriptors into a network, comparing with a fixed grid-based representation and POD. The coordinate model can evaluate at previously unseen locations without increasing its learned parameter count with the requested mesh.

**What happened.** These tests exposed a resolution/trainability problem in the grid-tied models and demonstrated useful coordinate transfer. They supplied the *true family parameters* to the decoder, so they were expressivity/surrogate tests rather than unknown-latent PDE solves.

**Status.** Foundational representation evidence with that scope. **Code:** [concat](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/concat/poisson2d_diag.py). **Evidence:** August coordinate-decoder sections of the preserved lab log.

### FiLM coordinate decoder

**Idea.** Let the latent inputs modulate feature scales and offsets throughout a coordinate network rather than only concatenating inputs at the beginning. Fourier features help express spatial variation.

**What happened.** FiLM improved the representation tests and transferred across meshes. Over-bandwidth Fourier features on coarse training grids also produced severe off-grid aliasing; the affected runs were superseded by bandwidth-corrected runs. A good on-grid score was therefore insufficient.

**Status.** Successful representation mechanism, subsequently tested in harder solve settings. **Code:** [film](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/film/poisson2d_film.py). Do not treat the true-parameter-conditioned experiments as auto-decoder ROM performance.

### Time-conditioned FiLM for heat, Burgers and wave

**Idea.** Treat time as another coordinate/input so a network represents a whole solution family across space and time.

**What happened.** The early heat, Burgers and wave studies showed representation gains but still used known family descriptors. Their full-field inference timings measured a surrogate query, not a reduced PDE solve. Some reported comparisons also had different losses or precisions, which the log explicitly records.

**Status.** Historical representation branch. **Original testbed code:** [film_heat](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/film_heat/heat2d_film.py), [film_burgers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/film_burgers/burgers2d_film.py), [film_wave](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/film_wave/wave2d_film.py). The later latent-stepping implementations are linked next; their solved latent variables should not be confused with the earlier supplied physical parameters.

### Auto-decoder FiLM with latent stepping and weak residuals

**Idea.** Learn solution latent codes offline; online, determine the codes from the PDE rather than handing the decoder the true solution-family parameters. For transients, advance the latent solution in time.

**What happened.** This made the representation into an actual ROM, exposing objective, initialization, conditioning and stopping failures hidden by supervised parameter-to-field reconstruction. Pointwise residual minimization amplified small grid-scale decoder errors. Projecting onto smooth weak tests was the important correction.

**Status.** An actual solved-manifold predecessor of the separable bank. **Code:** [latent_burgers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/latent_burgers/blat_common.py), [latent_heat](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/latent_heat/hlat_common.py), [latent_wave](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/latent_wave/wlat_common.py). The historical wave records contain later retractions; only fresh verified wave campaigns should support current physical claims.

### Multistage residual fitting

**Idea.** Fit the error left by one network with another, frequency-adjusted network, hoping to carry single-function precision into a parametric decoder and reduced solve.

**What happened.** Very accurate single-function fits did not transfer into a robust parametric-ROM result. The study also corrected Fourier units, boundary residual construction, energy weighting and overstatements about finite-budget latent fits. Held-out-derived latent artifacts were quarantined.

**Status.** Abandoned tested direction. Its single-function precision is not evidence of a machine-precision parametric decoder. **Code:** [staged](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/staged/ms_parametric.py), [staged_ad](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/staged_ad/ms_autodecoder.py). The lab log records a GPU-provenance gap for the historical cell; no fresh certification is claimed.

### Residual-FiLM, grouped-FiLM and coordinate-warp implementation

**Idea.** Reduce the cost of a joint coordinate decoder by sharing modulation across channel groups or introducing residual blocks; the implementation also contains a latent-dependent coordinate warp.

**What happened.** Grouped-FiLM gave useful model-size tradeoffs. Residual-FiLM did not meet the tested accuracy requirements. Better decoder reconstruction did not reliably improve the actual reduced solution, and favorable kernel timing did not imply a favorable complete solve. The grouped models were selected across repeated training seeds rather than from a favorable mean alone.

**Status.** Grouped-FiLM is a bounded successful compression of the pure coordinate architecture, not a general FOM crossover. Coordinate-warp support is present in code; no accepted warp benchmark is identified in the reviewed summary, so it is **implemented/unestablished**, not a demonstrated success or failure. **Code:** [film_arch](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/film_arch/nda_arch.py). **Raw numerical source:** [nda_numbers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/nda_numbers/summary.json).

The generated table below uses the configurations recommended by the saved gate audits, with each recorded seed separate. These are historical reconstruction/weak-solve errors, not paired runtime comparisons or worst-case accuracy guarantees. Full individual settings, including controls, insufficient quadrature and unsuccessful alternative widths and residual-FiLM, are in [coordinate-search.csv](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/coordinate-search.csv). Inclusion in that CSV is not endorsement.

| PDE / cell | Seed | Decoder | Parameters | M / m | Fitted mean error | Full weak mean error | EQ mean / median / worst error |
|---|---:|---|---:|---|---:|---:|---|
| Poisson / `nda_peq112m448_s0_r28` | 0 | groupfilm | 58419 | 112 / 448 | 0.470985% | 0.499615% | 0.517393% / 0.457201% / 0.746687% |
| Poisson / `nda_peq112m448_s1_r28` | 1 | groupfilm | 58419 | 112 / 448 | 0.460966% | 0.505153% | 0.514404% / 0.461805% / 0.866642% |
| Poisson / `nda_peq112m448_s2_r28` | 2 | groupfilm | 58419 | 112 / 448 | 0.510488% | 0.564766% | 0.591389% / 0.543521% / 1.066347% |
| Burgers / `nda_beq128m592_s0_r32` | 0 | groupfilm | 140449 | 128 / 592 | 0.768783% | 0.961799% | 0.981227% / 0.871488% / 1.971603% |
| Burgers / `nda_beq128m592_s1_r32` | 1 | groupfilm | 140449 | 128 / 592 | 0.721809% | 0.933736% | 0.976960% / 0.834356% / 2.089665% |
| Burgers / `nda_beq128m592_s2_r32` | 2 | groupfilm | 140449 | 128 / 592 | 0.743087% | 0.978542% | 0.999951% / 0.785813% / 2.459731% |

## Late August and September: the separable bank family

### Cached learned coordinate bank plus nonlinear coefficient head

**Idea.** Separate the expensive coordinate network from the changing latent head. Evaluate the spatial bank once at required points, then solve using cached matrices and the small head. The bank is learned rather than obtained from POD.

**What happened.** This removed the spatial network from the inner solve and became the durable main architecture. It also made representation diagnostics clearer: bank capacity and head capacity are separate bottlenecks. A head can only explore fields inside the bank's span. An undersized hidden layer can unintentionally cap bank rank.

**Status.** Core retained architecture. **Code:** [separable](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/separable/sep_common.py). **Design:** [separable EQ decoder](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-08-22-separable-eq-decoder-design.md). **Historical synthesis:** [separable results digest](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-09-03-separable-decoder-results-digest.md).

The first cached implementation explicitly matched the incumbent weak residual and Jacobian. Later exact-linear-term and off-mesh changes must be treated as explicit formulation changes rather than assumed equivalent by architecture alone.

### Head fitting: wider/deeper MLPs, latent Fourier features and staged training

**Idea.** Hold the spatial bank fixed and improve the coefficient map through width, depth, latent dimension, latent features, normalization or optimization changes.

**What happened.** These studies often reduced training error without closing the held-out or solved-error gap. Latent Fourier features performed poorly in the recorded Burgers campaign. Later training-only campaigns did not establish that more head training solved the main cost/accuracy problem.

**Status.** Tested variants of one family, not independent demonstrations that nonlinear heads dominate. **Code:** [headfit](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/headfit/sep_hfit.py). The canonical log's head-fitting, head-ablation and bank/head campaigns record the scope changes and negative gates.

### Exact linear contractions and tensorized Burgers evaluation

**Idea.** Precompute linear projected terms exactly; evaluate nonlinear terms with appropriate cached or tensorized operations. For polynomial structure, precomputed contractions can replace repeated spatial work.

**What happened.** This improved how the same bank/head model was solved. Burgers required care to preserve the discrete upwind operator in the relevant formulation. Numerical speedups from these implementations do not establish a superior decoder representation by themselves.

**Status.** Retained implementation lineage, with subsequent off-mesh formulations separately qualified. **Code:** [tensor](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/tensor/b2d_tensor_common.py). “Tensorized solver” here does not automatically mean the old grid-axis CP decoder.

### Boundary- and field-aware banks: Stokes and wave

**Idea.** Adapt the learned bank and tests to vector fields, incompressibility or a coupled displacement/velocity state, rather than blindly reusing a scalar-field reconstruction loss.

**What happened.** These extensions showed why physical constraints and the actual evolved state matter. Wave reconstruction, velocity scaling, time dependence and physical evolution exposed failures that scalar snapshot scores did not capture. Later fresh-wave verification superseded earlier unsupported wave claims.

**Status.** PDE-specific branches, not a universal extension result. **Code:** [stokes](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/stokes/stk2d_common.py), [wavebank](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/wavebank/wav2d_bank.py), [latent_wave](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/latent_wave/wlat_common.py). See the lab log's Stokes and fresh-wave entries for the admitted scope; do not reuse an old-wave headline without its replacement audit.

### Protected-anchor head

**Idea.** Preserve independent linear latent directions and place nonlinear corrections in an orthogonal complement in field-normalized coordinates:

$$q(z)=Uz+Vf(z),\qquad U^TU=I,\quad U^TV=0.$$

This protects against rank loss of the latent Jacobian under the stated coordinates, but does not guarantee a well-conditioned or accurate manifold.

**What happened.** The geometric protection worked in the bounded Burgers screen; representation and tangent errors did not improve sufficiently and acceptance gates failed.

**Status.** Tested negative for the declared screen. **Code:** [anchor](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/anchor/b3d_arch_anchor.py). **Saved run:** [anchor_numbers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/anchor_numbers/result.json).

### Learned quadratic coefficient head

**Idea.** Replace the MLP coefficient map with learned linear and pairwise latent-product terms inside the same learned bank.

**What happened.** It improved average and median representation and tangent metrics against matched MLP controls, while worsening the worst-case error. It still failed the declared mean and worst-error gates, so it was not promoted to a rollout or cost victory.

**Status.** Partial representation improvement; failed overall screen. **Code:** [quadratic](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/quadratic/b3d_arch_quadratic.py). **Saved run:** [quadratic_numbers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/quadratic_numbers/result.json). This is different from the later classical POD-based quadratic-manifold baseline.

### Shared solution encoder during head training

**Idea.** Organize the latent codes with an encoder trained from solution observations while retaining the same coefficient-decoder function class. The encoder is an offline training choice, not necessarily online work.

**What happened.** Improvements were inconsistent across optimizer repeats. Direct encoder reconstruction was separately worse than fitting the latent coordinates. Representation gates still failed.

**Status.** Tested bounded negative, not evidence that encoders can never help. **Code:** [encoder](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/encoder/b3d_arch_encoder.py). **Saved run:** [encoder_numbers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/encoder_numbers/result.json).

### Smooth mixture of local heads

**Idea.** Blend several coefficient maps using differentiable latent-dependent routing. Do not route using unavailable true family labels.

**What happened.** Routing and diversity controls did not show the declared collapse, but validation mean/worst behavior did not improve enough to pass. More specialists did not establish useful generalization.

**Status.** Tested negative for the matched screen. **Code:** [mixture](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/mixture/b3d_arch_mixture.py). **Saved run:** [mixture_numbers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/mixture_numbers/result.json).

The following common-screen table is recomputed from the final retained fit-error arrays, with the recorded outlier counts checked against those arrays. All arms used the same frozen spatial bank and cohort; the MLP rows are the matched controls. The nonlinear fits remain finite-budget local fits, not globally certified manifold distances. No new head passed the combined admission rule, and this table contains no rollout or timing claim.

| Head | Optimizer seed | Fitted validation mean | Median | Worst | States above 15% | Nonstationary fits |
|---|---:|---:|---:|---:|---:|---:|
| mlp128 | 200 | 5.5137% | 4.6721% | 17.0157% | 5 | 0 |
| mlp128 | 201 | 5.4921% | 4.6544% | 17.1652% | 4 | 0 |
| mlp192 | 200 | 5.6234% | 4.5991% | 18.7348% | 4 | 0 |
| mlp192 | 201 | 5.5568% | 4.7320% | 20.0435% | 3 | 1 |
| anchor | 200 | 6.0720% | 5.1155% | 17.7030% | 5 | 0 |
| anchor | 201 | 5.9485% | 5.0666% | 17.8785% | 6 | 0 |
| quadratic | 200 | 4.6894% | 3.8924% | 20.3053% | 2 | 0 |
| quadratic | 201 | 4.6886% | 3.8873% | 20.2729% | 2 | 0 |
| encoder | 200 | 5.4097% | 4.5502% | 16.1582% | 2 | 0 |
| encoder | 201 | 5.4940% | 4.6763% | 17.6365% | 4 | 0 |
| mixture | 200 | 5.5190% | 4.6529% | 21.2651% | 4 | 0 |
| mixture | 201 | 5.5036% | 4.6197% | 18.1472% | 3 | 0 |

### Modified CP with latent-dependent spatial factors

**Idea.** Retain CP structure, but add latent-modulated corrections to the axis factors themselves. This changes the spatial functions with the latent state, unlike ordinary CP where only channel coefficients change. A zero correction recovers the ordinary CP control.

**What happened.** The bounded Burgers and reflective/absorbing-wave pilot compared CP, modified CP and FiLM with frozen selections. Modified CP did not establish a useful matched-target win; the wave cases exposed severe physical inaccuracies even where numerical stationarity was satisfied.

**Status.** Tested negative for the registered physical targets. **Code:** [modcp](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/modcp/decoders.py). **Evidence:** the completed modified-CP evaluation entry in the preserved lab log and the report/retention paths listed there. Do not confuse finite solver completion with physically accurate evolution.

### Nonlinear head plus nested linear corrections

**Idea.** Keep the learned head as a compact starting manifold, then add selected coefficient-space directions:

$$u=B\bigl(h(z)+C_qy\bigr).$$

Increasing correction rank expands the accessible space without retraining. Some linear correction variables can be eliminated analytically in suitable linear problems.

**What happened.** The correction ladders provided a real post-training accuracy/cost control. In linear heat and wave settings the full learned-bank linear evolution could be both more accurate and cheaper than restricting the solution to a nonlinear head. In nonlinear Burgers, improvement depended on tests, quadrature and solving, and did not automatically beat the best classical solver.

**Status.** Important intermediate method; later paper framing moved toward the bank-prefix knob, although not every historical correction result was removed. **Code:** [correction](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/correction/correction_core.py), [linear_heat](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/linear_heat/run_linear.py). The correction basis is not an additional neural decoder trained online.

### Ordered learned-bank prefixes and free coefficients

**Idea.** Rotate/order a trained bank once and retain a chosen prefix. Solve directly for its coefficients, or retain a head folded into that bank where appropriate.

**What happened.** This produced a practical deployment-time knob and became central in the Burgers campaigns. It also sharpened the interpretation: increasing active bank rank is not the same experiment as increasing nonlinear latent dimension. Strong linear baselines can exploit similar reuse and truncation ideas.

**Status.** Main current Burgers representation path. **Code:** [ordered](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/ordered/qcore.py), [current_step](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/current_step/t2core.py); the retained nonlinear-head path is [current_head](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/current_head/hcore.py). The current fork is the consolidated JCP branch identified in the code index, not frozen `main`.

### Code-free/variable-projection bank training and wider heat banks

**Idea.** Fit a coordinate bank by repeatedly solving for the best coefficients during offline training, rather than asking stored auto-decoder codes to carry every adjustment. Train a head afterward when needed.

**What happened.** The heat bank work repaired representation limitations and obtained useful high-resolution results against named iterative comparators. Direct linear evolution in the learned bank, exact transform methods and coarse-grid controls remained crucial competing paths.

**Status.** Useful bank-training method; favorable high-resolution iterative-FOM ratios are not a general nonlinear-manifold advantage. **Code:** [heat_vp](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/heat_vp/train_vp.py), [banktrain](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/banktrain/train_bank.py). The different PDE trainers should not be assumed identical recipes.

### Masked sparse-output autoencoder baseline

**Idea.** Reproduce an alternative nonlinear-manifold ROM whose decoder's output connections are masked so a small subset of outputs can be evaluated efficiently.

**What happened.** The reproduction gate required corrections before passing as an adapted reproduction. The resulting shared-family comparison belongs in the alternative-decoder record, separate from the project's learned-bank architecture.

**Status.** Implemented and benchmarked baseline; use its lane report for exact comparison conditions rather than treating a successful reproduction gate as method superiority. **Code:** [masked](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/masked/kimae.py). Evidence: the September `nmrom-baselines` closing entry and original branch in the index.

### Classical quadratic manifold

**Idea.** Augment a POD-like linear basis with a fitted quadratic map:

$$u(a)=u_{\mathrm{ref}}+Va+W\,\operatorname{vech}(aa^T).$$

This is fitted with regularized linear algebra rather than training the earlier neural coefficient head.

**What happened.** It improved on matched-dimensional linear controls in the recorded Burgers panel, at added cost, but did not establish superiority over the project's better head configurations or the strong FOM frontier.

**Status.** Useful independently structured nonlinear baseline. **Code:** [qman](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/qman/qman.py). Evidence: the September quadratic-manifold panel in the log. It must remain distinct from the learned-bank quadratic-head screen above.

## Navier–Stokes: constraints and translation change the story

### Fixed-frame incompressible bank and nonlinear head

**Idea.** Build a physically appropriate incompressible representation, then fit a low-dimensional nonlinear head in that bank.

**What happened.** The fixed-frame Navier–Stokes campaign failed its registered representation-advantage gate against POD at matched latent size. Enlarging the data and exploring correction directions did not convert that failed gate into the planned successful main experiment. Exploratory follow-ups remain exploratory.

**Status.** Scoped negative for that family and recipe. **Code:** [ns2d](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/ns2d/ns2d_decoder.py). Evidence: the closed September NS campaign in the canonical log. This is not the current obstacle-flow campaign.

### Shift-aware / moving-frame decoder

**Idea.** Separate bulk translation from changing shape. Evaluate a centered representation at shifted coordinates, with the shift determined by the online physics solve rather than supplied by truth:

$$u(x)=\widetilde u(x-c;z).$$

**What happened.** Centering exposed a much more compact representation for the tested moving structures. The implementation evolved through centered stored bases, explicit shift variables and faster moving-frame solvers. Thus a “shift decoder” result is not automatically a coordinate-network-bank result.

**Status.** Meaningful positive branch on its specified family. **Code:** [shift](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/shift/shift_rom.py). No general result for arbitrary obstacles, multiple interacting structures or out-of-family flows follows.

### Nonlinear head and ordered span in a moving frame

**Idea.** Compare a nonlinear coefficient head with an ordered linear span while solving for the frame shift in both.

**What happened.** In this branch, the head retained useful shape compression and performed well on its held-out family; this is an important counterexample to saying nonlinear heads never help. Speed was compared with named spectral and iterative formulations, and invalid early timing blocks were superseded.

**Status.** Scoped positive result, not transferable unchanged to fixed-frame Burgers. **Code:** [shift_head](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/shift_head/head_rom.py). Evidence: the September shift-head closing entry, including its timing retractions.

### Divergence-free coordinate-network bank from a vector potential

**Idea.** Take the curl of a periodic neural vector potential to form a divergence-free spatial bank, train in the centered frame, and reuse the bank across meshes.

**What happened.** The nonlinear head came close to its bank's representation floor on the tested family, but the learned bank remained less accurate than its POD reference. The original admission targets were revised after the pilot by user decision; the coarse-mesh consistency target was not fully met. Aliasing was a proposed explanation, not an experimentally isolated cause.

**Status.** Qualified moving-frame coordinate-bank result with explicit deviations and open fit/transfer questions. **Code:** [curlbank](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/curlbank/coordnet.py), [banktrain](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/banktrain/train_bank.py). Evidence: the October coordinate-bank closing entry. A shared bank across meshes did not establish a single shared head across meshes.

## October: what improved, what did not, and what is proposed

### Wider, smoother and boundary-compatible banks

**Idea.** Increase bank capacity, penalize derivative error through Sobolev-style training, or improve boundary compatibility and training coverage.

**What happened.** Wider and smoother banks improved representation or derivative diagnostics, but those gains did not reliably carry into solved trajectory error. Later steep-front retraining and stepping/quadrature work did not remove the registered robustness limitation.

**Status.** Real trained variants, with no general larger-bank-is-better conclusion. **Code:** [wide](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/wide/w3d.py), [smooth](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/smooth/smoothtrain.py). The later boundary-compatible variants and outcomes are documented in the [improvement campaign report](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-21-improvement-campaign/final/report.md).

### Off-mesh quadrature and improved time integration

**Idea.** Integrate the continuous learned bank and its derivatives away from the original grid, and improve the time discretization/solver.

**What happened.** These changes produced important accuracy gains and qualified mesh-independent reduced costs. They are mostly changes to the equations and their evaluation, not new neural decoder families. The off-mesh analytical-advection formulation is not identical to the earlier discrete upwind target; the distinction must be explicit.

**Status.** Main retained method development, with conditional reference and quadrature qualifications. **Code:** [ordered](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/ordered/qcore.py), [current_step](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/current_step/t2core.py). The [full Burgers campaign report](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-17-full-experiment-plan/final/report.md) and [refinement report](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-26-burgers3d-refinement/2026-10-27-burgers3d-refinement.md) supersede earlier broad speed claims.

### Continuous POD and neural-operator controls

Continuous/interpolated POD, hyper-reduced POD, FNO, U-Net, DeepONet/POD-DeepONet and Transolver belong in the comparison history. Most are **comparators**, not successive versions of our decoder. A direct neural operator predicts a solution from problem inputs; a manifold ROM solves physics in a learned representation. Their output and timing contracts must match before comparing them.

The later multi-mesh experiment found that a continuous POD could also be reused across meshes, withdrawing the comparative multi-mesh cost claim. Stronger operator and FOM comparisons also overturned simplistic claims that competing learned methods lack knobs or that a same-grid iterative FOM is the decisive comparator. **Representative operator code:** [operators](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/operators/ops3d.py). **Evidence:** full and improvement campaign reports above, plus [old_readme](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/old_readme/README.md) for the historical operator variants. This report does not reproduce every operator architecture's training package.

### Same-sample reconstruction against classical interpolation and smoothing

**Question.** Does the learned bank reconstruct more accurately when classical alternatives receive comparable sampled information?

**What happened.** The final exposed-cohort screen did not pass either larger-campaign accuracy gate. A narrower comparison with coarse linear interpolation was more favorable than the comparison with the selected stronger classical methods. The strict coarse-data control did not restore the broader advantage. An initial filename-collision result was retracted and excluded.

**Status.** Accepted negative screen with conditional reference and data-provenance qualifications; not a new trained decoder. **Evidence and code package:** [same-sample reconstruction report](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-27-burgers-reconstruction/2026-10-27-burgers-same-sample-reconstruction.md).

### Prefix coarse-to-fine initialization

**Idea.** Use a smaller prefix of the ordered bank to initialize the larger-bank solve, including affine and pure-prefix transfer variants.

**What happened.** Both bounded initialization arms failed the cost screen against the unchanged temporal-warm fine solver. The campaign stopped before stronger accuracy qualification and expansion.

**Status.** Scoped negative for this initialization policy, not a rejection of all multilevel methods. **Code:** [prefix](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/prefix/prefixcore.py). Evidence: October prefix-pilot closing entry and its original branch checks.

### Separate-decoder latent continuation

**Idea.** Transfer between distinct smaller and larger latent heads, optionally taking coarse physics corrections before returning to the fine head.

**What happened.** Both the physics-assisted and transfer-only variants added substantial overhead without reducing fine corrections enough. The fresh stationarity and reproduction checks passed, but the cost gate failed. Strong reference-based accuracy checks were not run after the stop.

**Status.** Completed bounded negative pilot; no retraining, repeated correction-cycle campaign or higher-dimensional expansion followed. **Code:** [transfer](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/transfer/transfer.py). **Generated measurements:** [latent_numbers](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/latent_numbers/summary.json).

| Arm | Median query (ms) | Median paired time ratio to direct | Slower timing pairs / all pairs | Fine corrections total | Cost gate |
|---|---:|---:|---|---:|---|
| direct | 35.28064 | 1.000000 | 0 / 42 | 405 | baseline |
| physics | 120.56014 | 3.376892 | 42 / 42 | 407 | fail |
| transfer | 94.02371 | 2.616369 | 42 / 42 | 405 | fail |

These are paired timings from the same allocation and baseline, not speed ratios assembled from different jobs. A ratio above unity means slower than direct. The table measures this bounded initializer's cost; it does not establish physical error, optimal manifold distance or a general multilevel impossibility.

### Obstacle-flow ROM — status at the historical cutoff

The current obstacle campaign is distinct from the older periodic/moving-frame NS examples. It has been repairing and qualifying geometry, normals, reference solves, quadrature, domain and memory/assembly behavior. At this report's lab-log snapshot, training, a qualified reduced solve and strong matched-output comparisons remained outstanding.

**Status.** Active pipeline/reference qualification. Do not present its capacity probes as a trained decoder or a successful end-to-end ROM. Source location: `exp/2026-10-27-ns-obstacle-rom/experiments/` with the coordinator and dimension-specific worktrees recorded in the canonical log. This report deliberately does not pin a changing branch tip as a finalized scientific result.

### Proposed global-plus-local tree bank

**Idea.** Keep a global coarse bank and add compactly supported local correction banks on active patches. Adapt global rank, local rank and patch refinement while solving all active coefficients through the coupled physics equations:

$$u(x)=u_{\mathrm{bc}}(x)+B_0(x)c_0+
\sum_{P\in\mathcal A}w_P(x)B_P(T_P^{-1}x)c_P.$$

**What is new.** Allocate representational capacity where it helps instead of only increasing a global bank prefix. Coordinate evaluation at selected points is already available in the current method; that alone cannot be the new advantage.

**Status.** Proposed and reviewed, not trained or benchmarked. **Plan:** [tree_plan](https://github.com/tahmidawal/Tunable-NM-ROM/blob/dd6c6c0931dbbb21f3da39a940d2df90781e1a49/reports/2026-10-28-decoder-history/sources/tree_plan/PLAN.md). The plan begins with local POD/polynomial representation controls, then a stable fixed-space physics solve and an audited adaptation policy, before neural feature training. No numerical result or speed promise should be attached to it yet.

## What the whole history says

### What we should retain

- **Coordinate queryability and caching.** They enable mesh-free evaluation and reduced kernels whose cost can be independent of the requested output mesh. Full-field output still costs work proportional to what is requested.
- **Separating the bank from the head.** Measure the bank's best fit, the head's fitted error and the actual solved error separately. This makes it possible to see which part needs work.
- **Direct ordered-bank solving as a serious option.** It is not a failed nonlinear decoder; it is a legitimate linear trial representation for a potentially nonlinear PDE.
- **Moving frames where the physics warrants them.** Translation is a specific structural difficulty with a successful scoped response in the historical NS branch.
- **Strong classical comparisons.** Tuned POD, coarse grids and spectral/transform methods repeatedly changed the interpretation of apparent wins.

### What we should stop treating as evidence

- Low autoencoder error as proof of stable latent evolution or a reliable cold solve.
- More bank features, head width or training as proof that the online solution improves.
- A small final latent gradient as proof of small physical error.
- Better mean error with unreported catastrophic cases as an overall improvement.
- Flat inner-kernel time as proof of flat full-query time or superiority to the best FOM.
- A historical speedup without its reference tolerance, output contract, GPU pairing and baseline definition.
- Code existence, a smoke test or a capacity calculation as proof that an architecture passed an experiment.

### What now — recommended interpretation and next sequence

**For the current research claim:** use the qualified bank/solver results to describe mesh-independent reduced computation, coordinate evaluation, memory behavior and conditional accuracy improvements. Keep narrow-family speed results narrow. Do not lead with a general neural speed advantage that the stronger Burgers comparisons do not support.

**For an architecture change:** the tree plan is a plausible next question because it changes where capacity is allocated. First establish that local approximation actually improves the error-versus-active-size tradeoff on the intended data, including simple classical local bases. If it does not, neural training is unlikely to rescue the premise. If it does, test whether a stable coupled physics solve realizes the gain and whether selection, rejected trials and audits erase the savings.

**For obstacle flow:** finish the existing reference and end-to-end validation obligations before claiming a decoder result. Do not infer success from periodic translation examples.

**For the recent multilevel negatives:** preserve the early stops. Another initializer experiment should specify the mechanism expected to reduce expensive fine work; simply adding transfer steps repeats a cost already measured to be unhelpful in the tested policies.

These are recommendations from the historical evidence, not authorizations or commitments to run new experiments. This report creates no new experimental branch and merges none of the existing lanes.

## Retractions and qualifications to carry forward

| Historical claim or shortcut | Current reading |
|---|---|
| Original heat public rollout as a baseline | Known broken behavior; use corrected experiment revisions. |
| May SIREN/affine speed headlines | Historical named-CG comparisons, not re-certified under later fairness rules. |
| May best-cycle FAS accuracy | Truth-assisted best-cycle harvesting is not a deployable stopping rule. |
| FiLM supplied-family reconstruction as ROM performance | Representation/surrogate evidence until a latent PDE solve is actually performed. |
| Coarse-grid Fourier success measured only on-grid | Aliased versions were superseded; off-grid validation is necessary. |
| Old strong-form/staged-training precision as a ROM guarantee | Objective and provenance corrections limit the claim; parametric success was not established. |
| Old wave claims reused in later work | Follow fresh verified-wave evidence and explicit retractions. |
| Nonlinear skip guarantees full rank | Not for an unrestricted additive correction; protection requires additional geometry. |
| Favorable average head error | Include median, worst case, failures and the actual admission rule. |
| Large speedup against fixed-iteration or over-solved FOM | Compare with a tolerance/output-matched, tuned alternative. |
| Old POD nonconvergence or superiority statements | Some failures were solver/stopping artifacts; later tuned controls supersede them. |
| One bank across meshes proves a unique multi-mesh advantage | Continuous POD reuse withdrew the comparative claim. |
| Off-mesh accuracy is only a quadrature effect | Exact derivatives and changed operator targets matter; later references also changed. |
| October reconstruction development attempt | Filename-collision attempt excluded; use the final accepted screen. |
| Failed prefix/latent initializer proves all multilevel ideas fail | Only the bounded policies were tested; stronger accuracy stages were not run. |
| Tree plan or obstacle capacity work proves a successful new decoder | Neither has an admitted decoder performance result in this snapshot. |

## Source and reproduction notes

The collaborator PDF is built by `export_pdf.py`, using the simpler opening and dated update beside the original report. The GitHub links point to committed evidence; the companion GitHub code index provides original branch and file locations. The paragraphs below describe how the underlying historical report and its numerical tables were generated.

The editable narrative is `report.source.md`; `build.py` produces `2026-10-28-decoder-history.md`, the code index and the coordinate-search CSV. Run it using the repository's absolute Python interpreter path. It imports no JAX and performs no GPU work.

`sources.json` records the original locations, pinned commits where available, local reading copies and SHA-256 hashes. The first build pins sources; subsequent builds verify and reuse those snapshots rather than silently refreshing history. The coordinate table is generated from the architecture-study JSON. The head table recomputes descriptive statistics from saved fit arrays. The recent continuation table is generated from its retained summary JSON. No numerical table is manually transcribed from log prose. The accompanying raw head-screen JSONs let a reader inspect the state-level evidence without relying on this narrative.

This is a source-grounded historical review, not a rerun of all experiments or an independent re-audit of every raw array. Where the canonical log identifies accepted audits, the report relies on those scoped conclusions; where it records a limitation, the limitation remains. Dates identify recorded work, not newly generated measurement dates.

## Plain-language glossary

**Decoder:** map from a compact set of coordinates to a field such as temperature or velocity. **Encoder:** map from an observed field to compact coordinates. **Autoencoder:** encoder and decoder trained together. **Auto-decoder:** decoder trained with fitted latent codes, without requiring a learned encoder. **Latent code / $z$:** compact unknowns used by a nonlinear decoder. **Head:** network mapping latent codes to bank coefficients. **Bank / basis:** collection of spatial functions used to assemble a field. **Prefix / ordered bank:** the first chosen functions from a fixed ordering. **Span:** all linear combinations of a set of functions. **Manifold:** fields reachable through the decoder's generally nonlinear coordinate map.

**CP:** sum of separable products of axis factors. **LinearCP:** historical CP model with a linear skip plus a nonlinear coefficient head. **MLP:** ordinary layered fully connected neural network. **FiLM:** feature-wise scaling and shifting controlled by another input. **SIREN:** coordinate network using sine activations. **INR:** implicit neural representation, a network queried by coordinates. **Fourier features:** sinusoidal coordinate inputs. **ViT:** vision transformer. **Cross-attention:** queries using information from a separate token collection. **Protected anchor:** independent linear latent directions insulated from nonlinear corrections by an orthogonal construction. **Mixture head:** several maps combined by a learned routing function. **Quadratic manifold:** representation containing pairwise products of reduced coordinates. **Sobolev training:** training that also measures derivative errors. **Variable projection:** optimizing nonlinear features while solving their linear coefficients inside the training objective.

**PDE:** partial differential equation governing the field. **FOM:** full-order numerical PDE solver. **ROM:** reduced-order solver using fewer unknowns. **NM-ROM:** ROM using a nonlinear trial manifold. **POD:** proper orthogonal decomposition, a data-derived linear basis optimized for a specified snapshot norm. **POD hyper-reduction:** approximating expensive operations in a POD model using a smaller selected set of evaluations. **LSPG:** least-squares Petrov–Galerkin, choosing reduced coordinates by minimizing a tested/discrete residual. **Weak tests / $M$:** smooth functions used to turn field residuals into a smaller system. **Quadrature / $m$:** points and weights approximating an integral or projected operation. **EQ:** empirically fitted quadrature. **NNLS:** nonnegative least squares, used to fit nonnegative quadrature weights. **Tensor contraction:** combining multidimensional arrays by summing shared indices. **FAS:** full approximation scheme, a multilevel correction method for nonlinear equations.

**$K$:** dimension of a nonlinear latent code. **$R$:** full spatial-bank width. **$r$:** active bank-prefix width. **$q$:** number of added correction directions. **$C_q$:** matrix defining those directions. **Boundary lift / mask:** construction that supplies or enforces boundary values. **Moving frame / shift:** describing a translating structure relative to a moving origin. **Curl / divergence-free:** a vector-potential construction that enforces an incompressibility constraint in its stated representation. **Quadtree / octree:** hierarchical subdivision of space into smaller patches. **Local bank:** features supported on a patch. **Polynomial control:** simple non-neural features used to test whether localization itself helps.

**Fitted error / oracle diagnostic:** error after fitting reduced coordinates to a known field; a finite nonlinear fit is not automatically the globally best possible error. **Bank floor:** best-approximation diagnostic in the specified bank and norm; its meaning depends on how the projection is computed. **Full weak error:** error after solving with the unsampled weak objective. **EQ error:** error after solving with the quadrature-approximated objective. **Relative field error:** discrepancy divided by the stated reference norm. **Mean / median / worst:** average, middle value and largest value over the named population. **Seed:** recorded random-number initialization. **Parameters:** number of trainable model entries in the coordinate table. **Cell:** exact archived experiment/configuration identifier. **Gate:** predeclared acceptance test. **Stationarity:** the optimizer has a small first-order residual/gradient by its stopping criterion, not necessarily a physically correct solution. **Cold start:** solve without a nearby previous solution. **Warm start:** initialize from a previous or approximate solution. **Censoring:** reaching a computational cap before the requested stopping condition. **Tangent error:** failure of the decoder's local derivative directions to represent the desired field direction.

**Query time:** cost of the particular experiment's requested solve/output contract. **Paired time ratio:** method time divided by baseline time for paired observations from the same experiment; above unity is slower. **Slower pairs / all pairs:** count of timing comparisons in which the method was slower, and number compared. **Fine corrections total:** sum of fine-solver correction iterations across the pilot cases. **Transfer-only:** move between coarse and fine representations without the added coarse physics correction. **Physics arm:** transfer with bounded coarse physics correction. **Direct:** unchanged temporal-warm fine solver. **Admission / qualification:** passing stated numerical checks within a specific scope. **Held-out:** not used for fitting that assessed model. **Sealed:** deliberately withheld until a frozen decision rule is ready. **Exposed / development:** already examined and therefore available for method selection. **Conditional reference:** a reference whose error interpretation depends on specified assumptions or checks. **Retraction / superseded:** a claim withdrawn or replaced by better evidence. **Provenance:** record of which source, data, configuration and hardware produced an artifact. **SHA-256:** content fingerprint used to detect changed bytes.
