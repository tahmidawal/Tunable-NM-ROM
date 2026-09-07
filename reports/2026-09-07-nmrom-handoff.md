# NM-ROM handoff: separable decoders, Burgers 3D and fresh waves

Dated handoff snapshot for the results finalized on 2026-09-07. The completed architecture screens and fresh-wave results are independently reviewed, bounded validation evidence; the engineering acceptance targets remain provisional and no tested compressed head passes its full target. This handoff adds no numerical experiment.

**Read [LAB-LOG.md](../LAB-LOG.md) first in the next session.** It remains the canonical record, including retractions and later changes. This document is a fixed transfer note, not a second mutable project-status file.

## What the next person needs to know

- The separable architecture is implemented: a learned coordinate network supplies spatial functions, and a small latent head supplies their coefficients. The spatial bank supports substantially better representation than the tested compressed heads.
- Burgers 3D: the quadratic head is the strongest mean-reconstruction improvement in the matched screen, but unseen initial states still produce large errors. Representation gates block promotion to rollout and cost claims.
- Fresh reflective and absorbing waves: the full-bank linear model meets the declared initial-normalized physical-error ceilings. Every tested compressed head misses the full target. Good snapshot fitting and energy balance do not establish accurate evolution.
- All earlier wave experiments are excluded from trusted evidence by the user. The fresh benchmark, its verification and its checkpoints are the only wave lineage to continue.
- Wave dynamics are currently scalar and two-dimensional. Three-dimensional waves and Navier–Stokes remain future extensions; surface plots do not change the PDE dimension.

## User intent and decisions to preserve

The user wants the NM-ROM framework to transfer across PDEs and dimensions, starting with the current smooth/localized data style. The immediate sequence includes Burgers 3D, ordinary and reflective waves, and later incompressible Navier–Stokes. Broader initial-condition families are a later question. Do not condition the neural head on Gaussian centers, widths or other family descriptors. A Gaussian-like data generator is allowed; a family-specific online parameter fit is not the selected direction. Separate trained weights per PDE/boundary are the tested scope; unchanged-weight transfer is unproven.

The most recent visualization wording was “Can I not get it in images please.” This was interpreted as requesting still images showing successive evolution times. That follow-up was interrupted by this handoff request, so a newly arranged still-image sequence remains open. Existing static field grids, GIFs and an interactive time viewer are already saved below.

## Architecture and mathematical contract

The general decoder is $u(x;z)=b(x)\sum_{j=1}^{r}g_j(x)h_j(z)$. The factor $b$ enforces the appropriate essential boundary constraint; $g$ is a trained coordinate network, $h$ is the coefficient head, and $z\in\mathbb{R}^{k}$ is the state solved for online. The coordinate network is independent of the latent code, so spatial operators can be assembled or projected offline. The bank is learned; POD appears only as an explicit baseline. Mass-weighted QR changes bank coordinates without changing its span. Coefficient PCA initializes the heads and codes; it does not replace the coordinate network with POD.

Earlier non-wave implementation includes weak residuals, empirical quadrature for nonlinear terms, exact reduced linear operators, and tensor contractions on applicable PDEs. That lineage is in the consolidated branch and canonical log. Its older cost numbers are not evidence of speed for this Burgers 3D screen or the fresh wave campaign.

For waves, displacement and velocity share a consistent latent state:

$$u=Gh(z),\qquad v=GJ_h(z)w,\qquad \dot z=w,$$

$$\dot w=\underset{a}{\operatorname{argmin}}\,\left\|J_h(z)a+H_h(z)[w,w]+D_rJ_h(z)w+K_rh(z)\right\|_2^2.$$

Here the mass-orthonormal bank is fixed, the reduced stiffness and boundary damping operators are fixed for each physical case, and the Hessian term accounts for curvature of the decoder. The least-squares solve uses QR without ridge regularization; each Runge–Kutta stage has rank and finite-value guards. Velocity is the Jacobian lift of latent velocity, not an independently decoded field. The full-bank linear comparator is propagated independently by a matrix exponential.

The reference satisfies $M\dot v+Cv+Ku=0$, $\dot u=v$, with $E(u,v)=\tfrac12(v^TMv+u^TKu)$ and $\dot E=-v^TCv$. Mass uses tensor-product trapezoid weights; stiffness uses edge differences; absorber damping includes every boundary face and corner contribution. Fixed Dirichlet walls cause sign-inverting reflection. The first-order Sommerfeld absorber is approximate, especially for oblique incidence. Its boundary-model reflection is separate from discretization error. Its constant-displacement nullspace also requires an independent mean-field check.

## Burgers 3D: completed representation screen

The fixed bank uses 33 grid nodes per axis, 128 spatial functions and 32 latent coordinates. Each matched repeat uses 60000 updates, learning rate 0.0003, batch 4096, 8192 training states and 256 validation states. The unrestricted bank projection has mean 2.182214% and worst 8.013366% relative field error.

The preliminary mean ceiling is 4.305823% from the inherited POD comparison; the worst-state ceiling is 15.000000%. These remain provisional promotion criteria: even a screen pass requires the full inherited pilot, including negative controls and a recomputed POD comparator.

Errors below are per-state relative field L2 reconstruction errors after multistart latent fitting, summarized across validation states. They are not rollout errors. Percent displays are rounded; linked JSONs retain full precision.

| Head | Seed | Mean % | Median % | Worst % | Outliers | Nonstationary fits |
|---|---|---|---|---|---|---|
| MLP 128 | 200 | 5.513715 | 4.672066 | 17.015734 | 5 | 0 |
| MLP 128 | 201 | 5.492084 | 4.654377 | 17.165202 | 4 | 0 |
| MLP 192 | 200 | 5.623399 | 4.599069 | 18.734846 | 4 | 0 |
| MLP 192 | 201 | 5.556837 | 4.731979 | 20.043463 | 3 | 1 |
| Protected anchor | 200 | 6.072002 | 5.115527 | 17.703013 | 5 | 0 |
| Protected anchor | 201 | 5.948452 | 5.066594 | 17.878533 | 6 | 0 |
| Quadratic | 200 | 4.689399 | 3.892390 | 20.305274 | 2 | 0 |
| Quadratic | 201 | 4.688630 | 3.887288 | 20.272908 | 2 | 0 |
| Shared encoder | 200 | 5.409712 | 4.550165 | 16.158243 | 2 | 0 |
| Shared encoder | 201 | 5.494040 | 4.676315 | 17.636507 | 4 | 0 |
| Smooth mixture | 200 | 5.519008 | 4.652903 | 21.265062 | 4 | 0 |
| Smooth mixture | 201 | 5.503608 | 4.619719 | 18.147213 | 3 | 0 |

Quadratic reduces mean reconstruction error by 14.629309%–14.950290% and mean tangent error by 17.715947%–18.147681% relative to the matched narrow MLP. These are relative reductions, not percentage-point changes. It still fails both mean and worst-state criteria. Its large outliers are unseen initial states, so a rollout cannot be rescued merely by giving it an inaccurate starting representation.

The protected anchor achieves its intended geometry but worsens accuracy. Shared encoder training gives inconsistent gains, and direct encoder output has an additional reconstruction gap. The smooth mixture improves training and tangent fitting without improving validation mean or worst error; the recorded collapse tests do not explain its failure. The wider MLP is also not an improvement. These findings are limited to the tested forms and budget.

Optimizer repeats share a data cohort. Quadratic initialization is deterministic; its repeats vary minibatch randomness. Local stationarity is not proof of globally optimal fitting. The earlier warm-refined incumbent had a different training history and must not be presented as a matched architecture control. No candidate is promoted to a Burgers 3D rollout, final test or cost claim.

Detailed state/tangent/geometry diagnostics and source-job provenance: [B3D-ARCH-NOTES.md](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/B3D-ARCH-NOTES.md). Independent campaign audit: [campaign-audit.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/review/campaign-audit.json).

## Fresh waves: verified reference, failed compressed-head target

The first fresh verification attempts, `verify01` and `verify02`, failed their resolution gates and remain historical failures. Before any scientific neural comparison, the family was changed to a Gaussian core with a smooth compact cutoff and narrower center coverage. No old wave code, bank, checkpoint, data or gate was inherited as trusted infrastructure.

Reference `verify03`, job `3338045`, source `c69bf87439fd2d0ad1673c852e95ec0f322b8c26`, passed 334 declared gates and independent review. Its acceptance is bounded to the recorded family and mesh. Absorber reference uncertainty is empirical/conditional, not a uniform theorem.

| Frozen setting | Value |
|---|---|
| Mesh intervals per axis | 256 |
| Learned bank / latent / weak test dimensions | 64 / 16 / 64 |
| Training / validation / sealed final trajectories | 64 / 16 / 16 |
| Training / validation / sealed final seeds | 690601 / 690602 / 690603 |
| Optimizer seeds | 691200, 691201 |
| Final time / saved-time interval | 2.4 / 0.05 |
| Primary time step | 0.0025 |
| Original time-step ladder | 0.005, 0.0025, 0.00125 |
| Bank / head training updates | 6000 / 10000 |
| Gaussian width range | [0.12, 0.16] |
| Compact support half-width range | [0.36, 0.42] |

The provisional engineering target requires every validation trajectory to complete with displacement, velocity and energy-state errors at most 10.000000%, with finest-step-pair differences at most 1.000000% on each physical scale. Selected latent fits must also satisfy stationarity, rank and doubled-budget stability gates. These are validation results; the final cohort is still sealed.

### How to read the wave errors

Let $\delta u=u_{\rm ROM}-u_{\rm ref}$, $\delta v=v_{\rm ROM}-v_{\rm ref}$, $U_0=\|u_{\rm ref}(0)\|_M$, $E_0=E(u_{\rm ref}(0),v_{\rm ref}(0))$, and $V_0=\sqrt{2E_0}$. The stored errors are

$$e_u(t)=\frac{\|\delta u(t)\|_M}{U_0},\qquad e_v(t)=\frac{\|\delta v(t)\|_M}{V_0},\qquad e_E(t)=\sqrt{\frac{E(\delta u(t),\delta v(t))}{E_0}}.$$

**Every rollout table below summarizes each trajectory’s maximum over saved times, then takes the mean, median and worst across trajectories.** It is not a time average or a continuous-time supremum. The energy-state metric is the energy norm of the state difference, not the difference between two solution energies. Normalization is fixed by the initial state.

Instantaneous relative displacement error instead divides by $\|u_{\rm ref}(t)\|_M$. An absorbing field can become very small, making that ratio large while the initial-normalized error is small. The illustrated case below demonstrates the distinction; it is not a cohort summary.

| Boundary | Case | Seed | Time | Error / initial norm % | Error / current norm % |
|---|---|---|---|---|---|
| Reflective | 0 | 691200 | 2.4 | 11.978251 | 26.975450 |
| Absorbing | 0 | 691200 | 2.4 | 0.434964 | 227.007777 |

### Fresh linear baselines

This table and the original nonlinear rollout table report the energy-state error $e_E$. The complete wave report also gives displacement and velocity errors separately.

| Boundary | Linear model | Displacement dimension | Mean % | Median % | Worst % | Energy outliers |
|---|---|---|---|---|---|---|
| Reflective | Fresh POD | 16 | 25.383938 | 24.546967 | 42.849892 | 16 |
| Reflective | Fresh POD | 64 | 1.242758 | 1.322470 | 2.253366 | 0 |
| Reflective | Learned bank | 64 | 3.775129 | 3.499136 | 7.706046 | 0 |
| Absorbing | Fresh POD | 16 | 17.071857 | 16.601976 | 24.546187 | 16 |
| Absorbing | Fresh POD | 64 | 1.436503 | 1.413507 | 2.523749 | 0 |
| Absorbing | Learned bank | 64 | 3.166651 | 3.078373 | 4.627047 | 0 |

The full learned-bank model has 128 displacement-plus-velocity state coordinates versus 32 for the compressed wave heads. It meets all declared initial-normalized physical-error ceilings, but this dimension mismatch leaves compression and nonlinear dynamics coupled. The fresh POD baseline at the small dimension also struggles. Neither comparison proves that a nonlinear head cannot work.

### All original nonlinear rollout results

| Boundary | Head / objective | Seed | Mean % | Median % | Worst % | Energy outliers | Original time check |
|---|---|---|---|---|---|---|---|
| Reflective | MLP | 691200 | 52.043064 | 52.446276 | 88.221576 | 16 | pass |
| Reflective | Quadratic | 691200 | 94.709984 | 93.365144 | 125.285284 | 16 | unresolved |
| Reflective | MLP + velocity | 691200 | 50.232241 | 47.239933 | 85.597601 | 16 | pass |
| Reflective | Quadratic + velocity | 691200 | 99.154972 | 99.077625 | 140.688866 | 16 | unresolved |
| Reflective | MLP | 691201 | 53.336116 | 46.880293 | 90.439581 | 16 | pass |
| Reflective | Quadratic | 691201 | 103.349507 | 106.953933 | 143.316837 | 16 | unresolved |
| Reflective | MLP + velocity | 691201 | 48.733558 | 40.290811 | 104.252782 | 16 | pass |
| Reflective | Quadratic + velocity | 691201 | 101.415694 | 105.866991 | 137.269932 | 16 | unresolved |
| Absorbing | MLP | 691200 | 10.509178 | 8.445702 | 21.006954 | 6 | pass |
| Absorbing | Quadratic | 691200 | 12.862977 | 10.279752 | 22.206066 | 8 | pass |
| Absorbing | MLP + velocity | 691200 | 10.461219 | 8.533811 | 21.918079 | 5 | pass |
| Absorbing | Quadratic + velocity | 691200 | 22.000166 | 18.375950 | 50.242536 | 15 | pass |
| Absorbing | MLP | 691201 | 10.900708 | 8.555043 | 22.105436 | 6 | pass |
| Absorbing | Quadratic | 691201 | 12.490009 | 10.332178 | 22.410752 | 9 | pass |
| Absorbing | MLP + velocity | 691201 | 9.717106 | 8.281624 | 20.148499 | 4 | pass |
| Absorbing | Quadratic + velocity | 691201 | 17.416324 | 17.705120 | 28.670165 | 14 | pass |

**Full target passes — Reflective: 0/8; Absorbing: 0/8.** All original trajectories completed. All MLP runs and absorbing quadratic runs pass their original time-step checks, so their large errors cannot be dismissed as the observed time-step failure affecting reflective quadratic runs. Velocity-tangent training improves fitting diagnostics without consistently improving actual evolution.

MLP has 28032 head parameters; quadratic has 9792. Equal update counts do not make this a parameter- or compute-matched comparison. Optimizer repeats share the same data split. Nonstationary snapshot fits are retained as gate failures in the full report; later snapshot fits are not the initial fits used to start the rollout.

### Frozen-checkpoint continuation of reflective quadratic runs

Follow-ups retained each original bank, head, physical operators and stored initial latent position and velocity. They repeated the original finest step for hardware parity and then reduced the step further. No retraining, refitting or final-test opening occurred; the original primary-step verdict remains unchanged.

| Head / objective | Seed | Job | Finest-pair passes | Unresolved case | Finest-step energy median % | Worst % | Energy outliers |
|---|---|---|---|---|---|---|---|
| Quadratic | 691200 | 3339304 | 15/16 | 2 | 93.368538 | 125.964388 | 16 |
| Quadratic + velocity | 691200 | 3339553 | 16/16 | none | 99.078603 | 141.386389 | 16 |
| Quadratic | 691201 | 3339615 | 16/16 | none | 106.954021 | 143.312854 | 16 |
| Quadratic + velocity | 691201 | 3339647 | 16/16 | none | 105.895201 | 136.927990 | 16 |

The added steps are 0.000625, 0.0003125. Every temporally resolved case still misses accuracy. The remaining case must stay labeled unresolved; its error cannot be attributed entirely to architecture. The first continuation regenerated normalization scales with roundoff differences; the independent audit recomputed metrics with the original scales. Later continuations used the stored scales exactly.

## Saved code, evidence and visualizations

The complete wave write-up is [2026-09-07-fresh-wave-head-transfer.md](../reports/2026-09-07-fresh-wave-head-transfer.md). It contains all displacement, velocity, representation, phase, fitting, reference-uncertainty and refinement tables omitted from this compact handoff.

| Location | Contents |
|---|---|
| [Fresh wave implementation](../worktrees/2026-09-06-wave-head-transfer/experiments/fresh-wave-head) | `fresh_fom.py`, `fresh_models.py`, `fresh_learning.py`, `fresh_rom.py`, `fresh_evaluate.py`, `fresh_campaign.py` and `fresh_checkpoint_refine.py` |
| [Frozen campaign config](../worktrees/2026-09-06-wave-head-transfer/experiments/fresh-wave-head/campaign-config.json) | Family, seeds, schedules, gate definitions; use with `FROZEN-MATH.json` |
| [Burgers architecture implementation](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder) | `b3d_arch_bench.py`, `b3d_arch_baseline.py`, `b3d_arch_pilot.py`; candidate implementations/checkpoints live in their owning worktrees |
| [Complete fresh-wave campaign archive](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign) | Submissions, outputs, source/checksum manifests, cleanup receipts and movie export |
| [Independent review evidence](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/review) | Model/operator/full-field/refinement audit scripts and JSONs; final review disposition |
| [Reflective still-image grid](../reports/figures/2026-09-07-fresh-wave-reflective-displacement-fields.png) | Reference, full-bank linear, MLP and absolute-error fields at saved times |
| [Absorbing still-image grid](../reports/figures/2026-09-07-fresh-wave-absorbing-displacement-fields.png) | Same layout; velocity grids and PDF exports are beside these files |
| [Interactive wave evolution](../reports/animations/2026-09-07-wave-evolution.html) | Standalone play/pause, time slider, boundary/view and playback-speed controls |
| [Reflective animation](../reports/animations/2026-09-07-reflective-wave-surface.gif) / [Absorbing animation](../reports/animations/2026-09-07-absorbing-wave-surface.gif) | Reference and saved MLP prediction, fixed amplitude scales; top views also available |

Movies contain actual saved-time observations, without temporal interpolation. The same selected validation case is used for both boundaries. Movie export regenerated only the reference fields and checked parity against the original snapshots and rollout metrics. It did not train a model or recompute a ROM.

Historical validation completed before this handoff: Burgers benchmark, architecture-component and actual-head integration checks; wave FOM, model, learning and evaluation component checks; cluster preflights; independent raw-array and provenance audits; GIF frame decoding and browser-control checks. This documentation session only regenerates tables and checks document inputs/links. It does not rerun numerical tests.

### Exact primary result files

- Burgers `mlp_control`, job `3332190`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp128/out/result.json).
- Burgers `mlp_control`, job `3332191`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp192/out/result.json).
- Burgers `protected_anchor`, job `3336232`: [result.json](../worktrees/2026-09-06-b3d-anchor/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Burgers `quadratic_head`, job `3336338`: [result.json](../worktrees/2026-09-06-b3d-quadratic/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Burgers `shared_encoder`, job `3336240`: [result.json](../worktrees/2026-09-06-b3d-encoder/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Burgers `smooth_mixture`, job `3336230`: [result.json](../worktrees/2026-09-06-b3d-mixture/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Waves `reflective01`, job `3338483`, NVIDIA A100-PCIE-40GB, source `fdcc6491657363005cc9060a0cae21dac320a974`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/reflective01/cluster/out/campaign/result.json).
- Waves `absorbing02`, job `3338658`, NVIDIA H100 PCIe, source `fdcc6491657363005cc9060a0cae21dac320a974`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/absorbing02/cluster/out/campaign/result.json).
- Continuation `refquad20001`, source `4b2d71741f2829451260e1172fd6ab84edd2555f`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/refquad20001/cluster/out/refinement/result.json).
- Continuation `refquadvel20002`, source `1e4a489882c8181d56966be87cae01539e4150a3`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/refquadvel20002/cluster/out/refinement/result.json).
- Continuation `refquad20101`, source `1e4a489882c8181d56966be87cae01539e4150a3`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/refquad20101/cluster/out/refinement/result.json).
- Continuation `refquadvel20101`, source `1e4a489882c8181d56966be87cae01539e4150a3`: [result.json](../worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign/refquadvel20101/cluster/out/refinement/result.json).

Native wave checkpoints are tracked on the wave branch. Complete checked cluster archives and root audits are tracked on the repair branch. Architecture checkpoints remain with their owning candidate branches. All completed campaign directories were checksum-pulled and removed from the cluster; local archives are the recovery source.

## Branch map and ownership at handoff

Canonical documentation is on `main`, which was at `1e1a3503a1a1f9e4ec8e2387187aabece8322228` immediately before this handoff. Main is the frozen public scientific baseline with the known broken heat rollout; do not use it as an experiment base. Branch hashes below identify the saved state before this documentation addition, not floating future tips.

| Worktree under repository root | Saved commit | Role |
|---|---|---|
| [2026-09-04-separable-tensor-consolidated](../worktrees/2026-09-04-separable-tensor-consolidated) | `da479125b13a` | Consolidated earlier non-wave lineage; contains excluded historical wave material |
| [2026-09-06-burgers3d-repair](../worktrees/2026-09-06-burgers3d-repair) | `3cfd5f64fa18` | Root coordinator, controls, complete archives, independent audits and plotting |
| [2026-09-06-wave-head-transfer](../worktrees/2026-09-06-wave-head-transfer) | `906cbe6f6f9f` | Fresh wave science implementation, native outputs and trained checkpoints |
| [2026-09-06-b3d-anchor](../worktrees/2026-09-06-b3d-anchor) | `f0622144e951` | Protected anchor candidate |
| [2026-09-06-b3d-quadratic](../worktrees/2026-09-06-b3d-quadratic) | `4dbe77cbb3a6` | Quadratic candidate |
| [2026-09-06-b3d-encoder](../worktrees/2026-09-06-b3d-encoder) | `d2ef9e12f554` | Shared encoder candidate |
| [2026-09-06-b3d-mixture](../worktrees/2026-09-06-b3d-mixture) | `9ffaf558e294` | Smooth mixture candidate |

Each listed experiment branch is `exp/<worktree-directory-name>`. Read across trees as needed, but write scientific work only in the owning tree. The canonical root lab log and reports are the required shared-document destinations. No experiment branches were merged. The question of merging the wave branch into the repair branch was previously raised and remains unanswered; do not merge implicitly.

## Recommended next experiment — proposed, not run

**First test the effect of latent dimension with the current verified wave reference and frozen learned bank.** The full-bank comparison changes dimension and evolution structure together. A controlled dimension study is the clearest way to reduce that ambiguity before adding more head complexity.

1. Freeze the current reference implementation, data split, learned bank, physical metrics and original results. Choose a small increasing set of latent dimensions up to the bank dimension before training. Keep the final cohort sealed during model selection.
2. At each dimension, fit an affine head and the current MLP in the same learned bank. Give them the same training membership and initialization subspace, and record parameter counts and budgets. Retain fresh POD as a separate comparator. Matching displacement dimension also matches the displacement-plus-velocity state dimension.
3. Measure unrestricted bank projection, best recorded multistart head fit, tangent error, initial-condition fit, and autonomous rollout separately. Keep rank, nonstationary fits, failed cases, all timestep repetitions and outliers. This distinguishes lack of representation from poor evolution without treating a local fit as a proven global optimum.
4. Preserve initial-normalized metrics for comparison and also show absolute field error and instantaneous relative error. Declare how vanishing reference amplitudes are flagged before examining new results. Monitor phase and the absorbing mean as well as energy; do not relabel the old acceptance metric after seeing results.
5. If accurate reconstruction is available at a useful dimension but autonomous evolution still fails, test a separately controlled trajectory-aware training objective or state-manifold design. The existing velocity-tangent penalty is insufficient evidence of successful dynamics. If affine and nonlinear models both need nearly the full bank, reconsider the desired compression before claiming an architecture fix.

For Burgers 3D, prioritize representation of the unseen initial-state outliers under the fixed-bank protocol, preserving separate initial/later state summaries. Re-run the full pilot before any rollout promotion. Keep larger-PDE expansion and speed benchmarking behind demonstrated accuracy. These are recommendations, not newly authorized cluster jobs.

## Operational instructions for resuming

- Read the canonical log and repository [AGENTS.md](../AGENTS.md). For a new wave experiment, propose the current fresh-wave branch as the base because it owns the verified implementation and checkpoints. Obtain the required base/worktree/namespace agreement before creating a new tree. For simultaneous experiments, propose separate owners and namespaces first.
- Use `/home/tahmid/Dev/.venv/bin/python` locally and `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python` on the cluster. Run real numerical work on the cluster GPU partition with f64 and `JAX_DEFAULT_MATMUL_PRECISION=highest`; require `jax_backend=gpu` in every log. Local GPU work is only short, bounded smoke testing through `jaxrun`.
- Use a unique paralab job directory, regenerate data from its seed, stage committed code directly with checksums, and check `squeue` before and after submission. The existing approved wave namespace is `/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/`. Do not infer permission to reuse it for a new concurrent campaign.
- Existing repair `cluster/stage_fresh_wave.py`, `submit_fresh_wave.py` and `collect_fresh_wave.py` preserve source/job/archive provenance. Pull and verify results before removing the exact remote directory. Never use bare `scancel`; use the applicable guarded helper with exact numeric IDs.
- Preserve weak-form minimization, sufficiently more test modes than latent coordinates, decoder-output quadrature fitting when used, and the FOM-exact upwind operator in Burgers weak advection. Do not substitute random strong-form collocation. Hyper-reduce initialization too before claiming a grid-independent online path.
- Compare cost and accuracy from the same solver invocation and GPU. Warm the GPU before timing and persist repetition arrays. No cross-job timing inference is supported here; the fresh wave cold start remains full-field and there is no grid-independent speed claim.
- Preserve archives, the unopened final cohort and unrelated user edits. Append results and retractions to the canonical lab log before closing every session. Ask about merges when branches finish; do not merge unprompted.

The handoff session starts no simulations and leaves no numerical job to monitor. New experiments, branch merging and the still-image evolution follow-up are the remaining decisions/tasks; this document does not mark them completed.

## Reproducing this document

Tables and numerical prose are generated from the linked run JSONs. The adjacent manifest records their SHA-256 hashes, frozen branch metadata and the output hash. Regeneration stops if an input has changed. Existing worktrees or restored equivalent archived paths are required; this command does no training, fitting or simulation.

```bash
/home/tahmid/Dev/.venv/bin/python reports/gen_2026_09_07_nmrom_handoff.py \
  --repo /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude
```

Run from the repository root. Source: [gen_2026_09_07_nmrom_handoff.py](gen_2026_09_07_nmrom_handoff.py); manifest: [2026-09-07-nmrom-handoff.manifest.json](2026-09-07-nmrom-handoff.manifest.json).

## Plain-language glossary

- **NM-ROM / ROM / FOM:** nonlinear-manifold reduced-order model / reduced model / full-order reference model. A PDE is a partial differential equation.
- **Spatial bank, rank, r:** stored learned spatial functions; rank describes how many are linearly independent. **Latent dimension, k:** number of coordinates in the compressed displacement state. Waves also evolve latent velocity.
- **Head, MLP, quadratic, affine:** the map from latent coordinates to bank coefficients; respectively a feed-forward neural network, a polynomial including quadratic products, or a linear map plus a constant offset.
- **Protected anchor / shared encoder / smooth mixture:** tested heads that preserve a fixed coordinate component, share a learned field-to-code map, or blend coefficient predictions from multiple experts.
- **POD / PCA / QR:** data-derived linear basis / principal-component coordinates / orthogonal matrix factorization. Their role must be stated; none makes the learned spatial network a POD network.
- **Reconstruction / projection / tangent error:** error in fitting a snapshot / error using arbitrary coefficients in a specified linear span / error representing physical velocity using the decoder Jacobian.
- **Jacobian / Hessian / manifold:** first derivative / second derivative of the decoder / the set of fields produced by varying its latent code.
- **Rollout / autonomous evolution:** advancing the reduced state through time from its initial fit, without refitting to the reference at later times.
- **M, K, C, reduced operators:** mass weights, spatial stiffness and boundary damping in the reference equations, and their projections into the learned bank. **Weak test modes:** functions used to average/project the PDE residual. This use of mass M differs from test-count notation used in some earlier project documents.
- **Empirical quadrature / NNLS / hyper-reduction:** weighted spatial sampling fitted to decoder outputs / fitting nonnegative weights / evaluating the online residual without traversing the full grid. **Cold start:** computing the initial reduced state.
- **Mean / median / worst:** arithmetic average / middle value / largest error. Burgers rows pool validation states; wave rollout rows first take each trajectory’s maximum over saved times. Percent columns multiply dimensionless errors by one hundred.
- **Outlier / nonstationary fit:** a case above the stated error ceiling (nonfinite cases also fail) / a selected optimization result failing the recorded local convergence condition. Neither is removed from acceptance.
- **Seed / repeat / validation / sealed final cohort:** recorded random number initialization / optimization run / data used to assess and select models / reserved data still unopened for final evaluation.
- **Time check / finest-pair passes / unresolved case:** comparison of trajectories at the smallest declared timesteps / count meeting that tolerance / case still failing it. Passing a time check does not mean the physical solution is accurate. Case indices start at zero.
- **Initial-normalized / instantaneous relative / energy-state error:** division by fixed initial scales / division by the current reference field norm / energy norm of the difference in displacement and velocity. **Phase:** position within an oscillation; undefined when its amplitude vanishes.
- **Gate / provisional target / negative control:** required check / declared engineering threshold without universal scientific status / deliberately invalid input that should trigger a failure.
- **Dirichlet / Sommerfeld / nullspace:** fixed displacement wall / approximate outgoing-wave boundary rule / states an operator does not detect. **Navier–Stokes:** equations for fluid velocity and pressure; incompressibility imposes a divergence constraint.
- **Worktree / commit / job / checkpoint / SHA-256:** separate working directory for a branch / saved code revision / scheduler run identifier / stored trained model state / content fingerprint used to verify files.
- **f64 / highest precision / GPU preflight:** double precision / required matrix-multiplication setting / check that a cluster calculation is actually using the accelerator before accepting its results.
