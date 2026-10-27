# Burgers 2D reconstruction screen: lane contract

Status: implementation in preparation; no numerical release yet. The coordinator's frozen design governs cross-dimensional methods and selection. This lane implements only reconstruction inputs and independent coefficient fits, with no reduced rollout or fine PDE correction.

## Frozen evidence and data

Fork: `93f399faad67f94ea2b56fe25f13003143e64694`. Namespace: `/cluster/tufts/paralab/tawal01/b2d-recon-20261027/`. Maximum total actual allocation: four GPU-hours including failures. One job per directory. No sealed, historical test, or OOD input is read. Exposed cohorts are dev6 and val32 only. References are the already qualified E0 ST stored fields, copied as frozen evidence with source hashes; training and coarse FOM inputs are regenerated from seeds on the cluster.

Coarse intervals are 32 and 64, plus 128 for the inherited coarse-solution-data control; fine intervals are 256. The internal arrays contain only interior point values, with zero Dirichlet walls understood. Every reconstruction consumes the identical supplied coarse field and the shared grid. Every time is independent. Physical input descriptors appear only in data generation and audit metadata, never in `fit_decode`.

Inherited checkpoints are `smooth-bank` base and coarse, including their original rotations. The base has 512 columns and uses 256 nodes per axis; coarse uses 129 nodes. The latter is solution-data-only coarse training with an inherited fine-developed architecture/recipe. It is not labelled coarse-only at lower observation resolutions. No neural parameters are updated.

## Training match and POD

The original smoothtrain generator is copied byte-identically. For each bank, regenerate all 576 upwind/backward-Euler training trajectories at its own recorded node count and the exact 16,384 state picks (seed 0). Check original state-pick hash, full shape and sum/squared-sum fingerprints at relative tolerance $10^{-9}$; preserve the newly generated array hash and any lack of bitwise agreement. POD uses the same raw states with equal row weights and no mean, matching the bank data loss's global MSE weighting. The learned bank has an extra Gram regularizer and nonlinear architecture; the POD objective is not asserted identical to all training terms.

The initial POD construction is a fixed-seed randomized subspace iteration, rank 512, oversampling 64, three powers. Save singular values, captured energy, orthogonality and covariance-eigenvector residuals. These quantify approximation; no exact-optimal POD claim. The continuous extension is the full native tensor sine expansion, preserving physical amplitudes. Training at 256 nodes has 255 intervals and is not silently relabelled 256 intervals. Fine and coarse evaluations use the same native expansion, including high modes that alias on coarse observations.

## Actual coarse FOM input

Use vendored central-advection CN with four BE half-step startup stages (the E0 numerical family). Time steps are 0.0025, 0.00125, 0.000625, 0.0003125. Require all nonlinear residuals at or below $2\times10^{-11}$ and finite outputs. Accept the first successive pair whose maximum evolved-output gap, divided by fixed physical initial norm, is at most $10^{-5}$; use the finer member for the main supplied field. This is an empirical temporal-input check, not a rigorous PDE error bound. Retain both members and reconstruct them to expose amplification. Keep all failed cases visible; do not silently remove them from validation.

Outputs occur exactly at $t=0,0.05,0.10,0.15,0.20,0.25$. Initial-field checks remain separate. Reference input is direct nodal restriction of the fine E0 array, repeated with other saved E0 levels. Fine accuracy uses the original accepted fine-lattice judge, not a restricted judge.

## Retention and run gate

Every source/checkpoint/exposed-reference hash is in `INPUTS.json`. Large checkpoints and array inputs remain outside Git and are traceable to retained originals. Jobs preserve data/fit coefficients, reference identities, final and penultimate coarse arrays, timing repetitions and normalization. Regenerable training arrays may be retained remotely during preparation, but local duplicate copies are unnecessary once hashes, fingerprints, seeds, source and final POD bases are preserved and independently admitted. All unique results must be collected with checksums before remote removal.

No numeric job is submitted until the coordinator's independent design/code/exact-job review releases it. GPU preflight, f64 and highest matrix precision are mandatory.

## Glossary

- Interval: distance between adjacent grid nodes; L intervals has L+1 nodes per axis.
- Bank: frozen learned spatial functions; a field is their coefficient-weighted sum.
- Prefix: first retained bank columns, with a separately fitted coefficient for each.
- POD: spatial modes fitted to snapshot data by a singular-value decomposition.
- Snapshot: one solution field at one recorded time.
- E0 ST: existing space-and-time-refined reference campaign, with conditional numerical uncertainty.
- CN: Crank–Nicolson time stepping; BE: backward Euler startup or training stepping.
- Oracle: a diagnostic fit allowed to see the full reference; never an admissible reconstruction.

## A1: one extra input-qualification rung before reconstruction outcomes

Initial preparation reveals coarse-CNR inputs that miss the unchanged temporal gap threshold at the last registered time step. Before any reconstruction outcome is computed, permit exactly one extra rung, $\Delta t=0.00015625$, on those failed-admission cases only. Increase the runtime stage-record capacity to 2000 without editing the vendored PDE equations. Retain all original levels and metadata; supplement writes a separate directory. No threshold relaxation, additional rung, or dropping failed cases. Any remaining failures are descriptive inputs without a qualified spatial-error interpretation. This amendment requires the same independent exact-code/job review.
