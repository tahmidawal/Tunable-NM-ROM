# Proposed fixed-bank head refinement with additional training coverage

This is an unlaunched bounded accuracy experiment following the frozen runtime
pilot. The current nonlinear head's initial-state fit remains much worse than the
unrestricted spatial bank. Faster stopping improves query cost but cannot alter
that representable-state gap. The proposal isolates additional head optimization
from additional training coverage while preserving the current architecture.

Freeze the pilot spatial network, Fourier lift and output scale exactly. Retain
the same latent dimension, bank dimension and coefficient-head architecture.
Only the coefficient head, linear skip and per-training-snapshot codes may change.
No descriptor-conditioned input, encoder, CP model or new spatial basis is added.

Run two head/code refinement arms, both initialized from the same saved head:

- The original training cohort, to measure the effect of more optimization alone.
- The original cohort plus new independently seeded training draws from the same
  pinned single-bump distribution, to measure the effect of better coverage.

Proposed bounded settings are the original 32 trajectories plus 128 new training
trajectories generated with seed 790713. Keep all six original output times.
Use 8000 updates per arm, fixed snapshot minibatches of size 192, and two paired
minibatch-seed repeats, 790714 and 790715. All parameter shapes and the number of
optimizer updates/sampled snapshots are matched; actual offline time is measured.
These numbers are prospective settings, not results or an already launched job.

Use exact QR-compressed full-field reconstruction with per-snapshot relative
normalization. New training codes start from the nearest existing decoded training
snapshot in this field-derived metric. Only training fields initialize training
codes. The original validation draws are unchanged and remain outside all losses;
the final cohort remains unopened. The initial-fit and snapshot-fit evaluation
uses the same strict multistart procedure on every arm. The unrestricted frozen
bank projection is an invariant control and must remain unchanged.

First compare initial-state reconstruction separately from later snapshots, then
compare medians, worst cases and counts missing each existing accuracy target.
Preserve local-fit gradients, start-selection details and nonstationary attempts.
The immediate development target is every initial case below the existing 5%
threshold; a failure remains informative, and improvement over the frozen head
alone does not isolate coverage if the original-cohort refinement improves equally.

Only after this representation comparison should the already verified weak heat
rollout be tested on the new head, using both strict and runtime-selected solver
tolerances with the same paired complete-query/DST protocol. Better reconstruction
is not an automatic rollout or speed guarantee. Rebuild operators from the same
spatial bank on the original two meshes and retain the common observation norm.

This remains a deliberately restricted-family diagnostic. It does not substitute
for the declared broader heat cohort with separate single/multiple-component
reporting, per-resolution optimization or independent final confirmation.

## Plain-language glossary

- **Bank / head / latent code:** learned spatial functions / nonlinear coefficient
  map / compressed coordinate assigned to a training snapshot.
- **Frozen / refinement / coverage:** unchanged weights / additional optimization /
  how many different training states the model sees.
- **QR / multistart / stationarity:** exact orthonormal compression of field fitting /
  fitting from several starting codes / sufficiently small objective gradient.
- **Minibatch / paired repeat:** fixed-size training subset / corresponding runs
  with controlled optimizer sampling seeds.
- **Initial reconstruction / rollout / complete query:** initial-field fit /
  autonomous evolution / full initial-input to requested-field-output task.
- **Validation / final cohort / invariant control:** development-only evaluation
  cases / untouched independent confirmation cases / a diagnostic that must stay
  unchanged when only the head is refined.
