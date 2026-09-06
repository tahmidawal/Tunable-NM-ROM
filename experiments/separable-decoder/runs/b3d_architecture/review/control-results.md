# Independent control-results review

Codex subagent `/root/anchor_head` independently verified the pull manifests,
checkpoint hashes, GPU/completion logs, summary statistics and gate counts
against raw arrays. It also verified unchanged membership, identical bank QR
matrix and anchor, and roundoff-level differences in shared scales and regenerated
validation arrays. The controls are accepted as bounded negative representation
results, with no rollout promotion.

Increasing width improves training reconstruction but worsens validation mean
and worst error in both optimizer repeats. Tangent error improves slightly;
this does not establish better dynamics. Every measured selected-fit Jacobian
retains full latent rank. The unresolved fit in the wider arm is a different
state from its stationary worst case, so that failure does not disappear by
resolving the one convergence ambiguity.

Scope: one fixed data cohort and training schedule. These results do not prove
that capacity cannot help, that latent dimension is insufficient, or that local
latent fitting found global representation minima. Exact statistics are generated
in B3D-ARCH-NOTES.md from the raw JSON files. No files or jobs were changed by
the reviewing subagent.
