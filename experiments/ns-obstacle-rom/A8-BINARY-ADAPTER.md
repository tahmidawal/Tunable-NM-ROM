# Binary-mesh and oriented-normal adapter update

This source update is unexecuted and does not admit reference data or training. The
reviewed A6 source remains at `d1b4eb952abe0e454d06f3406a5cd26eab09e770`, with changed
files additionally preserved in `legacy/d1b4eb952/`. Its historical synthetic results
are unchanged.

The producer candidate now saves a newly generated `mesh.vol.bin` using
`netgen-binary-archive-v6.2.2608` and verifies it in a fresh interpreter under policy
`saved_curved_binary_verified_fresh_process_no_recurve`. The adapter explicitly requires
all three identifiers. There is no text-mesh conversion, automatic re-curve, or claim
that a binary save can recover precision already lost in an earlier text file.

The future outer training manifest retains schema `ns-obstacle-fe-training-v1` and
adds `mesh_filename` and `mesh_format`; `mesh_reload` names the fresh-process policy.
Each case binds the binary mesh, `mesh_identity.json`, both producer normal certificates
and `EXPORT-MANIFEST.json`, in addition to the existing state, array and layout files.
The existing genuine G2, training-population, source-review and common-lift requirements
remain mandatory. A successful native producer control alone cannot create this admission.

The adapter vendors the exact reviewed A8 helper and its scalar dependency, with provenance
in `A8-HELPER-PROVENANCE.json`. After binary loading it independently reruns the topology,
actual returned-coefficient-function, analytic-orientation and facet-moment controls.
Raw normals, per-label factors and actual oriented normals are retained and compared with
the producer exports. Native volume-facet normals are never multiplied by these factors.

`TrainingFEAdapter.open(manifest_path, evidence_directory=...)` requires a fresh directory
for the native control certificate, so a failed orientation check retains its samples.
The production adapter still rejects G1 inputs before importing NGSolve. A separate
low-level native fixture is needed to exercise adapter primitives before admitted training
data exist; inventing a passed training manifest is not an allowed fixture strategy.

The point, archive and working-memory caps are unchanged. The fine reference projection
already exceeds some of them, so measured native control resources and an independently
reviewed memory plan are required before increasing caps. Reference quadrature is not
weakened to fit implementation limits.

## Glossary

G1 is the initial solver-control stage. G2 independently qualifies reference solutions and
data exports. A binary mesh preserves the native archived floating-point coefficients;
its actual round-trip fidelity still needs measured confirmation. A boundary normal is
a unit direction perpendicular to the curved boundary. Outward-fluid normals point into
an excluded solid. A coefficient function is the finite-element library's evaluable field.
