# A8 independent-review reconciliation and release — 2026-10-28

The initial Codex source review found no blocking PDE or normal-consumer defect at
`26e9b8272309a28d0f006fe8a6aa688f6a39f456`. Independently, Claude returned
**NEEDS-FIXES** for certificate coverage, witness-margin reporting and audit documentation.
The original response, packet, hashes and successful consultation receipt are preserved.
Claude did not rerun or approve the final code; the independent Codex delta review below
verifies the requested resolutions.

## Resolution

- R1: every certificate point now evaluates the actual returned coefficient function,
  checks it against the freshly evaluated raw normal times its factor, checks raw replay,
  and records point/label/global errors. Analytic checks use the returned values.
- R2: adopt Claude's explicitly permitted reporting alternative. Keep the original pointwise
  adjacency inference and numerical guard. Report observed per-label margins and accept the
  possibility of a false stop on another mesh; no identity with the saved mesh is asserted.
  Do not adopt an unmeasured midpoint algorithm or a new angular threshold.
- D1/D2: replace the incomplete line-based audit with an AST inventory covering complete
  statements in the FOM and preflight; document the guard as a numerical near-zero check,
  not a mesh-quality certificate. The earlier audit-completeness claim is withdrawn.
- The standalone preflight now persists its own protected sibling certificate even when the
  helper fails. This evidence-only improvement changes no mathematical operation.

Two qualifications are retained. Claude's statement that outer point evaluations were
unchecked was too broad: the full `mapped_probe` body already checked every outer label at
its lower-order points, but that body was omitted from the consultation packet. The new
all-certificate-point check still improves coverage. Its thin-witness scenario was hypothetical;
the actual saved-array margins are reported in the lane's generated replay record. They do
not guarantee future mesh quality, and none of the rejection thresholds is relaxed.

Claude's suggested runtime checklist does not add retrospective G1 force or analytic-trace
acceptance gates. The frozen summary script and scientific configurations remain unchanged.
Native results, original coarse-control limitations and all later G2 requirements stay visible.

## Exact final source and independent delta review

Reviewed final commit: `262a0ef0546bdbecb1507fd3a48d22c10b137067`, mirror `3251c8429`.
Candidate manifest SHA256: `29a9d061f70f73272f63baebe7f763f55f091478b3404ed5b40a958e687cad0e`.
Helper SHA256: `3a13ea7ca6c9110997b1c44fd04006bdf9d5f089121beb91b2988a3ca6d7779b`.

The read-only numerical reviewer verified every one of the eighteen staged-input hashes,
sizes and exact commit bytes. Only the preflight and helper changed among staged inputs
after the initial review. FOM, configs, scientific gates and batch remain byte-identical.
The reviewer independently reproduced the AST coverage and saved-array replay, verified
all requested fixes and found **no remaining source blocker**. No native FE job or source
edit was performed by the reviewer.

## Bounded execution release

Permit one isolated `g1-smoke07` control retry after fresh exact staging and coordinator
remote-byte verification: one H200 GPU, sixteen reserved CPUs, 64 GiB, twenty minutes,
the existing pinned dependencies and GPU/f64/highest checks. Check actual queue and disk
before and after submission; never duplicate a job. Preserve all failures and certificates.
The source is isolated from G2 exporter changes. Its previous geometric diagnostic remains
unchanged, and the original failed G1 result is not rewritten.

This release is 2D G1 only. Native success remains unmeasured. No G2 reference, snapshot,
training, timing comparison or 3D repair is admitted. The unresolved 3D geometric-identity
failure proceeds through a separate diagnostic proposal.
