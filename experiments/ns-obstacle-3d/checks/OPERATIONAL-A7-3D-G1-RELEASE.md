# Operational amendment A7 — independent tiny 3D controls, 2026-10-28

This prospective release replaces the full-2D-G1-success prerequisite **only for the
exact tiny 3D infrastructure candidate identified below**. It does not change a scientific
acceptance criterion, reclassify a failed control, or release reference data or training.
The frozen A6 design and adapter bindings remain the scientific specification; this file
is an additional execution record, not an undocumented rewrite of those bindings.

## Reason and evidence

The user explicitly permits 3D infrastructure verification while 2D runs. Job 6567998 has
now passed the shared pinned-dependency, GPU/f64/highest, mapping, scalar linear inverse,
gradient and traction API checks. The empty-channel control agrees with the analytic
solution and balances; it required zero Newton updates because its Stokes solution is
already the exact Poiseuille solution. It does not validate a nonlinear obstacle solve.
The manufactured obstacle case stopped before solving on its mapped circle-normal check.
Full 2D G1 remains failed, and the actual failing vector and mesh were not saved.

The independent Codex numerical reviewer accepts these partial controls and finds no
demonstrated shared blocker requiring 3D infrastructure to wait for the 2D circle
diagnosis. The 3D candidate uses its own CSG geometry, cube/API preflight and boundary
orientation controls; those must succeed independently. Normal orientation remains a
runtime risk. It is not assumed correct from the 2D channel result.

## Exact permitted execution

- Source commit: `1c2fd8fa141ded0d1dcdc15c850a5a8682572b00`.
- Source SHA256: `84c66d03a84844866f1298a6dae74449a2708ea64c848560e83f8b64ea1b68aa`.
- Config SHA256: `637837aecc59da58b9c0f7c582be9f27d4aa97707baf711038fd57e06f541350`.
- Batch SHA256: `62f3b25e32632641da16257864e77c0bb7fa5e4f2e3cf4850c3243d334b1a7a7`.
- Input manifest SHA256: `5336879b7d02f722144cb3b738d5c99e1cebfca599755d3d81852b61a06836b9`.
- Directory: `/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/g1-smoke06`.
- Cases: sphere/pipe geometry, empty-pipe analytic control, manufactured flow only.
- One GPU on the `gpu` partition, sixteen reserved CPUs, four requested FE threads,
  64 GiB, twenty minutes. Existing hard process limits, 15000 mixed DOF and 6000 element
  guards, conservative factorization estimate and memory stops remain unchanged.

The coordinator independently verified every staged checksum on 2026-10-28; the receipt
is `state/3D-SMOKE06-ROOT-RECHECK-2026-10-28.txt`. Check the actual queue and disk again
before submitting and record the resulting job ID immediately. Do not duplicate a live
or completed allocation. Count all elapsed allocation against the existing campaign cap.

The candidate's older A3 campaign snapshot and textual full-2D prerequisite remain
unchanged as historical inputs. This explicit release supersedes that prerequisite for
this candidate alone. All other gates remain closed, including physical sphere solves,
reference admission, snapshots, bank/POD training, transients and larger sweeps. Preserve
every failure, collect all evidence with checksums, and obtain independent result review.

## Terms

G1 means bounded runtime controls. CSG means constructive solid geometry. DOF means a
finite-element unknown. A partial control validates only the operation it actually tests.
