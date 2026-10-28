## 2026-10-27

### Navier–Stokes obstacle campaign — discovery and design checkpoint (Codex coordinator)
- Started user-authorized fixed-cylinder 2D channel and fixed-sphere 3D circular-pipe campaign. Read canonical log, rules and recent Burgers/improvement/refinement/geometry/spectral evidence. Resolved fork 93f399faa / code-only parent 5a64899fd; created four isolated 2026-10-27-ns-obstacle{,-2d,-3d,-rom} worktrees and separate paralab namespaces. All committed work is mirrored; no merge or sealed-cohort access.
- Discovery established that inherited periodic NS2D/NS3D solvers do not provide obstacle/inlet/outlet/pressure/force references. New curved body-fitted NGSolve/Netgen mixed Taylor–Hood pipeline proposed, with pinned lane-local dependencies instead of shared-JAX changes. DFG2D-1 is the literature anchor; 3D Re10 pipe/sphere is explicitly project-defined. Confined-sphere source conventions/access/precision limits retained.
- Central prospective DESIGN.md, source inventory, job ledger and review packets saved in coordinator lane. Initial ceiling 12 allocated GPU-hours including CPU FE work and failures; no allocations yet. Two independent Codex reviewers approve dependency/control implementation only; their findings are preserved and addressed by pre-data A1/A2, including pressure normalization, exact-versus-approximate geometry, manufactured convection, separate steady/transient gates, aligned output times, outlet sensitivity and exact tensors for both bank/POD.
- First Claude design request timed out after 240 seconds without stdout/stderr or a diagnosis; not marked passed. One shorter bounded retry is in progress. Reference/control source preparation may continue; numerical submissions, production reference acceptance and training remain gated.
- No new numerical findings or retractions of earlier empirical results. Open: Claude review, code and exact-job reviews, reference gates, then conditional steady ROM/POD/FOM pilots and reporting. Current campaign is active, not complete.


## 2026-10-27

### Navier–Stokes obstacle campaign — reviewed implementation and first allocation (Codex coordinator)
- Independent Codex design/implementation and Claude design/numerical-implementation consultations are saved in coordinator reviews; design amendments A3/A4 resolve test spaces, pressure conventions, honest exact-tensor controls and data/representation gates. The timed-out first Claude design attempt is retained alongside the successful shorter retry, not counted as a passed review.
- Retracted earlier boundary-diagnostic readiness after direct pinned NGSolve source inspection showed that full VectorH1 gradients are volume-only. Both reference lanes now use `BoundaryFromVolumeCF` for the fluid-side gradient before combining it with boundary normals. Separate-face affine stress tests cover the actual boundary integration path. Independently reviewed source: 2D `1ac56b2d7`, 3D `983a50967`; runtime tests remain unexecuted after the import failure. No earlier empirical result was changed.
- One exact 2D control allocation 6566869 (`ctol_nsobs2d_g1`, `g1-smoke03`, pax008/H200) failed in 33 seconds, charged 0.0091666667 GPU-hours. Source commit `1ac56b2d7c3305f4e961149c31e087b422d2600f`, manifest `0dde96f33990f4e05054b4390de595d782db790231954eb31fcf636e0756ea13`. JAX GPU/f64 and all wheel hashes passed. Netgen import raised `KeyError: 'tkernel'`; no analytic, manufactured or physical FE case ran. Disk was not full. Saved failure summary and full archive SHA `02a97de2b7ede7b66554152dd654c0c251515500271a65d1d8066342b9e1082e`, with all extracted files verified. Remote-duplicate cleanup tracked by the 2D ledger.
- Measured cause: pinned OCCT wheel native libraries were absent after pip's target installation, so metadata discovery could not locate TKernel. Preparing a single ordinary infrastructure repair using the same hash-locked wheels and prescribed interpreter in a job-owned install prefix; no shared Python/JAX modification. Independent exact-diff/config review precedes any retry. The 3D lane remains gated behind dependency success and 2D controls.
- ROM/POD infrastructure has a separately reviewed retained-output aliasing repair (`511faa165`) and deterministic repeated-buffer regression. FE lifting/supremizer/test integration is under preparation only; no training or numerical admission. All failures count against the unchanged initial ceiling. No live campaign allocation, no merges, no unique-data removal or sealed/OOD access. Work remains active; all physical reference, pilot, timing and final reporting gates are open work.

## 2026-10-28

### Incompressible obstacle-flow campaign — reset recovery (Codex coordinator)

The user requested continuation after a runtime reset. Read the canonical log and current
repository rules, recovered the existing four worktrees without creating replacement
branches, verified their heads and the coordinator mirror, and checked the actual Tufts
queue (empty). No allocation was duplicated. Replacement workers own the original 2D and
ROM lanes; the numerical reviewer is read-only.

The fourth 2D allocation, job 6567998, is terminal: shared API/UMFPACK/gradient controls
passed, the empty channel completed, and the manufactured obstacle case stopped before
solve on a mapped-normal sign check. The cylinder benchmark was never reached. Its
complete checksummed collection is retained at 2D commit b1121569e; this is partial
control evidence, not a passed full G1 or admitted reference. Total charged allocation
remains 81 GPU-seconds (0.0225 GPU-hours). The precise failed normal was not recorded,
so no sign-repair cause is claimed. Unchanged-geometry diagnostics and an independent
channel-result audit are pending.

The unfinished A6 ROM source and synthetic failure records survived and are being
finished against the frozen A6 specification. Earlier narrow source readiness remains
superseded pending this implementation review and actual FE/data controls. The tiny
3D candidate's remote checksums were independently reverified, but no 3D job was
released. Provisional empty-channel field plots were generated directly from saved
arrays with hashes, under the coordinator's experiments/ns-obstacle/figures directory.
No merge, sealed-population access or unique-data deletion occurred.
