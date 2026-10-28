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

### Obstacle-flow controls — measured boundary-normal convention (Codex coordinator)

After reset recovery, independent review accepted the retained 2D API and empty-channel
controls. The channel needed zero Newton updates because its Stokes solution already
solves Poiseuille flow; it does not validate nonlinear obstacle convergence. Operational
A7 prospectively released only the exact tiny 3D infrastructure controls. Job 6610494
passed cube/API/UMFPACK/nonsymmetric-gradient controls, then failed the raw sphere-normal
check before any flow solve; full evidence is retained at 3D commit 091970665 and mirror
98f28ebd3, with checksum-verified remote-duplicate cleanup recorded at c66a5ff21.

A separately reviewed unchanged-geometry 2D diagnostic, job 6610978, completed and was
independently audited. Ordinary boundary integration agrees with raw mapped normals and
has relative divergence-theorem defect 0.0175678828431; the adjacent-volume facet and
all-element-boundary paths give 9.80909507011e-15. Offset/adjacency witnesses consistently
place the positive raw circle-normal direction in the fluid. The fresh channel-minus-circle
mesh is geometrically correct within its control. This withdraws the source assumption
that raw boundary normals automatically mean outward-fluid normals. It does not recreate
the missing original failed mesh. No obstacle solve or physical reference was admitted.

Implementation amendment A8 authorizes a scoped, topology-verified orientation repair
for ordinary boundary evaluation, preserving actual curved normals and raw evidence.
Exact implementation/configuration plus Codex and Claude review precede its single repair
retry. The 3D normal diagnostic is separately reviewed before staging; its initial missing
installer-binding defect was caught before execution and repaired without changing geometry.

ROM A6 commit d1b4eb952 (mirror 532cfa22a) passed independent source and synthetic algebra
review. Native FE/data integration remains closed, and no fictitious G2 admission is allowed
to test the adapter. Generated control tables and provisional field figures live in the
coordinator tree; final reports and presentation remain future campaign deliverables.
The ledger currently charges 135 GPU-seconds (0.0375 GPU-hours). No merge,
sealed-population access or unique-data deletion occurred.

### Obstacle-flow A8 retry and unresolved 3D geometry (Codex coordinator)

The 3D diagnostic job 6611259 completed, but native exterior-facet and all-element-boundary geometric identities failed; no 3D flow or normal repair is admitted. Its arrays and pinned source diagnosis are retained in the 3D lane, with independently verified archive and remote-duplicate cleanup. Negative facet Jacobians are a hypothesis, not a measured cause. Pinned Netgen source shows text mesh curved coefficients written with limited decimal precision. The next source-only proposal separates strict retained-file reload parity from an independently regenerated unchanged-recipe diagnostic; neither can reconstruct the original unsaved in-memory state. A binary archive path requires a clean-process measured round trip before G2 selection.

A8 normal-repair review preserved Claude's initial NEEDS-FIXES. The final isolated source 262a0ef0546bdbecb1507fd3a48d22c10b137067 checks every actual returned boundary coefficient-function value, reports witness margins, preserves the accepted scoped false-stop risk, and includes a complete callsite audit. Independent Codex delta review resolved the requested fixes before staging. The original force/trace/geometry gates were not relaxed.

Job 6612209 was submitted in g1-smoke07 and, while still pending with zero runtime, its H200-specific request was changed once to H100 using scontrol. Exact ID/owner/name/pending-state guards and before/after receipts are retained. No cancellation, duplication or source change occurred. It completed in 31 seconds on pax063 with all thirteen G1 run checks passing across the channel, manufactured and coarse DFG cases. All 820 collected files pass checksums; archive SHA256 is 962e5cf5ac28760bf4edde5659aaeab4a8cd638c8152df4a1afccd2cd8ce4a80. Independent numerical review and lane backup/cleanup are still pending. Coarse force and pressure discrepancies remain diagnostic; successful G1 controls confer no reference admission. Total terminal charge is now 194 GPU-seconds (0.0538888889 GPU-hours).

A coordinator-spawned automatic Git repack consumed a large temporary pack. Exact process ownership was verified before terminating only that repack and removing its unpublished incomplete temporary file. Existing Git packs and all experiment evidence were retained; the event is recorded in GIT-AUTO-GC-INTERRUPTION.json. Subsequent commands disable auto-GC only per command/process, without a persistent configuration change.

No training, method comparison, sealed-data access, worktree merge or unique experiment-data deletion occurred. The next 2D work is G2 source/export and resource qualification after independent G1 admission.

### User direction: complete the 2D ROM first (Codex coordinator)

The user explicitly narrowed the active objective to making the 2D ROM work and asked to make it the goal. A durable conversation goal is now active, covering qualified references, reproducible exports, learned bank and ordered prefixes, fair POD, actual reduced solves, velocity/pressure/forces/constraints, quadrature validation and matched-cost comparison with generated report/presentation/handoff. Smoke/source success is not the completion target. All further 3D work is deferred; its incomplete, unexecuted diagnostic source is preserved at ba83a8855 / mirror d3a1e9db4, with no active process or allocation.

Independent saved-array review accepts the 2D smoke07 G1 controls. The operational condition is now closed: retention a893210f5 / mirror98b803bd1 and checksum-backed remote cleanup c3a6c8a76 / mirror777213ba6 are complete, with the exact remote job directory independently confirmed absent. All stricter G2 blockers remain visible, including coarse divergence, force consistency and benchmark pressure/lift discrepancies. Generated control tables and an all-saved-point cylinder field figure preserve the coarse/provisional scope.

The 2D owner is implementing one bounded native producer and separate-process binary mesh reload control with unchanged physical settings and reload tolerance. The resumed ROM owner works in the existing clean ROM lane, integrating the final A8 normal contract and explicit mesh persistence contract before completing the actual bank/POD driver. No new native job or training has been released. Existing resource ceilings, scientific thresholds and stop rules remain unchanged.

A second previously detached Git auto-repack from the 2D lane was identified by exact process/cwd/stderr and stopped; only its unpublished incomplete temporary pack was removed after all related processes exited. Existing packs and experiment data remained intact. The event is retained in GIT-AUTO-GC-INTERRUPTION-2D.json; resumed lane commands use per-command/process gc.auto=0.

No sealed cohorts were opened, no branches merged, and no unique experiment data deleted.

### Continued 2D work: native producer accepted (Codex coordinator)

The user reaffirmed autonomous continuation toward useful 2D results. Actual state was checked after interruption: no campaign jobs were queued and native01 job 6613426 had completed. Disconnected agents were replaced only after verifying they were absent; existing source and collection work were continued without duplicate submission.

Independent read-only review accepts the native producer and fresh-process binary reload. All fifteen replay errors are exactly zero, with distinct producer/child interpreters, positive mapped weights/Jacobians, no quadrature inside the analytic solid, unchanged normal factors and zero prescribed essential-trace differences. The coarse physical solution agrees with smoke07 to roundoff. Full evidence is retained at 13083eb5b / mirror2ef3375f2; checksum-backed remote cleanup e88019fc2 / mirror57b8ca3af is complete. All nine allocations total230GPU-seconds; no physical reference or training admission follows.

Next source work is an initial DFG/manufactured h0,h1,h2 ladder with full common-rule comparisons and a separate native ROM-interface fixture using the real producer output. Producer-only capacity changes are being projected separately from unchanged ROM-fitting caps. The actual bank/POD driver is being completed in its own lane. 3D remains deferred; no sealed data, merge or unique-data deletion occurred.

### Continued-chat ownership reconciliation and next releases (Codex coordinator)

The continued chat initially could not see the parent chat's worker registry. Actual app thread histories later revealed an old ROM worker had overlapped package edits. New writes were stopped during the check; the old ROM worker is now authoritatively interrupted, its commands terminal, and the old reference worker completed. Both source variants and leftover edits were preserved, including legacy/558d2dce8-overlap. The current sole writers are reference_2d_run and rom_2d_run under continued chat01a18408-7af8-7f60-8742-5f283032a493; parent chat01a17ee7-e42a-7233-96fe-7bb7d9c849fa must not restart its writers against these trees. The requested goal was restored in the continued chat because its goal state had not transferred.

Independent review and root packaging checks released exactly one initial six-case h-refinement allocation (sourcefd5354794) and one native ROM interface allocation (reconciled sourcecb9db577e). The latter includes actual boundary velocity, pressure and stress-gradient replay after a review-requested coverage fix. The former uses a complete common fine rule with explicit tight point lookup and recorded inside-triangle checks, and raises only producer point capacity while preserving physical gates and downstream ROM caps. Each job remains one GPU, sixteen CPUs,64GiB,twenty minutes; no more than two active campaign jobs. Submission receipts and outcomes are pending; no training is released.

### First actual ROM interface control completed (Codex coordinator)

The selected ROM interface job6613813 completed successfully on H200/pax009 in30allocated seconds. Its real producer-input replay and boundary field/gradient paths report success at roundoff, and its bounded metric/component/eigen controls completed. Collection and independent saved-result review are pending; this is native integration evidence, not training or an end-to-end reduced solve. The separate reference ladder job6613821 is submitted once and pending at the last observed queue state. Total terminal charge is260GPU-seconds; both exact submission manifests are in the ledger. No automatic retry, physical admission or training was released.

### Native interface accepted; first refinement diagnostic stop (Codex coordinator)

Independent saved-result audit accepts native ROM interface job6613813 as low-level integration only:812 files and43 staged hashes verified, volume/boundary replay below6.68e-16, complete component maps and finite full-rank eigenbanks. Runtime eigen checks passed; sparse matrices were not retained for independent residual recomputation. Archive b35b31000bdbdcdd7ed5e9b96be679353448c5826618ab19b7b449c772a97924 is retained; mirror/remote cleanup closure is pending.

Ladder6613821 failed after44seconds at manufactured h1's directional Jacobian check, after Newton converged to1.1040663175355378e-11. All889files are independently verified, archive6f9fa30a7b6383824dd475264236443dd52b1a795911cdccb42d319a7abe1479. The three errors grow inversely with step size; cancellation is supported, not proved. Source review confirms the fixed-quadrature residual is quadratic, making larger centered differences a defensible prospective diagnostic amendment. Reviewer accepts steps[1e-2,1e-3,1e-4] for all future cases with unchanged thresholds/physics; source and full diagnostic-array persistence are being prepared for review before a bounded retry. Original failure remains failed.

DFG h1 improves scaled divergence from0.0219928 to0.00604354 and pressure drop from0.116873883245 to0.117236160719; physical qualification still fails. h2 and common-rule comparisons were not reached. All11allocations are terminal,304GPU-seconds(0.0844444444GPU-hours). No reference/training admission, sealed data, merge or unique-data deletion.

### Reviewed retry and actual reduced-driver integration (Codex coordinator)

The last continuation made concrete progress, not an idle wait: native interface acceptance and the failed spatial ladder were retained, and a prospective diagnostic amendment was independently assessed. After an interruption, app thread status confirmed all three prior worker turns interrupted/notLoaded with terminal commands; no live sync/push remained. Replacement workers reference_resume, rom_resume and numerical_resume now own the same isolated lanes/review. Parent original goal remains blocked on ownership and must not restart against these trees.

Ladder02 source c18b38a3bf082069ab92d763d6660a11d73389cc and inventory f91cd365fb22a81f9a67500ede71baad0102e62beac4cf25eb8f6b51083ecf94 pass independent and coordinator checks of all24runtime inputs. All six configurations differ only in the uniform larger centered-difference steps; full replay arrays precede unchanged acceptance checks. Exactly one same-resource allocation is released; original ladder01 failure remains failed. Its retention mirror and verified remote duplicate cleanup are now closed.

ROM driver db4d3e3dc passes source/array integration review after fixing a validation defect that had allowed a fitted arm to qualify against an unconverged full-rule control. Both invocations must now converge, with regression coverage and actual-state retention. Preparation entry66e637bb3 also passes source review and requires genuine G2/shared-lift admission before native execution. Actual preparation, training and physical solves remain unrun; a fair paired FOM callback is still needed. Generated control tables now include the retained partial refinement values. No sealed data, merge or unique-data deletion.

### Six reference solves complete; comparison failure retained (Codex coordinator)

Ladder02 job6614416 terminated FAILED1:0 after136seconds onpax009, batchMaxRSS1335680KiB. All six DFG/manufactured FOM cases completed; first common-rule comparison failed. The reviewed Jacobian amendment no longer blocks the solves. Preliminary DFGh2 has5444cells/61172DOFs and scaled divergence0.0014024773357450802, still above0.001; this is not reference admission. Collection and error inspection are in progress through verified live SSH handles; no automatic retry. All12allocations now total440GPU-seconds(0.1222222222GPU-hours).

Independent source review also accepts the genuine-admission preparation entry, reusable FOM query extraction for future native equivalence, and exact blocked moment prototype for array controls only. Production fine-mesh capacity remains unresolved: candidate-gradient arrays and prepared archive layout need blocking/per-array storage; owner is addressing this without reducing point sets, basis sizes or thresholds. One prospective h3 extension is being assessed from actualh2 counts under the existing design limit. No training, physical reduced solve, timing comparison, sealed-data access or merge has occurred.

### Ladder02 saved-array audit isolates lookup failure (Codex coordinator)

The archive825f469c36a66d61cbd0aea0329b00b58a2c2167f6d3a3e62125297ba956d7af is locally retained and all966files independently verified. All six saved Jacobian perturbation/action/residual arrays independently replay; maximum step error is below2.46e-10. The original ladder01 failed diagnostic remains failed; the prospective amendment now has measured support.

Exactly2of266756DFG common-rule points fail h0reference-triangle membership, with negative barycentric coordinates about1.89e-7 and3.06e-8. Both lie far downstream inside the channel, not near the obstacle. True curved-domain mismatch is unsupported by this evidence. Invalid-point reconstructed physical coordinates were not computed before the failed check; valid-point reconstruction is accurate to1.24e-16. Owner is preparing exact-coordinate inverse/neighbor lookup with unchanged membership/physical tolerances and full first-lookup diagnostics. No extrapolation, dropped points or shifted coordinates is authorized. A prospective comparison-only retry can reuse all six immutable completed solutions after source review; no duplicate solves are needed.

Blocked-moment and TSQR source controls are accepted only at array level. A separate bounded native block/eigen replay fixture is being prepared against real retained coarse producer inputs before production wiring; no training release.

### Two scoped native controls submitted; h3 source preparation

After independent exact-source reviews, native-block02job6614673 and comparison01job6614742 are each submitted exactly once. Their full manifests are in the coordinatorledger. The former tests real blocked evaluations/TSQR and retained sparse eigenchecks; the latter first certifies the two originally failed exact points then repeats both complete comparisons from the six immutable saved solutions, with zero PDE reruns. No production capacity or physical admission is inferred. Ladder02retention7276f4ae8/mirror887a06bb1 and cleanup3d0cd7700/mirrorf25743e01 are closed.

The prospective singleh3 capacity proposal passed independent arithmetic review from actualh2 counts: only producer30kcells/300kDOF/1.3Mpoints caps are authorized for source preparation;12Mnnz,48GiBworkingguard,64GiBallocation remain. Projected20.9635GiB and342.5seconds for two solves are estimates, with limitednnzheadroom; hardstops remain. Exactsource and successful comparisonrepair are required before execution. Independent p4capacity and complete physical controls remain outstanding.

The sixpanel provisional refinement figure is generated from retainedJSONs and passed independent evidence/label review. It displays allthreelevels and original thresholds; no common-field or ROMspeed claim. Coordinator artifacts/code are backed up; goal remains active.
