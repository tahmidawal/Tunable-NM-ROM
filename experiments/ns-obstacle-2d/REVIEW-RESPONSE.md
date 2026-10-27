# First smoke implementation review response

2026-10-27. The independent Codex numerical reviewer returned CONDITIONAL READY for G1 smoke on source commit `4e38c8265`, with no mathematical blocker reported through the coordinator. This is not G2 reference admission or training release. The central campaign/design binding is refreshed to predata-A4; the copied compatibility filename `campaign-A2.json` now contains version A4, identified by its internal version and SHA256. No equation, mesh setting, solver stopping policy, threshold, or numerical source changed in this rebinding.

The coordinator requires the independent Claude implementation consultation before submission. The remote directory may contain reviewed source bytes only until explicit release. Neither installer nor numerical solver has run.

The reviewer identifies later G2 work: save actual exact-obstacle distance and excluded-point/weight diagnostics for the volume rule; save boundary velocity gradients or independently reload and evaluate the FE state for force audit; and measure real CPU concurrency before comparative timings. Existing positive-weight/signed-Jacobian and sampled boundary checks do not close those requirements. The current exporter preserves mesh, FE coefficients, and full-rule field gradients; their independent reload still requires verification. These gaps remain explicit and are not G1 blockers.

All constants and manifest identities must match the staged release; a later source fix requires a new recorded attempt rather than silently replacing live files.

## Claude implementation consultation response — before second staged attempt

The independent Claude consultation re-derived the core mixed weak form, force sign, reaction force, gradient and physical balances, and manufactured fields without finding a mathematical blocker. Its raw review is preserved by the coordinator. The response adopts concrete execution checks while retaining original scientific tolerances.

- Added `preflight.py`: a tiny finite-element UMFPACK solve plus mapped volume/boundary point, Jacobian, weight and normal checks before actual flow cases. The existing `GetTrafo(Ngs_Element)`, `np.asarray(mip.jacobi)` and coefficient normal evaluation are retained because the pinned official bindings document their conversion/properties; this smoke now tests them cheaply. No speculative API replacement is made.
- Added normal checks for all four straight sides and inward-fluid normal at the obstacle before factorization. Manufactured exact closed-surface gradient and symmetric forces must agree to roundoff. Their symbolic witness remains independent of NGSolve derivatives.
- Exact pressure probes require `MeshPoint.nr >= 0` and finite interpolation. Invalid probes are saved as unresolved, with explicit G2 pressure ineligibility. No nearest-node substitution or nudged primary measurement is used.
- Boundary names are sorted. `TaskManager` wraps case assembly/solves, with actual effective CPU concurrency explicitly unmeasured until timing qualification.
- A batch exit trap always attempts the robust summary for ordinary failures, saving stage exit codes and missing/failed case states. A fundamental dependency/API failure stops later expensive cases. A scheduler hard kill can still prevent finalization; partial stage files remain evidence.
- The summary adds nongating raw DFG literature gaps, reaction/surface differences at both integration orders, and exact pressure-probe status. These do not create reference admission.
- The strict linear relative residual criterion remains `1e-10`. Claude's proposed relaxation was speculative and is not adopted. No iterative-refinement change is introduced absent an actual failure.
- No MKL/intel-openmp dependency is installed: current pinned PyPI metadata declares OpenBLAS. The import/tiny inverse preflight establishes runtime availability rather than assuming another build's dependencies.
- `SOURCE_COMMIT.txt` will be generated from the exact committed source at staging, transferred explicitly and independently recorded in the attempt manifest. The prior staged `g1-smoke01` is preserved; the candidate is for a distinct `g1-smoke02` after re-review.

Checks performed without NGSolve execution: Python syntax, shell syntax, symbolic witness equality, and the missing-case summary path. No packages installed and no numerical allocation submitted.

The job name now starts with `ctol_` so the existing explicit-ID cancellation helper can validate it if cancellation becomes necessary. No alternate cancellation policy is introduced. The first smoke's exact submission command specifies `--gres=gpu:h200:1`, subject to the coordinator checking current availability; this is recorded in the manifest. The hard twenty-minute limit, CPU FE interpretation, and total allocation accounting are unchanged.
