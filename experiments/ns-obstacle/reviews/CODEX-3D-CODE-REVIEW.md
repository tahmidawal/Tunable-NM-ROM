# Independent Codex 3D code and proposed smoke review

Reviewer: read-only physics_reviewer, 2026-10-27. Reviewed lane485d4252f,
reference3d.py SHA256 e730963026228d869d0ff6a7ae138d1eb2f5c13ff0c11d6167558ee9fffc6ef7,
configs/g1-smoke.json, installer/lock, batch, symbolic verifier and implementation notes.

Verdict: READY for proposed diagnostic smoke only, conditional on reviewed2Dsmoke passing
and exact staging verification. No blocking scientific or implementation defect found.

Inspection verified pipe-minus-sphere CSG and labels; manufactured derivatives, nonzero
convection and divergence-free field; source grad(u)u+grad(p)-nu Laplace(u) and natural
traction; mixed signs and no extra gauge; Newton update and independent free residual;
finite-divergence gradient balances; reversed fluid normal for solidforce and weakreaction;
analytic manufactured sphereforce; pinned job-local --no-deps install with JAXunchanged;
partition/interpreter/f64/highest/unique directory/hard timeout; pre-factor memory screens.
Official API references independently consulted:
https://docu.ngsolve.org/latest/netgen_tutorials/define_3d_geometries.html
https://docu.ngsolve.org/nightly/how_to/symbolic_integrators.html

Conditions:
1. Staged INPUTS.sha256 must bind source/config/campaign/installer/lock and SOURCE_COMMIT.txt
   to reviewed bytes; these staging artifacts were not part of the reviewed checkpoint.
2. smoke_complete_not_reference_admission is literal execution completion only.
3. case_seconds_max is cooperative between native operations; externaltimeout and Slurm
   are the hard allocation boundary, not that check alone.
4. fe.quadrature_order is assembly BONUS order whereas diagnostic order is absolute;
   rename/document before production.

Signed curved Jacobian, pointwise maximum trace, reload parity and full convergence remain
G2 blockers, not first diagnostic-smoke blockers. No physicalsphere reference, training,
temporal validation or performance claim released. Reviewer made no edits or numerical runs.
